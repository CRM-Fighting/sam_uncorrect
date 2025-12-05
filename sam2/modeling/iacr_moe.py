import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """
    专家层：保持与上一版 MoE 完全一致，以控制变量。
    使用 1x1 卷积 (等价于 Linear)。
    """

    def __init__(self, dim, hidden_ratio=4):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class InteractionAwareRouter(nn.Module):
    """
    【论文创新点】交互感知协作路由 (IACR)
    1. 生成全局上下文 Context
    2. 专家原型与 Context 进行 Self-Attention 交互
    3. 基于更新后的原型进行路由决策
    """

    def __init__(self, input_dim, num_experts, active_experts, prototype_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.prototype_dim = prototype_dim

        # 1. 专家原型 (可学习参数)
        # 初始化为正交矩阵更有利于收敛
        self.prototypes = nn.Parameter(torch.randn(num_experts, prototype_dim))    # 形状(11,64)
        nn.init.orthogonal_(self.prototypes)  # 正交初始化：避免原型初始冗余，加速收敛

        # 2. 投影层
        self.context_proj = nn.Linear(input_dim, prototype_dim)  # 压缩全局特征
        self.input_proj = nn.Conv2d(input_dim, prototype_dim, kernel_size=1)  # 压缩局部特征用于匹配

        # 3. 交互模块 (轻量级 MHSA)
        # batch_first=True: (Batch, Seq, Dim)
        self.interaction_attn = nn.MultiheadAttention(embed_dim=prototype_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(prototype_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # --- Step 1: 构建交互上下文 ---
        # 全局平均池化 GAP: (B, C, H, W) -> (B, C)
        global_context = x.mean(dim=(2, 3))
        # 投影: (B, proto_dim) -> (B, 1, proto_dim)
        global_context = self.context_proj(global_context).unsqueeze(1)

        # 扩展原型以匹配 Batch: (B, num_experts, proto_dim)
        protos = self.prototypes.unsqueeze(0).expand(B, -1, -1)

        # 拼接: [Context, P1, P2, ... Pn] -> (B, 1+N, proto_dim)
        interaction_seq = torch.cat([global_context, protos], dim=1)

        # --- Step 2: 专家交互 (Communication) ---
        # Self-Attention 让专家原型根据 Context 和彼此进行调整
        # Q=K=V=interaction_seq
        interacted_seq, _ = self.interaction_attn(interaction_seq, interaction_seq, interaction_seq)

        # 残差连接 + Norm
        interacted_seq = self.norm(interacted_seq + interaction_seq)

        # 提取更新后的专家原型 (去掉第一个 Context Token)
        # updated_protos: (B, num_experts, proto_dim)
        updated_protos = interacted_seq[:, 1:, :]

        # --- Step 3: 路由决策 ---
        # 将输入特征映射到原型空间: (B, proto_dim, H, W)
        x_routed = self.input_proj(x)

        # 变形以便进行点积: (B, H*W, proto_dim)
        x_flat = x_routed.flatten(2).transpose(1, 2)

        # 计算相似度 Logits
        # (B, HW, D) @ (B, D, N) -> (B, HW, N) -> (B, N, HW) -> (B, N, H, W)
        # updated_protos.transpose(1, 2) 形状为 (B, proto_dim, num_experts)
        logits = torch.bmm(x_flat, updated_protos.transpose(1, 2))
        logits = logits.transpose(1, 2).view(B, self.num_experts, H, W)

        # 归一化因子 (Scaling)
        logits = logits / (self.prototype_dim ** 0.5)

        if self.training:
            # 噪声强度可以调节，通常 1.0 / num_experts
            noise = torch.randn_like(logits) * (1.0 / self.num_experts)
            logits = logits + noise

        # Softmax 计算概率
        router_probs = F.softmax(logits, dim=1)

        # Top-K 选择
        topk_probs, topk_indices = torch.topk(router_probs, self.active_experts, dim=1)

        # 重新归一化权重
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

        return topk_probs, topk_indices, router_probs


class IACRMoEBlock(nn.Module):
    def __init__(self, dim, num_experts=11, active_experts=6):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts

        # 使用新的交互路由器
        self.router = InteractionAwareRouter(dim, num_experts, active_experts)

        # 专家层保持不变 (控制变量)
        self.experts = nn.ModuleList([ExpertLayer(dim) for _ in range(num_experts)])

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 路由
        routing_weights, selected_indices, all_probs = self.router(x)

        # 2. 专家计算 (加权叠加)
        final_output = torch.zeros_like(x)
        expert_mask = torch.zeros(B, self.num_experts, H, W, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, selected_indices, routing_weights)

        for i, expert in enumerate(self.experts):
            weight = expert_mask[:, i:i + 1, :, :]
            if weight.sum() > 0:
                final_output += weight * expert(x)

        # (A) 负载均衡损失 (Load Balancing)
        mean_prob = all_probs.mean(dim=(0, 2, 3))
        one_hot = torch.zeros_like(all_probs)
        one_hot.scatter_(1, selected_indices, 1.0)
        mean_load = one_hot.mean(dim=(0, 2, 3))
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        # (B) 原型正交损失 (Prototype Orthogonality Loss)
        # 目的：让初始的专家尽可能不同，覆盖不同语义
        # P @ P.T 应该接近 I (单位矩阵)
        prototypes = self.router.prototypes  # (N, D)
        # 归一化原型向量
        protos_norm = F.normalize(prototypes, dim=1)
        # 计算相关矩阵 (N, N)
        correlation = torch.mm(protos_norm, protos_norm.t())
        # 减去单位矩阵
        identity = torch.eye(self.num_experts, device=x.device)
        ortho_loss = torch.norm(correlation - identity, p='fro')

        # 总辅助损失 = 负载均衡 + 正交约束
        # 正交损失权重通常给小一点
        total_aux_loss = aux_loss + 0.5 * ortho_loss

        return x + final_output, total_aux_loss