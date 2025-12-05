import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """
    单个专家网络。
    使用 1x1 卷积保持空间尺寸，结构类似 MLP: (B, C, H, W) -> Hidden -> (B, C, H, W)
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


class TopKRouter(nn.Module):
    """
    门控网络 (Gating Network)
    输入特征图，输出每个像素应该激活哪几个专家。
    """

    def __init__(self, dim, num_experts, active_experts):
        super().__init__()
        self.gate = nn.Conv2d(dim, num_experts, kernel_size=1)
        self.active_experts = active_experts

    def forward(self, x):
        # x: (B, C, H, W) -> logits: (B, num_experts, H, W)
        logits = self.gate(x)

        # 计算概率
        scores = F.softmax(logits, dim=1)

        # 选出 Top-K
        # topk_probs: (B, k, H, W)
        # topk_indices: (B, k, H, W)
        topk_probs, topk_indices = torch.topk(scores, self.active_experts, dim=1)

        # 归一化选中的权重
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

        return topk_probs, topk_indices, scores


class SparseMoEBlock(nn.Module):
    """
    稀疏混合专家模块
    """

    def __init__(self, dim, num_experts=11, active_experts=6):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts

        self.router = TopKRouter(dim, num_experts, active_experts)
        self.experts = nn.ModuleList([ExpertLayer(dim) for _ in range(num_experts)])

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 路由选择
        routing_weights, selected_indices, all_probs = self.router(x)

        # 2. 专家计算 (为适应全卷积，采用加权叠加的方式)
        final_output = torch.zeros_like(x)

        # 构建 Mask: (B, num_experts, H, W)
        expert_mask = torch.zeros(B, self.num_experts, H, W, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, selected_indices, routing_weights)

        # 遍历专家
        for i, expert in enumerate(self.experts):
            # 获取该专家的权重图
            weight = expert_mask[:, i:i + 1, :, :]

            # 只有当该专家被选中时(权重>0)才计算，或者全部计算后乘权重
            # 为了代码简洁和避免 shape 问题，这里全部 forward
            if weight.sum() > 0:  # 简单的稀疏跳过
                final_output += weight * expert(x)

        # 3. 计算负载均衡损失 (Auxiliary Loss)
        # 目标：让每个专家被选中的概率尽可能平均，避免某些专家“累死”，某些“闲死”
        # mean_prob: 门控输出的平均概率 (Importance)
        # mean_load: 实际选中的平均频率 (Load)

        mean_prob = all_probs.mean(dim=(0, 2, 3))  # (num_experts,)

        one_hot = torch.zeros_like(all_probs)
        one_hot.scatter_(1, selected_indices, 1.0)
        mean_load = one_hot.mean(dim=(0, 2, 3))

        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        # 残差连接
        return x + final_output, aux_loss