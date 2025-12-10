import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """ 标准专家层 (用于共享专家) """

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


class GlobalGuidedAoERouter(nn.Module):
    """
    【最终修正版】Global Guided AoE Router
    包含防塌陷机制：Tanh限制 + 强噪声
    """

    def __init__(self, d_model, num_experts, d_low, topk=3, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_low = d_low
        self.topk = topk
        self.d_model = d_model

        # 1. AoE 特征生成
        self.w_down = nn.Linear(d_model, num_experts * d_low, bias=False)

        # 2. 交互感知模块
        # 【关键修复】维度改为 3D: [1, N, d_low]，确保与 B*T (3D) 兼容
        self.expert_pos_embed = nn.Parameter(torch.randn(1, num_experts, d_low))
        self.global_proj = nn.Linear(d_model, d_low)

        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=d_low, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_low)

        # 3. 双路门控 (Dual-Path Gating)
        self.local_scorer = nn.Linear(d_low, 1)

        self.global_gate_mlp = nn.Sequential(
            nn.Linear(d_low, d_low * 2),
            nn.GELU(),
            nn.Linear(d_low * 2, num_experts)
        )

        # 4. 执行
        self.act = nn.GELU()
        self.w_up = nn.Parameter(torch.randn(num_experts, d_low, d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_down.weight, std=0.02)
        nn.init.normal_(self.expert_pos_embed, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.local_scorer.weight, std=0.02)

    def _sparse_w_up(self, feats, indices):
        selected_w_up = self.w_up[indices]
        output = torch.matmul(feats.unsqueeze(2), selected_w_up).squeeze(2)
        return output

    def forward(self, x):
        """ x: [Batch, Tokens, d_model] """
        batch_size, num_tokens, d_model = x.shape
        # 使用 reshape 避免不连续内存报错
        x_flat = x.reshape(-1, d_model)

        # Step 1: AoE 生成
        all_expert_feats_flat = self.w_down(x_flat)
        expert_feats = all_expert_feats_flat.view(-1, self.num_experts, self.d_low)

        # Step 2: 交互 (Interaction)
        global_ctx_raw = x.mean(dim=1)
        global_ctx = self.global_proj(global_ctx_raw)

        # [B, d_low] -> [B*T, 1, d_low]
        global_ctx_expanded = global_ctx.unsqueeze(1).expand(batch_size, num_tokens, -1).reshape(-1, 1, self.d_low)

        # [B*T, N, d_low] + [1, N, d_low] -> [B*T, N, d_low] (自动广播，保持3D)
        expert_feats_with_pos = expert_feats + self.expert_pos_embed

        # 拼接 3D 张量 -> [B*T, 1+N, d_low]
        # 这里 input 都是 3D 的，不会报 "got 3 and 4" 的错
        interaction_seq = torch.cat([global_ctx_expanded, expert_feats_with_pos], dim=1)

        interacted_seq, _ = self.interaction_attn(interaction_seq, interaction_seq, interaction_seq)
        interacted_seq = self.norm(interacted_seq + interaction_seq)

        # 取出更新后的特征 (去掉第0个Context)
        refined_feats = interacted_seq[:, 1:, :]

        # Step 3: 全局引导决策 (Dual-Path)

        # 路径 A (Local)
        local_logits = self.local_scorer(refined_feats).squeeze(-1)

        # 路径 B (Global)
        global_bias = self.global_gate_mlp(global_ctx)

        # 【关键防塌陷 1】限制 Global Bias 幅度，防止它“独裁”
        global_bias = torch.tanh(global_bias) * 2.0

        global_bias_expanded = global_bias.unsqueeze(1).expand(-1, num_tokens, -1).reshape(-1, self.num_experts)

        final_logits = local_logits + global_bias_expanded

        # 【关键防塌陷 2】加大训练噪声到 1.0 (强迫 Router 探索)
        if self.training:
            final_logits = final_logits + torch.randn_like(final_logits) * 1.0

        router_probs = F.softmax(final_logits, dim=-1)

        # Step 4: 执行
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_low)
        selected_feats = torch.gather(expert_feats, 1, gather_indices)

        activated_feats = self.act(selected_feats)
        expert_outputs = self._sparse_w_up(activated_feats, topk_indices)

        weighted_outputs = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        # Aux Loss
        mean_prob = router_probs.mean(dim=0)
        one_hot = torch.zeros_like(router_probs).scatter_(1, topk_indices, 1.0)
        mean_load = one_hot.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        return weighted_outputs.reshape(batch_size, num_tokens, d_model), aux_loss


class SharedGlobalGuidedAoEBlock(nn.Module):
    """
    【最终版 Block】Shared Expert + Global Guided AoE
    """

    def __init__(self, dim, num_experts=8, active_experts=3):
        super().__init__()

        # 1. 共享专家
        self.shared_expert = ExpertLayer(dim)

        # 2. 路由专家
        self.moe = GlobalGuidedAoERouter(
            d_model=dim,
            num_experts=num_experts,
            d_low=dim // 4,
            topk=active_experts
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # A. 共享路径
        shared_out = self.shared_expert(x)

        # B. 路由路径
        x_tokens = x.flatten(2).transpose(1, 2)
        moe_out, aux_loss = self.moe(x_tokens)
        moe_out = moe_out.transpose(1, 2).reshape(B, C, H, W)

        # C. 融合
        return x + shared_out + moe_out, aux_loss