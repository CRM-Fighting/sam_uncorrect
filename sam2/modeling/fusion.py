import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Agent (学生: 预测权重) ---
class SamplingAgent(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, groups=in_channels // 2,
                      bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, f_ir, f_vis):
        x = torch.cat([f_ir, f_vis], dim=1)
        return torch.sigmoid(self.net(x))


# --- 2. Router (老师: 生成真值) ---
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, entropy_map):
        return torch.sigmoid(entropy_map * self.scale)


# --- 3. HyperNetwork (决策者: 决定修多少) ---
class HyperNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, weights):
        # 显著性分数：红蓝区域越多，分数越高 (权重越偏离0.5，信息量越大)
        salience_score = (weights - 0.5).abs().mean(dim=(2, 3))
        k = self.mlp(salience_score) * 0.8 + 0.1  # 限制在 10% ~ 90%
        return k


# --- 4. Mixer (精修工: 提升特征) ---
class MixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # need_weights=False 节省显存
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# --- 5. DynamicFusionModule (总控制器) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.agent = SamplingAgent(dim)
        self.router = Router()
        self.hypernet = HyperNetwork()
        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)
        self.MAX_TOKENS = 4096  # 显存安全阀

    def forward(self, f_ir, f_vis, gt_entropy=None):
        """
        f_ir, f_vis: [B, C, H, W]
        gt_entropy: [B, 1, H, W] (仅训练时提供)
        """
        B, C, H, W = f_ir.shape
        N = H * W

        # 1. Agent 预测权重
        pred_weights = self.agent(f_ir, f_vis)

        # 2. 训练/测试 分支
        if self.training and gt_entropy is not None:
            target_weights = self.router(gt_entropy)
            active_weights = target_weights
            # 蒸馏 Loss: 逼迫 Agent 模仿 Router
            aux_loss = F.mse_loss(pred_weights, target_weights.detach())
        else:
            active_weights = pred_weights
            aux_loss = torch.tensor(0.0, device=f_ir.device)

        # 3. 计算筛选比例 k
        k_ratios = self.hypernet(active_weights)
        avg_k = int(N * k_ratios.mean().item())
        avg_k = max(min(avg_k, self.MAX_TOKENS), 1)  # 安全截断

        # 4. 筛选 Top-k (基于显著性)
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        flat_weights = active_weights.flatten(2).transpose(1, 2)

        flat_salience = (flat_weights - 0.5).abs()
        _, topk_indices = torch.topk(flat_salience.squeeze(-1), avg_k, dim=1)
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, C)

        # 5. 分流与精修
        selected_ir = torch.gather(flat_ir, 1, gather_idx)
        selected_vis = torch.gather(flat_vis, 1, gather_idx)
        selected_w = torch.gather(flat_weights, 1, topk_indices.unsqueeze(-1))

        refined_ir = self.mixer_ir(selected_ir)
        refined_vis = self.mixer_vis(selected_vis)

        # 融合 Selected 部分
        fused_selected = refined_ir * selected_w + refined_vis * (1 - selected_w)

        # 6. 回填组合
        f_fused_flat = flat_ir + flat_vis  # 默认直接相加 (背景)
        f_fused_flat.scatter_(1, gather_idx, fused_selected)  # 覆盖前景

        f_final = f_fused_flat.transpose(1, 2).view(B, C, H, W)

        return f_final, aux_loss