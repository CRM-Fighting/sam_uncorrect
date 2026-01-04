import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np


# --- 1. Agent (预测权重) ---
class SamplingAgent(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, f_ir, f_vis):
        x = torch.cat([f_ir, f_vis], dim=1)
        return torch.sigmoid(self.net(x))


# --- 2. Router (教师) ---
class Router(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.constant_(self.net.bias, 0.0)

    def forward(self, entropy_map):
        return torch.sigmoid(self.net(entropy_map))


# --- 3. HyperNetwork (筛选器) ---
class HyperNetwork(nn.Module):
    def __init__(self, in_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, weights):
        # 找红蓝边缘 (0 或 1)
        importance = (weights - 0.5).abs()
        global_stat = importance.mean(dim=(2, 3)).view(-1, 1)
        k_ratio = self.mlp(global_stat)
        return k_ratio * 0.8 + 0.1

    # --- 4. Mixer (稀疏精修) ---


class MixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x


# --- 5. SparseReconstructionFusion (论文级命名) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.agent = SamplingAgent(dim)
        self.router = Router()
        self.hypernet = HyperNetwork()

        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)

        # 荧光笔增强系数
        self.highlight_scale = nn.Parameter(torch.tensor(1.0))

        # 可视化
        self.vis_count = 0
        self.vis_dir = "vis_reconstruction"
        os.makedirs(self.vis_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        N = H * W

        # 1. 准备全局背景特征 (Background Context)
        # 这是"填充"用的材料，对应简单的 Baseline 效果
        bg_context_feat = f_ir + f_vis

        # 2. Agent 预测权重
        weights = self.agent(f_ir, f_vis)  # [B, 1, H, W]

        aux_loss = torch.tensor(0.0, device=f_ir.device)
        if self.training and gt_entropy is not None:
            target = self.router(gt_entropy)
            aux_loss = F.mse_loss(weights, target.detach())

        # 3. 筛选逻辑：红蓝边缘
        score_map = (weights - 0.5).abs()
        k_ratio = self.hypernet(weights.detach())
        flat_score = score_map.flatten(2)  # [B, 1, N]

        # ★★★ 重构画布 (The Canvas) ★★★
        # 初始化一个全 0 的画布，我们后面要在上面"填空"
        # 这种写法在论文里叫 "Zero-initialized Feature Canvas"
        reconstructed_canvas = torch.zeros((B, C, N), device=f_ir.device)

        for b in range(B):
            # --- Step A: 确定哪些是前景(边缘)，哪些是背景 ---
            k = int(N * k_ratio[b].item())
            k = max(k, 64)

            # topk_idx: 选中的边缘索引 (Sparse Indices)
            _, topk_idx = torch.topk(flat_score[b], k, dim=1)  # [1, K]

            # --- Step B: 生成背景掩码 ---
            # 创建一个全1掩码，把选中的地方扣掉，剩下的就是背景
            # 这就是你说的 "剩余的像素点"
            all_indices = torch.arange(N, device=f_ir.device).unsqueeze(0)  # [1, N]
            # 为了简单起见，我们直接利用 scatter 的特性
            # 我们不需要显式获得背景索引，只需要用 scatter 覆盖即可

            # --- Step C: 处理前景 (Edge Processing Stream) ---
            idx_expanded = topk_idx.transpose(0, 1).expand(-1, C)  # [K, C]

            # 提取
            flat_ir_b = f_ir[b].flatten(1).transpose(0, 1)
            flat_vis_b = f_vis[b].flatten(1).transpose(0, 1)
            selected_ir = torch.gather(flat_ir_b, 0, idx_expanded).unsqueeze(0)
            selected_vis = torch.gather(flat_vis_b, 0, idx_expanded).unsqueeze(0)

            # 精修
            refined_ir = self.mixer_ir(selected_ir)
            refined_vis = self.mixer_vis(selected_vis)

            # 融合增量
            w_selected = torch.gather(weights[b].flatten(), 0, topk_idx.squeeze(0)).view(1, -1, 1)
            mixer_delta = refined_ir * w_selected + refined_vis * (1 - w_selected)

            # 荧光笔增强
            # 取出 Base 部分 (Base + Detail)
            flat_base_b = bg_context_feat[b].flatten(1).transpose(0, 1)  # [N, C]
            base_selected = torch.gather(flat_base_b, 0, idx_expanded).unsqueeze(0)

            score_selected = torch.gather(flat_score[b].flatten(), 0, topk_idx.squeeze(0)).view(1, -1, 1)
            modulation = 1.0 + score_selected * self.highlight_scale

            # 得到最终的边缘特征 (Enhanced Foreground Features)
            final_edge_feat = (base_selected + mixer_delta) * modulation

            # --- Step D: 处理背景 (Background Filling Stream) ---
            # 我们先假设全图都是背景
            final_flat_b = flat_base_b.clone()  # [N, C]

            # --- Step E: 填充重构 (In-painting / Filling) ---
            # ★★★ 你的核心需求在这里实现 ★★★
            # 逻辑：
            # 1. 画布上先铺满了背景 (Background Context)
            # 2. 我们把计算好的 "final_edge_feat" 强制覆盖(scatter)到对应的位置
            # 这相当于：先画边缘，再填背景（代码实现上是先铺背景再修边缘，效果完全等价，且效率更高）

            # 使用 scatter_ (注意没有add) 进行覆盖
            # 将增强后的边缘，填入到全图特征中
            final_flat_b.scatter_(0, idx_expanded, final_edge_feat.squeeze(0))

            # 放入 Batch 列表
            reconstructed_canvas[b] = final_flat_b.transpose(0, 1)

            # 可视化 (可选)
            if self.training and self.vis_count % 200 == 0 and b == 0:
                self._visualize_reconstruction(topk_idx, H, W, f_ir.device)
            if b == 0: self.vis_count += 1

        # 还原形状
        f_final = reconstructed_canvas.view(B, C, H, W)
        return f_final, aux_loss

    def _visualize_reconstruction(self, topk_idx, H, W, device):
        try:
            # 0为背景，255为边缘
            mask = torch.zeros((H * W), device=device)
            mask.scatter_(0, topk_idx.squeeze(0), 1.0)
            mask = mask.view(H, W).cpu().numpy()
            save_path = os.path.join(self.vis_dir, f"recon_mask_{self.vis_count}.png")
            cv2.imwrite(save_path, (mask * 255).astype(np.uint8))
        except:
            pass