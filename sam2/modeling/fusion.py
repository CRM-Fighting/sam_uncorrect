import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np


# --- 1. Agent (保留定义，防止报错，不参与计算) ---
class SamplingAgent(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, f_ir, f_vis):
        x = torch.cat([f_ir, f_vis], dim=1)
        logits = self.net(x)
        return logits


# --- 2. Cross-Mixer (跨模态交叉注意力) ---
class CrossMixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        # FFN 零初始化 (保留这个是为了训练稳定性，防止初始梯度爆炸)
        nn.init.constant_(self.ffn[-1].weight, 0)
        nn.init.constant_(self.ffn[-1].bias, 0)

    def forward(self, x_q, x_kv):
        q = self.norm_q(x_q)
        k = v = self.norm_kv(x_kv)

        attn_out, _ = self.attn(q, k, v, need_weights=False)

        x = x_q + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


# --- 3. DynamicFusionModule (主模块：去门控 + 去阈值) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        if dim == 96:
            self.stage_name = "Stage1"
        elif dim == 192:
            self.stage_name = "Stage2"
        elif dim == 384:
            self.stage_name = "Stage3"
        elif dim == 768:
            self.stage_name = "Stage4"
        else:
            self.stage_name = f"Dim{dim}"

        self.agent = SamplingAgent(dim)

        # Cross Mixer
        self.mixer_ir = CrossMixerBlock(dim, dropout=dropout)
        self.mixer_vis = CrossMixerBlock(dim, dropout=dropout)

        # [修改] 移除了 self.gamma (零初始化门控)

        # 可视化设置
        self.vis_count = 0
        self.vis_root = "vis_entropy_no_gate_no_limit_1_20"  # 更新文件夹名
        self.vis_train_dir = os.path.join(self.vis_root, "train")
        self.vis_val_dir = os.path.join(self.vis_root, "val")
        os.makedirs(self.vis_train_dir, exist_ok=True)
        os.makedirs(self.vis_val_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None, gt_entropy_vis=None, gt_entropy_ir=None):
        B, C, H, W = f_ir.shape
        device = f_ir.device

        # 0. 基础底座
        base_feat_sum = f_ir + f_vis

        # --- Phase 1: 熵图引导筛选 ---
        if gt_entropy is None:
            selection_mask = torch.zeros((B, 1, H, W), device=device)
        else:
            threshold = gt_entropy.mean(dim=(2, 3), keepdim=True)
            selection_mask = (gt_entropy > threshold).float()

        aux_loss = torch.tensor(0.0, device=device)

        # --- Phase 2: 跨模态交互精修 ---
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)

        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()
        mask_flat = selection_mask.flatten(2).transpose(1, 2)

        for b in range(B):
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)

            # [修改] 移除了 "if indices.numel() < 64" 的强制兜底逻辑
            # 改为：只要有选中的点就处理，没有就跳过 (保持原样)
            if indices.numel() > 0:
                sel_ir = flat_ir[b, indices].unsqueeze(0)
                sel_vis = flat_vis[b, indices].unsqueeze(0)

                # Cross Attention 互补增强
                enhanced_ir = self.mixer_ir(x_q=sel_ir, x_kv=sel_vis)
                enhanced_vis = self.mixer_vis(x_q=sel_vis, x_kv=sel_ir)

                refined_residual = (enhanced_ir - sel_ir) + (enhanced_vis - sel_vis)

                # [修改] 移除了 self.gamma，直接进行残差相加 (Standard Residual)
                final_canvas[b, indices] += refined_residual.squeeze(0)

        # --- Phase 3: 可视化 ---
        if self.vis_count % 200 == 0:
            self._visualize(
                gt_entropy_ir[0] if gt_entropy_ir is not None else None,
                gt_entropy_vis[0] if gt_entropy_vis is not None else None,
                gt_entropy[0] if gt_entropy is not None else None,
                selection_mask[0],
                H, W
            )
        self.vis_count += 1

        f_final = final_canvas.transpose(1, 2).view(B, C, H, W)
        return f_final, aux_loss

    def _to_heatmap(self, entropy_map, target_size):
        if entropy_map is None: return None

        if entropy_map.ndim == 3: entropy_map = entropy_map.squeeze(0)
        e_data = entropy_map.detach().cpu().numpy()

        e_min, e_max = e_data.min(), e_data.max()
        if e_max - e_min > 1e-5:
            e_norm = (e_data - e_min) / (e_max - e_min)
        else:
            e_norm = e_data

        pixel_val = (255 * (1 - e_norm)).astype(np.uint8)
        img_b = pixel_val
        img_g = pixel_val
        img_r = np.full_like(pixel_val, 255)

        e_color = cv2.merge([img_b, img_g, img_r])
        e_color = cv2.resize(e_color, target_size, interpolation=cv2.INTER_NEAREST)
        return e_color

    def _visualize(self, map_ir, map_vis, map_sum, selection_mask, H, W):
        try:
            mode = "train" if self.training else "val"
            save_dir = self.vis_train_dir if self.training else self.vis_val_dir
            target_vis_size = (256, 256)
            vis_list = []

            if map_ir is not None:
                vis_list.append(self._to_heatmap(map_ir, target_vis_size))

            if map_vis is not None:
                vis_list.append(self._to_heatmap(map_vis, target_vis_size))

            if map_sum is not None:
                vis_list.append(self._to_heatmap(map_sum, target_vis_size))

            if selection_mask is not None:
                mask_data = selection_mask.squeeze().detach().cpu().numpy()
                mask_vis = (255 * (1 - mask_data)).astype(np.uint8)
                mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                mask_bgr = cv2.resize(mask_bgr, target_vis_size, interpolation=cv2.INTER_NEAREST)
                vis_list.append(mask_bgr)

            if vis_list:
                combined = np.hstack(vis_list)
                filename = f"{mode}_step{self.vis_count}_{self.stage_name}.png"
                cv2.imwrite(os.path.join(save_dir, filename), combined)

        except Exception as e:
            print(f"Vis Error: {e}")