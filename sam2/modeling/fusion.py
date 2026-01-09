import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- 1. Agent (学生 / 辅助预测器) ---
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


# --- 2. Router (老师) ---
class Router(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 1x1 卷积等价于像素级的 Linear Layer
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1)

        # 初始化
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.constant_(self.net.bias, 0.0)

    def forward(self, entropy_map):
        return self.net(entropy_map)


# --- 3. HyperNetwork (动态阈值器) ---
class HyperNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, f_ir, f_vis):
        ctx_ir = f_ir.mean(dim=(2, 3))
        ctx_vis = f_vis.mean(dim=(2, 3))
        global_ctx = torch.cat([ctx_ir, ctx_vis], dim=1)
        threshold = self.mlp(global_ctx)
        return threshold  # [B, 1]


# --- 4. Mixer (稀疏精修模块) ---
class MixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        nn.init.constant_(self.ffn[-1].weight, 0)
        nn.init.constant_(self.ffn[-1].bias, 0)

    def forward(self, x):
        x_norm = self.norm(x)
        # 显存优化
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x


# --- 5. DynamicFusionModule (主模块) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # --- [修改 1] 根据通道数推断 Stage 名称，用于文件名区分 ---
        # Hiera-Tiny 典型通道配置: 96, 192, 384, 768
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
        self.router = Router(in_channels=1)
        self.hypernet = HyperNetwork(in_dim=dim * 2)
        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)

        # --- [修改 2] 更新可视化根目录为 vis_1_9 ---
        self.vis_count = 0
        self.vis_root = "vis_1_9"  # 修改文件夹名
        self.vis_train_dir = os.path.join(self.vis_root, "train")
        self.vis_val_dir = os.path.join(self.vis_root, "val")
        os.makedirs(self.vis_train_dir, exist_ok=True)
        os.makedirs(self.vis_val_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        N = H * W
        device = f_ir.device

        # 0. 基础底座
        base_feat_sum = f_ir + f_vis

        # --- Phase 1: Agent 预测 ---
        agent_logits = self.agent(f_ir, f_vis)
        aux_loss = torch.tensor(0.0, device=device)
        selection_mask_ste = None
        hard_mask_vis = None

        # --- Phase 2: 生成筛选标准 ---
        if self.training and gt_entropy is not None:
            # === 训练模式 ===
            driver_logits = self.router(gt_entropy)
            threshold = self.hypernet(f_ir, f_vis).view(B, 1, 1, 1)

            diff = driver_logits - threshold

            # 硬掩码 (Forward)
            hard_mask = (diff > 0).float()
            hard_mask_vis = hard_mask

            # 软掩码 (Backward)
            soft_mask = torch.sigmoid(diff * 5.0)

            # STE
            selection_mask_ste = hard_mask + (soft_mask - soft_mask.detach())

            # Agent Loss
            aux_loss = F.binary_cross_entropy_with_logits(agent_logits, hard_mask.detach())

        else:
            # === 推理模式 ===
            selection_mask_ste = (agent_logits > 0).float()
            hard_mask_vis = None

        # --- Phase 3: 稀疏精修 ---
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()
        mask_flat = selection_mask_ste.flatten(2).transpose(1, 2)

        for b in range(B):
            # 1. 获取索引
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)

            # 兜底机制
            if indices.numel() < 64:
                if self.training and gt_entropy is not None:
                    logits_for_topk = driver_logits[b].flatten()
                else:
                    logits_for_topk = agent_logits[b].flatten()
                _, indices = torch.topk(logits_for_topk, 64)

            # 2. 提取
            sel_ir = flat_ir[b, indices].unsqueeze(0)
            sel_vis = flat_vis[b, indices].unsqueeze(0)

            # 3. 精修
            ref_ir = self.mixer_ir(sel_ir)
            ref_vis = self.mixer_vis(sel_vis)

            # 4. 梯度桥接
            m_ste = mask_flat[b, indices].view(1, -1, 1)

            # 5. 融合
            refined_sum = (ref_ir + ref_vis) * m_ste

            # 6. 回填
            final_canvas[b, indices] = refined_sum.squeeze(0)

        # --- [修改 3] 可视化逻辑修改 ---
        # 改为每 100 步保存一次
        if self.vis_count % 100 == 0:
            student_mask_vis = (agent_logits[0] > 0).float()
            self._visualize_comparison(
                gt_entropy[0] if gt_entropy is not None else None,
                hard_mask_vis[0] if hard_mask_vis is not None else None,
                student_mask_vis,
                H, W, device
            )

        # [关键修复] 计数器必须在 if 外面增加，否则会卡在 1
        self.vis_count += 1

        f_final = final_canvas.transpose(1, 2).view(B, C, H, W)
        return f_final, aux_loss

    def _visualize_comparison(self, entropy_map, teacher_mask, student_mask, H, W, device):
        try:
            mode = "train" if self.training else "val"
            save_dir = self.vis_train_dir if self.training else self.vis_val_dir
            vis_list = []

            # 1. 熵加图 (Heatmap) - Router输入
            if entropy_map is not None:
                e_data = entropy_map.squeeze().detach().cpu().numpy()
                e_min, e_max = e_data.min(), e_data.max()
                if e_max - e_min > 1e-5:
                    e_norm = (e_data - e_min) / (e_max - e_min)
                else:
                    e_norm = e_data

                # BGR: White (255,255,255) -> Red (0,0,255)
                e_img_b = (255 * (1 - e_norm)).astype(np.uint8)
                e_img_g = (255 * (1 - e_norm)).astype(np.uint8)
                e_img_r = np.full_like(e_norm, 255, dtype=np.uint8)
                e_color = cv2.merge([e_img_b, e_img_g, e_img_r])
                vis_list.append(e_color)

            # 2. 老师/超网络选择 (BW) - GT Mask
            if teacher_mask is not None:
                t_data = teacher_mask.squeeze().detach().cpu().numpy()
                t_img = (t_data * 255).astype(np.uint8)
                t_img_bgr = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
                vis_list.append(t_img_bgr)

            # 3. 学生/Agent选择 (BW) - Pred Mask
            if student_mask is not None:
                s_data = student_mask.squeeze().detach().cpu().numpy()
                s_img = (s_data * 255).astype(np.uint8)
                s_img_bgr = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
                vis_list.append(s_img_bgr)

            if vis_list:
                combined_img = np.hstack(vis_list)
                # --- [修改 4] 命名包含 Stage 和 Step ---
                # 格式: {mode}_step{count}_{StageName}_compare.png
                # 例如: train_step100_Stage1_compare.png
                filename = f"{mode}_step{self.vis_count}_{self.stage_name}_compare.png"
                cv2.imwrite(os.path.join(save_dir, filename), combined_img)

        except Exception as e:
            print(f"Vis error: {e}")