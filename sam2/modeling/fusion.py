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


# --- 2. Mixer (稀疏精修模块) ---
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


# --- 3. DynamicFusionModule (主模块 - TCDG-Net 逻辑版) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 根据通道数推断 Stage 名称，用于文件名区分
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
        # 注意：移除了 Router 和 HyperNetwork
        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)

        # 可视化设置
        self.vis_count = 0
        self.vis_root = "vis_tcdg_logic_1_15"  # 文件夹名更新以示区分
        self.vis_train_dir = os.path.join(self.vis_root, "train")
        self.vis_val_dir = os.path.join(self.vis_root, "val")
        os.makedirs(self.vis_train_dir, exist_ok=True)
        os.makedirs(self.vis_val_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        device = f_ir.device

        # 0. 基础底座
        base_feat_sum = f_ir + f_vis

        # --- Phase 1: Agent 预测 (学生) ---
        agent_logits = self.agent(f_ir, f_vis)

        aux_loss = torch.tensor(0.0, device=device)
        selection_mask = None  # 最终用于 Mixer 的掩码
        teacher_mask_vis = None  # 用于可视化的老师掩码 (GT统计掩码)

        # --- Phase 2: 生成筛选标准 (TCDG-Net 统计逻辑) ---
        if self.training and gt_entropy is not None:
            # === 训练模式：使用统计规律作为 Teacher ===

            # [核心逻辑] TCDG-Net 图 4(b) 路径: Entropy -> GAP -> Threshold -> Mask
            # 1. 计算自适应阈值 T (GAP)
            # gt_entropy: [B, 1, H, W] -> threshold: [B, 1, 1, 1]
            # 计算每张图的全局平均熵值
            threshold = gt_entropy.mean(dim=(2, 3), keepdim=True)

            # 2. 生成二值掩码 (Screening)
            # 大于均值的区域为纹理 (1)，否则为内容 (0)
            teacher_mask = (gt_entropy > threshold).float()
            teacher_mask_vis = teacher_mask

            # 3. Agent Loss (蒸馏)
            # 强迫学生网络 (Agent) 学习这个统计分布
            aux_loss = F.binary_cross_entropy_with_logits(agent_logits, teacher_mask.detach())

            # 4. 确定 Mixer 使用的掩码
            # 训练时使用 Teacher Mask 指导 Mixer，保证只增强真正的纹理区域
            selection_mask = teacher_mask

        else:
            # === 推理模式：完全依赖学生 ===
            # 推理时没有 gt_entropy，使用 Agent 预测的结果
            selection_mask = (agent_logits > 0).float()
            teacher_mask_vis = None

        # --- Phase 3: 稀疏精修 (Sparse Refinement) ---
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()
        mask_flat = selection_mask.flatten(2).transpose(1, 2)

        for b in range(B):
            # 1. 获取索引
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)

            # 兜底机制：如果筛选出的点太少 (例如极端平滑的图)
            # 使用 Agent 的 Logits (软值) 选出 Top 64 个点，防止 Mixer 报错
            if indices.numel() < 64:
                logits_for_topk = agent_logits[b].flatten()
                _, indices = torch.topk(logits_for_topk, 64)

            # 2. 提取 (Gather)
            sel_ir = flat_ir[b, indices].unsqueeze(0)
            sel_vis = flat_vis[b, indices].unsqueeze(0)

            # 3. 精修 (Mixer)
            ref_ir = self.mixer_ir(sel_ir)
            ref_vis = self.mixer_vis(sel_vis)

            # 4. 融合
            # TCDG-Net 思想：只把增强后的纹理加回去，内容区域保持基座输出
            refined_sum = ref_ir + ref_vis

            # 5. 回填 (Scatter)
            final_canvas[b, indices] = refined_sum.squeeze(0)

        # --- 可视化 (每100步保存一次) ---
        if self.vis_count % 100 == 0:
            student_mask_vis = (agent_logits[0] > 0).float()
            self._visualize_comparison(
                gt_entropy[0] if gt_entropy is not None else None,
                teacher_mask_vis[0] if teacher_mask_vis is not None else None,
                student_mask_vis,
                H, W, device
            )
        self.vis_count += 1

        f_final = final_canvas.transpose(1, 2).view(B, C, H, W)
        return f_final, aux_loss

    def _visualize_comparison(self, entropy_map, teacher_mask, student_mask, H, W, device):
        try:
            mode = "train" if self.training else "val"
            save_dir = self.vis_train_dir if self.training else self.vis_val_dir
            vis_list = []

            # 1. 熵图 (Heatmap)
            if entropy_map is not None:
                # 简单防错：确保尺寸一致
                if entropy_map.shape[-1] != W:
                    e_show = F.interpolate(entropy_map.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze()
                else:
                    e_show = entropy_map.squeeze()

                e_data = e_show.detach().cpu().numpy()
                e_min, e_max = e_data.min(), e_data.max()
                # 归一化到 0-1
                if e_max - e_min > 1e-5:
                    e_norm = (e_data - e_min) / (e_max - e_min)
                else:
                    e_norm = e_data

                # BGR: White (255,255,255) -> Red (0,0,255)
                # 这种配色方案：值越小越白，值越大越红
                e_img_b = (255 * (1 - e_norm)).astype(np.uint8)
                e_img_g = (255 * (1 - e_norm)).astype(np.uint8)
                e_img_r = np.full_like(e_norm, 255, dtype=np.uint8)
                e_color = cv2.merge([e_img_b, e_img_g, e_img_r])
                vis_list.append(e_color)

            # 2. 老师/统计掩码 (BW)
            if teacher_mask is not None:
                t_data = teacher_mask.squeeze().detach().cpu().numpy()
                t_img = (t_data * 255).astype(np.uint8)
                t_img_bgr = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
                vis_list.append(t_img_bgr)

            # 3. 学生/Agent预测 (BW)
            if student_mask is not None:
                s_data = student_mask.squeeze().detach().cpu().numpy()
                s_img = (s_data * 255).astype(np.uint8)
                s_img_bgr = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
                vis_list.append(s_img_bgr)

            if vis_list:
                combined_img = np.hstack(vis_list)
                # 文件名包含 Stage 和 Step
                filename = f"{mode}_step{self.vis_count}_{self.stage_name}_compare.png"
                cv2.imwrite(os.path.join(save_dir, filename), combined_img)

        except Exception as e:
            print(f"Vis error: {e}")