import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np


# --- 1. Agent (保持不变) ---
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


# --- 2. Mixer (保持不变，确保有 Dropout) ---
class MixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Attention 中开启 dropout
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        nn.init.constant_(self.ffn[-1].weight, 0)
        nn.init.constant_(self.ffn[-1].bias, 0)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x


# --- 3. DynamicFusionModule (核心修改版) ---
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
        self.mixer_ir = MixerBlock(dim, dropout=dropout)
        self.mixer_vis = MixerBlock(dim, dropout=dropout)

        # 可视化设置
        self.vis_count = 0
        self.vis_root = "vis_debug_gap_fix"  # 改名，标志着我们正在修复 Gap
        self.vis_train_dir = os.path.join(self.vis_root, "train")
        self.vis_val_dir = os.path.join(self.vis_root, "val")
        os.makedirs(self.vis_train_dir, exist_ok=True)
        os.makedirs(self.vis_val_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        device = f_ir.device

        # 0. 基础底座 (Baseline)
        base_feat_sum = f_ir + f_vis

        # --- Phase 1: Agent 预测 ---
        # Agent 必须学会自己看图说话
        agent_logits = self.agent(f_ir, f_vis)  # shape: [B, 1, H, W]

        aux_loss = torch.tensor(0.0, device=device)
        teacher_mask_vis = None

        # --- Phase 2: 训练时的监督 ---
        if self.training and gt_entropy is not None:
            # 1. 准备 Teacher (仅用于计算 Loss)
            threshold = gt_entropy.mean(dim=(2, 3), keepdim=True)
            teacher_mask = (gt_entropy > threshold).float()
            teacher_mask_vis = teacher_mask

            # 2. 计算蒸馏 Loss
            aux_loss = F.binary_cross_entropy_with_logits(agent_logits, teacher_mask.detach())
        else:
            teacher_mask_vis = None

        # --- Phase 3: 筛选 Mask (核心修改点) ---
        # ★★★ 无论训练还是推理，统一使用 Agent 的预测结果 ★★★
        # 这样 Mixer 就能适应 Agent 的不完美，消除 Training-Inference Gap
        # 使用 sigmoid + 0.5 阈值，这等价于 logits > 0
        selection_mask = (agent_logits > 0).float()

        # [可选优化] 训练初期 Agent 太烂怎么办？
        # 如果担心初期崩盘，可以保留一个“保底机制”：
        # selection_mask = teacher_mask if (self.training and torch.rand(1)<0.5) else (agent_logits>0).float()
        # 但为了复现 70% 的基线，我们先相信 Agent 即使全错，Mixer 也能学会“啥也不干”（输出0），
        # 从而退化回 base_feat_sum (即相加融合)，保证下限不低于 70%。

        # --- Phase 4: 稀疏精修 ---
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()
        mask_flat = selection_mask.flatten(2).transpose(1, 2)

        for b in range(B):
            # 找出 Agent 认为需要精修的点
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)

            # 兜底：如果 Agent 啥也没选到 (点太少)，或者选太多(显存爆)，这里只处理太少的情况
            if indices.numel() < 64:
                # 即使 Agent 觉得全是背景，我们也强行选几个点跑一下，防止 Mixer 梯度断掉
                logits_for_topk = agent_logits[b].flatten()
                _, indices = torch.topk(logits_for_topk, 64)

            sel_ir = flat_ir[b, indices].unsqueeze(0)
            sel_vis = flat_vis[b, indices].unsqueeze(0)

            ref_ir = self.mixer_ir(sel_ir)
            ref_vis = self.mixer_vis(sel_vis)

            refined_sum = ref_ir + ref_vis

            # 这里的 refined_sum 是“增量”。
            # 如果 Mixer 学会了输出 0，那么 final_canvas 就等于 base_feat_sum (相加融合)
            final_canvas[b, indices] = refined_sum.squeeze(0)

        # --- 可视化 ---
        if self.vis_count % 200 == 0:
            student_mask_vis = (agent_logits > 0).float()
            self._visualize_comparison(
                gt_entropy[0] if gt_entropy is not None else None,
                teacher_mask_vis[0] if teacher_mask_vis is not None else None,
                student_mask_vis[0],
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
            target_vis_size = (256, 256)

            if entropy_map is not None:
                if entropy_map.ndim == 3: entropy_map = entropy_map.squeeze(0)
                e_data = entropy_map.detach().cpu().numpy()
                e_min, e_max = e_data.min(), e_data.max()
                if e_max - e_min > 1e-5:
                    e_norm = (e_data - e_min) / (e_max - e_min)
                else:
                    e_norm = e_data
                e_img = (e_norm * 255).astype(np.uint8)
                e_img = cv2.applyColorMap(e_img, cv2.COLORMAP_JET)
                e_img = cv2.resize(e_img, target_vis_size, interpolation=cv2.INTER_NEAREST)
                vis_list.append(e_img)

            if teacher_mask is not None:
                t_data = teacher_mask.squeeze().detach().cpu().numpy()
                t_img = (t_data * 255).astype(np.uint8)
                t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
                t_img = cv2.resize(t_img, target_vis_size, interpolation=cv2.INTER_NEAREST)
                vis_list.append(t_img)

            if student_mask is not None:
                s_data = student_mask.squeeze().detach().cpu().numpy()
                s_img = (s_data * 255).astype(np.uint8)
                s_img = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
                s_img = cv2.resize(s_img, target_vis_size, interpolation=cv2.INTER_NEAREST)
                vis_list.append(s_img)

            if vis_list:
                combined_img = np.hstack(vis_list)
                filename = f"{mode}_step{self.vis_count}_{self.stage_name}.png"
                cv2.imwrite(os.path.join(save_dir, filename), combined_img)
        except Exception as e:
            print(f"Vis error: {e}")