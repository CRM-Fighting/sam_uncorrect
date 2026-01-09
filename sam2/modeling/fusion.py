import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- 1. Agent (学生 / 辅助预测器) ---
# 作用: 测试时替代 Router，独立判断。
class SamplingAgent(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
            # 输出 Logits，不加 Sigmoid
        )

    def forward(self, f_ir, f_vis):
        x = torch.cat([f_ir, f_vis], dim=1)
        logits = self.net(x)
        return logits


# --- 2. Router (老师) ---
# ★★★ 结构已改为线性层 (1x1 Conv) ★★★
# 作用: 训练时根据熵图打分。
class Router(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 1x1 卷积等价于像素级的 Linear Layer (Wx + b)
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1)

        # 初始化
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.constant_(self.net.bias, 0.0)

    def forward(self, entropy_map):
        return self.net(entropy_map)


# --- 3. HyperNetwork (动态阈值器) ---
# 作用: 根据全局双模态特征，决定阈值。
class HyperNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        # in_dim = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, 1)
            # 输出 Logits 阈值
        )

    def forward(self, f_ir, f_vis):
        # 分别计算全局均值，保留模态独立性
        ctx_ir = f_ir.mean(dim=(2, 3))
        ctx_vis = f_vis.mean(dim=(2, 3))

        # 拼接 (Concatenation)
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
        # 零初始化
        nn.init.constant_(self.ffn[-1].weight, 0)
        nn.init.constant_(self.ffn[-1].bias, 0)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x


# --- 5. DynamicFusionModule (主模块) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.agent = SamplingAgent(dim)

        # 线性 Router
        self.router = Router(in_channels=1)

        # HyperNet 输入翻倍
        self.hypernet = HyperNetwork(in_dim=dim * 2)

        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)

        # 可视化配置
        self.vis_count = 0
        self.vis_root = "vis_reconstruction"
        self.vis_train_dir = os.path.join(self.vis_root, "train")
        self.vis_val_dir = os.path.join(self.vis_root, "val")
        os.makedirs(self.vis_train_dir, exist_ok=True)
        os.makedirs(self.vis_val_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        N = H * W
        device = f_ir.device

        # 0. 基础底座 (Background)
        # 未选中的点保留此值 (Residual Connection)
        base_feat_sum = f_ir + f_vis

        # --- Phase 1: Agent 预测 ---
        agent_logits = self.agent(f_ir, f_vis)
        aux_loss = torch.tensor(0.0, device=device)

        # 最终用于计算的 mask (带梯度)
        selection_mask_ste = None

        # 用于可视化的变量
        hard_mask_vis = None

        # --- Phase 2: 生成筛选标准 ---
        if self.training and gt_entropy is not None:
            # === 训练模式 ===

            # 1. Router 打分 (Pixel-level)
            driver_logits = self.router(gt_entropy)

            # 2. HyperNet 定阈值 (Image-level)
            threshold = self.hypernet(f_ir, f_vis).view(B, 1, 1, 1)

            # 3. 比较: Logits - Threshold
            diff = driver_logits - threshold

            # 4. 硬掩码 (Forward用)
            hard_mask = (diff > 0).float()
            hard_mask_vis = hard_mask  # 存下来画图用

            # 5. 软掩码 (Backward用)
            # 温度系数 5.0 可调，越大梯度越陡
            soft_mask = torch.sigmoid(diff * 5.0)

            # ★★★ STE (直通估计器) ★★★
            # 实现了: 前向硬筛选，反向传梯度给 Router 和 HyperNet
            selection_mask_ste = hard_mask + (soft_mask - soft_mask.detach())

            # 6. Agent 模仿 Loss
            aux_loss = F.binary_cross_entropy_with_logits(agent_logits, hard_mask.detach())

        else:
            # === 推理模式 ===
            # Agent 独立判断，硬截断
            selection_mask_ste = (agent_logits > 0).float()
            hard_mask_vis = None  # 推理时没有 GT mask

        # --- Phase 3: 稀疏精修 (Sparse Processing) ---

        flat_ir = f_ir.flatten(2).transpose(1, 2)  # [B, N, C]
        flat_vis = f_vis.flatten(2).transpose(1, 2)

        # 初始画布 = 基础底座 (未选中点保留原样)
        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()

        # 将 Mask 拉平
        mask_flat = selection_mask_ste.flatten(2).transpose(1, 2)  # [B, N, 1]

        for b in range(B):
            # 1. 获取索引 (用于 Gather 省算力)
            # 以前向传播的硬值 (0或1) 为准
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)

            # 兜底机制 (防止空选导致报错)
            if indices.numel() < 64:
                if self.training and gt_entropy is not None:
                    logits_for_topk = driver_logits[b].flatten()
                else:
                    logits_for_topk = agent_logits[b].flatten()
                _, indices = torch.topk(logits_for_topk, 64)

            # 2. 提取特征
            sel_ir = flat_ir[b, indices].unsqueeze(0)
            sel_vis = flat_vis[b, indices].unsqueeze(0)

            # 3. 分别精修
            ref_ir = self.mixer_ir(sel_ir)
            ref_vis = self.mixer_vis(sel_vis)

            # 4. ★★★ 梯度桥接 ★★★
            # 乘上 STE Mask (数值为1.0，但带着梯度)
            m_ste = mask_flat[b, indices].view(1, -1, 1)

            # 5. 纯相加融合 (Selected Points)
            # 选中点 = 精修IR + 精修Vis
            refined_sum = (ref_ir + ref_vis) * m_ste

            # 6. 回填 (Replacement)
            # 选中点被替换，未选中点保持 base_feat_sum
            final_canvas[b, indices] = refined_sum.squeeze(0)

        # --- 可视化 (训练时对比三者) ---
        if self.vis_count % 200 == 0:
            # 取 batch 中第一张图
            student_mask_vis = (agent_logits[0] > 0).float()

            self._visualize_comparison(
                gt_entropy[0] if gt_entropy is not None else None,
                hard_mask_vis[0] if hard_mask_vis is not None else None,
                student_mask_vis,
                H, W, device
            )
            self.vis_count += 1

        # 恢复维度
        f_final = final_canvas.transpose(1, 2).view(B, C, H, W)
        return f_final, aux_loss

    def _visualize_comparison(self, entropy_map, teacher_mask, student_mask, H, W, device):
        try:
            mode = "train" if self.training else "val"
            save_dir = self.vis_train_dir if self.training else self.vis_val_dir

            vis_list = []

            # 1. 熵加图 (Heatmap)
            if entropy_map is not None:
                e_data = entropy_map.squeeze().detach().cpu().numpy()
                e_min, e_max = e_data.min(), e_data.max()
                if e_max - e_min > 1e-5:
                    e_norm = (e_data - e_min) / (e_max - e_min)
                else:
                    e_norm = e_data
                e_img = (e_norm * 255).astype(np.uint8)
                e_color = cv2.applyColorMap(e_img, cv2.COLORMAP_INFERNO)
                vis_list.append(e_color)

            # 2. 老师选择 (BW)
            if teacher_mask is not None:
                t_data = teacher_mask.squeeze().detach().cpu().numpy()
                t_img = (t_data * 255).astype(np.uint8)
                t_img_bgr = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
                vis_list.append(t_img_bgr)

            # 3. 学生选择 (BW)
            if student_mask is not None:
                s_data = student_mask.squeeze().detach().cpu().numpy()
                s_img = (s_data * 255).astype(np.uint8)
                s_img_bgr = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
                vis_list.append(s_img_bgr)

            if vis_list:
                # 水平拼接
                combined_img = np.hstack(vis_list)
                cv2.imwrite(os.path.join(save_dir, f"{mode}_step{self.vis_count}_compare.png"), combined_img)

        except Exception as e:
            print(f"Vis error: {e}")