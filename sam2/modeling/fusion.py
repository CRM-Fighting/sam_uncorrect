import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- 1. Agent (学生: 负责在测试时模仿老师) ---
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
        logits = self.net(x)
        return torch.sigmoid(logits), logits


# --- 2. Router (老师: 训练时利用 GT 指导全场) ---
class Router(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.constant_(self.net.bias, 0.0)

    def forward(self, entropy_map):
        return torch.sigmoid(self.net(entropy_map))


# --- 3. HyperNetwork (动态阈值器) ---
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
        # 统计全局信息
        global_stat = weights.mean(dim=(2, 3)).view(-1, 1)
        threshold = self.mlp(global_stat)
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

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x


# --- 5. DynamicFusionModule ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.agent = SamplingAgent(dim)
        self.router = Router()
        self.hypernet = HyperNetwork()

        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)

        self.highlight_scale = nn.Parameter(torch.tensor(1.0))

        # --- 可视化配置 (修改部分) ---
        self.vis_count = 0
        self.vis_root = "vis_reconstruction"

        # 分别定义训练和验证的保存路径
        self.vis_train_dir = os.path.join(self.vis_root, "train")
        self.vis_val_dir = os.path.join(self.vis_root, "val")

        # 自动创建文件夹
        os.makedirs(self.vis_train_dir, exist_ok=True)
        os.makedirs(self.vis_val_dir, exist_ok=True)

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        N = H * W
        device = f_ir.device

        bg_context_feat = f_ir + f_vis

        # --- Phase 1: 角色分配 (Teacher vs Student) ---

        # 1. 学生 (Agent) 始终要预测
        weights_student, student_logits = self.agent(f_ir, f_vis)

        # 2. 决定谁来掌舵 (Driver)
        if self.training and gt_entropy is not None:
            # ★★★ 训练模式 ★★★
            weights_teacher = self.router(gt_entropy)
            weights_driver = weights_teacher
            aux_loss = F.mse_loss(weights_student, weights_teacher.detach())
        else:
            # ★★★ 推理/验证模式 ★★★
            weights_driver = weights_student
            aux_loss = torch.tensor(0.0, device=device)

        # --- Phase 2: 基于掌舵者的权重进行后续操作 ---

        # 3. HyperNet 决定阈值
        threshold = self.hypernet(weights_driver)  # [B, 1]

        # 4. 生成软掩码
        soft_mask = torch.sigmoid((weights_driver.view(B, -1) - threshold) * 50.0)

        # 5. 重构画布
        reconstructed_canvas = torch.zeros((B, C, N), device=device)

        for b in range(B):
            # --- Step A: 稀疏索引 ---
            flat_mask_b = soft_mask[b]
            active_indices = torch.nonzero(flat_mask_b > 0.5).squeeze(1)

            if active_indices.numel() < 64:
                _, active_indices = torch.topk(weights_driver[b].flatten(), 64)

            # --- Step B: 提取特征 ---
            idx_expanded = active_indices.unsqueeze(1).expand(-1, C)
            flat_ir_b = f_ir[b].flatten(1).transpose(0, 1)
            flat_vis_b = f_vis[b].flatten(1).transpose(0, 1)
            selected_ir = torch.gather(flat_ir_b, 0, idx_expanded).unsqueeze(0)
            selected_vis = torch.gather(flat_vis_b, 0, idx_expanded).unsqueeze(0)

            # 提取软权重
            mask_selected = torch.gather(flat_mask_b, 0, active_indices).view(1, -1, 1)
            w_driver_selected = torch.gather(weights_driver[b].flatten(), 0, active_indices).view(1, -1, 1)

            # --- Step C: Mixer 精修 ---
            refined_ir = self.mixer_ir(selected_ir)
            refined_vis = self.mixer_vis(selected_vis)

            # --- Step D: 融合 ---
            fusion_content = refined_ir * w_driver_selected + refined_vis * (1 - w_driver_selected)
            mixer_delta = fusion_content * mask_selected

            # --- Step E: 回填 ---
            flat_base_b = bg_context_feat[b].flatten(1).transpose(0, 1)
            base_selected = torch.gather(flat_base_b, 0, idx_expanded).unsqueeze(0)
            modulation = 1.0 + mask_selected * self.highlight_scale
            final_edge_feat = (base_selected + mixer_delta) * modulation

            final_flat_b = flat_base_b.clone()
            final_flat_b.scatter_(0, idx_expanded, final_edge_feat.squeeze(0))
            reconstructed_canvas[b] = final_flat_b.transpose(0, 1)

            # --- 可视化 ---
            # 修改：仅在训练时采样可视化，或者你也可以保留验证集的可视化
            # 这里逻辑是：每200步保存一次，自动根据 training 状态分文件夹
            if self.vis_count % 200 == 0 and b == 0:
                self._visualize_reconstruction(
                    student_logits[b],
                    active_indices,
                    threshold[b],
                    H, W, device
                )

            if b == 0: self.vis_count += 1

        f_final = reconstructed_canvas.view(B, C, H, W)
        return f_final, aux_loss

    def _visualize_reconstruction(self, logits, active_indices, threshold_val, H, W, device):
        try:
            # 判断当前是训练还是验证，决定存到哪个文件夹
            if self.training:
                mode = "train"
                save_dir = self.vis_train_dir
            else:
                mode = "val"
                save_dir = self.vis_val_dir

            # --- 1. 选中像素图 ---
            mask = torch.zeros((H * W), device=device)
            mask.scatter_(0, active_indices, 1.0)
            mask = mask.view(H, W).cpu().numpy()

            pixel_save_path = os.path.join(
                save_dir,  # <--- 使用对应的子目录
                f"{mode}_step{self.vis_count}_ch{self.dim}_pixels.png"
            )
            cv2.imwrite(pixel_save_path, (mask * 255).astype(np.uint8))

            # --- 2. 熵差图 ---
            entropy_data = logits.squeeze().detach().cpu().numpy()
            limit = max(abs(entropy_data.min()), abs(entropy_data.max()), 1e-4)

            plt.figure(figsize=(6, 6))
            plt.imshow(entropy_data, cmap='bwr', vmin=-limit, vmax=limit)
            plt.axis('off')
            plt.tight_layout(pad=0)

            entropy_save_path = os.path.join(
                save_dir,  # <--- 使用对应的子目录
                f"{mode}_step{self.vis_count}_ch{self.dim}_entropy.png"
            )
            plt.savefig(entropy_save_path, dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Visualization failed: {e}")
            pass