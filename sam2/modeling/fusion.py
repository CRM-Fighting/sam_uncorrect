import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft  # [新增] 频域操作需要
from torch.utils.checkpoint import checkpoint


# ==========================================
# 1. 基础组件：稀疏相对位置编码
# ==========================================
class SparseRelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_dist=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.num_relative_distance = (2 * max_dist + 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, coords_q, coords_k):
        relative_coords = coords_q[:, :, None, :] - coords_k[:, None, :, :]
        relative_coords += self.max_dist
        relative_coords[:, :, :, 0] = relative_coords[:, :, :, 0].clamp(0, 2 * self.max_dist)
        relative_coords[:, :, :, 1] = relative_coords[:, :, :, 1].clamp(0, 2 * self.max_dist)
        relative_position_index = relative_coords[:, :, :, 0] * (2 * self.max_dist + 1) + \
                                  relative_coords[:, :, :, 1]
        relative_position_index = relative_position_index.long()

        bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            relative_position_index.shape[0], relative_position_index.shape[1],
            relative_position_index.shape[2], self.num_heads
        )
        return bias.permute(0, 3, 1, 2).contiguous()


# ==========================================
# 2. 熵特征注入器
# ==========================================
class EntropyInjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim * 2)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, entropy):
        style = self.net(entropy)
        scale, shift = style.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x


# ==========================================
# [新增 A] 频域校正模块 (From FreqSal)
# 作用：利用互补模态的幅度(纹理)增强当前模态，保留相位(结构)
# ==========================================
class FrequencyRectificationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 简单的门控，用于决定频域混合的程度
        # 这里为了轻量化，没有使用复杂的卷积，而是直接混合
        pass

    def forward(self, x_main, x_aux):
        B, C, H, W = x_main.shape
        # 1. FFT 变换 (Real FFT)
        # norm='backward' 保证能量守恒
        fft_main = torch.fft.rfft2(x_main, norm='backward')
        fft_aux = torch.fft.rfft2(x_aux, norm='backward')

        # 2. 提取幅度 (Amplitude) 和 相位 (Phase)
        amp_main, pha_main = torch.abs(fft_main), torch.angle(fft_main)
        amp_aux = torch.abs(fft_aux)

        # 3. 幅度混合：用 Aux 的幅度补充 Main 的幅度
        # 这种简单的加法在频域等价于增强纹理细节
        amp_fused = amp_main + amp_aux

        # 4. 逆变换回空间域
        # 关键：强制使用 Main 的相位，保证物体结构（如形状、位置）不发生畸变
        fft_new = torch.polar(amp_fused, pha_main)
        x_rect = torch.fft.irfft2(fft_new, s=(H, W), norm='backward')

        # 残差连接：原始特征 + 0.2 * 频域增强特征 (避免训练初期震荡)
        return x_main + 0.2 * x_rect


# ==========================================
# [新增 B] 自适应高频注入 (From BiLIE)
# 作用：使用拉普拉斯算子提取边缘，防止深层特征平滑化
# ==========================================
class AdaptiveHighFreqEnhancement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 固定拉普拉斯卷积核 (边缘检测算子)
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        # register_buffer 保证 kernel 不会作为参数更新，但会随模型保存
        self.register_buffer('kernel', kernel)

        # 自适应门控，决定哪些区域需要注入高频信息
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 动态适配通道数
        k = self.kernel.expand(C, 1, 3, 3)

        # 提取高频 (Edges)
        # padding=1 保持尺寸一致, groups=C 保证逐通道处理
        high_freq = F.conv2d(x, k, padding=1, groups=C)

        # 自适应注入：只在 Gate 响应高的地方加边缘
        return x + high_freq * self.gate(x)


# ==========================================
# [新增 C] 可学习形态学层 (Morphological Layer)
# 作用：修复断裂 (Curve) 和 粘连
# ==========================================
class LearnableMorphologicalLayer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 学习形态学操作的权重 (决定是腐蚀还是膨胀，以及强度)
        self.morph_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Tanh()  # Tanh 输出 [-1, 1]，允许正负反馈
        )

    def forward(self, x):
        # 1. 软膨胀 (Soft Dilation) ≈ MaxPool
        dilated = F.max_pool2d(x, self.kernel_size, stride=1, padding=self.padding)

        # 2. 软腐蚀 (Soft Erosion) ≈ -MaxPool(-x)
        eroded = -F.max_pool2d(-x, self.kernel_size, stride=1, padding=self.padding)

        # 3. 形态学梯度 (Morphological Gradient) = 膨胀 - 腐蚀
        # 这捕捉了物体的骨架和边缘结构
        skeleton = dilated - eroded

        # 4. 将骨架信息加回原特征
        return x + skeleton * self.morph_gate(x)


# ==========================================
# 3. [原有] 特征校正模块 (From CMX 论文)
# ==========================================
class FeatureRectificationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, dim),  # 使用GN更稳
            nn.Sigmoid()
        )

    def forward(self, feat_main, feat_aux):
        # 拼接两个模态
        cat = torch.cat([feat_main, feat_aux], dim=1)
        # 生成门控权重
        gate = self.conv(cat)
        # 校正：原始特征 + 加权后的原始特征 (类似于残差增强)
        rectified = feat_main + feat_main * gate
        return rectified


# ==========================================
# 4. [稳健版] 门控多尺度修补
# ==========================================
class MultiScaleRefinementBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 2, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        feat_local = self.local_branch(x)
        feat_context = self.context_branch(x)

        global_gate = self.global_branch(x)
        feat_local = feat_local * global_gate
        feat_context = feat_context * global_gate

        stack = torch.cat([identity, feat_local, feat_context], dim=1)
        gates = self.gate_conv(stack)

        g_local = gates[:, 0:1, :, :]
        g_context = gates[:, 1:2, :, :]

        out = identity + g_local * feat_local + g_context * feat_context

        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)

        out = self.proj(out)
        return out


# ==========================================
# 5. 混合差分注意力
# ==========================================
class DifferentialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, lambda_init=0.8):
        super().__init__()
        assert num_heads % 2 == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.lambda_q1 = nn.Parameter(torch.zeros(num_heads // 2, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(num_heads // 2, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(num_heads // 2, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(num_heads // 2, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

        self.rpe = SparseRelativePositionBias(num_heads=num_heads, max_dist=128)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv, coords_q, coords_k, alpha_map=None):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        rpe_bias = self.rpe(coords_q, coords_k)

        attn_score_std = (q @ k.transpose(-2, -1)) * self.scale
        attn_score_std = attn_score_std + rpe_bias
        attn_probs_std = torch.softmax(attn_score_std, dim=-1)
        x_std = (attn_probs_std @ v)

        n_half = self.num_heads // 2
        attn_probs_1 = attn_probs_std[:, :n_half]
        attn_probs_2 = attn_probs_std[:, n_half:]

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        lambda_full = lambda_full.view(-1, 1, 1)

        attn_diff = attn_probs_1 - lambda_full * attn_probs_2
        x_diff_part = (attn_diff @ v[:, :n_half])

        x_std_1 = x_std[:, :n_half]
        if alpha_map is not None:
            alpha = alpha_map.unsqueeze(1)
            x_final_1 = x_std_1 + alpha * x_diff_part
        else:
            x_final_1 = x_std_1 + 0.5 * x_diff_part

        x_final_2 = x_std[:, n_half:]
        x = torch.cat([x_final_1, x_final_2], dim=1)
        x = x.transpose(1, 2).reshape(B, N_q, C)

        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiffCrossMixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.injector = EntropyInjector(dim)
        self.diff_attn = DifferentialCrossAttention(
            dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x_q, x_kv, coords_q, coords_k, alpha_map=None, entropy_tokens=None):
        if entropy_tokens is not None:
            x_q = self.injector(x_q, entropy_tokens)
        q = self.norm_q(x_q)
        k = self.norm_kv(x_kv)
        attn_out = self.diff_attn(x_q=q, x_kv=k, coords_q=coords_q, coords_k=coords_k, alpha_map=alpha_map)
        x = x_q + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ==========================================
# 6. 主模块 (DynamicFusionModule)
# ==========================================
class DynamicFusionModule(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        # [修改] 1. 频域校正模块 (Priority A: FreqSal)
        # 先在频域对齐模态，再做后续处理
        self.freq_rect_ir = FrequencyRectificationModule(dim)
        self.freq_rect_vis = FrequencyRectificationModule(dim)

        # [保持] 2. 空间校正模块 (From CMX)
        self.rect_ir = FeatureRectificationModule(dim)
        self.rect_vis = FeatureRectificationModule(dim)

        # [新增] 3. 自适应高频增强 (Priority A: BiLIE)
        self.high_freq_enhance = AdaptiveHighFreqEnhancement(dim)

        # [新增] 4. 形态学层 (Priority C: Morphological)
        self.morph_layer = LearnableMorphologicalLayer(dim)

        self.init_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LayerNorm(dim)
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.mixer_ir = DiffCrossMixerBlock(dim, num_heads=4, dropout=dropout)
        self.mixer_vis = DiffCrossMixerBlock(dim, num_heads=4, dropout=dropout)
        self.context_refine = MultiScaleRefinementBlock(dim)
        self.aux_head = nn.Conv2d(dim, 9, kernel_size=1)

    def forward(self, f_ir, f_vis, gt_entropy=None, gt_entropy_vis=None, gt_entropy_ir=None):
        # --- Modality Dropout (From U3M) ---
        if self.training:
            prob = torch.rand(1).item()
            if prob < 0.1:  # 10% 概率丢弃 IR
                f_ir = torch.zeros_like(f_ir)
            elif prob < 0.2:  # 10% 概率丢弃 VIS
                f_vis = torch.zeros_like(f_vis)

        # --- Step 1: 频域校正 (New) ---
        # 用 VIS 的纹理补 IR，用 IR 的结构补 VIS
        f_ir = self.freq_rect_ir(f_ir, f_vis)
        f_vis = self.freq_rect_vis(f_vis, f_ir)

        # --- Step 2: 空间校正 (Original) ---
        f_ir = self.rect_ir(f_ir, f_vis)
        f_vis = self.rect_vis(f_vis, f_ir)

        # --- 融合逻辑 ---
        B, C, H, W = f_ir.shape
        device = f_ir.device

        base_feat_cat = torch.cat([f_ir, f_vis], dim=1)
        base_feat_sum = self.init_fusion[0](base_feat_cat)
        base_feat_sum = base_feat_sum.permute(0, 2, 3, 1)
        base_feat_sum = self.init_fusion[1](base_feat_sum)
        base_feat_sum = base_feat_sum.permute(0, 3, 1, 2)

        if gt_entropy is None:
            # 即使没有 Entropy，也要走一遍增强流程
            # 对 base_feat_sum 做 refine 和增强
            f_final = self.context_refine(base_feat_sum)
            f_final = self.high_freq_enhance(f_final)  # 增强边缘
            f_final = self.morph_layer(f_final)  # 修复断裂
            return f_final, torch.tensor(0.0, device=device), None

        threshold = gt_entropy.mean(dim=(2, 3), keepdim=True)
        selection_mask = (gt_entropy > threshold).float()

        flat_entropy = gt_entropy.flatten(2).transpose(1, 2)
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        mask_flat = selection_mask.flatten(2).transpose(1, 2)

        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()
        aux_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)
            if indices.numel() > 0:
                y_coords = torch.div(indices, W, rounding_mode='floor')
                x_coords = indices % W
                coords = torch.stack([y_coords, x_coords], dim=1).unsqueeze(0).float().to(device)

                sel_ir = flat_ir[b, indices].unsqueeze(0)
                sel_vis = flat_vis[b, indices].unsqueeze(0)
                sel_entropy = flat_entropy[b, indices].unsqueeze(0)
                alpha_vals = self.alpha_net(sel_entropy)

                enhanced_ir = checkpoint(self.mixer_ir, sel_ir, sel_vis, coords, coords, alpha_vals, sel_entropy,
                                         use_reentrant=False)
                enhanced_vis = checkpoint(self.mixer_vis, sel_vis, sel_ir, coords, coords, alpha_vals, sel_entropy,
                                          use_reentrant=False)

                residual = (enhanced_ir - sel_ir) + (enhanced_vis - sel_vis)
                fusion_weight = 0.2 + 0.8 * torch.tanh(sel_entropy)
                weighted_residual = residual * fusion_weight
                final_canvas[b, indices] += weighted_residual.squeeze(0)

        f_final = final_canvas.transpose(1, 2).view(B, C, H, W)

        # --- Step 3: 后处理增强 (Refine -> HighFreq -> Morph) ---
        f_final = self.context_refine(f_final)

        # [新增] BiLIE: 注入高频边缘
        f_final = self.high_freq_enhance(f_final)

        # [新增] Morph: 修复车道线断裂
        f_final = self.morph_layer(f_final)

        aux_logits = self.aux_head(f_final)

        return f_final, aux_loss, aux_logits