import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ==========================================
# 1. 基础组件：稀疏相对位置编码 (Sparse RPE)
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
        # coords: [B, N, 2]
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
# 2. 熵特征注入器 (Entropy Injector)
# ==========================================
class EntropyInjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 将 1 维熵映射为 Scale 和 Shift 参数
        self.net = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim * 2)  # 输出 scale 和 shift
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, entropy):
        # x: [B, N, C]
        # entropy: [B, N, 1]

        # 计算调节参数
        style = self.net(entropy)  # [B, N, 2*C]
        scale, shift = style.chunk(2, dim=-1)

        # AdaIN 风格注入： x = x * (1 + scale) + shift
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x


# ==========================================
# 3. [重构] 选择性多尺度修补 (Selective Multi-Scale Refinement)
# 核心思想：Concat -> Softmax Selection -> Weighted Sum
# 解决问题：既保留护栏细节(3x3)，又连接减速带(7x7)
# ==========================================
class MultiScaleRefinementBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 分支1: 局部细节 (3x3) - 专注细小物体（护栏、锥桶）
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # DW-Conv
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)  # PW-Conv
        )

        # 分支2: 长距离上下文 (7x7) - 专注大物体连接（减速带、弯道）
        self.context_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

        # 分支3: 全局先验 (Global) - 专注整体光照调节
        # 这是一个门控网络，不直接参与加权竞争，而是调节前两个分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # [新增] 选择网络 (The Selector)
        # 输入: 拼接后的特征 (Identity + Local + Context) -> 3*C
        # 输出: 3个权重图 (Identity权重, Local权重, Context权重)
        self.select_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 3, 1)  # 输出3个通道
        )

        # 最后的融合投影
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # 1. 计算各分支特征
        feat_local = self.local_branch(x)
        feat_context = self.context_branch(x)

        # 全局门控：像调色板一样，调节所有分支的基调
        global_gate = self.global_branch(x)  # [B, C, 1, 1]
        feat_local = feat_local * global_gate
        feat_context = feat_context * global_gate

        # 2. [核心] Concat + 选择权重计算
        # 我们把原始特征、局部修补、上下文修补拼在一起
        # 形状变 [B, 3C, H, W]
        stack = torch.cat([identity, feat_local, feat_context], dim=1)

        # 生成权重 [B, 3, H, W] -> Softmax 归一化
        # 这一步决定了每个像素点“听谁的”
        weights = F.softmax(self.select_conv(stack), dim=1)

        # 拆分权重
        w_id = weights[:, 0:1, :, :]
        w_lo = weights[:, 1:2, :, :]
        w_co = weights[:, 2:3, :, :]

        # 3. 加权融合 (Weighted Sum)
        # 在护栏处，w_lo 会很大；在减速带处，w_co 会很大
        out = w_id * identity + w_lo * feat_local + w_co * feat_context

        out = self.proj(out)
        return out


# ==========================================
# 4. 混合差分注意力 (Pixel-wise Alpha)
# ==========================================
class DifferentialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 差分参数
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

        # Standard Attention
        attn_score_std = (q @ k.transpose(-2, -1)) * self.scale
        attn_score_std = attn_score_std + rpe_bias
        attn_probs_std = torch.softmax(attn_score_std, dim=-1)
        x_std = (attn_probs_std @ v)

        # Differential Attention
        n_half = self.num_heads // 2
        attn_probs_1 = attn_probs_std[:, :n_half]
        attn_probs_2 = attn_probs_std[:, n_half:]

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        lambda_full = lambda_full.view(-1, 1, 1)

        attn_diff = attn_probs_1 - lambda_full * attn_probs_2
        x_diff_part = (attn_diff @ v[:, :n_half])

        # Dynamic Fusion
        x_std_1 = x_std[:, :n_half]
        if alpha_map is not None:
            # alpha_map: [B, N, 1] -> [B, 1, N, 1]
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

        # 熵注入模块
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
        # 1. 先进行熵注入 (Entropy Injection)
        if entropy_tokens is not None:
            x_q = self.injector(x_q, entropy_tokens)

        # 2. 注意力计算
        q = self.norm_q(x_q)
        k = self.norm_kv(x_kv)
        attn_out = self.diff_attn(x_q=q, x_kv=k, coords_q=coords_q, coords_k=coords_k, alpha_map=alpha_map)
        x = x_q + attn_out

        # 3. FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ==========================================
# 5. 主模块 (DynamicFusionModule)
# ==========================================
class DynamicFusionModule(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        # [新增] 初始融合层：把 Concat 后的 2C 特征融合回 C
        # 作用：智能地决定背景区域该保留多少IR，多少VIS，而不是傻傻相加
        self.init_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LayerNorm(dim)
        )

        # Alpha 生成器
        self.alpha_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 交互模块
        self.mixer_ir = DiffCrossMixerBlock(dim, num_heads=4, dropout=dropout)
        self.mixer_vis = DiffCrossMixerBlock(dim, num_heads=4, dropout=dropout)

        # [升级] 多尺度修补模块
        self.context_refine = MultiScaleRefinementBlock(dim)

    def forward(self, f_ir, f_vis, gt_entropy=None, gt_entropy_vis=None, gt_entropy_ir=None):
        B, C, H, W = f_ir.shape
        device = f_ir.device

        # [修改] Concat -> Conv -> Norm
        # 这样比直接相加好，因为不会丢失原始特征的独立性
        base_feat_cat = torch.cat([f_ir, f_vis], dim=1)  # [B, 2C, H, W]
        base_feat_sum = self.init_fusion[0](base_feat_cat)  # [B, C, H, W]

        # LayerNorm (手动处理维度)
        base_feat_sum = base_feat_sum.permute(0, 2, 3, 1)  # [B, H, W, C]
        base_feat_sum = self.init_fusion[1](base_feat_sum)
        base_feat_sum = base_feat_sum.permute(0, 3, 1, 2)  # [B, C, H, W]

        if gt_entropy is None:
            return base_feat_sum, torch.tensor(0.0, device=device)

        # --- 1. 筛选 ---
        threshold = gt_entropy.mean(dim=(2, 3), keepdim=True)
        selection_mask = (gt_entropy > threshold).float()

        flat_entropy = gt_entropy.flatten(2).transpose(1, 2)
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        mask_flat = selection_mask.flatten(2).transpose(1, 2)

        final_canvas = base_feat_sum.flatten(2).transpose(1, 2).clone()
        aux_loss = torch.tensor(0.0, device=device)

        # --- 2. 交互 ---
        for b in range(B):
            indices = torch.nonzero(mask_flat[b, :, 0] > 0.5).squeeze(1)
            if indices.numel() > 0:
                y_coords = torch.div(indices, W, rounding_mode='floor')
                x_coords = indices % W
                coords = torch.stack([y_coords, x_coords], dim=1).unsqueeze(0).float().to(device)

                sel_ir = flat_ir[b, indices].unsqueeze(0)
                sel_vis = flat_vis[b, indices].unsqueeze(0)

                # 获取稀疏熵
                sel_entropy = flat_entropy[b, indices].unsqueeze(0)

                # 计算 Alpha
                alpha_vals = self.alpha_net(sel_entropy)

                # 传入 sel_entropy 进行特征注入
                enhanced_ir = checkpoint(self.mixer_ir, sel_ir, sel_vis, coords, coords, alpha_vals, sel_entropy,
                                         use_reentrant=False)
                enhanced_vis = checkpoint(self.mixer_vis, sel_vis, sel_ir, coords, coords, alpha_vals, sel_entropy,
                                          use_reentrant=False)

                residual = (enhanced_ir - sel_ir) + (enhanced_vis - sel_vis)

                # 软加权贴回
                fusion_weight = 0.2 + 0.8 * torch.tanh(sel_entropy)
                weighted_residual = residual * fusion_weight
                final_canvas[b, indices] += weighted_residual.squeeze(0)

        f_final = final_canvas.transpose(1, 2).view(B, C, H, W)

        # --- 3. [升级] 选择性多尺度修补 ---
        f_final = self.context_refine(f_final)

        return f_final, aux_loss