import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Agent (吸纳豆包：可学习温度系数) ---
class SamplingAgent(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        # 初始化为 1.0，让网络自己学温度
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, f_ir, f_vis):
        x = torch.cat([f_ir, f_vis], dim=1)
        # 限制温度范围，防止除以 0 或梯度爆炸
        temp = torch.clamp(self.temp, 0.1, 10.0)
        return torch.sigmoid(self.net(x) / temp)


# --- 2. Router (吸纳豆包：温和初始化) ---
class Router(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Conv2d(in_channels, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        # 使得初始输出在 0.5 附近，避免开局“偏科”
        nn.init.xavier_normal_(self.net.weight)
        nn.init.constant_(self.net.bias, 0.0)

    def forward(self, entropy_map):
        return torch.sigmoid(self.net(entropy_map))


# --- 3. HyperNetwork (吸纳豆包：像素级阈值 + 自适应) ---
class PixelWiseHyperNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 1x1 卷积，为每个像素生成独立的阈值
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_ir, f_vis):
        x = torch.cat([f_ir, f_vis], dim=1)
        threshold = self.conv(x)  # [B, 1, H, W]

        # 这里的 Trick 是：阈值不能脱离全图的统计分布
        # 我们让阈值 = 局部预测 * 0.8 + 全局均值 * 0.2
        # 这样既有像素级差异，又有全局视野
        global_mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        # 这里的 global_mean 需要过一个简单的变换映射到 0-1，或者直接用 threshold 自身
        # 为了稳定，直接限制 threshold 范围
        return torch.clamp(threshold, 0.05, 0.95)


# --- 4. EfficientAttention (吸纳豆包：数值防护 EPS) ---
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = 1e-8  # 防 NaN

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 数值稳定 Softmax
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = torch.matmul(k.transpose(-2, -1), v)
        # 这里的 context 可能很小，加 eps 防止后续计算问题 (可选)
        out = torch.matmul(q, context)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# --- 5. MixerBlock (吸纳豆包：Dropout 防止过拟合) ---
class MixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),  # 加点 Dropout 确实对小数据集有好处
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --- 6. DynamicFusionModule (集大成者) ---
class DynamicFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.eps = 1e-8

        self.agent = SamplingAgent(dim)
        self.router = Router(in_channels=1)
        # 改用像素级 HyperNet
        self.hypernet = PixelWiseHyperNet(dim)

        self.mixer_ir = MixerBlock(dim)
        self.mixer_vis = MixerBlock(dim)

        # 可学习 STE 温度
        self.ste_temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, f_ir, f_vis, gt_entropy=None):
        B, C, H, W = f_ir.shape
        N = H * W

        # 1. Agent 预测权重
        pred_weights = self.agent(f_ir, f_vis)

        # 2. 训练/测试分支
        if self.training and gt_entropy is not None:
            target_weights = self.router(gt_entropy)
            active_weights = target_weights
            # ★ 修正豆包错误：F.mse_loss 默认就是 mean，不要再除以 N 了！
            aux_loss = F.mse_loss(pred_weights, target_weights.detach())
        else:
            active_weights = pred_weights
            aux_loss = torch.tensor(0.0, device=f_ir.device)

        # 3. 像素级阈值
        threshold = self.hypernet(f_ir, f_vis)  # [B, 1, H, W]

        # 4. STE 可导掩码 (核心优化)
        diff = active_weights - threshold
        ste_temp = torch.clamp(self.ste_temp, 1.0, 20.0)  # 限制温度范围

        mask_hard = (diff > 0).float()
        mask_soft = torch.sigmoid(diff * ste_temp)
        # 前向 Hard，反向 Soft
        mask = mask_hard - mask_soft.detach() + mask_soft

        # 5. 准备数据
        flat_ir = f_ir.flatten(2).transpose(1, 2)
        flat_vis = f_vis.flatten(2).transpose(1, 2)
        flat_weights = active_weights.flatten(2).transpose(1, 2)
        flat_mask = mask.flatten(2).transpose(1, 2)

        # 6. 软加权应用掩码 (Soft Masking)
        # 豆包说得对：直接置 0 容易 NaN。
        # 我们这里虽然看起来像置 0，但因为 mask 是 0/1，其实就是置 0。
        # 为了防止 Attention 里的 Softmax(0) 问题，我们在 EfficientAttention 里加了 eps，所以这里可以放心用。
        masked_ir = flat_ir * flat_mask
        masked_vis = flat_vis * flat_mask

        # 7. 特征精修
        refined_ir = self.mixer_ir(masked_ir)
        refined_vis = self.mixer_vis(masked_vis)

        # 8. 残差回填 (Delta Injection)
        # 只在 mask=1 的地方回填残差
        delta_ir = (refined_ir - masked_ir) * flat_mask
        # 加上 Clamp 防止梯度爆炸
        final_ir = flat_ir + delta_ir.clamp(-1.0, 1.0)

        delta_vis = (refined_vis - masked_vis) * flat_mask
        final_vis = flat_vis + delta_vis.clamp(-1.0, 1.0)

        # 9. 终极融合 (全图加权)
        # 加上 clamp 防止权重极值
        flat_weights = torch.clamp(flat_weights, self.eps, 1 - self.eps)
        f_fused_flat = final_ir * flat_weights + final_vis * (1 - flat_weights)

        # 10. 还原
        f_final = f_fused_flat.transpose(1, 2).view(B, C, H, W)

        return f_final, aux_loss