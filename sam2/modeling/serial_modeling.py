import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # 核心修改: 引入 checkpoint
from sam2.modeling.fusion import DynamicFusionModule
from sam2.modeling.segformer_head import SegFormerHead


def _get_hiera_dim(trunk):
    """
    暴力获取 Hiera 主干的 embed_dim，兼容各种修改版
    """
    # 策略 1: 查 PatchEmbed (最稳，直接看第一层卷积输出多少)
    if hasattr(trunk, 'patch_embed') and hasattr(trunk.patch_embed, 'proj'):
        proj = trunk.patch_embed.proj
        if hasattr(proj, 'out_channels'):  # Conv2d
            return proj.out_channels
        if hasattr(proj, 'out_features'):  # Linear
            return proj.out_features

    # 策略 2: 查 PosEmbed (次稳，看位置编码的最后一维)
    if hasattr(trunk, 'pos_embed') and trunk.pos_embed is not None:
        return trunk.pos_embed.shape[-1]

    # 策略 3: 查 Blocks 的 Norm 层
    if hasattr(trunk, 'blocks') and len(trunk.blocks) > 0:
        b0 = trunk.blocks[0]
        # 只要是 LayerNorm，就有 normalized_shape
        if hasattr(b0, 'norm1') and hasattr(b0.norm1, 'normalized_shape'):
            return b0.norm1.normalized_shape[0]

    # 策略 4: 查常见属性名
    for attr in ['embed_dim', 'dim', 'd_model', 'num_features']:
        if hasattr(trunk, attr):
            return getattr(trunk, attr)

    raise AttributeError("无法自动检测 Hiera 的维度！请检查 backbone 结构。")


class SerialSAM2Backbone(nn.Module):
    """
    【共享 MoE 版】串行骨干网
    结构：Hiera Stage -> Shared MoE -> Hiera Stage -> ...
    RGB 和 IR 共用同一组 MoE 参数。
    """

    def __init__(self, base_sam, moe_class, feature_channels=None, num_experts=8, active_experts=3):
        super().__init__()
        self.base_sam = base_sam

        # 1. 自动检测特征维度 (使用暴力检测函数)
        if feature_channels is None:
            embed_dim = _get_hiera_dim(base_sam.image_encoder.trunk)

            # Hiera 的层级倍率通常是 [1, 2, 4, 8]
            feature_channels = [embed_dim * (2 ** i) for i in range(4)]
            print(f"🔧 [Auto-Detect] Backbone Embed Dim: {embed_dim} (Checkpointing Enabled)")
            print(f"🔧 [Auto-Detect] Feature Channels: {feature_channels}")

        # 2. 冻结原始 SAM2
        for param in self.base_sam.parameters():
            param.requires_grad = False

        # 3. 构建共享的串行 MoE 层 (Shared MoE)
        self.shared_moe_layers = nn.ModuleList([
            moe_class(dim=ch, num_experts=num_experts, active_experts=active_experts)
            for ch in feature_channels
        ])

    def run_serial_stream(self, image, moe_layers):
        """ 单模态串行前向传播 """
        trunk = self.base_sam.image_encoder.trunk

        # --- 1. Patch Embed & Pos Embed ---
        x = trunk.patch_embed(image)

        if trunk.pos_embed is not None:
            pos_embed = trunk.pos_embed
            B, H, W, C = x.shape
            # 自动尺寸适配
            if pos_embed.shape[-1] != C and pos_embed.shape[1] == C:
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            if pos_embed.shape[1] != H or pos_embed.shape[2] != W:
                pos_embed = pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bilinear', align_corners=False)
                pos_embed = pos_embed.permute(0, 2, 3, 1)
            x = x + pos_embed

        features = []
        total_aux_loss = 0.0

        # --- 2. 逐 Block 运行 ---
        stage_idx = 0
        for i, blk in enumerate(trunk.blocks):
            # ★★★ 核心修改：梯度检查点 (Gradient Checkpointing) ★★★
            # 只有当 x 需要梯度时（即已经过了第一个 MoE 层），开启 checkpoint 才有意义
            # 这对于冻结的 backbone 尤其重要，因为我们不需要存储中间激活值
            if self.training and x.requires_grad:
                # use_reentrant=False 是新版 PyTorch 推荐的，更安全
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

            # B. 检查 Stage 结尾，插入 MoE
            if i in trunk.stage_ends:
                # 1. 跑 MoE (共享参数)
                x_in = x.permute(0, 3, 1, 2)
                x_out, aux_loss = moe_layers[stage_idx](x_in)
                total_aux_loss += aux_loss

                # 2. 残差连接
                x = x + x_out.permute(0, 2, 3, 1)

                features.append(x.permute(0, 3, 1, 2))
                stage_idx += 1

        return features, total_aux_loss

    def forward(self, img_rgb, img_ir):
        feats_rgb, loss_rgb = self.run_serial_stream(img_rgb, self.shared_moe_layers)
        feats_ir, loss_ir = self.run_serial_stream(img_ir, self.shared_moe_layers)
        return feats_rgb, feats_ir, (loss_rgb + loss_ir)


class SerialSegModel(nn.Module):
    """
    基础分割模型
    """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__()
        # 1. 初始化 Backbone (自动检测维度)
        self.backbone = SerialSAM2Backbone(base_sam, moe_class, feature_channels=None)

        # 2. 获取检测到的维度 (复用检测逻辑)
        embed_dim = _get_hiera_dim(base_sam.image_encoder.trunk)
        channels = [embed_dim * (2 ** i) for i in range(4)]

        # 3. 初始化后续层
        self.fusion_layers = nn.ModuleList([DynamicFusionModule(ch) for ch in channels])

        # ★★★ 确保 SegFormerHead 被初始化 ★★★
        self.segformer_head = SegFormerHead(in_channels=channels, num_classes=num_classes)

    def forward(self, vis, ir, gt_entropy_maps=None):
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused_features = []
        total_fusion_loss = 0.0

        for i in range(4):
            gt_ent = gt_entropy_maps[i] if (gt_entropy_maps is not None) else None
            f_out, f_loss = self.fusion_layers[i](feats_ir[i], feats_rgb[i], gt_ent)
            fused_features.append(f_out)
            total_fusion_loss += f_loss

        logits = self.segformer_head(fused_features)
        logits = F.interpolate(logits, size=vis.shape[2:], mode='bilinear')

        return logits, moe_loss, total_fusion_loss