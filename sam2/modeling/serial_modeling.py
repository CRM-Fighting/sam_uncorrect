import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.fusion import DynamicFusionModule
from sam2.modeling.segformer_head import SegFormerHead


class SerialSAM2Backbone(nn.Module):
    """
    【共享 MoE 版】串行骨干网
    结构：Hiera Stage -> Shared MoE -> Hiera Stage -> ...
    RGB 和 IR 共用同一组 MoE 参数。
    """

    def __init__(self, base_sam, moe_class, feature_channels=[96, 192, 384, 768], num_experts=8, active_experts=3):
        super().__init__()
        self.base_sam = base_sam

        # 1. 冻结原始 SAM2
        for param in self.base_sam.parameters():
            param.requires_grad = False

        # 2. 构建共享的串行 MoE 层 (Shared MoE)
        self.shared_moe_layers = nn.ModuleList([
            moe_class(dim=ch, num_experts=num_experts, active_experts=active_experts)
            for ch in feature_channels
        ])

    def run_serial_stream(self, image, moe_layers):
        """ 单模态串行前向传播 """
        trunk = self.base_sam.image_encoder.trunk

        # --- 1. Patch Embed & Pos Embed (带尺寸适配修复) ---
        x = trunk.patch_embed(image)

        if trunk.pos_embed is not None:
            pos_embed = trunk.pos_embed
            # 自动处理 1024x1024 预训练权重与 640x480 输入的尺寸不匹配问题
            B, H, W, C = x.shape

            # 检查格式是否为 NCHW (例如 [1, 96, 256, 256]), 如果是则转为 NHWC
            if pos_embed.shape[-1] != C and pos_embed.shape[1] == C:
                pos_embed = pos_embed.permute(0, 2, 3, 1)

            # 如果尺寸不匹配，进行插值
            if pos_embed.shape[1] != H or pos_embed.shape[2] != W:
                pos_embed = pos_embed.permute(0, 3, 1, 2)  # NHWC -> NCHW
                pos_embed = F.interpolate(
                    pos_embed, size=(H, W), mode='bilinear', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1)  # NCHW -> NHWC

            x = x + pos_embed

        features = []
        total_aux_loss = 0.0

        # --- 2. 逐 Block 运行，并在 Stage 结束时插入 MoE ---
        # 修复点：Hiera 没有 .stages 属性，只有 .blocks 和 .stage_ends

        stage_idx = 0
        for i, blk in enumerate(trunk.blocks):
            # A. 跑 Hiera Block (冻结)
            x = blk(x)

            # B. 检查当前 Block 是否是某个 Stage 的结尾
            # trunk.stage_ends 存储了每个 stage 最后一个 block 的索引 (例如 [1, 4, 20, 23])
            if i in trunk.stage_ends:
                # 到了 Stage 结尾，执行 MoE

                # 1. 跑 MoE (共享参数)
                # MoE 期望输入: [B, C, H, W]
                x_in = x.permute(0, 3, 1, 2)
                x_out, aux_loss = moe_layers[stage_idx](x_in)
                total_aux_loss += aux_loss

                # 2. 残差连接 (x是NHWC, x_out是NCHW -> 转回NHWC)
                x = x + x_out.permute(0, 2, 3, 1)

                # 3. 收集特征 (输出 NCHW)
                features.append(x.permute(0, 3, 1, 2))

                # 进入下一个 Stage
                stage_idx += 1

        return features, total_aux_loss

    def forward(self, img_rgb, img_ir):
        feats_rgb, loss_rgb = self.run_serial_stream(img_rgb, self.shared_moe_layers)
        feats_ir, loss_ir = self.run_serial_stream(img_ir, self.shared_moe_layers)
        return feats_rgb, feats_ir, (loss_rgb + loss_ir)


class SerialSegModel(nn.Module):
    """
    基础分割模型 (适配 Dynamic Fusion + Shared MoE)
    """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__()
        self.backbone = SerialSAM2Backbone(base_sam, moe_class)
        channels = [96, 192, 384, 768]

        # 确保使用 DynamicFusionModule
        self.fusion_layers = nn.ModuleList([DynamicFusionModule(ch) for ch in channels])

        self.segformer_head = SegFormerHead(in_channels=channels, num_classes=num_classes)

    def forward(self, vis, ir, gt_entropy_maps=None):
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused_features = []
        total_fusion_loss = 0.0

        for i in range(4):
            # 获取对应层级的熵图 (如果存在)
            gt_ent = gt_entropy_maps[i] if (gt_entropy_maps is not None) else None

            # DynamicFusionModule 返回 (特征, 蒸馏Loss)
            f_out, f_loss = self.fusion_layers[i](feats_ir[i], feats_rgb[i], gt_ent)

            fused_features.append(f_out)
            total_fusion_loss += f_loss

        logits = self.segformer_head(fused_features)
        logits = F.interpolate(logits, size=vis.shape[2:], mode='bilinear')

        return logits, moe_loss, total_fusion_loss