import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.fusion import SimpleFusionModule
from sam2.modeling.segformer_head import SegFormerHead


class SerialSAM2Backbone(nn.Module):
    """
    【共享 MoE 版】串行骨干网 (修复 Hiera 遍历逻辑)
    结构：Hiera Stage -> Shared MoE -> Hiera Stage -> ...
    """

    def __init__(self, base_sam, moe_class, feature_channels=[96, 192, 384, 768], num_experts=8, active_experts=3):
        super().__init__()
        self.base_sam = base_sam

        # 1. 冻结原始 SAM2
        for param in self.base_sam.parameters():
            param.requires_grad = False

        # 2. 构建共享的串行 MoE 层
        self.shared_moe_layers = nn.ModuleList([
            moe_class(dim=ch, num_experts=num_experts, active_experts=active_experts)
            for ch in feature_channels
        ])

    def run_serial_stream(self, image, moe_layers):
        """ 单模态串行前向传播 """
        trunk = self.base_sam.image_encoder.trunk

        # 1. Patch Embedding (切块)
        x = trunk.patch_embed(image)

        # 2. Positional Embedding (位置编码 - 自动插值)
        if hasattr(trunk, "_get_pos_embed"):
            x = x + trunk._get_pos_embed(x.shape[1:3])
        elif trunk.pos_embed is not None:
            x = x + trunk.pos_embed

        features = []
        total_aux_loss = 0.0

        # --- 修复点：根据 stage_ends 手动拆分 blocks ---
        # Hiera 没有 .stages 属性，只有一个平铺的 .blocks 列表
        # stage_ends 记录了每个 Stage 最后一个 block 的索引 (例如 [1, 4, 20, 23])
        start_idx = 0

        # 遍历 4 个阶段 (假设 trunk.stage_ends 长度为 4)
        for i, end_idx in enumerate(trunk.stage_ends):
            # A. 取出当前 Stage 的所有 Blocks 并执行
            # 这里的 slice 操作是左闭右开，所以用 end_idx + 1
            stage_blocks = trunk.blocks[start_idx: end_idx + 1]
            start_idx = end_idx + 1  # 更新下一轮起点

            for block in stage_blocks:
                x = block(x)

            # 此时 x 是当前 Stage 的输出 (例如 Stage 1 跑完是 120x160)

            # B. 跑 MoE (共享参数)
            # Hiera 输出是 NHWC [B, H, W, C]，MoE 需要 NCHW [B, C, H, W]
            x_in = x.permute(0, 3, 1, 2)
            x_out, aux_loss = moe_layers[i](x_in)
            total_aux_loss += aux_loss

            # C. 残差连接 (MoE 结果加回主干)
            # 将增强后的特征加回数据流，供下一阶段使用
            x = x + x_out.permute(0, 2, 3, 1)

            # D. 收集特征 (转回 NCHW 供后续 Head 使用)
            features.append(x.permute(0, 3, 1, 2))

        return features, total_aux_loss

    def forward(self, img_rgb, img_ir):
        # RGB 和 IR 都传入 self.shared_moe_layers
        feats_rgb, loss_rgb = self.run_serial_stream(img_rgb, self.shared_moe_layers)
        feats_ir, loss_ir = self.run_serial_stream(img_ir, self.shared_moe_layers)

        # MoE Loss 是两者的总和
        return feats_rgb, feats_ir, (loss_rgb + loss_ir)


class SerialSegModel(nn.Module):
    """ 纯分割模型 (适配共享 MoE Backbone) """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__()
        self.backbone = SerialSAM2Backbone(base_sam, moe_class)
        channels = [96, 192, 384, 768]

        self.fusion_layers = nn.ModuleList([SimpleFusionModule(ch) for ch in channels])
        self.segformer_head = SegFormerHead(in_channels=channels, num_classes=num_classes)

    def forward(self, vis, ir):
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused = [self.fusion_layers[i](feats_rgb[i], feats_ir[i]) for i in range(4)]
        logits = self.segformer_head(fused)
        logits = F.interpolate(logits, size=vis.shape[2:], mode='bilinear')

        return logits, moe_loss