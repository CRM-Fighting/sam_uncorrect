import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.serial_modeling import SerialSegModel
from sam2.modeling.fusion import DynamicFusionModule


class MultiTaskSerialModel(SerialSegModel):
    """
    多任务模型:
    1. Backbone: Shared MoE (串行)
    2. Fusion: DynamicFusionModule (Cross-Mixer + 门控残差)
    3. SAM Aux: 仅使用 Stage 4 特征，通过简单的 Conv 预测 Mask
    """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__(base_sam, moe_class, num_classes)

        # 1. 融合层
        channels = [96, 192, 384, 768]
        self.fusion_layers = nn.ModuleList([
            DynamicFusionModule(dim=ch) for ch in channels
        ])

        # 2. 冻结 SAM 组件
        for param in self.backbone.base_sam.sam_prompt_encoder.parameters(): param.requires_grad = False
        for param in self.backbone.base_sam.sam_mask_decoder.parameters(): param.requires_grad = False

        self.backbone.base_sam.sam_mask_decoder.use_high_res_features = False

        # 3. 辅助任务头 (修复：输出通道改为 1，用于二分类 Loss)
        # 结构：Conv(768->256) -> ReLU -> Conv(256->1)
        self.aux_head = nn.Sequential(
            nn.Conv2d(channels[-1], 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, vis, ir, gt_semantic=None, gt_entropy_maps=None, gt_entropy_vis=None, gt_entropy_ir=None):
        # 1. 基础流 (MoE Backbone)
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        # 2. Fusion 流
        fused = []
        total_fusion_loss = 0
        for i in range(4):
            ent_sum = gt_entropy_maps[i] if (gt_entropy_maps is not None) else None
            ent_vis = gt_entropy_vis[i] if (gt_entropy_vis is not None) else None
            ent_ir = gt_entropy_ir[i] if (gt_entropy_ir is not None) else None

            f_out, f_loss = self.fusion_layers[i](
                f_ir=feats_ir[i],
                f_vis=feats_rgb[i],
                gt_entropy=ent_sum,
                gt_entropy_vis=ent_vis,
                gt_entropy_ir=ent_ir
            )

            fused.append(f_out)
            total_fusion_loss += f_loss

        # 主任务 SegFormer Head
        seg_logits = self.segformer_head(fused)
        seg_logits = F.interpolate(seg_logits, size=vis.shape[2:], mode='bilinear')

        # 3. SAM 辅助流 (修复：计算 Logits 并上采样)
        sam_preds = {}
        if gt_semantic is not None:  # 仅训练时计算
            # 使用 Stage 4 特征预测二值掩码
            aux_logits = self.aux_head(fused[-1])
            # 上采样到原图尺寸 (480x640) 以匹配 GT
            aux_logits = F.interpolate(aux_logits, size=vis.shape[2:], mode='bilinear', align_corners=False)

            sam_preds['rgb_s4'] = aux_logits
            sam_preds['ir_s4'] = aux_logits

        return seg_logits, sam_preds, moe_loss, total_fusion_loss