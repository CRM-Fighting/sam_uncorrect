import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.serial_modeling import SerialSegModel
from sam2.modeling.fusion import DynamicFusionModule


class MultiTaskSerialModel(SerialSegModel):
    """
    多任务模型 (修复版):
    1. Backbone: Shared MoE (串行)
    2. Fusion: DynamicFusionModule (软残差门控)
    3. SAM Aux: 仅使用 Stage 4 特征 (向高分代码看齐，禁用 High-Res)
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

        # ★★★ 关键修复：显式关闭高分特征，与第一份代码保持一致 ★★★
        self.backbone.base_sam.sam_mask_decoder.use_high_res_features = False

        # 3. 适配层 (Stage 4: 768 -> 256)
        self.sam_proj_s4 = nn.Conv2d(768, 256, kernel_size=1)

    def get_prompt(self, gt):
        B, H, W = gt.shape
        coords, labels = [], []
        for i in range(B):
            y, x = torch.where(gt[i] > 0)
            if len(y) > 0:
                idx = torch.randint(len(y), (1,)).item()
                coords.append([x[idx].item(), y[idx].item()])
                labels.append(1)
            else:
                coords.append([W // 2, H // 2])
                labels.append(0)
        return torch.tensor(coords, device=gt.device).unsqueeze(1).float(), \
            torch.tensor(labels, device=gt.device).unsqueeze(1)

    def run_sam_head(self, feat, gt, proj):
        # 1. 投影特征
        sam_feat = proj(feat)  # [B, 256, H/32, W/32]
        target_size = sam_feat.shape[-2:]

        # 2. 获取 Prompt
        pt_c, pt_l = self.get_prompt(gt)
        sparse, dense = self.backbone.base_sam.sam_prompt_encoder(points=(pt_c, pt_l), boxes=None, masks=None)

        # 3. 处理 PE 和 Dense 插值 (必要步骤)
        pe = self.backbone.base_sam.sam_prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != target_size:
            pe = F.interpolate(pe, size=target_size, mode='bilinear', align_corners=False)

        if dense.shape[-2:] != target_size:
            dense = F.interpolate(dense, size=target_size, mode='bilinear', align_corners=False)

        # 4. 解码 (★★★ 关键修复：high_res_features=None ★★★)
        low_res, iou_pred, _, _ = self.backbone.base_sam.sam_mask_decoder(
            image_embeddings=sam_feat,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=None  # 强制为 None，不干扰主特征
        )
        return low_res

    def forward(self, vis, ir, gt_semantic=None, gt_entropy_maps=None):
        # 1. 基础流 (MoE Backbone)
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        # 2. Fusion 流 (带蒸馏 Loss)
        fused = []
        total_fusion_loss = 0
        for i in range(4):
            # 获取对应的熵图 (如果存在)
            gt_ent = gt_entropy_maps[i] if (gt_entropy_maps is not None) else None

            # 调用软残差门控融合
            f_out, f_loss = self.fusion_layers[i](feats_ir[i], feats_rgb[i], gt_ent)

            fused.append(f_out)
            total_fusion_loss += f_loss

        # 主任务 SegFormer Head
        seg_logits = self.segformer_head(fused)
        seg_logits = F.interpolate(seg_logits, size=vis.shape[2:], mode='bilinear')

        # 3. SAM 辅助流 (★★★ 关键修复：移除 Neck 和 High-Res 逻辑 ★★★)
        sam_preds = {}
        if gt_semantic is not None:
            H, W = vis.shape[2:]

            # RGB Stage 4 Aux
            out_r = self.run_sam_head(feats_rgb[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['rgb_s4'] = F.interpolate(out_r, size=(H, W), mode='bilinear')

            # IR Stage 4 Aux
            out_i = self.run_sam_head(feats_ir[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['ir_s4'] = F.interpolate(out_i, size=(H, W), mode='bilinear')

        return seg_logits, sam_preds, moe_loss, total_fusion_loss