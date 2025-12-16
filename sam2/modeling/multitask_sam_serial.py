import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.serial_modeling import SerialSegModel


class MultiTaskSerialModel(SerialSegModel):
    """
    多任务模型：Shared Serial MoE + SAM2 Aux (Only Stage 4 Supervision)
    修复：
    1. 适配 SAM 2.1 接口 (repeat_image, 4个返回值)
    2. 修复 Dense Prompt 维度不匹配问题 (增加插值逻辑)
    """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__(base_sam, moe_class, num_classes)

        # 冻结 SAM 解码器组件
        for param in self.backbone.base_sam.sam_prompt_encoder.parameters(): param.requires_grad = False
        for param in self.backbone.base_sam.sam_mask_decoder.parameters(): param.requires_grad = False

        # 强制关闭 high_res_features，确保只使用 Stage 4 特征
        self.backbone.base_sam.sam_mask_decoder.use_high_res_features = False

        # SAM2 适配层 (Stage 4: 768 -> 256)
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
        target_size = sam_feat.shape[-2:]  # (15, 20)

        # 2. 获取 Prompt (Sparse & Dense)
        pt_c, pt_l = self.get_prompt(gt)
        sparse, dense = self.backbone.base_sam.sam_prompt_encoder(points=(pt_c, pt_l), boxes=None, masks=None)

        # --- 核心修复区 Start ---

        # 3. 处理 PE (位置编码) 插值
        pe = self.backbone.base_sam.sam_prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != target_size:
            pe = F.interpolate(pe, size=target_size, mode='bilinear', align_corners=False)

        # 4. 处理 Dense Prompt (密集提示) 插值 <--- 之前漏了这一步
        # dense 的默认形状是 [B, 256, 64, 64]，必须缩放到 [B, 256, 15, 20] 才能相加
        if dense.shape[-2:] != target_size:
            dense = F.interpolate(dense, size=target_size, mode='bilinear', align_corners=False)

        # --- 核心修复区 End ---

        # 5. 解码
        low_res_masks, iou_pred, _, _ = self.backbone.base_sam.sam_mask_decoder(
            image_embeddings=sam_feat,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,  # 现在的 dense 尺寸已经对齐了
            multimask_output=False,
            repeat_image=False,
            high_res_features=None
        )
        return low_res_masks

    def forward(self, vis, ir, gt_semantic=None):
        # 1. 基础分割流
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused = [self.fusion_layers[i](feats_rgb[i], feats_ir[i]) for i in range(4)]
        logits = self.segformer_head(fused)
        logits = F.interpolate(logits, size=vis.shape[2:], mode='bilinear')

        # 2. SAM 辅助流
        sam_preds = {}
        if gt_semantic is not None:
            H, W = vis.shape[2:]

            # RGB Stage 4
            out_r = self.run_sam_head(feats_rgb[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['rgb_s4'] = F.interpolate(out_r, size=(H, W), mode='bilinear')

            # IR Stage 4
            out_i = self.run_sam_head(feats_ir[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['ir_s4'] = F.interpolate(out_i, size=(H, W), mode='bilinear')

        return logits, sam_preds, moe_loss