import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.serial_modeling import SerialSegModel, _get_hiera_dim
from sam2.modeling.fusion import DynamicFusionModule


class MultiTaskSerialModel(SerialSegModel):
    """
    多任务模型 V7 (维度自适应版)
    """

    def __init__(self, base_sam, moe_class, num_classes=9):
        # 1. 调用父类初始化 (父类会自动检测维度并初始化 Backbone 和 Head)
        super().__init__(base_sam, moe_class, num_classes)

        # 2. 重新获取维度 (因为 super().__init__ 里的 fusion_layers 会被这里覆盖)
        embed_dim = _get_hiera_dim(base_sam.image_encoder.trunk)
        channels = [embed_dim * (2 ** i) for i in range(4)]

        # 3. 重新初始化 Fusion 层
        self.fusion_layers = nn.ModuleList([
            DynamicFusionModule(dim=ch) for ch in channels
        ])

        # 4. 冻结 SAM 组件
        for param in self.backbone.base_sam.sam_prompt_encoder.parameters(): param.requires_grad = False
        for param in self.backbone.base_sam.sam_mask_decoder.parameters(): param.requires_grad = False
        # 冻结 Neck
        for param in self.backbone.base_sam.image_encoder.neck.parameters(): param.requires_grad = False

        # 5. 适配层 (SAM Head 投影)
        # 自动使用最后一层维度
        self.sam_proj_s4 = nn.Conv2d(channels[-1], 256, kernel_size=1)

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

    def run_sam_head(self, feat, gt, proj, high_res_features=None):
        # 1. 投影特征
        sam_feat = proj(feat)

        # 2. 基础对齐: Stride 32 -> 16
        if high_res_features is not None:
            sam_feat = F.interpolate(
                sam_feat, scale_factor=2.0, mode='bilinear', align_corners=False
            )

        # 3. 强制对齐高分特征
        if high_res_features is not None:
            feat_s0, feat_s1 = high_res_features

            # feat_s1 必须是 sam_feat 的 2 倍
            target_h_s1 = sam_feat.shape[-2] * 2
            target_w_s1 = sam_feat.shape[-1] * 2
            if feat_s1.shape[-2] != target_h_s1 or feat_s1.shape[-1] != target_w_s1:
                feat_s1 = F.interpolate(feat_s1, size=(target_h_s1, target_w_s1),
                                        mode='bilinear', align_corners=False)

            # feat_s0 必须是 sam_feat 的 4 倍
            target_h_s0 = sam_feat.shape[-2] * 4
            target_w_s0 = sam_feat.shape[-1] * 4
            if feat_s0.shape[-2] != target_h_s0 or feat_s0.shape[-1] != target_w_s0:
                feat_s0 = F.interpolate(feat_s0, size=(target_h_s0, target_w_s0),
                                        mode='bilinear', align_corners=False)

            high_res_features = [feat_s0, feat_s1]

        # 4. 获取提示
        pt_c, pt_l = self.get_prompt(gt)
        sparse, dense = self.backbone.base_sam.sam_prompt_encoder(points=(pt_c, pt_l), boxes=None, masks=None)

        # 5. PE 和 Dense 插值
        pe = self.backbone.base_sam.sam_prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != sam_feat.shape[-2:]:
            pe = F.interpolate(pe, size=sam_feat.shape[-2:], mode='bilinear')

        if dense.shape[-2:] != sam_feat.shape[-2:]:
            dense = F.interpolate(dense, size=sam_feat.shape[-2:], mode='bilinear')

        # 6. 运行解码器
        low_res, iou_pred, _, _ = self.backbone.base_sam.sam_mask_decoder(
            image_embeddings=sam_feat,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features
        )
        return low_res

    def _prepare_high_res_features(self, neck_out):
        """ 拆包并投影 """

        def unwrap(t):
            return t[0] if isinstance(t, (list, tuple)) else t

        decoder = self.backbone.base_sam.sam_mask_decoder
        feat_s0 = decoder.conv_s0(unwrap(neck_out[0]))
        feat_s1 = decoder.conv_s1(unwrap(neck_out[1]))
        return [feat_s0, feat_s1]

    def forward(self, vis, ir, gt_semantic=None, gt_entropy_maps=None):
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused = []
        total_fusion_loss = 0
        for i in range(4):
            gt_ent = gt_entropy_maps[i] if (gt_entropy_maps is not None) else None
            f_out, f_loss = self.fusion_layers[i](feats_ir[i], feats_rgb[i], gt_ent)
            fused.append(f_out)
            total_fusion_loss += f_loss

        # 主任务
        seg_logits = self.segformer_head(fused)
        seg_logits = F.interpolate(seg_logits, size=vis.shape[2:], mode='bilinear')

        # 3. SAM 辅助流
        sam_preds = {}
        if gt_semantic is not None:
            H, W = vis.shape[2:]

            with torch.no_grad():
                neck_out_rgb = self.backbone.base_sam.image_encoder.neck(feats_rgb)
                neck_out_ir = self.backbone.base_sam.image_encoder.neck(feats_ir)

            high_res_rgb = self._prepare_high_res_features(neck_out_rgb)
            high_res_ir = self._prepare_high_res_features(neck_out_ir)

            out_r = self.run_sam_head(
                feats_rgb[-1], gt_semantic, self.sam_proj_s4,
                high_res_features=high_res_rgb
            )
            sam_preds['rgb_s4'] = F.interpolate(out_r, size=(H, W), mode='bilinear')

            out_i = self.run_sam_head(
                feats_ir[-1], gt_semantic, self.sam_proj_s4,
                high_res_features=high_res_ir
            )
            sam_preds['ir_s4'] = F.interpolate(out_i, size=(H, W), mode='bilinear')

        return seg_logits, sam_preds, moe_loss, total_fusion_loss