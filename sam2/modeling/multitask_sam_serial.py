import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.serial_modeling import SerialSegModel
from sam2.modeling.fusion import DynamicFusionModule


class MultiTaskSerialModel(SerialSegModel):
    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__(base_sam, moe_class, num_classes)

        channels = [96, 192, 384, 768]

        # 【修改】去掉 Stage 1 的 Fusion Layer，只保留后 3 个
        self.fusion_layers = nn.ModuleList([
            DynamicFusionModule(dim=ch) for ch in channels[1:]
        ])

        for param in self.backbone.base_sam.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.base_sam.sam_mask_decoder.parameters():
            param.requires_grad = False

        self.backbone.base_sam.sam_mask_decoder.use_high_res_features = False
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
        sam_feat = proj(feat)
        target_size = sam_feat.shape[-2:]
        pt_c, pt_l = self.get_prompt(gt)

        with torch.no_grad():
            sparse, dense = self.backbone.base_sam.sam_prompt_encoder(
                points=(pt_c, pt_l), boxes=None, masks=None
            )

        if dense.shape[-2:] != target_size:
            dense = F.interpolate(dense, size=target_size, mode='bilinear', align_corners=False)
        pe = self.backbone.base_sam.sam_prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != target_size:
            pe = F.interpolate(pe, size=target_size, mode='bilinear', align_corners=False)

        low_res, iou_pred, _, _ = self.backbone.base_sam.sam_mask_decoder(
            image_embeddings=sam_feat,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=None
        )
        return low_res

    def forward(self, vis, ir, gt_semantic=None, gt_entropy_maps=None, gt_entropy_vis=None, gt_entropy_ir=None):
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused = []
        total_fusion_loss = 0
        final_aux_logits = None

        # 【修改】Stage 1 (High Res) 直接拼接 (Early Concat)
        # 此时 c1_fused 维度为 96 + 96 = 192
        c1_fused = torch.cat([feats_rgb[0], feats_ir[0]], dim=1)
        fused.append(c1_fused)

        # 【修改】Stage 2, 3, 4 走 Fusion 模块
        # 注意：i 从 0 到 2，对应 channels[1:]，特征取 [i+1]
        for i in range(3):
            ent_sum = gt_entropy_maps[i + 1] if (gt_entropy_maps is not None) else None
            ent_vis = gt_entropy_vis[i + 1] if (gt_entropy_vis is not None) else None
            ent_ir = gt_entropy_ir[i + 1] if (gt_entropy_ir is not None) else None

            f_out, f_loss, aux_logit = self.fusion_layers[i](
                f_ir=feats_ir[i + 1],
                f_vis=feats_rgb[i + 1],
                gt_entropy=ent_sum,
                gt_entropy_vis=ent_vis,
                gt_entropy_ir=ent_ir
            )
            fused.append(f_out)
            total_fusion_loss += f_loss
            if i == 2: final_aux_logits = aux_logit

        # 【修改】Detail Feat 使用双模态拼接特征，供边缘提取
        detail_feat_combined = torch.cat([feats_rgb[0], feats_ir[0]], dim=1)

        ret = self.segformer_head(fused, detail_feat=detail_feat_combined)

        # 【修改】解包返回
        edge_prompt = None
        if self.training:
            if isinstance(ret, tuple):
                seg_logits, edge_prompt = ret
            else:
                seg_logits = ret
        else:
            seg_logits = ret

        seg_logits = F.interpolate(seg_logits, size=vis.shape[2:], mode='bilinear')

        sam_preds = {}
        if gt_semantic is not None:
            H, W = vis.shape[2:]
            out_r = self.run_sam_head(feats_rgb[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['rgb_s4'] = F.interpolate(out_r, size=(H, W), mode='bilinear')
            out_i = self.run_sam_head(feats_ir[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['ir_s4'] = F.interpolate(out_i, size=(H, W), mode='bilinear')

        return seg_logits, sam_preds, moe_loss, total_fusion_loss, final_aux_logits, edge_prompt