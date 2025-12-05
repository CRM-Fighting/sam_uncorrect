import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import SimpleFusionModule
from .segformer_head import SegFormerHead
# 【关键修改】引入 IACR MoE
from .iacr_moe import IACRMoEBlock


class MultiModalSegFormerIACR(nn.Module):
    def __init__(self, sam_model, feature_channels, num_classes=9):
        super().__init__()
        self.sam_model = sam_model

        # 冻结原有模块 (保持控制变量)
        for param in self.sam_model.parameters(): param.requires_grad = False

        self.fusion_layers = nn.ModuleList([SimpleFusionModule(ch) for ch in feature_channels])
        for param in self.fusion_layers.parameters(): param.requires_grad = False

        self.segformer_head = SegFormerHead(in_channels=feature_channels, num_classes=num_classes)
        for param in self.segformer_head.parameters(): param.requires_grad = False

        # === 新增：IACR MoE 模块 ===
        # 保持 num_experts=11, active_experts=6 以进行公平对比

        self.rgb_moe_layers = nn.ModuleList([
            IACRMoEBlock(dim=ch, num_experts=11, active_experts=6)
            for ch in feature_channels
        ])

        self.ir_moe_layers = nn.ModuleList([
            IACRMoEBlock(dim=ch, num_experts=11, active_experts=6)
            for ch in feature_channels
        ])

        # 只训练 MoE
        for param in self.rgb_moe_layers.parameters(): param.requires_grad = True
        for param in self.ir_moe_layers.parameters(): param.requires_grad = True

    def extract_features(self, image):
        backbone = self.sam_model.image_encoder.trunk
        out = backbone(image)
        if isinstance(out, (list, tuple)):
            features = list(out)
        elif isinstance(out, dict):
            features = [out[k] for k in sorted(out.keys())]
        return features[:4]

    def forward(self, image_rgb, image_ir):
        with torch.no_grad():
            raw_rgb = self.extract_features(image_rgb)
            raw_ir = self.extract_features(image_ir)

        enhanced_rgb = []
        enhanced_ir = []
        total_aux_loss = 0.0

        for i in range(4):
            # RGB
            out_rgb, loss_rgb = self.rgb_moe_layers[i](raw_rgb[i])
            enhanced_rgb.append(out_rgb)
            total_aux_loss += loss_rgb

            # IR
            out_ir, loss_ir = self.ir_moe_layers[i](raw_ir[i])
            enhanced_ir.append(out_ir)
            total_aux_loss += loss_ir

        fused_features = []
        for i in range(4):
            f = self.fusion_layers[i](enhanced_rgb[i], enhanced_ir[i])
            fused_features.append(f)

        logits = self.segformer_head(fused_features)
        logits = F.interpolate(logits, size=image_rgb.shape[2:], mode='bilinear', align_corners=False)

        return logits, total_aux_loss