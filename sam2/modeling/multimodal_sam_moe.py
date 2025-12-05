import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import SimpleFusionModule
from .segformer_head import SegFormerHead
from .moe import SparseMoEBlock


class MultiModalSegFormerMoE(nn.Module):
    def __init__(self, sam_model, feature_channels, num_classes=9):
        super().__init__()
        self.sam_model = sam_model

        # --- 1. 冻结原有模块 ---
        # SAM2 编码器
        for param in self.sam_model.parameters():
            param.requires_grad = False

        # 融合模块 (暂时冻结，这一步只训 MoE)
        self.fusion_layers = nn.ModuleList([SimpleFusionModule(ch) for ch in feature_channels])
        for param in self.fusion_layers.parameters():
            param.requires_grad = False

        # SegFormer 解码器 (暂时冻结)
        self.segformer_head = SegFormerHead(in_channels=feature_channels, num_classes=num_classes)
        for param in self.segformer_head.parameters():
            param.requires_grad = False

        # --- 2. 新增：可训练的 MoE 模块 ---
        # 为 RGB 和 IR 分别建立 4 个阶段的 MoE
        # Hiera 输出 4 层特征，所以需要 4 个 MoE Block

        self.rgb_moe = nn.ModuleList([
            SparseMoEBlock(dim=ch, num_experts=11, active_experts=6)
            for ch in feature_channels
        ])

        self.ir_moe = nn.ModuleList([
            SparseMoEBlock(dim=ch, num_experts=11, active_experts=6)
            for ch in feature_channels
        ])

        # 确保 MoE 是开启梯度的
        for param in self.rgb_moe.parameters(): param.requires_grad = True
        for param in self.ir_moe.parameters(): param.requires_grad = True

    def extract_features(self, image):
        """从 Hiera 骨干提取 4 层特征"""
        backbone = self.sam_model.image_encoder.trunk
        out = backbone(image)

        # 处理输出格式
        if isinstance(out, (list, tuple)):
            features = list(out)
        elif isinstance(out, dict):
            features = [out[k] for k in sorted(out.keys())]
        else:
            raise ValueError("Unknown backbone output format")

        # Hiera Tiny 应该输出 4 层
        return features[:4]

    def forward(self, image_rgb, image_ir):
        # 1. 骨干提取 (No Grad)
        with torch.no_grad():
            raw_rgb = self.extract_features(image_rgb)
            raw_ir = self.extract_features(image_ir)

        # 2. MoE 增强 (Trainable)
        # 分别处理 RGB 和 IR 的每一层
        enhanced_rgb = []
        enhanced_ir = []
        total_aux_loss = 0.0

        for i in range(4):
            # RGB 通过 MoE
            feat_rgb, loss_rgb = self.rgb_moe[i](raw_rgb[i])
            enhanced_rgb.append(feat_rgb)
            total_aux_loss += loss_rgb

            # IR 通过 MoE
            feat_ir, loss_ir = self.ir_moe[i](raw_ir[i])
            enhanced_ir.append(feat_ir)
            total_aux_loss += loss_ir

        # 3. 融合 (No Grad)
        fused_features = []
        for i in range(4):
            f = self.fusion_layers[i](enhanced_rgb[i], enhanced_ir[i])
            fused_features.append(f)

        # 4. 解码 (No Grad)
        # 注意：这里我们假设解码器已经加载了上一阶段训练好的权重
        # 所以它能理解 fused_features 的特征分布
        logits = self.segformer_head(fused_features)

        # 上采样回原图
        logits = F.interpolate(logits, size=image_rgb.shape[2:], mode='bilinear', align_corners=False)

        return logits, total_aux_loss