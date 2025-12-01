import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import SimpleFusionModule
from .segformer_head import SegFormerHead


class MultiModalSegFormer(nn.Module):
    def __init__(self, sam_model, feature_channels, num_classes=9):
        super().__init__()
        self.sam_model = sam_model

        # 1. 默认冻结 SAM2
        for param in self.sam_model.parameters():
            param.requires_grad = False

        # 2. 融合模块 (4个阶段)
        self.fusion_layers = nn.ModuleList([
            SimpleFusionModule(ch) for ch in feature_channels
        ])

        # 3. SegFormer 解码器
        self.segformer_head = SegFormerHead(in_channels=feature_channels, num_classes=num_classes)
        self.segformer_head.init_weights()  # 初始化权重

    def extract_features(self, image):
        """辅助函数：处理SAM2 backbone的返回值"""
        # 获取 image_encoder 的 trunk (即 Hiera)
        backbone = self.sam_model.image_encoder.trunk

        # Hiera 的 forward 可能会返回多层特征列表，或者直接返回最后一层
        # 我们需要确保拿到所有中间层特征。
        # 这里的调用方式取决于 sam2 版本，通常直接调用即可拿到多尺度特征
        backbone_out = backbone(image)

        # 情况A: 返回的是列表/元组 (标准 Hiera 行为)
        if isinstance(backbone_out, (list, tuple)):
            features = list(backbone_out)

        # 情况B: 返回的是字典
        elif isinstance(backbone_out, dict):
            # 这里的 key 可能是 "res2", "res3"... 需要按顺序排
            # 或者是 stage names
            sorted_keys = sorted(backbone_out.keys())
            features = [backbone_out[k] for k in sorted_keys]

        else:
            # 情况C: 只返回了最后一层 Tensor (不常见，除非 config 里修改了 return_interm_layers)
            raise ValueError(f"Backbone output type {type(backbone_out)} not supported. Expect list or dict.")

        # 严格校验：我们需要 4 层特征 [C1, C2, C3, C4]
        # 如果 backbone 输出包含不用的一层 (比如最初的 stem)，需要在这里切片
        if len(features) != 4:
            # Hiera Tiny 通常输出 4 个阶段。如果多了，可能需要 features = features[-4:]
            print(f"Warning: Backbone returned {len(features)} levels. Expected 4.")

        return features

    def forward(self, image_rgb, image_ir, use_segformer=True):
        # 1. 提取特征 (No Grad for Encoder)
        with torch.no_grad():
            features_rgb = self.extract_features(image_rgb)
            features_ir = self.extract_features(image_ir)

        # 2. 逐层融合
        fused_features = []
        for i in range(4):
            f_fused = self.fusion_layers[i](features_rgb[i], features_ir[i])
            fused_features.append(f_fused)

        # 3. 解码
        if use_segformer:
            seg_logits = self.segformer_head(fused_features)
            # 上采样回原图尺寸 (640x480)
            seg_logits = F.interpolate(seg_logits, size=image_rgb.shape[2:], mode='bilinear', align_corners=False)
            return seg_logits
        else:
            return fused_features