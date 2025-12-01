import torch
import torch.nn as nn


class SimpleFusionModule(nn.Module):
    def __init__(self, channel_dim):
        """
        初始化融合模块
        channel_dim: 该层特征图的通道数
        """
        super().__init__()
        # 目前是简单的相加，不需要参数，但为了以后改起来方便，预留了位置
        # 如果你想改成卷积融合，可以在这里定义 self.conv = nn.Conv2d(...)
        self.channel_dim = channel_dim

    def forward(self, feat_rgb, feat_ir):
        # 验证尺寸
        if feat_rgb.shape != feat_ir.shape:
            # 如果尺寸稍微有差异（通常不会），强制对齐
            feat_ir = nn.functional.interpolate(feat_ir, size=feat_rgb.shape[2:], mode='bilinear', align_corners=False)

        # 融合逻辑：目前是 Element-wise Add
        return feat_rgb + feat_ir