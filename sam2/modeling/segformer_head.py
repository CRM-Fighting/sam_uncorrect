import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim=256, num_classes=9):  # MSRS通常是9类(包含背景)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 4个阶段的特征维度对齐
        self.linear_c4 = MLP(input_dim=in_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=in_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=in_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=in_channels[0], embed_dim=embedding_dim)

        # 融合层
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(0.1)
        # 最终预测层
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def init_weights(self):
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # inputs: [c1, c2, c3, c4] 对应 1/4, 1/8, 1/16, 1/32 尺寸
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        # 1. MLP 投影
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # 2. 上采样到 C1 的尺寸 (通常是原图的 1/4)
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        # _c1 不需要上采样

        # 3. 拼接与融合
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # 4. 预测
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x