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
    def __init__(self, in_channels, embedding_dim=256, num_classes=9, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.linear_c4 = MLP(input_dim=in_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=in_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=in_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=in_channels[0], embed_dim=embedding_dim)

        # [新增] 细节聚合投影 (Detail Aggregation from CAINet)
        # 接收 Backbone 原始 Stage 1 特征
        self.detail_proj = nn.Sequential(
            nn.Conv2d(in_channels[0], embedding_dim, 1, bias=False),
            nn.GroupNorm(16, embedding_dim),
            nn.ReLU(inplace=True)
        )

        # 融合层 (GroupNorm版)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embedding_dim), # 解决 Batch=1 问题
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, detail_feat=None):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        # 基础融合
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # [新增] 细节注入 (Detail Aggregation)
        if detail_feat is not None:
            d_feat = self.detail_proj(detail_feat)
            if d_feat.shape[-2:] != _c.shape[-2:]:
                d_feat = F.interpolate(d_feat, size=_c.shape[-2:], mode='bilinear', align_corners=False)
            _c = _c + d_feat  # 残差连接

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x