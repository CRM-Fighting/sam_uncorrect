import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class BinarySUMLoss(nn.Module):
    """
    严格复现 SUM 论文公式 1 (Focal) 和 公式 2 (Dice)
    论文参考: Uncertainty-aware Fine-tuning of Segmentation Foundation Models [cite: 182-186]
    """

    def __init__(self, theta=0.6, focal_weight=20.0, dice_weight=1.0):
        super().__init__()
        self.theta = theta  # Dice 过滤阈值
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred_logits, gt_semantic, uncertain_map):
        """
        pred_logits: [B, 1, H, W]
        gt_semantic: [B, H, W] (原始9类标签)
        uncertain_map: [B, 1, H, W] (0~1, 值越大越不确定)
        """
        # 1. 标签二值化 (前景=1, 背景=0)
        gt_binary = (gt_semantic > 0).float().unsqueeze(1)

        # 2. 尺寸对齐
        if uncertain_map.shape[-2:] != pred_logits.shape[-2:]:
            uncertainty = F.interpolate(uncertain_map, size=pred_logits.shape[-2:], mode='bilinear')
        else:
            uncertainty = uncertain_map

        # 3. 计算确定性图 (ci = 1 - ui) [cite: 142]
        certainty = 1.0 - uncertainty

        # --- 公式 1: Uncertainty-aware Focal Loss (加权) [cite: 182] ---
        # Loss = ci * Focal(p, y)
        loss_focal_raw = sigmoid_focal_loss(pred_logits, gt_binary, alpha=0.25, gamma=2, reduction='none')
        loss_focal_weighted = loss_focal_raw * certainty
        term_focal = loss_focal_weighted.mean() * self.focal_weight

        # --- 公式 2: Uncertainty-aware Dice Loss (阈值截断) [cite: 186] ---
        # 只有 certainty > theta 的像素参与计算
        valid_mask = (certainty > self.theta).float()

        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * gt_binary * valid_mask).sum(dim=(2, 3))
        union = (pred_probs * valid_mask).sum(dim=(2, 3)) + (gt_binary * valid_mask).sum(dim=(2, 3))

        # 平滑项设为 1.0 防止除零
        dice_score = (2. * intersection + 1.0) / (union + 1.0)
        term_dice = (1.0 - dice_score.mean()) * self.dice_weight

        return term_focal + term_dice


class StandardSegLoss(nn.Module):
    """ SegFormer 分支专用 (CE + Dice) """

    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weights = torch.ones(num_classes)
        self.ce_weights[0] = 0.4  # 降低背景权重

    def forward(self, preds, labels):
        if self.ce_weights.device != preds.device:
            self.ce_weights = self.ce_weights.to(preds.device)

        loss_ce = F.cross_entropy(preds, labels, weight=self.ce_weights, ignore_index=255)

        pred_soft = F.softmax(preds, dim=1)
        target_one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        inter = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        loss_dice = 1 - (2. * inter + 1) / (union + 1)

        return 0.5 * loss_ce + 0.5 * loss_dice.mean()