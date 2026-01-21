import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class BinarySUMLoss(nn.Module):
    def __init__(self, theta=0.6, focal_weight=20.0, dice_weight=1.0):
        super().__init__()
        self.theta = theta
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred_logits, gt_semantic, uncertain_map):
        gt_binary = (gt_semantic > 0).float().unsqueeze(1)
        if uncertain_map.shape[-2:] != pred_logits.shape[-2:]:
            uncertainty = F.interpolate(uncertain_map, size=pred_logits.shape[-2:], mode='bilinear')
        else:
            uncertainty = uncertain_map
        certainty = 1.0 - uncertainty

        loss_focal_raw = sigmoid_focal_loss(pred_logits, gt_binary, alpha=0.25, gamma=2, reduction='none')
        loss_focal_weighted = loss_focal_raw * certainty
        term_focal = loss_focal_weighted.mean() * self.focal_weight

        valid_mask = (certainty > self.theta).float()
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * gt_binary * valid_mask).sum(dim=(2, 3))
        union = (pred_probs * valid_mask).sum(dim=(2, 3)) + (gt_binary * valid_mask).sum(dim=(2, 3))
        dice_score = (2. * intersection + 1.0) / (union + 1.0)
        term_dice = (1.0 - dice_score.mean()) * self.dice_weight

        return term_focal + term_dice


class StandardSegLoss(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        # [保持] 类别加权：背景 0.1，其他 1.0
        self.ce_weights = torch.tensor([0.1] + [1.0] * (num_classes - 1))

    def forward(self, preds, labels):
        if self.ce_weights.device != preds.device:
            self.ce_weights = self.ce_weights.to(preds.device)

        # 1. CE Loss
        loss_ce = F.cross_entropy(preds, labels, weight=self.ce_weights, ignore_index=255)

        # 2. Dice Loss (针对存在的类别)
        pred_soft = F.softmax(preds, dim=1)
        target_one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (pred_soft * target_one_hot).sum(dims)
        cardinality = pred_soft.sum(dims) + target_one_hot.sum(dims)

        dice_score = (2. * intersection + 1.0) / (cardinality + 1.0)
        dice_loss = 1.0 - dice_score

        return 0.5 * loss_ce + 0.5 * dice_loss.mean()