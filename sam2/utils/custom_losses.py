import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class BinarySUMLoss(nn.Module):
    # (保持原有的 BinarySUMLoss 不变)
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
    """ [修改] 优化权重的 SegLoss """
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weights = torch.ones(num_classes)
        self.ce_weights[0] = 0.4

    def forward(self, preds, labels):
        if self.ce_weights.device != preds.device:
            self.ce_weights = self.ce_weights.to(preds.device)

        loss_ce = F.cross_entropy(preds, labels, weight=self.ce_weights, ignore_index=255)

        pred_soft = F.softmax(preds, dim=1)
        target_one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        inter = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        loss_dice = 1 - (2. * inter + 1) / (union + 1)

        # [修改] 提高 Dice 权重: 0.5 -> 0.7, 降低 CE: 0.5 -> 0.3
        # 这样模型会更在意 IoU，而不是单纯的像素分类准确率
        return 0.3 * loss_ce + 0.7 * loss_dice.mean()