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

        # Focal Loss
        loss_focal_raw = sigmoid_focal_loss(pred_logits, gt_binary, alpha=0.25, gamma=2, reduction='none')
        loss_focal_weighted = loss_focal_raw * certainty
        term_focal = loss_focal_weighted.mean() * self.focal_weight

        # Dice Loss
        valid_mask = (certainty > self.theta).float()
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * gt_binary * valid_mask).sum(dim=(2, 3))
        union = (pred_probs * valid_mask).sum(dim=(2, 3)) + (gt_binary * valid_mask).sum(dim=(2, 3))
        dice_score = (2. * intersection + 1.0) / (union + 1.0)
        term_dice = (1.0 - dice_score.mean()) * self.dice_weight

        return term_focal + term_dice


class StandardSegLoss(nn.Module):
    def __init__(self, num_classes=9, class_weights=None, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # 边缘损失权重
        self.edge_loss_weight = 0.1

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = torch.tensor([0.1] + [1.0] * (num_classes - 1))

    def _generate_gt_edge(self, labels):
        """
        [工业级边缘提取] 利用 One-Hot 形态学梯度生成完美闭合边缘
        """
        # 1. 转 One-Hot [B, NumClasses, H, W]
        target_one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # 2. 拉普拉斯算子提取所有类别的边界 (逐通道计算)
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                         [-1, 8, -1],
                                         [-1, -1, -1]], dtype=torch.float32, device=labels.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(self.num_classes, 1, 1, 1)

        # groups=num_classes 保证通道独立
        edge_per_class = F.conv2d(target_one_hot, laplacian_kernel, padding=1, groups=self.num_classes)

        # 3. 聚合 & 二值化 (只要任意通道有边缘，就是边缘)
        edge_map = torch.abs(edge_per_class).sum(dim=1, keepdim=True)
        edge_map = (edge_map > 0.1).float()  # [B, 1, H, W]

        return edge_map

    def forward(self, preds, labels, pred_edge=None):
        device = preds.device
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)

        # 1. CE Loss
        loss_ce = F.cross_entropy(preds, labels, weight=self.class_weights, ignore_index=self.ignore_index)

        # 2. Dice Loss
        probs = F.softmax(preds, dim=1)
        valid_mask = (labels != self.ignore_index).float().unsqueeze(1)

        target_safe = labels.clone()
        target_safe[labels == self.ignore_index] = 0
        target_one_hot = F.one_hot(target_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * target_one_hot * valid_mask).sum(dim=(2, 3))
        cardinality = (probs * valid_mask).sum(dim=(2, 3)) + (target_one_hot * valid_mask).sum(dim=(2, 3))
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)

        # Weighted Dice
        weighted_dice_loss = ((1.0 - dice_score) * self.class_weights).mean()

        total_loss = loss_ce + 1.0 * weighted_dice_loss

        # 3. Edge Loss
        loss_edge_val = torch.tensor(0.0, device=device)
        if pred_edge is not None:
            # 实时生成 GT Edge
            with torch.no_grad():
                gt_edge = self._generate_gt_edge(labels)

            # 对齐尺寸
            if pred_edge.shape[-2:] != gt_edge.shape[-2:]:
                pred_edge = F.interpolate(pred_edge, size=gt_edge.shape[-2:], mode='bilinear')

            # 【核心修复】使用 autocast(enabled=False) 强制退出混合精度上下文
            # 这样 PyTorch 就允许使用 binary_cross_entropy 了
            with torch.cuda.amp.autocast(enabled=False):
                # 必须转为 float32 (FP32)
                pred_edge_safe = torch.clamp(pred_edge.float(), min=1e-6, max=1.0 - 1e-6)
                gt_edge_safe = gt_edge.float()

                loss_edge_val = F.binary_cross_entropy(pred_edge_safe, gt_edge_safe)

            total_loss += self.edge_loss_weight * loss_edge_val

        # 返回总 Loss 和 详细 Loss 字典
        loss_dict = {
            "CE": loss_ce.item(),
            "Dice": weighted_dice_loss.item(),
            "Edge": loss_edge_val.item()
        }
        return total_loss, loss_dict