# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized
"""
loss_fns.py 是 SAM 2 训练的「损失函数工具箱」，核心功能是定义模型预测结果与真实标签的差距计算规则—— 这些规则会指导模型通过反向传播更新参数，
最终实现 “分割更准确、IoU 预测更靠谱、遮挡判断更精准” 的目标。所有损失函数都基于 PyTorch 实现，贴合 SAM 2 的训练任务（图像 + 视频分割），
主要包含 4 类核心损失：掩码分割损失（focal + dice）、IoU 预测损失（MAE）、遮挡预测损失（交叉熵）、组合损失（加权求和）。

这份官方代码是 SAM 2 “交互式分割训练” 的专属损失工具，比之前推导的版本更复杂 —— 核心适配了 SAM 2 的两大特性：
多掩码预测：一个点击可能输出多个候选掩码（比如点击 “轮胎”，模型可能预测 “轮胎” 或 “整辆车”）；
多步骤交互：模拟用户多次点击修正（比如第一次点击没对准，第二次补充点击），需要累加每步损失。
所有函数 / 类都围绕 “如何精准计算‘多步骤、多掩码’与真实标签的差距” 设计，最终通过反向传播让模型学会 “选对掩码、修正错误”。
"""
def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    """
    功能：计算预测掩码与真实掩码的Dice损失（重叠度损失）
    Args:
        inputs: 模型预测的掩码 → 形状可能是 [N,1,H,W]（单掩码）或 [N,M,H,W]（多掩码，M是候选掩码数）
        targets: 真实掩码 → 形状 [N,1,H,W]（0=背景，1=目标）
        num_objects: 批次中的目标总数 → 用于归一化损失（避免批次大小影响）
        loss_on_multimask: 是否开启“多掩码模式” → True时inputs是[N,M,H,W]，False是[N,1,H,W]
    Returns:
        归一化后的Dice损失（标量）
    """

    # 步骤1：Sigmoid激活 → 把预测值（任意范围）转成0-1概率（符合“掩码概率”的物理含义）
    inputs = inputs.sigmoid()    # 语法：PyTorch逐元素激活，σ(x)=1/(1+e^-x)，比如输入10→0.999，输入-10→0.000
    # 步骤2：展平空间维度 → 把[H,W]的图像转成1维数组（方便批量计算所有像素的损失）
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)   # 语法：flatten(2) → 从第2维（H）开始展平，前2维（N,M）保留
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)     # 步骤3：计算Dice系数 → 核心是“交集/并集”，衡量重叠度 # 分子：2*交集 → sum(-1)是对最后一维（H*W）求和，得到[N,M]或[N]
    denominator = inputs.sum(-1) + targets.sum(-1)    # 分母：并集（预测像素数+真实像素数）

    # 步骤4：Dice损失 = 1 - Dice系数 → 系数越大（重叠度高），损失越小
    # +1是“平滑项” → 避免分母为0（比如预测和真实都是全0时，分母=0会报错）
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects    # 多掩码时loss是[N,M]，直接除以num_objects（标量）
    return loss.sum() / num_objects   # 单掩码时loss是[N]，先求和再归一化
    """
    Dice 损失不关注 “单个像素是否分类对”，只关注 “预测和真实的重叠面积”—— 特别适合 SAM 2 中 “小目标分割”（比如分割 “手指尖”“小零件”）。
    """

"""
Focal Loss是为了解决类别不平衡问题而设计的，特别是在目标检测中正负样本比例极度失衡的情况。
"""
def sigmoid_focal_loss(
    inputs,    # 模型原始输出（未经过sigmoid的logits）
    targets,    # 真实标签（0或1）
    num_objects,    # 批次中的目标数量（用于归一化）
    alpha: float = 0.25,    # 平衡正负样本的权重
    gamma: float = 2,   # 调节难易样本的权重
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()    # 将模型输出转换为概率值（0-1之间）
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")     #基础交叉熵损失
    p_t = prob * targets + (1 - prob) * (1 - targets)
    """
    这个公式的精妙之处：
    当 targets=1（正样本）时：p_t = prob
    当 targets=0（负样本）时：p_t = 1 - prob
    物理意义：p_t 表示模型预测正确的概率
    正样本：预测概率越高，p_t越大 → 模型越有信心
    负样本：预测概率越低，p_t越大 → 模型越有信心
    """
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects

# 3. IoU 预测损失（MAE）：优化“模型预测的IoU值与真实IoU的差距”
"""
训练模型自我评估分割质量的能力，让模型学会预测自己生成掩码的IoU分数。
"""
def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    参数说明：
    inputs：模型预测的掩码logits
    targets：真实掩码
    pred_ious：模型预测的IoU分数（这是关键！
    num_objects：目标数量（归一化用）
    loss_on_multimask：是否多掩码模式
    use_l1_loss：是否使用L1损失（论文中提到的新改进）
    """
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    """
    语法：assert 是Python的断言语句，如果条件为False会抛出异常
    作用：确保输入和目标的维度都是4维 [N, M, H, W]
    """
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    """
    # 假设 inputs 形状: [1, 2, 2, 2] (N=1, M=2, H=2, W=2)
    inputs = torch.tensor([[
        [[0.5, -0.3], [1.2, -0.8]],   # 掩码1
        [[-0.1, 0.9], [0.7, -0.5]]    # 掩码2
    ]])
    # flatten(2) 展平空间维度 → [1, 2, 4]
    inputs_flat = inputs.flatten(2)  
    # [[[0.5, -0.3, 1.2, -0.8],   # 掩码1展平
    #   [-0.1, 0.9, 0.7, -0.5]]]  # 掩码2展平
    # > 0 进行二值化
    pred_mask = inputs_flat > 0
    # [[[True, False, True, False],   # 掩码1二值化
    #   [False, True, True, False]]]  # 掩码2二值化
    """


    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    """
    # 假设 gt_mask 也是 [1, 2, 4]
    gt_mask = torch.tensor([[
        [[True, True, False, False],   # 真实掩码1
         [True, True, False, False]]   # 真实掩码2
    ]])
    
    # pred_mask & gt_mask 逐元素与运算
    intersection = pred_mask & gt_mask
    # [[[True & True, False & True, True & False, False & False],   # 掩码1交集
    #   [False & True, True & True, True & False, False & False]]]  # 掩码2交集
    # = [[[True, False, False, False],
    #     [False, True, False, False]]]
    
    # torch.sum(..., dim=-1) 对最后一个维度求和
    area_i = torch.sum(intersection, dim=-1).float()
    # 掩码1: True(1) + False(0) + False(0) + False(0) = 1
    # 掩码2: False(0) + True(1) + False(0) + False(0) = 1
    # area_i = [[1.0, 1.0]]
    # pred_mask | gt_mask 逐元素或运算
    union = pred_mask | gt_mask
    # [[[True | True, False | True, True | False, False | False],
    #   [False | True, True | True, True | False, False | False]]]
    # = [[[True, True, True, False],
    #     [True, True, True, False]]]
    
    area_u = torch.sum(union, dim=-1).float()
    # 掩码1: 1+1+1+0 = 3
    # 掩码2: 1+1+1+0 = 3  
    # area_u = [[3.0, 3.0]]
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    # torch.clamp(area_u, min=1.0) 确保分母至少为1，避免除0
    # 掩码1: 1.0 / 3.0 = 0.333
    # 掩码2: 1.0 / 3.0 = 0.333
    # actual_ious = [[0.333, 0.333]]
    """


    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects
    """
    # 假设 pred_ious = [[0.8, 0.6]]（模型预测的IoU分数）
    # L1损失（绝对值损失）
    l1_loss = |0.8-0.333| + |0.6-0.333| = 0.467 + 0.267 = 0.734
    # MSE损失（平方损失）  
    mse_loss = (0.8-0.333)² + (0.6-0.333)² = 0.218 + 0.071 = 0.289
    
    reduction="none"的作用：
    不进行降维，保持每个样本的独立损失
    返回形状与输入相同的损失张量
    """

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
