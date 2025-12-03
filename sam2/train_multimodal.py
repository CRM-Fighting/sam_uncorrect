import os
import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 引入模型组件
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam import MultiModalSegFormer


# ==========================================
# 1. 核心组件：加权损失函数 & mIoU 计算器
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) Logits
        # targets: (B, H, W) Labels
        inputs = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]

        # 将 Label 转 One-hot
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 计算 Dice
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 1 - Dice 均值
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=9, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha  # CE 权重 (降低)
        self.beta = beta  # Dice 权重 (提高，Dice对不平衡数据更鲁棒)

        # 【关键修改】类别加权：背景(0)设为0.1，其他物体设为1.0
        # 强迫模型关注非背景像素
        self.class_weights = torch.ones(num_classes)
        self.class_weights[0] = 0.1

        self.dice = DiceLoss(ignore_index=255)

    def forward(self, preds, labels):
        # 动态将权重移到当前设备
        if self.class_weights.device != preds.device:
            self.class_weights = self.class_weights.to(preds.device)

        # 加权交叉熵
        loss_ce = F.cross_entropy(preds, labels, weight=self.class_weights, ignore_index=255)
        loss_dice = self.dice(preds, labels)

        return self.alpha * loss_ce + self.beta * loss_dice


class IOUEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_classes ** 2)
        return count.reshape(self.num_classes, self.num_classes)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def get_metrics(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)

        # 像素准确率 (PA)
        pixel_acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return miou, iou, pixel_acc


# ==========================================
# 2. 数据集定义
# ==========================================
class MSRSDataset(Dataset):
    def __init__(self, root_dirs, limit=None):
        self.ir_dir = root_dirs['ir']
        self.vis_dir = root_dirs['vis']
        self.label_dir = root_dirs['label']
        self.filenames = sorted([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])

        if limit is not None:
            self.filenames = self.filenames[:limit]
            print(f"注意：数据集已限制为前 {limit} 张图片。")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        ir_path = os.path.join(self.ir_dir, fname)
        vis_path = os.path.join(self.vis_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        # 读取与预处理 (480x640)
        target_size = (640, 480)
        vis = Image.open(vis_path).convert('RGB').resize(target_size, Image.BILINEAR)
        ir = Image.open(ir_path).convert('RGB').resize(target_size, Image.BILINEAR)
        label = Image.open(label_path).resize(target_size, Image.NEAREST)

        vis_t = torch.from_numpy(np.array(vis)).float().permute(2, 0, 1) / 255.0
        ir_t = torch.from_numpy(np.array(ir)).float().permute(2, 0, 1) / 255.0
        label_t = torch.from_numpy(np.array(label)).long()

        return vis_t, ir_t, label_t, fname  # 返回文件名以便可视化


# ==========================================
# 3. 辅助函数：冻结与绘图
# ==========================================
def configure_model_freezing(model, freeze_encoder=True, freeze_fusion=True, freeze_decoder=False):
    """冻结配置函数，控制哪些模块参与训练"""
    print("\n[冻结配置] 正在设置模型可训练模块...")

    # 1. 先把所有参数"锁住"（冻结）
    for param in model.parameters():
        param.requires_grad = False

    # 2. 遍历模型的所有子模块，根据名字决定是否"解锁"
    for name, param in model.named_parameters():
        # (A) SegFormer 解码器：我们需要训练它，所以解锁！
        if "segformer_head" in name:
            if not freeze_decoder:
                param.requires_grad = True

        # (B) 融合模块：如果是简单的相加，它没有参数；如果是卷积融合，需要根据情况解锁
        if "fusion_layers" in name:
            if not freeze_fusion:
                param.requires_grad = True

        # (C) SAM2 编码器：通常保持冻结
        if "sam_model" in name:
            if not freeze_encoder:  # 只有当你显式说"我不冻结编码器"时才解锁
                param.requires_grad = True

    # 统计一下真正参与训练的参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" -> 总参数量: {total_params / 1e6:.2f} M")
    print(f" -> 参与训练参数: {trainable_params / 1e6:.2f} M (主要集中在 SegFormer 解码器)")
    print("[冻结配置] 完成。\n")


def save_validation_sample(save_dir, epoch, fname, vis_t, label_t, pred_t):
    """保存每一轮的验证效果图：可见光 | 真实标签 | 预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 转换格式
    vis_np = (vis_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    label_np = label_t.cpu().numpy().astype(np.uint8) * 30  # 乘30为了让类别颜色拉开差距便于观察
    pred_np = pred_t.cpu().numpy().astype(np.uint8) * 30

    # 拼接
    label_img = np.stack([label_np] * 3, axis=-1)
    pred_img = np.stack([pred_np] * 3, axis=-1)
    combined = np.hstack([vis_np, label_img, pred_img])

    Image.fromarray(combined).save(os.path.join(save_dir, f"epoch_{epoch}_{fname}"))


def plot_curves(history, save_dir):
    plt.figure(figsize=(15, 5))

    # 1. Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch');
    plt.legend()

    # 2. mIoU 曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['val_miou'], 'g-', label='Val mIoU')
    plt.title('mIoU Curve (Higher is Better)')
    plt.xlabel('Epoch');
    plt.legend()

    # 3. Acc 曲线 (作为对比)
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Pixel Accuracy (Reference)')
    plt.xlabel('Epoch');
    plt.legend()

    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()


# ==========================================
# 4. 训练主流程
# ==========================================
def train_model_process(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = "checkpoints"
    vis_dir = os.path.join(save_dir, "vis_progress")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    model.to(device)

    # 设置冻结 (在这个阶段，只训练 SegFormer Head)
    configure_model_freezing(model, freeze_encoder=True, freeze_fusion=True, freeze_decoder=False)

    # 定义优化器 - 只优化需要训练的参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 定义损失函数 (使用改进的混合 Loss)
    criterion = SegmentationLoss(num_classes=9, alpha=0.3, beta=0.7)

    # mIoU 评估器
    evaluator = IOUEvaluator(num_classes=9)

    # 记录训练过程
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_miou": []  # 新增：记录每个epoch的mIoU
    }

    best_miou = 0.0
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # 每个 Epoch 分为训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0  # 简单像素级准确率
            total_pixels = 0

            # 重置评估器（仅验证阶段使用）
            if phase == 'val':
                evaluator.reset()

            # 进度条
            pbar = tqdm(dataloader, desc=f"{phase} Phase")

            # 只保存该轮次第一张图用于可视化（验证阶段）
            saved_sample = False

            for vis, ir, labels, fnames in pbar:
                vis = vis.to(device)
                ir = ir.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 只有训练阶段才开启梯度记录
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向传播 (传入两张图)
                    outputs = model(vis, ir)

                    # 计算 Loss
                    loss = criterion(outputs, labels)

                    # 简单计算预测类别用于计算准确率
                    preds = torch.argmax(outputs, dim=1)

                    # 反向传播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * vis.size(0)
                # 计算正确预测的像素数 (忽略 Label=255 的背景)
                mask = labels != 255
                running_corrects += torch.sum((preds == labels) & mask)
                total_pixels += torch.sum(mask)

                # 验证阶段：计算 mIoU 和保存可视化
                if phase == 'val':
                    # 计算 mIoU
                    evaluator.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

                    # 可视化保存 (每轮只存第一张验证图)
                    if not saved_sample:
                        save_validation_sample(vis_dir, epoch + 1, fnames[0], vis[0], labels[0], preds[0])
                        saved_sample = True

                # 更新进度条显示的 Loss
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / total_pixels if total_pixels > 0 else 0.0

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录历史
            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

                # 计算并记录 mIoU
                miou, class_ious, pixel_acc = evaluator.get_metrics()
                history["val_miou"].append(miou)
                print(f"Val mIoU: {miou:.4f} | Pixel Acc: {pixel_acc:.4f}")
                print(f"Class IoU: {np.round(class_ious, 3)}")

                # 深拷贝模型参数 (保存验证集 mIoU 最高的模型)
                if miou > best_miou:
                    best_miou = miou
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # 保存最佳模型
                    save_path = os.path.join(save_dir, "best_multimodal_model.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"Found better model (mIoU: {miou:.4f}), saving weights...")

        # 每轮都保存最新状态
        latest_path = os.path.join(save_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Best Val mIoU: {best_miou:.4f}")

    # 加载最佳权重
    model.load_state_dict(best_model_wts)

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_multimodal_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # 绘制训练曲线
    plot_curves(history, save_dir)
    print(f"训练曲线已保存到: {save_dir}/training_metrics.png")
    print(f"可视化样本已保存到: {vis_dir}")

    return pd.DataFrame(history)


# ==========================================
# 5. 入口
# ==========================================
def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 路径配置 - 保持原样不变
    train_dirs = {
        'vis': r"sam2/data/MSRS/train/vis",
        'ir': r"sam2/data/MSRS/train/ir",
        'label': r"sam2/data/MSRS/train/Segmentation_labels"
    }
    # 验证集切片
    val_dirs = {
        'vis': r"sam2/data/MSRS/test/vis",
        'ir': r"sam2/data/MSRS/test/ir",
        'label': r"sam2/data/MSRS/test/Segmentation_labels"
    }

    # 模型配置 - 保持原样不变
    sam2_config = "sam2.1/sam2.1_hiera_t.yaml"
    sam2_ckpt = "checkpoints/sam2.1_hiera_tiny.pt"

    # 构建模型
    print("1. 构建基础模型...")
    base_sam = build_sam2(sam2_config, sam2_ckpt, device="cpu")  # 初始先放CPU，后续自动转
    model = MultiModalSegFormer(base_sam, feature_channels=[96, 192, 384, 768], num_classes=9)

    # 准备数据
    print("2. 准备数据...")
    train_ds = MSRSDataset(train_dirs)
    val_ds = MSRSDataset(val_dirs, limit=100)  # 验证前100张

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    print(f"训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")

    # 开始训练
    print("3. 开始训练循环...")
    history = train_model_process(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0001)

    print("\n4. 训练完成！")


if __name__ == "__main__":
    main()