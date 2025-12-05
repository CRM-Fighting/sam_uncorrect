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

# 引入你的模型组件
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam_moe import MultiModalSegFormerMoE
from sam2.modeling.multimodal_sam_iacr import MultiModalSegFormerIACR
# ================= 全局配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
# 上一阶段（只训练SegFormer）得到的最佳模型路径
PRETRAINED_SEG_PATH = "checkpoints/best_msrs_model.pth"

# 结果保存路径
CHECKPOINT_DIR = "checkpoints_moe"
VIS_DIR = os.path.join(CHECKPOINT_DIR, "vis_plots")

TRAIN_DIRS = {
    'vi': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/vi",
    'ir': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/ir",
    'label': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/Segmentation_labels"
}
VAL_DIRS = {
    'vi': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi",
    'ir': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir",
    'label': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels"
}


# ================= 工具类：mIoU 计算器 =================
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
        return miou, iou


# ================= 工具类：绘图 =================
def plot_training_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 6))

    # 1. 总损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Total Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. mIoU 曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_miou'], 'b--', label='Train mIoU')
    plt.plot(epochs, history['val_miou'], 'g-', label='Val mIoU')
    plt.title('mIoU Curve (The Higher The Better)')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    # 3. MoE 负载均衡损失曲线 (观察专家是否坍塌)
    if 'train_aux_loss' in history:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['train_aux_loss'], 'orange', label='Aux Loss (Load Balance)')
        plt.title('MoE Aux Loss (Should Decrease)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线图已保存至: {save_path}")


# ================= 损失函数 =================
class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 针对 MSRS 数据集：背景(类0)占比极大，给予极低权重(0.1)
        # 强迫模型去学类1-8的特征
        self.weights = torch.ones(num_classes)
        self.weights[0] = 0.1

    def forward(self, preds, labels):
        if self.weights.device != preds.device:
            self.weights = self.weights.to(preds.device)

        # 1. Weighted Cross Entropy
        loss_ce = F.cross_entropy(preds, labels, weight=self.weights, ignore_index=255)

        # 2. Dice Loss (对类别不平衡更鲁棒)
        preds_soft = F.softmax(preds, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        inter = (preds_soft * labels_one_hot).sum(dim=(2, 3))
        union = preds_soft.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))
        # Dice = 2*I / U
        dice = 1 - (2. * inter + 1) / (union + 1)
        loss_dice = dice.mean()

        # 组合: 30% CE + 70% Dice
        return 0.3 * loss_ce + 0.7 * loss_dice


# ================= 数据集 =================
class MSRSDataset(Dataset):
    def __init__(self, root_dirs, limit=None):
        self.vis_dir = root_dirs['vi']
        self.ir_dir = root_dirs['ir']
        self.label_dir = root_dirs['label']
        self.filenames = sorted([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])
        if limit: self.filenames = self.filenames[:limit]

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        # 读取
        vis = Image.open(os.path.join(self.vis_dir, name)).convert('RGB').resize((640, 480))
        ir = Image.open(os.path.join(self.ir_dir, name)).convert('RGB').resize((640, 480))
        label = Image.open(os.path.join(self.label_dir, name)).resize((640, 480), Image.NEAREST)

        # 转 Tensor
        vis_t = torch.from_numpy(np.array(vis)).float().permute(2, 0, 1) / 255.0
        ir_t = torch.from_numpy(np.array(ir)).float().permute(2, 0, 1) / 255.0
        label_t = torch.from_numpy(np.array(label)).long()

        return vis_t, ir_t, label_t


# ================= 训练函数 =================
def train_iacr_process():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    print("Building IACR-MoE Model...")
    base_sam = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cpu")
    # 初始化 IACR 模型
    model = MultiModalSegFormerIACR(base_sam, [96, 192, 384, 768], num_classes=9)

    # 加载权重 (逻辑不变)
    if os.path.exists(PRETRAINED_SEG_PATH):
        print(f"Loading baseline weights from {PRETRAINED_SEG_PATH}...")
        pretrained_dict = torch.load(PRETRAINED_SEG_PATH, map_location="cpu")
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
    else:
        print("Error: Baseline weights not found!")
        return

    model.to(device)

    # 冻结检查
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable Params (IACR MoE only): {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")

    # 数据加载 (保持一致)
    train_loader = DataLoader(MSRSDataset(TRAIN_DIRS), batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(MSRSDataset(VAL_DIRS, limit=100), batch_size=2, shuffle=False, num_workers=2)

    # 优化器 (保持一致)
    optimizer = optim.AdamW(trainable_params, lr=0.0001)
    criterion = SegmentationLoss(num_classes=9)

    # Aux Weight 可以保持 0.05 或者根据正交损失适当调小一点点，建议保持 0.05 先试
    AUX_WEIGHT = 0.2

    # 记录器
    history = {
        'train_loss': [], 'val_loss': [],
        'train_miou': [], 'val_miou': [],
        'train_aux_loss': []
    }
    best_miou = 0.0

    # 评估器
    evaluator = IOUEvaluator(num_classes=9)

    # 6. 循环
    NUM_EPOCHS = 30
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # --- Train Phase ---
        model.train()
        train_loss_meter = 0
        train_aux_meter = 0
        evaluator.reset()

        pbar = tqdm(train_loader, desc="Train", ncols=100)
        for vis, ir, label in pbar:
            vis, ir, label = vis.to(device), ir.to(device), label.to(device)
            optimizer.zero_grad()

            # Forward: (logits, aux_loss)
            preds, aux_loss = model(vis, ir)

            # 损失计算
            seg_loss = criterion(preds, label)
            total_loss = seg_loss + AUX_WEIGHT * aux_loss

            total_loss.backward()
            optimizer.step()

            # 记录
            train_loss_meter += total_loss.item()
            train_aux_meter += aux_loss.item()

            # 计算训练集 mIoU (粗略)
            preds_mask = torch.argmax(preds, dim=1)
            evaluator.add_batch(label.cpu().numpy(), preds_mask.cpu().numpy())

            pbar.set_postfix({
                "L_Total": f"{total_loss.item():.3f}",
                "L_Aux": f"{aux_loss.item():.3f}"
            })

        train_miou, _ = evaluator.get_metrics()
        history['train_loss'].append(train_loss_meter / len(train_loader))
        history['train_aux_loss'].append(train_aux_meter / len(train_loader))
        history['train_miou'].append(train_miou)

        # --- Val Phase ---
        model.eval()
        val_loss_meter = 0
        evaluator.reset()

        with torch.no_grad():
            for vis, ir, label in tqdm(val_loader, desc="Val", ncols=100):
                vis, ir, label = vis.to(device), ir.to(device), label.to(device)

                # 验证时不需要 aux_loss
                preds, _ = model(vis, ir)

                loss = criterion(preds, label)
                val_loss_meter += loss.item()

                preds_mask = torch.argmax(preds, dim=1)
                evaluator.add_batch(label.cpu().numpy(), preds_mask.cpu().numpy())

        val_miou, class_ious = evaluator.get_metrics()
        history['val_loss'].append(val_loss_meter / len(val_loader))
        history['val_miou'].append(val_miou)

        print(f"Validation Result -> Loss: {val_loss_meter / len(val_loader):.4f} | mIoU: {val_miou:.4f}")
        print(f"Class IoU: {np.round(class_ious, 3)}")  # 打印每一类的IoU，重点看非0类

        # --- 保存策略 ---
        if val_miou > best_miou:
            best_miou = val_miou
            save_path = os.path.join(CHECKPOINT_DIR, "best_moe_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"★ New Best mIoU! Model saved to {save_path}")

        # 实时绘图
        plot_training_curves(history, VIS_DIR)

    print(f"\nAll Done. Best Validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    train_iacr_process()