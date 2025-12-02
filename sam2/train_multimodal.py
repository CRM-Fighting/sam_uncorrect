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

# 引入之前的模型组件
# 请确保你的 sam2 文件夹在当前目录下
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam import MultiModalSegFormer


# ==========================================
# 1. 新设计的混合损失函数 (Combined Loss)
#    CrossEntropy 关注像素分类准确度
#    Dice Loss 关注分割区域的重合度
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits
        # targets: (B, H, W) labels

        # 将 Logits 转为概率 (Softmax)
        inputs = F.softmax(inputs, dim=1)

        # 将 targets 转为 One-hot 编码
        # 注意处理 ignore_index (例如背景或无效区域)
        # 这里简化处理，假设 targets 都在 0~C-1 之间
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 计算 Dice 系数
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 1 - Dice 作为 Loss
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=9, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # CrossEntropy 权重
        self.beta = beta  # Dice 权重
        # CE Loss 自动处理多分类
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = DiceLoss(ignore_index=255)

    def forward(self, preds, labels):
        loss_ce = self.ce(preds, labels)
        loss_dice = self.dice(preds, labels)
        return self.alpha * loss_ce + self.beta * loss_dice


# ==========================================
# 2. 自定义 MSRS 数据集 (Dataset)
#    支持双模态读取 + 验证集截取
# ==========================================
class MSRSDataset(Dataset):
    def __init__(self, root_dirs, limit=None):
        """
        root_dirs: 字典，包含 'ir', 'vis', 'label' 的路径
        limit: 整数，如果设置，只加载前 limit 张图片 (用于验证集只取100张)
        """
        self.ir_dir = root_dirs['ir']
        self.vis_dir = root_dirs['vis']
        self.label_dir = root_dirs['Segmentation_labels']

        # 获取文件名列表 (假设三种图文件名一致)
        self.filenames = sorted([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])

        # 【要求4实现】如果设置了 limit，只截取前 limit 张
        if limit is not None:
            self.filenames = self.filenames[:limit]
            print(f"注意：数据集已限制为前 {limit} 张图片。")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # 拼接路径
        ir_path = os.path.join(self.ir_dir, fname)
        vis_path = os.path.join(self.vis_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        # 读取图片
        # 可见光转 RGB, 红外转 RGB (适配 SAM2 输入), Label 保持原样
        vis = Image.open(vis_path).convert('RGB')
        ir = Image.open(ir_path).convert('RGB')
        label = Image.open(label_path)  # 通常是 P 模式或 L 模式

        # 统一调整尺寸 (480x640)
        target_size = (640, 480)
        vis = vis.resize(target_size, Image.BILINEAR)
        ir = ir.resize(target_size, Image.BILINEAR)
        label = label.resize(target_size, Image.NEAREST)  # 标签必须用最近邻插值，保持整数类别

        # 转 Tensor 并归一化
        vis_t = torch.from_numpy(np.array(vis)).float().permute(2, 0, 1) / 255.0
        ir_t = torch.from_numpy(np.array(ir)).float().permute(2, 0, 1) / 255.0
        label_t = torch.from_numpy(np.array(label)).long()

        return vis_t, ir_t, label_t


# ==========================================
# 3. 冻结配置函数 (详细注释版)
# ==========================================
def configure_model_freezing(model, freeze_encoder=True, freeze_fusion=True, freeze_decoder=False):
    """
    【小白解说】冻结参数是什么意思？
    在深度学习中，“训练”就是更新模型的参数（权重）。
    - requires_grad = True  -> 开启梯度 -> 训练时这个模块的参数会改变（学习）。
    - requires_grad = False -> 关闭梯度 -> 训练时这个模块的参数保持不动（冻结）。

    为什么我们要冻结？
    1. SAM2 的编码器已经在大数据集上训练得很好了，我们不想破坏它的特征提取能力。
    2. 我们的显存/算力有限（CPU环境），只训练解码器计算量小，跑得快。
    """
    print("\n[冻结配置] 正在设置模型可训练模块...")

    # 1. 先把所有参数“锁住”（冻结）
    for param in model.parameters():
        param.requires_grad = False

    # 2. 遍历模型的所有子模块，根据名字决定是否“解锁”
    for name, param in model.named_parameters():

        # (A) SegFormer 解码器：我们需要训练它，所以解锁！
        if "segformer_head" in name:
            if not freeze_decoder:
                param.requires_grad = True

        # (B) 融合模块：如果是简单的相加，它没有参数；如果是卷积融合，需要根据情况解锁
        # 这里预留位置，如果你以后想训练融合层，把 freeze_fusion 设为 False 即可
        if "fusion_layers" in name:
            if not freeze_fusion:
                param.requires_grad = True

        # (C) SAM2 编码器：通常保持冻结
        if "sam_model" in name:
            if not freeze_encoder:  # 只有当你显式说“我不冻结编码器”时才解锁
                param.requires_grad = True

    # 统计一下真正参与训练的参数量，让你心里有数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" -> 总参数量: {total_params / 1e6:.2f} M")
    print(f" -> 参与训练参数: {trainable_params / 1e6:.2f} M (主要集中在 SegFormer 解码器)")
    print("[冻结配置] 完成。\n")


# ==========================================
# 4. 数据处理入口 (Data Process)
# ==========================================
def get_dataloaders(batch_size=2):
    # 定义路径 (根据你的描述)
    # 注意：Windows 路径建议用 r"" 防止转义，或者用 /
    # 请确保这些路径下确实有图片文件
    train_dirs = {
        'ir': r"sam2/data/MSRS/train/ir",
        'vis': r"sam2/data/MSRS/train/vis",
        'label': r"sam2/data/MSRS/train/Segmentation_labels"  # 假设这是Label文件夹名
    }

    val_dirs = {
        'ir': r"sam2/data/MSRS/test/ir",
        'vis': r"sam2/data/MSRS/test/vis",
        'label': r"sam2/data/MSRS/test/Segmentation_labels"
    }

    print("正在加载数据集...")
    # 训练集：加载所有 1083 张
    train_ds = MSRSDataset(train_dirs, limit=None)

    # 验证集：【要求4】只提取前 100 张用于验证
    val_ds = MSRSDataset(val_dirs, limit=100)

    # 创建 DataLoader
    # num_workers=0 是为了在 Windows CPU 环境下避免多进程报错
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")

    return train_loader, val_loader


# ==========================================
# 5. 训练主流程 (Train Model Process)
# ==========================================
def train_model_process(model, train_loader, val_loader, num_epochs=5, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)

    # 1. 设置冻结 (在这个阶段，只训练 SegFormer Head)
    configure_model_freezing(model, freeze_encoder=True, freeze_fusion=True, freeze_decoder=False)

    # 2. 定义优化器
    # filter() 确保优化器只接收那些 requires_grad=True 的参数，否则会报错
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 3. 定义损失函数 (使用我们新写的混合 Loss)
    criterion = SegmentationLoss(num_classes=9, alpha=0.6, beta=0.4)  # alpha偏重CE, beta偏重Dice

    # 记录训练过程
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []  # 分割任务准确率计算较慢，这里主要记录Loss，Acc暂用简化代替
    }

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

            # 进度条
            pbar = tqdm(dataloader, desc=f"{phase} Phase")

            for vis, ir, labels in pbar:
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

                # 更新进度条显示的 Loss
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / total_pixels

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录历史
            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

                # 深拷贝模型参数 (保存验证集 Loss 最低的模型)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("Found better model, saving weights...")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Loss: {best_loss:.4f}")

    # 加载最佳权重
    model.load_state_dict(best_model_wts)

    # 保存最终模型
    save_path = "sam2/best_msrs_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return pd.DataFrame(history)


# ==========================================
# 6. 绘图函数 (保持原风格)
# ==========================================
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_process)), train_process["train_loss"], 'ro-', label="Train loss")
    plt.plot(range(len(train_process)), train_process["val_loss"], 'bs-', label="Val loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_process)), train_process["train_acc"], 'ro-', label="Train acc")
    plt.plot(range(len(train_process)), train_process["val_acc"], 'bs-', label="Val acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.show()
    print("图表已生成。")


# ==========================================
# 7. 主程序入口 (Main)
# ==========================================
def main():
    # A. 配置路径和参数
    # 请确保这里的 yaml 和 pt 路径是正确的
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
    # 注意：这里假设 model_train.py 在根目录或者 sam2 目录外层
    # 如果文件在 sam2/ 里面，路径要相应调整

    sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_ckpt = "../checkpoints/sam2.1_hiera_tiny.pt"

    # Hiera-Tiny 通道数配置
    backbone_channels = [96, 192, 384, 768]

    # B. 构建模型
    print("1. 构建基础模型...")
    device = "cpu"  # 暂时强制写 CPU，如果以后有显卡，train_model_process 里会自动检测
    try:
        base_sam = build_sam2(sam2_config, sam2_ckpt, device=device)
    except FileNotFoundError:
        print("错误：找不到配置文件或权重文件，请检查 sam2_config 和 sam2_ckpt 路径。")
        return

    model = MultiModalSegFormer(base_sam, feature_channels=backbone_channels, num_classes=9)

    # C. 获取数据
    print("2. 准备数据...")
    try:
        # 这里的 batch_size=2 适合 CPU 训练
        train_loader, val_loader = get_dataloaders(batch_size=2)
    except Exception as e:
        print(f"数据加载出错: {e}")
        return

    # D. 开始训练
    print("3. 开始训练循环...")
    # num_epochs 可以改小一点先测试，比如 2
    history = train_model_process(model, train_loader, val_loader, num_epochs=3, learning_rate=0.0001)

    # E. 绘图
    print("4. 绘制结果...")
    matplot_acc_loss(history)


if __name__ == "__main__":
    main()