import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam import MultiModalSegFormer

# ================= 配置区 (根据你的 sam2.1_hiera_t.yaml 修改) =================
import os

# 1. 设置路径 (请根据你实际解压后的文件夹层级修改根目录)
# 假设你的脚本在 crm-fighting/firstproject/ 目录下运行
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# MSRS 数据集路径 (请修改为你电脑上的实际绝对路径)
MSRS_ROOT = r"E:\Datasets\MSRS"

# SAM2 配置文件路径
# 注意：build_sam2 函数通常在 sam2/configs 下查找，或者传入绝对路径
# 根据你提供的路径: configs/sam2.1/sam2.1_hiera_t.yaml
SAM2_CONFIG = os.path.join("sam2.1", "sam2.1_hiera_t.yaml")

# SAM2 权重文件路径
# 根据你提供的路径: ../checkpoints/sam2.1_hiera_tiny.pt
SAM2_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")

# 2. 模型参数
# 根据 sam2.1_hiera_t.yaml 确认的通道数 [Stage1, Stage2, Stage3, Stage4]
BACKBONE_CHANNELS = [96, 192, 384, 768]

# MSRS 数据集类别 (通常是 9 类)
NUM_CLASSES = 9

# 训练参数 (Windows CPU 建议设置)
BATCH_SIZE = 2
EPOCHS = 5
LR = 0.0001
# =========================================================================

class MSRSDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root = root_dir
        self.split = split
        # 假设 MSRS 结构: root/Vi/train/xxx.png, root/Ir/train/xxx.png
        # 你需要根据实际文件夹结构修改下面的路径拼接逻辑
        self.rgb_dir = os.path.join(root_dir, 'Visible', split)  # 示例
        self.ir_dir = os.path.join(root_dir, 'Infrared', split)
        self.mask_dir = os.path.join(root_dir, 'Label', split)

        self.filenames = [f for f in os.listdir(self.rgb_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # 1. 读取图片
        rgb_path = os.path.join(self.rgb_dir, fname)
        ir_path = os.path.join(self.ir_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)  # 假设标签名和图片名一致

        rgb = Image.open(rgb_path).convert('RGB')
        ir = Image.open(ir_path).convert('L')  # 红外通常是单通道灰度
        mask = Image.open(mask_path)  # 标签通常是索引图 (P模式)

        # 2. 预处理 (Resize/ToTensor/Normalize)
        # 既然是 640x480，我们直接转 Tensor 归一化
        # IR 需要复制成 3 通道以适配 SAM2 输入
        ir = ir.convert('RGB')

        # 简单的数据增强/转换
        # 为了演示，直接转 numpy -> tensor
        rgb = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1) / 255.0
        ir = torch.from_numpy(np.array(ir)).float().permute(2, 0, 1) / 255.0

        # 处理 Mask (Long Tensor)
        mask = torch.from_numpy(np.array(mask)).long()

        return rgb, ir, mask


def configure_model_freezing(model, freeze_sam=True, freeze_fusion=False, freeze_head=False):
    """控制冻结的辅助函数"""
    print(f"\n[Config] Freeze SAM: {freeze_sam}, Fusion: {freeze_fusion}, Head: {freeze_head}")
    for name, param in model.named_parameters():
        param.requires_grad = True  # Reset

        if "sam_model" in name and freeze_sam:
            param.requires_grad = False
        if "fusion_layers" in name and freeze_fusion:
            param.requires_grad = False
        if "segformer_head" in name and freeze_head:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Config] Trainable Params: {trainable / 1e6:.2f}M\n")


def main():
    device = torch.device("cpu")  # 你的环境
    print(f"Device: {device}")

    # 1. 构建基础模型
    base_sam = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    model = MultiModalSegFormer(base_sam, BACKBONE_CHANNELS, NUM_CLASSES)
    model.to(device)

    # 2. 设置冻结策略 (只训练 Head 和 Fusion)
    configure_model_freezing(model, freeze_sam=True, freeze_fusion=False, freeze_head=False)

    # 3. 数据加载
    # 如果你还没有整理好MSRS，可以用 dummy data 测试代码逻辑
    # dataset = MSRSDataset(MSRS_ROOT, split="train")
    print("注意: 如果没有真实数据，请先创建一个虚拟Dataset类进行测试")
    # 暂时使用虚拟数据运行演示
    dataset = FakeDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 优化器与损失
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()  # 多分类损失

    # 5. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        for i, (rgb, ir, mask) in enumerate(dataloader):
            rgb, ir, mask = rgb.to(device), ir.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, ir)  # (B, 9, 480, 640)

            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "msrs_segformer_sam2.pth")
    print("模型已保存！")


# 虚拟数据集类，用于验证代码是否跑通
class FakeDataset(Dataset):
    def __len__(self): return 10

    def __getitem__(self, idx):
        # 模拟 640x480 的 MSRS 数据
        return torch.randn(3, 480, 640), torch.randn(3, 480, 640), torch.randint(0, 9, (480, 640))


if __name__ == "__main__":
    main()