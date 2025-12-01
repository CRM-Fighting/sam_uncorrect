import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam import MultiModalSegFormer

# 简单的颜色映射表，用于显示9类分割结果
COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]
])


def preprocess_image(image_path, device):
    """读取图片并转换为模型可接受的 Tensor 格式"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到文件: {image_path}")

    # 1. 读取图片
    img = Image.open(image_path).convert('RGB')  # 确保转为3通道

    # 2. 调整尺寸 (如果图片不是 640x480，可以在这里强制 resize，或者保持原图)
    # img = img.resize((640, 480))

    # 3. 转为 numpy 并归一化到 [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # 4. 转为 Tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor.to(device), img_np


def main():
    device = "cpu"
    print(f"Using device: {device}")

    # ================= 配置区 =================
    # 请确保这些路径是你电脑上的真实路径
    # 注意：Windows路径分隔符问题，建议使用 r"..." 或 "/"
    path_rgb = r"F:\MultiModal_Project\sam2\data\MSRS\train\vi\00001D.png"  # 可见光通常在 vi 文件夹
    path_ir = r"F:\MultiModal_Project\sam2\data\MSRS\train\ir\00001D.png"  # 红外通常在 ir 文件夹

    config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint_path = "../checkpoints/sam2.1_hiera_tiny.pt"
    # ==========================================

    # 加载模型
    backbone_channels = [96, 192, 384, 768]
    base_sam = build_sam2(config_path, checkpoint_path, device=device)
    model = MultiModalSegFormer(base_sam, backbone_channels, num_classes=9)
    model.eval()

    # 读取并处理图片
    print(f"正在读取测试图片...")
    try:
        # 注意：这里我交换了你的 dummy_rgb 和 dummy_ir 的路径变量名
        # 因为通常 'vi' 是 visible(rgb), 'ir' 是 infrared
        tensor_rgb, view_rgb = preprocess_image(path_rgb, device)
        tensor_ir, view_ir = preprocess_image(path_ir, device)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请检查 path_rgb 和 path_ir 的路径是否正确，或者修改代码使用随机数据测试。")
        return

    print("模型推理中...")
    with torch.no_grad():
        # 这里传入的必须是 Tensor
        logits = model(tensor_rgb, tensor_ir)

    print(f"输出 Logits 尺寸: {logits.shape}")

    # 转为类别索引
    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 可视化
    print("显示结果窗口...")
    color_mask = COLORS[pred_mask % 9]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(view_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Input IR")
    plt.imshow(view_ir)  # 如果是红外图，也可以用 plt.imshow(view_ir.mean(axis=2), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("SegFormer Prediction")
    plt.imshow(color_mask.astype(np.uint8))
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()