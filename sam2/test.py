import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import cv2  # 确保导入cv2用于图像融合
from matplotlib.colors import ListedColormap

# 导入SAM2相关模块
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# 1. 配置路径
checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
image_path = "../images/train/Infrared/00001.png"
output_dir = "../segmentation_results"
os.makedirs(output_dir, exist_ok=True)

# 2. 初始化设备
device = "cpu"
print(f"使用 {device} 设备进行处理")

# 3. 构建模型（忽略_C模块警告）
model = build_sam2(model_cfg, checkpoint, device=device)

# 4. 初始化掩码生成器
mask_generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=64,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    box_nms_thresh=0.6,
    min_mask_region_area=50
)

# 5. 读取图像并生成掩码
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
masks = mask_generator.generate(image_np)  # 即使有警告，仍能生成掩码

# 6. 生成最终分割图
if len(masks) == 0:
    print("未分割出任何物体，请检查图像或调整参数")
else:
    # 生成随机颜色（每个物体一种颜色）
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

    # 叠加所有掩码
    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)
    for i, mask in enumerate(masks):
        seg = mask["segmentation"]
        mask_overlay[seg] = colors[i]

    # 融合原图与掩码（半透明）
    combined = cv2.addWeighted(image_np, 0.5, mask_overlay, 0.5, 0)

    # 保存最终分割图
    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.title(f"所有物体分割结果（共{len(masks)}个物体）")
    plt.axis("off")
    output_path = os.path.join(output_dir, "final_segmentation.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"最终分割图已保存至：{output_path}")