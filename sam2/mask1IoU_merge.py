import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
import math
from collections import Counter

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# Step 1：读取图像
image_path = "../images/train/"
image_name_ir = f"Infrared/00001.png"
image_name_vis = f"Visible/00001.png"

image_ir = cv2.imread(image_path + image_name_ir)
image_vis = cv2.imread(image_path + image_name_vis)

if image_ir is None or image_vis is None:
    raise FileNotFoundError("图像文件未正确加载，请检查路径。")

# Step 2：加载 SAM 模型 - 强制使用CPU
checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# 强制使用CPU，避免CUDA错误
device = "cpu"
print(f"Using {device} device")

# 创建保存结果的目录
output_dir = "../results/"
os.makedirs(output_dir, exist_ok=True)

try:
    model = build_sam2(model_cfg, checkpoint, device=device)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    # 如果模型加载失败，创建一个简单的演示
    print("创建示例掩码进行演示...")

    # 创建示例掩码数据
    h, w = 600, 800
    masks_ir = [
        {
            "segmentation": np.random.rand(h, w) > 0.9,
            "area": 1000,
            "bbox": [100, 100, 200, 200]
        },
        {
            "segmentation": np.random.rand(h, w) > 0.9,
            "area": 1500,
            "bbox": [300, 200, 150, 150]
        }
    ]
    masks_vis = [
        {
            "segmentation": np.random.rand(h, w) > 0.9,
            "area": 1200,
            "bbox": [120, 110, 180, 190]
        },
        {
            "segmentation": np.random.rand(h, w) > 0.9,
            "area": 1300,
            "bbox": [320, 210, 140, 140]
        }
    ]


    # 简化的mask叠加函数
    def simple_mask_overlay(masks, image_shape):
        """简化的mask叠加"""
        overlay = np.zeros(image_shape[:2], dtype=np.uint8)
        for mask_data in masks:
            mask = mask_data['segmentation']
            overlay[mask] += 1
        return overlay


    # 在循环中使用
    overlay_ir = simple_mask_overlay(masks_ir, (h, w))
    overlay_vis = simple_mask_overlay(masks_vis, (h, w))

    # 简单显示
    plt.figure(figsize=(5, 12))
    plt.subplot(2, 1, 1)
    plt.imshow(overlay_ir, cmap='hot')
    plt.title('Infrared Mask superposition (示例)')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(overlay_vis, cmap='hot')
    plt.title('Visible Mask superposition (示例)')
    plt.colorbar()

    # 保存整体图片
    plt.savefig(os.path.join(output_dir, 'mask_superposition_demo.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"整体图片已保存到: {os.path.join(output_dir, 'mask_superposition_demo.png')}")

    # 分别保存两个子图
    # 保存红外图像
    plt.figure(figsize=(5, 6))
    plt.imshow(overlay_ir, cmap='hot')
    plt.title('Infrared Mask superposition (示例)')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'infrared_mask_demo.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形，避免重叠

    # 保存可见光图像
    plt.figure(figsize=(5, 6))
    plt.imshow(overlay_vis, cmap='hot')
    plt.title('Visible Mask superposition (示例)')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'visible_mask_demo.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形，避免重叠

    print(f"红外图像已保存到: {os.path.join(output_dir, 'infrared_mask_demo.png')}")
    print(f"可见光图像已保存到: {os.path.join(output_dir, 'visible_mask_demo.png')}")

    plt.show()
else:
    # Step 3：使用自动Mask生成器进行全图分割
    mask_generator = SAM2AutomaticMaskGenerator(model)
    masks_ir = mask_generator.generate(image_ir)
    masks_vis = mask_generator.generate(image_vis)


    # 简化的mask叠加函数
    def simple_mask_overlay(masks, image_shape):
        """简化的mask叠加"""
        overlay = np.zeros(image_shape[:2], dtype=np.uint8)
        for mask_data in masks:
            mask = mask_data['segmentation']
            overlay[mask] += 1
        return overlay


    # 在循环中使用
    overlay_ir = simple_mask_overlay(masks_ir, image_ir.shape)
    overlay_vis = simple_mask_overlay(masks_vis, image_vis.shape)

    # 简单显示
    plt.figure(figsize=(5, 12))
    plt.subplot(2, 1, 1)
    plt.imshow(overlay_ir, cmap='hot')
    plt.title('Infrared Mask superposition')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(overlay_vis, cmap='hot')
    plt.title('Visible Mask superposition')
    plt.colorbar()

    # 保存整体图片
    plt.savefig(os.path.join(output_dir, 'mask_superposition.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"整体图片已保存到: {os.path.join(output_dir, 'mask_superposition.png')}")

    # 分别保存两个子图
    # 保存红外图像
    plt.figure(figsize=(5, 6))
    plt.imshow(overlay_ir, cmap='hot')
    plt.title('Infrared Mask superposition')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'infrared_mask.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形，避免重叠

    # 保存可见光图像
    plt.figure(figsize=(5, 6))
    plt.imshow(overlay_vis, cmap='hot')
    plt.title('Visible Mask superposition')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'visible_mask.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形，避免重叠

    print(f"红外图像已保存到: {os.path.join(output_dir, 'infrared_mask.png')}")
    print(f"可见光图像已保存到: {os.path.join(output_dir, 'visible_mask.png')}")

    plt.show()
