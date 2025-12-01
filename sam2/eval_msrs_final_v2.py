import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch

# ================= 🔧 配置区域 (修改这里) =================

# 1. 填入你的图片路径
IR_IMAGE_PATH = r"F:\sam2-main\images\vis\00007N.png"  # 红外图
VIS_IMAGE_PATH = r"F:\sam2-main\images\ir\00007N.png"  # 可见光图
GT_IMAGE_PATH = r"F:\sam2-main\images\ground\00007N.png"  # GT标签图 (RGB)

# 2. 模型配置
CHECKPOINT_PATH = r"../checkpoints/sam2.1_hiera_tiny.pt"
CONFIG_PATH = r"configs/sam2.1/sam2.1_hiera_t.yaml"

# =========================================================

# --- 自动环境修复 ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    print("[环境] SAM 2 库加载成功。")
except ImportError:
    print("[错误] 无法导入 SAM2，请检查环境。")
    exit()


# --- 核心工具函数 ---

def get_unique_colors(image_rgb):
    """
    提取 GT 中所有出现的颜色（代表不同的物体类别）
    排除纯黑背景 [0, 0, 0]
    """
    # 将 (H, W, 3) 展平为 (N, 3)
    pixels = image_rgb.reshape(-1, 3)
    # 获取唯一颜色行
    unique_colors = np.unique(pixels, axis=0)

    valid_colors = []
    for color in unique_colors:
        if np.sum(color) > 0:  # 排除背景黑色
            valid_colors.append(color)
    return valid_colors


def preprocess_mask(binary_mask):
    """
    【形态学优化】
    把二值掩码处理得更干净，断开粘连的物体，填补内部空洞
    """
    mask = (binary_mask * 255).astype(np.uint8)
    # 3x3 的核适合大部分分辨率
    kernel = np.ones((3, 3), np.uint8)

    # 1. 开运算：断开细小连接（防止两辆车粘在一起被算成一辆）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # 2. 闭运算：填补空洞（防止一个物体因为噪点被拆成两半）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union


def evaluate_multi_class_instances(sam_masks, gt_image_rgb):
    """
    【双重遍历评估逻辑】
    1. 外层：遍历颜色（类别）。
    2. 内层：遍历连通域（同类别的独立物体）。
    """
    # 1. 找出图中有几种颜色的物体
    target_colors = get_unique_colors(gt_image_rgb)
    if len(target_colors) == 0:
        return None

    if len(sam_masks) == 0:
        return {"mIoU": 0.0, "Recall": 0.0, "Num_Instances": 0}

    # 预处理 SAM2 预测结果
    pred_binary_masks = [m['segmentation'] for m in sam_masks]

    # 收集所有独立物体的 IoU
    all_instance_ious = []

    print(f"      -> 发现 {len(target_colors)} 种物体类别(颜色)")

    # --- 第一层循环：遍历类别（颜色）---
    for color_idx, color in enumerate(target_colors):
        # 提取该颜色的所有区域 (例如：提取出所有的“蓝色”)
        # axis=-1 表示在RGB通道上比对
        raw_category_mask = np.all(gt_image_rgb == color, axis=-1)

        # 形态学优化，变干净
        clean_category_mask = preprocess_mask(raw_category_mask)

        # --- 第二层循环：连通域分析（拆分同色物体）---
        # num_labels: 这一类里有几个独立物体 (包含背景0)
        num_labels, labels_im = cv2.connectedComponents(clean_category_mask, connectivity=8)

        # 从 1 开始遍历（0是背景）
        for i in range(1, num_labels):
            # 提取具体的某一个物体（例如：蓝色的第2辆车）
            gt_instance_mask = (labels_im == i)

            # 过滤掉噪点 (小于100像素的色块忽略)
            if gt_instance_mask.sum() < 100:
                continue

            # --- 匹配环节 ---
            # 拿着这个具体的真物体，去 SAM2 的结果里找最佳匹配
            best_iou = 0.0
            for pred_mask in pred_binary_masks:
                # 快速筛选：如果没有交集，直接跳过 (大幅提速)
                # if not np.logical_and(gt_instance_mask, pred_mask).any(): continue

                iou = calculate_iou(gt_instance_mask, pred_mask)
                if iou > best_iou:
                    best_iou = iou

            all_instance_ious.append(best_iou)

    if not all_instance_ious:
        return None

    # 计算全局指标
    miou = np.mean(all_instance_ious)
    recall = sum(1 for i in all_instance_ious if i > 0.5) / len(all_instance_ious)

    return {
        "mIoU": miou,
        "Recall": recall,
        "Num_Instances": len(all_instance_ious),  # 实际参与评估的物体总数
        "Num_Classes": len(target_colors)
    }


# --- 主执行 ---

def run_main():
    if not os.path.exists(GT_IMAGE_PATH):
        print(f"找不到 GT 图片: {GT_IMAGE_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n加载模型 ({device})...")

    try:
        ckpt = str(current_file.parent / CHECKPOINT_PATH)
        cfg = str(current_file.parent / CONFIG_PATH)
        if not os.path.exists(ckpt): ckpt = CHECKPOINT_PATH
        model = build_sam2(cfg, ckpt, device=device)
        generator = SAM2AutomaticMaskGenerator(model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 读取 GT 并转 RGB
    print(f"读取 GT: {Path(GT_IMAGE_PATH).name}")
    gt_rgb = cv2.imread(GT_IMAGE_PATH)
    if gt_rgb is None: return
    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)

    # --- 评估红外 ---
    if os.path.exists(IR_IMAGE_PATH):
        print(f"\n正在评估红外 (IR)...")
        img_ir = cv2.imread(IR_IMAGE_PATH)
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)

        masks = generator.generate(img_ir)
        print(f"      -> SAM2 输出了 {len(masks)} 个 Mask")

        res = evaluate_multi_class_instances(masks, gt_rgb)
        if res:
            print(f"  ------------------------------")
            print(f"  ★ IR mIoU:   {res['mIoU']:.4f}")
            print(f"  ★ IR Recall: {res['Recall']:.2%}")
            print(f"  ------------------------------")
            print(f"  (基于 {res['Num_Classes']} 个类别中的 {res['Num_Instances']} 个独立物体)")

    # --- 评估可见光 ---
    if os.path.exists(VIS_IMAGE_PATH):
        print(f"\n正在评估可见光 (VIS)...")
        img_vis = cv2.imread(VIS_IMAGE_PATH)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        masks = generator.generate(img_vis)
        res = evaluate_multi_class_instances(masks, gt_rgb)
        if res:
            print(f"  ------------------------------")
            print(f"  ★ VIS mIoU:   {res['mIoU']:.4f}")
            print(f"  ★ VIS Recall: {res['Recall']:.2%}")
            print(f"  ------------------------------")

    print("\n评估结束。")


if __name__ == "__main__":
    run_main()