import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch

# ================= 🔧 配置区域 (修改这里) =================

# 1. 图片路径
IR_IMAGE_PATH = r"F:\sam2-main\images\ir\00009N.png"
VIS_IMAGE_PATH = r"F:\sam2-main\images\vis\00009N.png"
GT_IMAGE_PATH = r"F:\sam2-main\images\ground\00009N.png"

# 2. 模型权重与配置
# 注意：如果你的脚本在 sam2/ 目录下，这里的相对路径 ../checkpoints 是对的
CHECKPOINT_PATH = r"../checkpoints/sam2.1_hiera_tiny.pt"
CONFIG_PATH = r"configs/sam2.1/sam2.1_hiera_t.yaml"

# =========================================================

# --- 1. 强力环境修复与诊断模块 ---
current_file = Path(__file__).resolve()
# 假设脚本在 sam2/ 目录下，我们需要把 sam2-main/ (父目录的父目录) 加入路径
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"[调试] 已将项目根目录加入 path: {project_root}")

try:
    # 尝试导入，如果报错会捕获并提示安装
    import hydra
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    print("[环境] SAM 2 库加载成功。")

except ImportError as e:
    print(f"\n{'=' * 60}")
    print(f"❌ 严重错误: 无法导入 SAM2 环境。")
    print(f"   报错信息: {e}")
    print(f"{'=' * 60}")

    # 智能提示
    err_msg = str(e)
    if "hydra" in err_msg:
        print("💡 原因: 缺少 hydra-core 库 (SAM2 的核心依赖)。")
        print("🛠️ 解决: 请在终端(Terminal)运行以下命令安装:")
        print("   pip install hydra-core iopath")
    elif "No module named 'sam2'" in err_msg:
        print("💡 原因: Python 找不到 sam2 文件夹。")
        print(f"   当前脚本位置: {current_file}")
        print(f"   请确认 'sam2' 文件夹就在 {project_root} 下。")

    # 强制退出，避免后面报错
    sys.exit(1)


# --- 2. 核心工具函数 ---

def get_unique_colors(image_rgb):
    """提取 GT 中的所有类别颜色 (忽略纯黑背景)"""
    pixels = image_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    valid_colors = []
    for color in unique_colors:
        if np.sum(color) > 0:  # 排除背景 [0,0,0]
            valid_colors.append(color)
    return valid_colors


def preprocess_mask(binary_mask):
    """
    【形态学优化】关键步骤！
    解决 '两个物体粘在一起' 和 '物体内部有噪点' 的问题
    """
    mask = (binary_mask * 255).astype(np.uint8)

    # 定义核大小 (3x3 适合大部分情况，如果物体很大可改 5x5)
    kernel = np.ones((3, 3), np.uint8)

    # 1. 开运算 (Opening): 先腐蚀后膨胀 -> 断开细小粘连，去除噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. 闭运算 (Closing): 先膨胀后腐蚀 -> 填补内部空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def calculate_iou(mask1, mask2):
    """计算二值 IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union


def evaluate_predictions_instance_aware(sam_masks, gt_image_rgb):
    """
    【最终版评估逻辑】
    颜色提取 -> 形态学优化 -> 连通域拆分 -> 最佳匹配
    """
    target_colors = get_unique_colors(gt_image_rgb)
    if len(target_colors) == 0:
        return None

    # 预处理 SAM2 预测结果
    if len(sam_masks) == 0:
        return {"mIoU": 0.0, "Recall": 0.0, "Num_Instances": 0}

    pred_binary_masks = [m['segmentation'] for m in sam_masks]
    all_instance_ious = []

    # 第一层循环：遍历颜色（类别）
    for color in target_colors:
        # A. 提取该颜色的原始掩码
        raw_class_mask = np.all(gt_image_rgb == color, axis=-1)

        # B. 【优化】形态学处理 (变干净)
        clean_class_mask = preprocess_mask(raw_class_mask)

        # C. 连通域分析：拆解成独立物体
        # connectivity=8 表示 8 邻域连通
        num_labels, labels_im = cv2.connectedComponents(clean_class_mask, connectivity=8)

        # D. 遍历该类别下的每一个独立物体 (从1开始，0是背景)
        for i in range(1, num_labels):
            gt_instance_mask = (labels_im == i)

            # 过滤极小噪点 (例如 < 50 像素)
            if gt_instance_mask.sum() < 50:
                continue

            # E. 寻找 SAM2 的最佳匹配
            best_iou = 0.0
            for pred_mask in pred_binary_masks:
                # 快速筛选优化 (可选)
                # if not np.logical_and(gt_instance_mask, pred_mask).any(): continue

                iou = calculate_iou(gt_instance_mask, pred_mask)
                if iou > best_iou:
                    best_iou = iou

            all_instance_ious.append(best_iou)

    if not all_instance_ious:
        return None

    # 计算指标
    miou = np.mean(all_instance_ious)
    recall = sum(1 for i in all_instance_ious if i > 0.5) / len(all_instance_ious)

    return {
        "mIoU": miou,
        "Recall": recall,
        "Num_Instances": len(all_instance_ious),
        "Num_Classes": len(target_colors)
    }


# --- 3. 主程序 ---

def run_main():
    # 检查图片文件
    if not os.path.exists(GT_IMAGE_PATH):
        print(f"[文件错误] 找不到 GT: {GT_IMAGE_PATH}")
        return

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/4] 加载模型 ({device})...")

    try:
        # 处理相对路径
        ckpt = str(current_file.parent / CHECKPOINT_PATH)
        cfg = str(current_file.parent / CONFIG_PATH)
        if not os.path.exists(ckpt): ckpt = CHECKPOINT_PATH

        model = build_sam2(cfg, ckpt, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(model)
    except Exception as e:
        print(f"[模型加载失败] 请检查权重路径。\n错误: {e}")
        return

    # 读取 GT
    print(f"[2/4] 读取 GT: {Path(GT_IMAGE_PATH).name}")
    gt_rgb = cv2.imread(GT_IMAGE_PATH)
    if gt_rgb is None: return
    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)

    # 评估红外
    if os.path.exists(IR_IMAGE_PATH):
        print(f"\n[3/4] 评估红外图像 (IR)...")
        img_ir = cv2.imread(IR_IMAGE_PATH)
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(img_ir)
        print(f"      -> SAM2 生成了 {len(masks)} 个 Mask")

        res = evaluate_predictions_instance_aware(masks, gt_rgb)
        if res:
            print(f"      ----------------------------------")
            print(f"      ★ IR mIoU:   {res['mIoU']:.4f}")
            print(f"      ★ IR Recall: {res['Recall']:.2%}")
            print(f"      ----------------------------------")
            print(f"      (共发现 {res['Num_Classes']} 类，拆解为 {res['Num_Instances']} 个独立物体)")

    # 评估可见光
    if os.path.exists(VIS_IMAGE_PATH):
        print(f"\n[4/4] 评估可见光图像 (VIS)...")
        img_vis = cv2.imread(VIS_IMAGE_PATH)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(img_vis)
        res = evaluate_predictions_instance_aware(masks, gt_rgb)
        if res:
            print(f"      ----------------------------------")
            print(f"      ★ VIS mIoU:   {res['mIoU']:.4f}")
            print(f"      ★ VIS Recall: {res['Recall']:.2%}")
            print(f"      ----------------------------------")

    print("\n评估结束。")


if __name__ == "__main__":
    run_main()