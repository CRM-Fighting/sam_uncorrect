import numpy as np
import torch
import cv2
import os
import gc  # 引入垃圾回收
from pathlib import Path
from tqdm import tqdm

# --- 尝试导入 SAM 2 ---
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库。请确保已安装。")
    exit()

# ================= 配置区域 (绝对路径) =================

# 1. 数据集根目录
DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")

# 2. 输出目录 (会自动在此目录下生成 val 和 test 文件夹)
OUTPUT_ROOT = DATASET_ROOT / "entropy_maps_add"

# 3. 基础项目路径推断
PROJECT_BASE = Path("/home/mmsys/disk/MCL/MultiModal_Project")
SAM2_BASE = PROJECT_BASE / "sam2"

# 4. 模型权重绝对路径
CHECKPOINT ="../checkpoints/sam2.1_hiera_tiny.pt"

# 5. 配置文件绝对路径
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# 4个阶段的下采样倍率
TOKEN_SIZES = {'stage1': 4, 'stage2': 8, 'stage3': 16, 'stage4': 32}
IMG_W, IMG_H = 640, 480

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dataset Path: {DATASET_ROOT}")
print(f"Output Path:  {OUTPUT_ROOT}")
print("-" * 50)


# ================= 辅助函数 =================
def simple_mask_overlay(masks, image_shape):
    overlay = np.zeros(image_shape[:2], dtype=np.int32)
    if len(masks) == 0: return overlay
    for mask_data in masks:
        overlay[mask_data['segmentation']] += 1
    return overlay


def calculate_token_entropy(token):
    values, counts = np.unique(token, return_counts=True)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0: return 0.0
    return -np.sum(probabilities * np.log2(probabilities))


def create_entropy_map(heatmap, token_size):
    H, W = heatmap.shape
    num_h = H // token_size
    num_w = W // token_size
    tokens = heatmap.reshape(num_h, token_size, num_w, token_size).transpose(0, 2, 1, 3)
    entropy_map = np.zeros((num_h, num_w), dtype=np.float32)
    for i in range(num_h):
        for j in range(num_w):
            entropy_map[i, j] = calculate_token_entropy(tokens[i, j])
    return entropy_map


# ================= 一致性检查函数 =================
def check_data_consistency(subset_name):
    """
    检查指定子集的数据完整性
    """
    target_path = DATASET_ROOT / subset_name

    if not target_path.exists():
        print(f"\n[错误] 数据集目录不存在: {target_path}")
        return False

    print(f"\n{'=' * 20} 正在检查 {subset_name} 数据集一致性 {'=' * 20}")

    vi_dir = target_path / "vi"
    ir_dir = target_path / "ir"
    label_dir = target_path / "Segmentation_labels"

    # 检查文件夹是否存在
    if not vi_dir.exists() or not ir_dir.exists() or not label_dir.exists():
        print(f"严重错误：{subset_name} 下缺少必要目录！")
        if not vi_dir.exists(): print(" -> 缺失 vi")
        if not ir_dir.exists(): print(" -> 缺失 ir")
        if not label_dir.exists(): print(" -> 缺失 Segmentation_labels")
        return False

    # 获取所有文件名
    files_vi = set(f for f in os.listdir(vi_dir) if f.endswith('.png'))
    files_ir = set(f for f in os.listdir(ir_dir) if f.endswith('.png'))
    files_lb = set(f for f in os.listdir(label_dir) if f.endswith('.png'))

    # 取并集
    all_files = files_vi | files_ir | files_lb

    consistent = True
    print(f"[{subset_name}] 共扫描到 {len(all_files)} 个唯一文件名，开始比对...")

    for fname in sorted(list(all_files)):
        missing_parts = []
        if fname not in files_vi: missing_parts.append("vi")
        if fname not in files_ir: missing_parts.append("ir")
        if fname not in files_lb: missing_parts.append("Segmentation_labels")

        if missing_parts:
            consistent = False
            # 只要缺失 vi 或 ir，就无法生成，必须打印警告
            if "vi" in missing_parts or "ir" in missing_parts:
                print(f"[数据缺失] 文件: {fname} 缺失 -> {', '.join(missing_parts)}")

    if consistent:
        print(f">> 检查通过：{subset_name} 数据一一对应。")
    else:
        print(f">> 警告：{subset_name} 存在不一致数据，生成时将跳过缺失文件。")

    print("=" * 60 + "\n")
    return True


# ================= 核心处理逻辑 =================
def process_subset(subset_name, mask_generator):
    vis_dir = DATASET_ROOT / subset_name / "vi"
    ir_dir = DATASET_ROOT / subset_name / "ir"

    # 再次确认目录存在
    if not vis_dir.exists() or not ir_dir.exists():
        print(f"跳过 {subset_name}: 目录结构不完整")
        return

    print(f"开始处理数据集: {subset_name}")

    # 创建输出目录： OUTPUT_ROOT / subset_name / stageX
    # 例如: .../entropy_maps_add/val/stage1
    output_dirs = {}
    for stage in TOKEN_SIZES.keys():
        d = OUTPUT_ROOT / subset_name / stage
        d.mkdir(parents=True, exist_ok=True)
        output_dirs[stage] = d

    # 以可见光目录为基准进行遍历
    files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])

    for fname in tqdm(files, desc=f"Processing {subset_name}"):
        path_vis = vis_dir / fname
        path_ir = ir_dir / fname
        save_name_npy = Path(fname).stem + ".npy"

        # 断点续传检查 (如果 stage1 已经有结果，假定该图已处理完毕)
        if (output_dirs['stage1'] / save_name_npy).exists():
            continue

        # 基础存在性检查
        if not path_vis.exists() or not path_ir.exists():
            continue

        try:
            img_vis = cv2.imread(str(path_vis))
            img_ir = cv2.imread(str(path_ir))

            if img_vis is None or img_ir is None:
                print(f"\n[读取失败] 图片损坏: {fname}")
                continue

            img_vis = cv2.resize(img_vis, (IMG_W, IMG_H))
            img_ir = cv2.resize(img_ir, (IMG_W, IMG_H))

            # === 核心生成 ===
            with torch.no_grad():
                masks_vis = mask_generator.generate(img_vis)
                masks_ir = mask_generator.generate(img_ir)

            heatmap_vis = simple_mask_overlay(masks_vis, (IMG_H, IMG_W))
            heatmap_ir = simple_mask_overlay(masks_ir, (IMG_H, IMG_W))

            # === 计算并保存4个阶段 ===
            for stage_name, token_size in TOKEN_SIZES.items():
                map_vis = create_entropy_map(heatmap_vis, token_size)
                map_ir = create_entropy_map(heatmap_ir, token_size)
                diff_map = map_ir + map_vis
                np.save(output_dirs[stage_name] / save_name_npy, diff_map)

            # === 显式垃圾回收 ===
            del masks_vis, masks_ir, heatmap_vis, heatmap_ir
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"\n[处理异常] 图片 {fname} 出错: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue


if __name__ == "__main__":
    # === 修改点：只包含 val 和 test ===
    TARGET_SUBSETS = ["val", "test"]

    # 1. 检查一致性
    valid_subsets = []
    for subset in TARGET_SUBSETS:
        if check_data_consistency(subset):
            valid_subsets.append(subset)

    if not valid_subsets:
        print("未找到有效的 val 或 test 目录，程序退出。")
        exit()

    # 2. 加载模型
    print(f"正在加载 SAM 2 模型 (Config: {MODEL_CFG})...")
    model = build_sam2(str(MODEL_CFG), str(CHECKPOINT), device=DEVICE)
    mask_generator = SAM2AutomaticMaskGenerator(model)

    # 3. 循环处理
    print(f"\n即将处理以下数据集: {valid_subsets}")
    for subset in valid_subsets:
        print(f"\n>>> 进入 {subset} 处理阶段...")
        process_subset(subset, mask_generator)

    print("\n所有任务 (Val, Test) 处理完成。")