import numpy as np
import torch
import cv2
import os
import gc
from pathlib import Path
from tqdm import tqdm

# --- 尝试导入 SAM 2 ---
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库。请确保已安装。")
    exit()

# ================= 配置区域 =================

# 1. 数据集根目录
DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")

# 2. 输出目录 (保持和之前一致)
OUTPUT_ROOT = DATASET_ROOT / "entropy_maps_add"

# 3. 模型权重
CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# 4个阶段的下采样倍率
TOKEN_SIZES = {'stage1': 4, 'stage2': 8, 'stage3': 16, 'stage4': 32}
IMG_W, IMG_H = 640, 480

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dataset Path: {DATASET_ROOT}")
print(f"Output Path:  {OUTPUT_ROOT}")
print("-" * 50)


# ================= 辅助函数 (保持不变) =================
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


# ================= 一致性检查 =================
def check_data_consistency(subset_name):
    target_path = DATASET_ROOT / subset_name
    if not target_path.exists(): return False

    vi_dir = target_path / "vi"
    ir_dir = target_path / "ir"
    if not vi_dir.exists() or not ir_dir.exists(): return False

    files_vi = set(f for f in os.listdir(vi_dir) if f.endswith('.png'))
    return len(files_vi) > 0  # 简单检查，假设之前已经检查过完整性


# ================= 核心处理逻辑 (修改版) =================
def process_subset(subset_name, mask_generator):
    vis_dir = DATASET_ROOT / subset_name / "vi"
    ir_dir = DATASET_ROOT / subset_name / "ir"

    print(f"开始补全数据集: {subset_name} 的 VI 和 IR 熵图")

    # === 修改点：只创建 vi 和 ir 目录，不处理 sum ===
    vi_output_dirs = {}
    ir_output_dirs = {}

    for stage in TOKEN_SIZES.keys():
        # 1. 可见光熵图目录 (OUTPUT_ROOT/subset/vi/stageX)
        d_vi = OUTPUT_ROOT / subset_name / "vi" / stage
        d_vi.mkdir(parents=True, exist_ok=True)
        vi_output_dirs[stage] = d_vi

        # 2. 红外熵图目录 (OUTPUT_ROOT/subset/ir/stageX)
        d_ir = OUTPUT_ROOT / subset_name / "ir" / stage
        d_ir.mkdir(parents=True, exist_ok=True)
        ir_output_dirs[stage] = d_ir

    files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])

    for fname in tqdm(files, desc=f"Processing {subset_name}"):
        save_name_npy = Path(fname).stem + ".npy"

        # 断点续传：如果 VI 文件夹里已经有这个文件，说明处理过了 (假设IR也就有了)
        if (vi_output_dirs['stage1'] / save_name_npy).exists():
            continue

        path_vis = vis_dir / fname
        path_ir = ir_dir / fname

        try:
            img_vis = cv2.imread(str(path_vis))
            img_ir = cv2.imread(str(path_ir))
            if img_vis is None or img_ir is None: continue

            img_vis = cv2.resize(img_vis, (IMG_W, IMG_H))
            img_ir = cv2.resize(img_ir, (IMG_W, IMG_H))

            # === 推理 (这是必须的，无法跳过) ===
            with torch.no_grad():
                masks_vis = mask_generator.generate(img_vis)
                masks_ir = mask_generator.generate(img_ir)

            heatmap_vis = simple_mask_overlay(masks_vis, (IMG_H, IMG_W))
            heatmap_ir = simple_mask_overlay(masks_ir, (IMG_H, IMG_W))

            # === 计算并保存 (只保存单模态) ===
            for stage_name, token_size in TOKEN_SIZES.items():
                map_vis = create_entropy_map(heatmap_vis, token_size)
                map_ir = create_entropy_map(heatmap_ir, token_size)

                # 仅保存 VI 和 IR，不再做相加和保存相加图
                np.save(vi_output_dirs[stage_name] / save_name_npy, map_vis)
                np.save(ir_output_dirs[stage_name] / save_name_npy, map_ir)

            del masks_vis, masks_ir, heatmap_vis, heatmap_ir, map_vis, map_ir
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error {fname}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue


if __name__ == "__main__":
    # === 包含所有阶段 ===
    TARGET_SUBSETS = ["train", "val", "test"]

    print(f"正在加载 SAM 2 模型 (Config: {MODEL_CFG})...")
    model = build_sam2(str(MODEL_CFG), str(CHECKPOINT), device=DEVICE)
    mask_generator = SAM2AutomaticMaskGenerator(model)

    for subset in TARGET_SUBSETS:
        if check_data_consistency(subset):
            print(f"\n>>> 处理 {subset} ...")
            process_subset(subset, mask_generator)

    print("\n所有单模态熵图补全完成。")