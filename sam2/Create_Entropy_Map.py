import numpy as np
import torch
import cv2
import os
import gc  # 引入垃圾回收，防止默认密度下内存溢出
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
DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")
OUTPUT_ROOT = DATASET_ROOT / "entropy_maps"

# 4个阶段的下采样倍率
TOKEN_SIZES = {'stage1': 4, 'stage2': 8, 'stage3': 16, 'stage4': 32}
IMG_W, IMG_H = 640, 480

# 模型配置
CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    要求5：在进行之前先检查训练集中 Segmentation_labels, vi, ir 的图像是否对应
    """
    print(f"\n{'=' * 20} 正在检查 {subset_name} 数据集一致性 {'=' * 20}")

    vi_dir = DATASET_ROOT / subset_name / "vi"
    ir_dir = DATASET_ROOT / subset_name / "ir"

    # === 修改点：文件夹名称改为 Segmentation_labels ===
    label_dir = DATASET_ROOT / subset_name / "Segmentation_labels"

    # 检查文件夹是否存在
    if not vi_dir.exists() or not ir_dir.exists() or not label_dir.exists():
        print(f"严重错误：数据目录不完整！请检查 vi, ir, Segmentation_labels 文件夹是否存在于 {subset_name} 下。")
        # 打印具体哪个不存在
        if not vi_dir.exists(): print(f"  -> 缺失: {vi_dir}")
        if not ir_dir.exists(): print(f"  -> 缺失: {ir_dir}")
        if not label_dir.exists(): print(f"  -> 缺失: {label_dir}")
        return False

    # 获取所有文件名（不带路径）
    files_vi = set(f for f in os.listdir(vi_dir) if f.endswith('.png'))
    files_ir = set(f for f in os.listdir(ir_dir) if f.endswith('.png'))
    files_lb = set(f for f in os.listdir(label_dir) if f.endswith('.png'))

    # 取并集，找出所有出现过的文件名
    all_files = files_vi | files_ir | files_lb

    consistent = True
    print(f"共扫描到 {len(all_files)} 个唯一文件名，开始比对...")

    for fname in sorted(list(all_files)):
        missing_parts = []
        if fname not in files_vi: missing_parts.append("可见光(vi)")
        if fname not in files_ir: missing_parts.append("红外(ir)")
        if fname not in files_lb: missing_parts.append("标签(Segmentation_labels)")

        if missing_parts:
            consistent = False
            print(f"[数据不一致] 文件: {fname} 缺失于 -> {', '.join(missing_parts)}")

    if consistent:
        print(">> 检查通过：vi, ir, Segmentation_labels 所有图像一一对应。")
    else:
        print(">> 警告：发现数据不一致（详见上方日志）。程序将继续执行，但在生成时会跳过缺失文件。")

    print("=" * 60 + "\n")
    return consistent


# ================= 核心处理逻辑 =================
def process_subset(subset_name, mask_generator):
    print(f"开始处理数据集: {subset_name}")
    vis_dir = DATASET_ROOT / subset_name / "vi"
    ir_dir = DATASET_ROOT / subset_name / "ir"

    # 创建输出目录
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

        # 断点续传检查
        if (output_dirs['stage1'] / save_name_npy).exists():
            continue

        # --- 严格检查与打印 ---
        missing_flag = False
        if not path_vis.exists():
            print(f"\n[缺失报错] 没找到可见光图: {fname}")
            missing_flag = True
        if not path_ir.exists():
            print(f"\n[缺失报错] 没找到红外图: {fname}")
            missing_flag = True
        if missing_flag: continue

        try:
            img_vis = cv2.imread(str(path_vis))
            img_ir = cv2.imread(str(path_ir))

            if img_vis is None:
                print(f"\n[读取失败] 可见光文件损坏或无法读取: {fname}")
                continue
            if img_ir is None:
                print(f"\n[读取失败] 红外文件损坏或无法读取: {fname}")
                continue

            img_vis = cv2.resize(img_vis, (IMG_W, IMG_H))
            img_ir = cv2.resize(img_ir, (IMG_W, IMG_H))

            # === 内存保护：使用 no_grad 并显式清理 ===
            # 这里不改变采样密度，但必须手动管理内存，否则默认密度下非常容易崩
            with torch.no_grad():
                masks_vis = mask_generator.generate(img_vis)
                masks_ir = mask_generator.generate(img_ir)

            heatmap_vis = simple_mask_overlay(masks_vis, (IMG_H, IMG_W))
            heatmap_ir = simple_mask_overlay(masks_ir, (IMG_H, IMG_W))

            for stage_name, token_size in TOKEN_SIZES.items():
                map_vis = create_entropy_map(heatmap_vis, token_size)
                map_ir = create_entropy_map(heatmap_ir, token_size)
                diff_map = map_ir - map_vis
                np.save(output_dirs[stage_name] / save_name_npy, diff_map)

            tqdm.write(f"红外图-{fname}，可见光-{fname}，差异熵图-{save_name_npy}")

            # === 显式垃圾回收 (防止 OOM) ===
            del masks_vis, masks_ir, heatmap_vis, heatmap_ir
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            # 捕获内存溢出或其他错误，打印并不中断程序
            print(f"\n[处理异常] 图片 {fname} 出错: {e}")
            # 尝试清理内存以挽救后续循环
            torch.cuda.empty_cache()
            gc.collect()
            continue


if __name__ == "__main__":
    # 1. 检查数据
    # 如果不通过，程序会打印警告但继续（根据原代码逻辑）
    # 如果你希望能强制退出，可以在 check_data_consistency 返回 False 时 exit()
    check_data_consistency("train")

    # 2. 加载模型
    print("正在加载 SAM 2 模型...")
    model = build_sam2(MODEL_CFG, CHECKPOINT, device=DEVICE)

    # === 保持默认参数 (高采样密度) ===
    # 注意：这会消耗大量显存和计算时间
    mask_generator = SAM2AutomaticMaskGenerator(model)

    # 3. 只处理训练集
    process_subset("train", mask_generator)

    print("\n所有任务处理完成。")