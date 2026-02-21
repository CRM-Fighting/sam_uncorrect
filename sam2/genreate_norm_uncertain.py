import numpy as np
import torch
import cv2
import os
from pathlib import Path
import sys

# 尝试导入进度条
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 库可以显示进度条 (pip install tqdm)")

# 尝试导入 SAM2
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库，请检查环境。")
    exit()


# ==========================================
#          核心算法：计算 ci (Float32)
# ==========================================

def generate_ci_map_and_stats(masks, image_shape):
    """
    计算论文公式参数 ci 和 最大堆叠数。
    返回: (ci_map, max_stack_value)
    """
    h, w = image_shape[:2]
    overlay_count = np.zeros((h, w), dtype=np.float32)

    if len(masks) == 0:
        return overlay_count, 0.0

    # 1. 累加掩码覆盖次数
    for mask_data in masks:
        overlay_count[mask_data['segmentation']] += 1.0

    # 2. 获取最大堆叠数
    max_stack = np.max(overlay_count)

    # 3. 归一化
    if max_stack > 0:
        ci_map = overlay_count / max_stack
    else:
        ci_map = overlay_count

    return ci_map.astype(np.float32), max_stack


def process_paired_data(vis_dir, ir_dir, output_root, mask_generator):
    """
    成对处理函数：同时读取红外和可见光，以便同时打印统计信息
    """
    vis_path = Path(vis_dir)
    ir_path = Path(ir_dir)

    # 准备输出路径
    out_vis_path = Path(output_root) / "train/vi"
    out_ir_path = Path(output_root) / "train/ir"
    out_vis_path.mkdir(parents=True, exist_ok=True)
    out_ir_path.mkdir(parents=True, exist_ok=True)

    # 获取文件列表 (以可见光目录为基准)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in vis_path.iterdir() if f.suffix.lower() in valid_extensions])

    print(f"\n任务: 处理 Train 数据集 (成对处理)")
    print(f"  红外源: {ir_path}")
    print(f"  可见源: {vis_path}")
    print(f"  数量: {len(files)} 对")

    iterator = tqdm(files, desc="Processing Pairs", unit="pair") if HAS_TQDM else files

    for f_vis in iterator:
        fname = f_vis.name
        f_ir = ir_path / fname

        # --- 1. 检查文件是否存在 ---
        if not f_ir.exists():
            msg = f"红外——{fname}——未找到"
            if HAS_TQDM:
                tqdm.write(msg)
            else:
                print(msg)
            continue

        # --- 2. 读取图片 ---
        img_vis = cv2.imread(str(f_vis))
        img_ir = cv2.imread(str(f_ir))

        if img_vis is None:
            msg = f"可见光——{fname}——未找到"
            if HAS_TQDM:
                tqdm.write(msg)
            else:
                print(msg)
            continue

        if img_ir is None:
            msg = f"红外——{fname}——未找到"
            if HAS_TQDM:
                tqdm.write(msg)
            else:
                print(msg)
            continue

        try:
            # --- 3. 处理红外 (IR) ---
            # 这里的 mask_generator 已经是 Large 版本了
            masks_ir = mask_generator.generate(img_ir)
            ci_map_ir, max_stack_ir = generate_ci_map_and_stats(masks_ir, img_ir.shape)

            # --- 4. 处理可见光 (Vis) ---
            masks_vis = mask_generator.generate(img_vis)
            ci_map_vis, max_stack_vis = generate_ci_map_and_stats(masks_vis, img_vis.shape)

            # --- 5. 保存 .npy ---
            save_name = f_vis.stem + ".npy"
            np.save(str(out_ir_path / save_name), ci_map_ir)
            np.save(str(out_vis_path / save_name), ci_map_vis)

            # --- 6. 打印指定格式 ---
            stats_msg = f"{fname}—红外：{int(max_stack_ir)}-可见光：{int(max_stack_vis)}"

            if HAS_TQDM:
                tqdm.write(stats_msg)
            else:
                print(stats_msg)

        except Exception as e:
            err_msg = f"[异常] 处理 {fname} 时出错: {e}"
            if HAS_TQDM:
                tqdm.write(err_msg)
            else:
                print(err_msg)


if __name__ == "__main__":
    # --- 1. 路径配置 (绝对路径) ---
    DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")
    OUTPUT_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map")

    train_vi_dir = DATASET_ROOT / "train/vi"
    train_ir_dir = DATASET_ROOT / "train/ir"

    # --- 2. 加载 SAM2 Large 模型 【已修改】 ---
    # 权重路径：确保你有这个文件，通常 Large 版本的权重名为 sam2.1_hiera_large.pt
    checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    # Config路径：使用 hiera_l (Large)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"正在加载 SAM2 Large 模型到 {device} ...")
    print(f"配置文件: {model_cfg}")
    print(f"权重文件: {checkpoint}")

    try:
        model = build_sam2(model_cfg, checkpoint, device=device)
        # 初始化 Mask Generator，参数可按需调整
        mask_generator = SAM2AutomaticMaskGenerator(model)
        print("Large 模型加载成功。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print(
            "请检查：\n1. checkpoints 目录下是否有 sam2.1_hiera_large.pt\n2. configs/sam2.1/ 下是否有 sam2.1_hiera_l.yaml")
        exit()

    # --- 3. 执行成对处理 (只处理训练集) ---
    process_paired_data(train_vi_dir, train_ir_dir, OUTPUT_ROOT, mask_generator)

    print("\n所有训练集处理完成。")