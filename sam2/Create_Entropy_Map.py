import numpy as np
import torch
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# --- 尝试导入 SAM 2 ---
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库。请确保已安装。")
    exit()

# ================= 配置区域 (请根据实际情况修改) =================

# 1. 数据集根目录
# 请确保目录结构为:
# .../MSRS/train/vi/, .../MSRS/train/ir/
# .../MSRS/val/vi/,   .../MSRS/val/ir/
DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")

# 2. 输出目录
# 结果将生成在: DATASET_ROOT/entropy_maps/
OUTPUT_ROOT = DATASET_ROOT / "entropy_maps"

# 3. Hiera 四个阶段对应的 Token 尺寸 (下采样倍率)
# Stage 1: /4, Stage 2: /8, Stage 3: /16, Stage 4: /32
TOKEN_SIZES = {
    'stage1': 4,
    'stage2': 8,
    'stage3': 16,
    'stage4': 32
}

# 4. 图像标准化尺寸 (必须与训练代码一致!)
# 宽度 640, 高度 480
IMG_W, IMG_H = 640, 480

# 5. 模型配置
CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 核心算法函数 =================

def simple_mask_overlay(masks, image_shape):
    """
    生成热度图 (叠加次数图)
    masks: SAM2 生成的掩码列表
    image_shape: (H, W)
    """
    overlay = np.zeros(image_shape[:2], dtype=np.int32)
    if len(masks) == 0:
        return overlay

    # 累加每个掩码覆盖的区域
    for mask_data in masks:
        overlay[mask_data['segmentation']] += 1
    return overlay


def calculate_token_entropy(token):
    """计算单个 Token (小块) 的信息熵"""
    # 统计小块内数值的频率
    values, counts = np.unique(token, return_counts=True)
    probabilities = counts / counts.sum()
    # 去除0概率以避免 log(0)
    probabilities = probabilities[probabilities > 0]

    if len(probabilities) == 0:
        return 0.0

    # 香农熵公式
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def create_entropy_map(heatmap, token_size):
    """
    将热度图切分为 Grid，并计算每个 Grid 的熵
    heatmap: (H, W)
    token_size: 下采样倍率 (如 4, 8, 16, 32)
    """
    H, W = heatmap.shape

    # 计算切分后的网格数量
    # 例如 480/32 = 15, 640/32 = 20
    num_h = H // token_size
    num_w = W // token_size

    # 使用 reshape 和 transpose 快速切块
    # [H, W] -> [num_h, size, num_w, size] -> [num_h, num_w, size, size]
    tokens = heatmap.reshape(num_h, token_size, num_w, token_size).transpose(0, 2, 1, 3)

    # 计算熵图
    entropy_map = np.zeros((num_h, num_w), dtype=np.float32)
    for i in range(num_h):
        for j in range(num_w):
            entropy_map[i, j] = calculate_token_entropy(tokens[i, j])

    return entropy_map


def process_subset(subset_name, mask_generator):
    """
    处理单个子集 (train 或 val)
    """
    print(f"\n{'=' * 20} 开始处理数据集: {subset_name} {'=' * 20}")

    # 定义输入路径 (注意文件夹名是 vi 和 ir)
    vis_dir = DATASET_ROOT / subset_name / "vi"
    ir_dir = DATASET_ROOT / subset_name / "ir"

    if not vis_dir.exists() or not ir_dir.exists():
        print(f"错误: 找不到输入文件夹 -> {vis_dir} 或 {ir_dir}")
        return

    # 预先创建所有输出目录
    # 结构: entropy_maps/train/stage1/
    output_dirs = {}
    for stage in TOKEN_SIZES.keys():
        d = OUTPUT_ROOT / subset_name / stage
        d.mkdir(parents=True, exist_ok=True)
        output_dirs[stage] = d
        print(f"  [准备目录] {d}")

    # 获取文件列表 (以可见光文件夹为准)
    files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
    print(f"  [文件数量] 找到 {len(files)} 对图像")

    # 使用 tqdm 显示进度
    for fname in tqdm(files, desc=f"Generating {subset_name}"):
        path_vis = vis_dir / fname
        path_ir = ir_dir / fname

        # 确保红外图也存在
        if not path_ir.exists():
            continue

            # 1. 读取图像
        img_vis = cv2.imread(str(path_vis))
        img_ir = cv2.imread(str(path_ir))

        if img_vis is None or img_ir is None:
            continue

        # 2. 强制 Resize (关键!)
        # 必须缩放到与训练时一致的尺寸 (640x480)，否则生成的 map 尺寸会对不上
        img_vis = cv2.resize(img_vis, (IMG_W, IMG_H))
        img_ir = cv2.resize(img_ir, (IMG_W, IMG_H))

        # 3. SAM 2 生成掩码
        # 生成器返回 list of dicts
        masks_vis = mask_generator.generate(img_vis)
        masks_ir = mask_generator.generate(img_ir)

        # 4. 生成热度图 (Heatmap)
        # 形状为 (480, 640)
        heatmap_vis = simple_mask_overlay(masks_vis, (IMG_H, IMG_W))
        heatmap_ir = simple_mask_overlay(masks_ir, (IMG_H, IMG_W))

        # 5. 循环生成 4 个阶段的差异熵图
        for stage_name, token_size in TOKEN_SIZES.items():
            # 分别计算熵
            map_vis = create_entropy_map(heatmap_vis, token_size)
            map_ir = create_entropy_map(heatmap_ir, token_size)

            # 计算差异图 (IR - VIS)
            # 正值: IR 信息量大; 负值: VIS 信息量大
            diff_map = map_ir - map_vis

            # 6. 保存为 .npy
            # 文件名保持与原图一致 (如 00005.npy)
            save_name = Path(fname).stem + ".npy"
            save_path = output_dirs[stage_name] / save_name

            np.save(save_path, diff_map)


# ================= 主执行入口 =================

if __name__ == "__main__":
    print(f"正在初始化 SAM 2 模型 ({DEVICE})...")

    # 构建 SAM2
    model = build_sam2(MODEL_CFG, CHECKPOINT, device=DEVICE)
    mask_generator = SAM2AutomaticMaskGenerator(model)

    # 1. 处理训练集 (train)
    process_subset("train", mask_generator)

    # 2. 处理验证集 (val)
    process_subset("val", mask_generator)

    print("\n" + "=" * 40)
    print("处理完成！")
    print(f"所有熵图已保存至: {OUTPUT_ROOT}")
    print("文件夹结构示例:")
    print(f"  {OUTPUT_ROOT}/train/stage1/00xxx.npy (尺寸 120x160)")
    print(f"  {OUTPUT_ROOT}/train/stage4/00xxx.npy (尺寸 15x20)")
    print("=" * 40)