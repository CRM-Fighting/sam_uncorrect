import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
# 熵图的根目录 (请根据您的实际路径修改)
ENTROPY_ROOT = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/entropy_maps_add"

# 数据集子集 (这里取训练集)
SUBSET = "train"

# 要可视化的图片数量
NUM_IMAGES = 10

# 输出文件夹
OUTPUT_DIR = "visualization_results_with_gap"

# 阶段列表
STAGES = ["stage1", "stage2", "stage3", "stage4"]


# ================= 工具函数 =================

def load_npy(path):
    """加载 .npy 文件"""
    if not os.path.exists(path):
        # 尝试打印路径以便调试，但不要报错停止
        # print(f"Warning: File not found: {path}")
        return None
    return np.load(path)


def visualize_stage(vi_map, ir_map, sum_map, stage_name, save_dir, img_name):
    """
    可视化单个阶段的 VI, IR, Sum 熵图，并标注 GAP 均值
    """
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 计算 GAP (全局平均值)
    gap_value = 0.0
    if sum_map is not None:
        gap_value = np.mean(sum_map)

    # 设置总标题
    fig.suptitle(f"Image: {img_name} | {stage_name} | Sum Entropy GAP: {gap_value:.6f}", fontsize=18, fontweight='bold')

    # 定义要画的图和标题
    maps = [
        ("Visible Entropy (VI)", vi_map, "Blues"),
        ("Infrared Entropy (IR)", ir_map, "Reds"),
        (f"Sum Entropy (VI+IR)\nGAP Mean: {gap_value:.4f}", sum_map, "viridis")  # 在标题里显示 GAP
    ]

    for ax, (title, data, cmap) in zip(axes, maps):
        if data is None:
            ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
            ax.axis('off')
            continue

        # 绘制热力图
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=14)
        ax.axis('off')  # 去掉坐标轴刻度，更美观

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        # 特殊处理：如果是 Stage4 (15x20) 或者其他小图，直接在格子上印数字
        # 这样老师能看到每个像素的具体熵值
        if data.shape[0] <= 32:
            for (j, i), val in np.ndenumerate(data):
                # 字体大小根据图片尺寸自动调整
                fontsize = 6 if data.shape[0] > 20 else 7
                # 颜色反转：深色背景用白字，浅色背景用黑字
                text_color = 'white' if val > data.max() / 2 else 'black'

                ax.text(i, j, f"{val:.1f}", ha='center', va='center',
                        color=text_color, fontsize=fontsize)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, f"{img_name}_{stage_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return gap_value


def main():
    # 1. 准备路径
    vi_root = os.path.join(ENTROPY_ROOT, SUBSET, "vi")
    ir_root = os.path.join(ENTROPY_ROOT, SUBSET, "ir")

    # 2. 获取图片列表 (以 stage1 的 vi 为基准)
    stage1_vi_dir = os.path.join(vi_root, "stage1")
    if not os.path.exists(stage1_vi_dir):
        print(f"Error: Directory not found: {stage1_vi_dir}")
        return

    all_files = sorted([f for f in os.listdir(stage1_vi_dir) if f.endswith(".npy")])
    target_files = all_files[:NUM_IMAGES]

    print(f"🚀 Found {len(all_files)} images. Processing first {len(target_files)}...")
    print(f"📂 Results will be saved to: {os.path.abspath(OUTPUT_DIR)}\n")

    # 3. 循环处理每一张图片
    for fname in tqdm(target_files):
        img_name = os.path.splitext(fname)[0]  # 去掉 .npy 后缀

        # 为每张图片创建一个文件夹
        img_save_dir = os.path.join(OUTPUT_DIR, img_name)
        os.makedirs(img_save_dir, exist_ok=True)

        print(f"\nProcessing {img_name}...")

        # 4. 循环处理每一个阶段
        for stage in STAGES:
            # 构建文件路径
            vi_path = os.path.join(vi_root, stage, fname)
            ir_path = os.path.join(ir_root, stage, fname)

            # 加载数据
            vi_data = load_npy(vi_path)
            ir_data = load_npy(ir_path)

            # 计算 Sum
            sum_data = None
            if vi_data is not None and ir_data is not None:
                sum_data = vi_data + ir_data

            # 可视化并获取 GAP 值
            gap = visualize_stage(vi_data, ir_data, sum_data, stage, img_save_dir, img_name)

            print(f"  - {stage}: GAP Mean = {gap:.4f}")

    print(f"\n✅ All done! Please check the folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()