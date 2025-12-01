import numpy as np
import torch
import cv2  # 用于加载、绘制和拼接
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import math
import random  # 用于生成随机掩码颜色

# 假设 sam2 库在您的 Python 路径中
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库。")
    print("请确保您已按照 INSTALL.md 安装了 'sam2'。")
    print("您可能需要运行: pip install -e .")
    exit()

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 核心辅助函数 ---

def simple_mask_overlay(masks, image_shape):
    """生成热度图 (Heatmap)"""
    print(f"  正在创建 {image_shape[0]}x{image_shape[1]} 的热度图...")
    overlay = np.zeros(image_shape[:2], dtype=np.int32)

    if len(masks) == 0:
        print("  警告: SAM 2 没有找到任何掩码。")
        return overlay

    for mask_data in masks:
        overlay[mask_data['segmentation']] += 1

    print(f"  热度图创建完毕。最大重叠次数: {np.max(overlay)}")
    return overlay


def tokenize_heatmap(heatmap, token_shape):
    """将热度图均匀分割为 tokens。"""
    H, W = heatmap.shape
    token_h, token_w = token_shape
    if H % token_h != 0 or W % token_w != 0:
        raise ValueError(f"热度图尺寸 ({H}, {W}) 无法被 token 尺寸 ({token_h}, {token_w}) 整除。")
    num_tokens_h, num_tokens_w = H // token_h, W // token_w
    tokens = heatmap.reshape(num_tokens_h, token_h, num_tokens_w, token_w).transpose(0, 2, 1, 3)
    return tokens, num_tokens_h, num_tokens_w


def calculate_token_entropy(token):
    """计算单个 token 的信息熵。"""
    values, counts = np.unique(token, return_counts=True)
    probabilities = counts / counts.sum()  # <-- 分子=counts, 分母=counts.sum()
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0: return 0.0
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def create_entropy_map(tokens, num_tokens_h, num_tokens_w):
    """遍历 tokens 生成熵图。"""
    entropy_map = np.zeros((num_tokens_h, num_tokens_w), dtype=np.float32)
    for i in range(num_tokens_h):
        for j in range(num_tokens_w):
            entropy_map[i, j] = calculate_token_entropy(tokens[i, j])
    return entropy_map


# --- 2. 图像生成与保存函数 ---

def create_segmentation_image(image, masks_list, mask_count):
    """
    (新) 在原图上绘制所有分割掩码，并标注数量。
    """
    print(f"  正在绘制 {mask_count} 个掩码到分割图上...")
    overlay = np.zeros_like(image, dtype=np.uint8)
    final_image = image.copy()

    for mask_data in masks_list:
        color = [random.randint(50, 255) for _ in range(3)]
        overlay[mask_data['segmentation']] = color

    final_image = cv2.addWeighted(final_image, 1.0, overlay, 0.5, 0)

    # --- 添加掩码数量标注 ---
    text = f"分割掩码数量 (Masks): {mask_count}"
    cv2.rectangle(final_image, (0, 0), (450, 45), (0, 0, 0), -1)
    cv2.putText(final_image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return final_image


def save_map_visual_with_legend(map_data, path, title, interpolation='bilinear', cmap='hot', vmin=None, vmax=None,
                                label=None):
    """
    (已更新) 使用 Matplotlib 保存图像，带 colorbar (图例)。
    支持 'hot' (热图) 和 'coolwarm' (差异图) 色彩。
    """
    plt.figure(figsize=(8, 6))

    if label is None:
        if '热度图' in title:
            label = '掩码叠加次数'
        elif '熵图' in title:
            label = '信息熵'
        else:
            label = '数值'

    im = plt.imshow(map_data, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=16)
    plt.colorbar(im, label=label)
    plt.axis('off')

    plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


def stitch_images(image_paths_list, output_path, standard_width=800):
    """
    (新) 将所有图像垂直拼接成一张长图。
    """
    print(f"  正在拼接 {len(image_paths_list)} 张图像...")
    all_images_resized = []

    for path in image_paths_list:
        img = cv2.imread(str(path))
        if img is None:
            print(f"    警告: 无法读取用于拼接的图像 {path}")
            continue

        h, w, _ = img.shape
        new_h = int(h * (standard_width / w))
        resized_img = cv2.resize(img, (standard_width, new_h), interpolation=cv2.INTER_AREA)
        all_images_resized.append(resized_img)

    try:
        final_image = cv2.vconcat(all_images_resized)
        cv2.imwrite(str(output_path), final_image)
        print(f"  已保存拼接长图到: {output_path}")
    except Exception as e:
        print(f"    错误: 拼接图像失败: {e}")


# --- 3. 工作流函数 ---
def process_and_save_maps(heatmap, output_dir, base_name, token_sizes):
    """
    接收热度图，生成并保存所有热度图和熵图。
    返回:
    1. saved_paths_dict: 包含所有 (5) 张已保存图像路径的字典。
    2. all_entropy_maps_data: 包含 (4) 个熵图原始Numpy数组的字典。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths_dict = {}  # 用于存储路径
    all_entropy_maps_data = {}  # 用于存储数据

    # --- 1. 保存热度图 ---
    print("  正在保存热度图...")
    save_path_heatmap_png = output_dir / f"{base_name}_2_heatmap_visual.png"
    np.save(output_dir / f"{base_name}_2_heatmap_raw.npy", heatmap)
    save_map_visual_with_legend(
        heatmap,
        save_path_heatmap_png,
        title=f"{base_name} 热度图 (Heatmap)",
        interpolation='bilinear'
    )
    saved_paths_dict['heatmap'] = save_path_heatmap_png

    # --- 2. 循环生成和保存熵图 ---
    print("  正在生成和保存多尺度熵图...")
    for token_size in token_sizes:
        print(f"    计算尺度: {token_size}x{token_size}")
        token_shape = (token_size, token_size)
        tokens, num_h, num_w = tokenize_heatmap(heatmap, token_shape)
        entropy_map = create_entropy_map(tokens, num_h, num_w)

        all_entropy_maps_data[token_size] = entropy_map  # <-- 存储数据

        map_name = f"{base_name}_3_entropy_map_{token_size}x{token_size}"
        save_path_entropy_png = output_dir / f"{map_name}_visual.png"
        np.save(output_dir / f"{map_name}_raw.npy", entropy_map)

        save_map_visual_with_legend(
            entropy_map,
            save_path_entropy_png,
            title=f"{base_name} 熵图 (Entropy Map) {token_size}x{token_size}",
            interpolation='nearest'  # <-- 关键：保持块状
        )
        saved_paths_dict[f'entropy_{token_size}'] = save_path_entropy_png

    return saved_paths_dict, all_entropy_maps_data


# --- 4. 主执行脚本 ---
if __name__ == "__main__":

    # --- A. 配置参数 ---
    script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

    ir_base_folder = script_dir / "../images/ir/"
    vis_base_folder = script_dir / "../images/vis/"
    output_base_dir = script_dir / "../results/"

    # !!! --- 用户修改区: 请在这里定义您 5 对图像的文件名 --- !!!
    image_basenames_to_process = [
        "00005N.png",
        "00006N.png",
        "00007N.png",
        "00009N.png",
        "000010N.png",
        "00183D.png",
    ]
    # --- 结束修改区 ---

    checkpoint = script_dir / "../checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = script_dir / "configs/sam2.1/sam2.1_hiera_t.yaml"

    device = "cpu"
    print(f"正在强制使用 {device} 设备")

    token_sizes_to_process = [4, 8, 16, 32]
    stitched_image_width = 800

    # --- B. 加载模型 ---

    # !!! --- 关键警告 --- !!!
    # 下面的 'build_sam2'
    # 除非您已经手动修改了 'sam2/build_sam.py' 文件 (强制CPU)，
    # 否则此脚本将在此处崩溃并显示 'AssertionError'。
    # !!! --- 关键警告 --- !!!

    print(f"正在加载模型: {model_cfg.name}...")
    model = build_sam2(str(model_cfg), str(checkpoint), device=device)
    model.to(device)
    print("SAM 2 模型加载成功。")

    mask_generator = SAM2AutomaticMaskGenerator(model)

    # --- C. 循环处理每对图像 ---
    for basename in image_basenames_to_process:

        print(f"\n--- 开始处理图像对: {basename} ---")

        # 1. 定义路径
        path_ir = ir_base_folder / basename
        path_vis = vis_base_folder / basename

        pair_output_dir = output_base_dir / Path(basename).stem
        output_ir_dir = pair_output_dir / "ir"
        output_vis_dir = pair_output_dir / "vis"
        output_diff_dir = pair_output_dir / "diff"

        output_ir_dir.mkdir(parents=True, exist_ok=True)
        output_vis_dir.mkdir(parents=True, exist_ok=True)
        output_diff_dir.mkdir(parents=True, exist_ok=True)

        base_name_ir = f"{Path(basename).stem}_ir"
        base_name_vis = f"{Path(basename).stem}_vis"

        # 2. 加载图像
        image_ir = cv2.imread(str(path_ir))
        image_vis = cv2.imread(str(path_vis))

        if image_ir is None or image_vis is None:
            print(f"  错误: 找不到 {path_ir} 或 {path_vis}。跳过此图像对。")
            continue

        print(f"  图像尺寸: {image_ir.shape[1]}x{image_ir.shape[0]}")

        # --- 处理红外图像 ---
        print("  正在处理红外图像...")
        path_orig_ir = output_ir_dir / f"{base_name_ir}_0_input.png"
        cv2.imwrite(str(path_orig_ir), image_ir)

        masks_ir = mask_generator.generate(image_ir)
        img_seg_ir = create_segmentation_image(image_ir, masks_ir, len(masks_ir))
        path_seg_ir = output_ir_dir / f"{base_name_ir}_1_segmentation_map.png"
        cv2.imwrite(str(path_seg_ir), img_seg_ir)

        heatmap_ir = simple_mask_overlay(masks_ir, image_ir.shape)
        map_paths_ir, entropy_maps_ir_data = process_and_save_maps(
            heatmap_ir, output_ir_dir, base_name_ir, token_sizes_to_process
        )

        # --- 处理可见光图像 ---
        print("  正在处理可见光图像...")
        path_orig_vis = output_vis_dir / f"{base_name_vis}_0_input.png"
        cv2.imwrite(str(path_orig_vis), image_vis)

        masks_vis = mask_generator.generate(image_vis)
        img_seg_vis = create_segmentation_image(image_vis, masks_vis, len(masks_vis))
        path_seg_vis = output_vis_dir / f"{base_name_vis}_1_segmentation_map.png"
        cv2.imwrite(str(path_seg_vis), img_seg_vis)

        heatmap_vis = simple_mask_overlay(masks_vis, image_vis.shape)
        map_paths_vis, entropy_maps_vis_data = process_and_save_maps(
            heatmap_vis, output_vis_dir, base_name_vis, token_sizes_to_process
        )

        # --- 处理差异熵图 (熵减) ---
        print("  正在处理差异熵图 (IR - VIS)...")
        for token_size in token_sizes_to_process:
            print(f"    计算尺度: {token_size}x{token_size}")

            map_ir = entropy_maps_ir_data[token_size]
            map_vis = entropy_maps_vis_data[token_size]
            diff_map = map_ir - map_vis

            base_name_diff = f"diff_map_{token_size}x{token_size}"
            save_path_npy = output_diff_dir / f"{base_name_diff}_raw.npy"
            save_path_png = output_diff_dir / f"{base_name_diff}_visual.png"

            np.save(save_path_npy, diff_map)

            v_abs_max = np.max(np.abs(diff_map))
            v_min, v_max = (None, None) if v_abs_max == 0 else (-v_abs_max, v_abs_max)

            save_map_visual_with_legend(
                diff_map, save_path_png,
                title=f"差异熵图 (IR - VIS) {token_size}x{token_size}",
                interpolation='nearest', cmap='coolwarm',
                vmin=v_min, vmax=v_max, label='熵值差异 (IR - VIS)'
            )

        # --- 拼接长图 ---
        print("  正在拼接长图...")
        # 拼接红外
        paths_to_stitch_ir = [
            path_orig_ir, path_seg_ir, map_paths_ir['heatmap'],
            map_paths_ir['entropy_4'], map_paths_ir['entropy_8'],
            map_paths_ir['entropy_16'], map_paths_ir['entropy_32'],
        ]
        stitch_path_ir = pair_output_dir / f"{Path(basename).stem}_ir_9_ALL_STITCHED.png"
        stitch_images(paths_to_stitch_ir, stitch_path_ir, standard_width=stitched_image_width)

        # 拼接可见光
        paths_to_stitch_vis = [
            path_orig_vis, path_seg_vis, map_paths_vis['heatmap'],
            map_paths_vis['entropy_4'], map_paths_vis['entropy_8'],
            map_paths_vis['entropy_16'], map_paths_vis['entropy_32'],
        ]
        stitch_path_vis = pair_output_dir / f"{Path(basename).stem}_vis_9_ALL_STITCHED.png"
        stitch_images(paths_to_stitch_vis, stitch_path_vis, standard_width=stitched_image_width)

    print("\n--- 所有处理已完成 ---")