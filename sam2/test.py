import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# --- 引入您的模型定义 ---
from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel

# ================= 配置区域 =================
CKPT_PATH = "checkpoints/1_27日23点权重/best_model.pth"
ENTROPY_ROOT = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/entropy_maps_add"
TEST_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'
}

# 输出文件夹
OUTPUT_DIR = "1_22日17点训练结果"

CLASS_NAMES = [
    "Background", "Car", "Person", "Bike", "Curve",
    "CarStop", "Guardrail", "ColorCone", "Bump"
]
NUM_CLASSES = 9
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"


# ================= 数据集定义 =================
class MSRSTestDataset(Dataset):
    def __init__(self, dirs, entropy_root=None):
        self.vis_dir = dirs['vi']
        self.ir_dir = dirs['ir']
        self.lbl_dir = dirs['label']
        self.entropy_vis_dirs = {}
        self.entropy_ir_dirs = {}

        if entropy_root:
            sub = 'test'
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                self.entropy_vis_dirs[stage] = os.path.join(entropy_root, sub, 'vi', stage)
                self.entropy_ir_dirs[stage] = os.path.join(entropy_root, sub, 'ir', stage)

        raw_files = sorted([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])
        self.files = []
        for f in raw_files:
            if os.path.exists(os.path.join(self.ir_dir, f)) and \
                    os.path.exists(os.path.join(self.lbl_dir, f)):
                self.files.append(f)
        print(f"✅ Found {len(self.files)} valid test samples.")

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def _load_gray_as_tensor(self, path):
        if not os.path.exists(path): return torch.zeros((1, 15, 20))
        arr = np.load(path)
        t = torch.from_numpy(arr).float()
        if t.ndim == 2: t = t.unsqueeze(0)
        return t

    def __getitem__(self, i):
        n = self.files[i]
        # 加载图片
        v_img = Image.open(os.path.join(self.vis_dir, n)).convert('RGB')
        i_img = Image.open(os.path.join(self.ir_dir, n)).convert('RGB')
        lbl = Image.open(os.path.join(self.lbl_dir, n))

        # 强制 Resize 到 480x640
        v_img = v_img.resize((640, 480), Image.BILINEAR)
        i_img = i_img.resize((640, 480), Image.BILINEAR)
        lbl = lbl.resize((640, 480), Image.NEAREST)

        v_tensor = TF.to_tensor(v_img)
        v = self.normalize(v_tensor)
        i_tensor = TF.to_tensor(i_img)
        i_new = self.normalize(i_tensor)
        l = torch.from_numpy(np.array(lbl)).long()

        list_vis, list_ir, list_sum = [], [], []
        target_sizes = [[120, 160], [60, 80], [30, 40], [15, 20]]

        if self.entropy_vis_dirs:
            npy_name = n.replace('.png', '.npy')
            idx = 0
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                p_vis = os.path.join(self.entropy_vis_dirs[stage], npy_name)
                p_ir = os.path.join(self.entropy_ir_dirs[stage], npy_name)
                t_vis = self._load_gray_as_tensor(p_vis)
                t_ir = self._load_gray_as_tensor(p_ir)
                t_sum = t_vis + t_ir

                size = target_sizes[idx]
                t_vis = TF.resize(t_vis, size, interpolation=TF.InterpolationMode.NEAREST)
                t_ir = TF.resize(t_ir, size, interpolation=TF.InterpolationMode.NEAREST)
                t_sum = TF.resize(t_sum, size, interpolation=TF.InterpolationMode.NEAREST)

                list_vis.append(t_vis)
                list_ir.append(t_ir)
                list_sum.append(t_sum)
                idx += 1

        return v, i_new, l, list_sum, list_vis, list_ir, n


# ================= 可视化工具 =================
def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    person = [64, 64, 0]
    bike = [0, 128, 192]
    curve = [0, 0, 192]
    car_stop = [128, 128, 0]
    guardrail = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump = [192, 64, 0]
    palette = np.array([unlabelled, car, person, bike, curve, car_stop, guardrail, color_cone, bump], dtype=np.uint8)
    return palette


def save_comparison_vis(pred_mask, gt_mask, save_path):
    """
    拼接保存：左边是真实标签(GT)，右边是预测结果(Prediction)
    """
    palette = get_palette()

    # 1. 生成彩色 GT 掩码
    gt_color_arr = palette[gt_mask]
    gt_img = Image.fromarray(gt_color_arr)

    # 2. 生成彩色预测掩码
    pred_color_arr = palette[pred_mask]
    pred_img = Image.fromarray(pred_color_arr)

    # 3. 拼接图像 (宽度 x2)
    w, h = gt_img.size
    combined_img = Image.new('RGB', (w * 2, h))

    # 粘贴: 左 GT, 右 Pred
    combined_img.paste(gt_img, (0, 0))
    combined_img.paste(pred_img, (w, 0))

    # 4. 保存
    combined_img.save(save_path)


# ================= 评估器 =================
class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self):
        self.confusion_matrix.fill(0)

    def add_batch(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_matrix

    def get_miou_and_pa(self):
        """
        计算mIoU和全局像素准确率
        返回:
            miou: 平均交并比
            pa: 全局像素准确率
            class_iou: 每个类别的IoU
        """
        intersection = np.diag(self.confusion_matrix)  # 各类别正确预测的像素数
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection

        # 计算mIoU (剔除Union为0的类别)
        valid_mask = union > 0
        if valid_mask.sum() == 0:
            miou = 0.0
        else:
            iou = intersection[valid_mask] / (union[valid_mask] + 1e-10)
            miou = np.mean(iou)

        # 计算全局像素准确率 (PA)
        pa = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)

        # 每个类别的IoU
        class_iou = np.zeros(self.num_classes)
        class_iou[valid_mask] = intersection[valid_mask] / (union[valid_mask] + 1e-10)

        return miou, pa, class_iou

    def get_class_pa(self):
        """
        计算每个类别的像素准确率 (Class-wise Pixel Accuracy)
        返回:
            class_pa: 每个类别的像素准确率
        """
        class_pa = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            # 该类别真实像素总数
            gt_pixels = np.sum(self.confusion_matrix[cls, :])
            if gt_pixels == 0:
                class_pa[cls] = 0.0
            else:
                # 该类别预测正确的像素数 / 该类别真实像素总数
                class_pa[cls] = self.confusion_matrix[cls, cls] / (gt_pixels + 1e-10)
        return class_pa


# ================= 主函数 =================
def test():
    print(f"🚀 Loading model from {CKPT_PATH}...")
    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    try:
        checkpoint = torch.load(CKPT_PATH, map_location='cuda')
        model.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint not found at {CKPT_PATH}")
        return

    model.eval()

    if not os.path.exists(TEST_DIRS['vi']):
        print(f"⚠️ Test dir {TEST_DIRS['vi']} not found, switching to 'val'...")
        TEST_DIRS['vi'] = TEST_DIRS['vi'].replace('test', 'val')
        TEST_DIRS['ir'] = TEST_DIRS['ir'].replace('test', 'val')
        TEST_DIRS['label'] = TEST_DIRS['label'].replace('test', 'val')

    dataset = MSRSTestDataset(TEST_DIRS, ENTROPY_ROOT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    global_evaluator = Evaluator(NUM_CLASSES)
    total_image_miou = 0.0
    total_image_pa = 0.0  # 新增：累加单张图片的PA
    valid_samples = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Visualizations (GT vs Pred) will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    print(f"🚀 Start testing on {len(dataset)} images...")

    with torch.no_grad():
        for i, (v, i_img, l, l_sum, l_vis, l_ir, fname) in enumerate(tqdm(dataloader)):
            v, i_img, l = v.cuda(), i_img.cuda(), l.cuda()
            fname = fname[0]

            e_sum = [e.cuda() for e in l_sum]
            e_vis = [e.cuda() for e in l_vis]
            e_ir = [e.cuda() for e in l_ir]

            seg_out, _, _, _ = model(
                vis=v, ir=i_img, gt_semantic=None,
                gt_entropy_maps=e_sum, gt_entropy_vis=e_vis, gt_entropy_ir=e_ir
            )

            pred = torch.argmax(seg_out, dim=1).cpu().numpy().squeeze()
            gt = l.cpu().numpy().squeeze()  # Ground Truth

            # 1. 计算指标
            single_eval = Evaluator(NUM_CLASSES)
            single_eval.add_batch(gt, pred)
            s_miou, s_pa, _ = single_eval.get_miou_and_pa()  # s_pa是单张图片的PA

            total_image_miou += s_miou
            total_image_pa += s_pa  # 新增：累加单张PA
            valid_samples += 1
            global_evaluator.add_batch(gt, pred)

            # 2. 可视化拼接 (GT 左, Pred 右)
            save_name = fname
            save_path = os.path.join(OUTPUT_DIR, save_name)
            save_comparison_vis(pred, gt, save_path)

    # --- 最终打印指标 ---
    g_miou, g_pa, class_iou = global_evaluator.get_miou_and_pa()  # g_pa是全局PA
    class_pa = global_evaluator.get_class_pa()  # 新增：获取每个类别的PA
    avg_img_miou = total_image_miou / max(valid_samples, 1)
    avg_img_pa = total_image_pa / max(valid_samples, 1)  # 新增：平均单张图片PA

    print("\n" + "=" * 60)
    print(f"📊 Final Test Results")
    print(f"   Global Mean IoU:    {g_miou * 100:.2f}%")
    print(f"   Global Pixel Acc:   {g_pa * 100:.2f}%")  # 新增：打印全局PA
    print(f"   Avg Image mIoU:     {avg_img_miou * 100:.2f}%")
    print(f"   Avg Image Pixel Acc:{avg_img_pa * 100:.2f}%")  # 新增：打印平均单张PA
    print("-" * 60)
    print(f"{'Class Name':<15} | {'IoU (%)':<10} | {'PA (%)':<10}")  # 新增：PA列
    print("-" * 60)
    for idx, (iou, pa) in enumerate(zip(class_iou, class_pa)):  # 新增：遍历类别PA
        print(f"{CLASS_NAMES[idx]:<15} | {iou * 100:.2f}%     | {pa * 100:.2f}%")
    print("-" * 60)
    print(f"✅ All visualizations (GT vs Pred) saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    test()