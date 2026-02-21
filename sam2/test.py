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
# [注意] 这里我默认指向你最新的 Early Concat + Edge Sup 权重目录
# 请根据实际情况修改 EXP_NAME
EXP_NAME = "2月6日17点权重"
CKPT_PATH = f"checkpoints/{EXP_NAME}/best_model.pth"
OUTPUT_DIR = f"{EXP_NAME}_Test_Result"

# 路径配置
TEST_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'
}

NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_TTA = True  # 是否开启测试时增强

# MSRS 类别定义
CLASSES = [
    'Background', 'Car', 'Person', 'Bike', 'Curve',
    'Car Stop', 'Guardrail', 'Color Cone', 'Bump'
]


class MSRSTestDataset(Dataset):
    def __init__(self, dirs):
        self.vis_dir = dirs['vi']
        self.ir_dir = dirs['ir']
        self.lbl_dir = dirs['label']

        self.files = sorted([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])

        # 验证文件配对
        self.valid_files = []
        for f in self.files:
            if os.path.exists(os.path.join(self.ir_dir, f)) and os.path.exists(os.path.join(self.lbl_dir, f)):
                self.valid_files.append(f)

        print(f"✅ Found {len(self.valid_files)} valid test samples.")

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.valid_files)

    def _safe_to_tensor(self, pic):
        if isinstance(pic, Image.Image):
            arr = np.array(pic)
        else:
            arr = pic
        img = torch.tensor(arr, dtype=torch.float32)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1)
        return img / 255.0

    def __getitem__(self, index):
        fname = self.valid_files[index]

        v_path = os.path.join(self.vis_dir, fname)
        i_path = os.path.join(self.ir_dir, fname)
        l_path = os.path.join(self.lbl_dir, fname)

        v_img = Image.open(v_path).convert('RGB')
        i_img = Image.open(i_path).convert('RGB')
        lbl = Image.open(l_path)

        # Resize to standard size for inference
        v_img = v_img.resize((640, 480), Image.BILINEAR)
        i_img = i_img.resize((640, 480), Image.BILINEAR)
        lbl = lbl.resize((640, 480), Image.NEAREST)

        v_tensor = self.normalize(self._safe_to_tensor(v_img))
        i_tensor = self.normalize(self._safe_to_tensor(i_img))
        l_tensor = torch.tensor(np.array(lbl)).long()

        return v_tensor, i_tensor, l_tensor, fname


class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def get_miou_and_pa(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)

        pixel_acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        return miou, pixel_acc, iou

    def get_class_pa(self):
        class_pa = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return class_pa


def colorize_mask(mask):
    palette = [
        0, 0, 0,  # Background
        64, 0, 128,  # Car
        64, 64, 0,  # Person
        0, 128, 192,  # Bike
        0, 0, 192,  # Curve
        128, 128, 0,  # Car Stop
        64, 64, 128,  # Guardrail
        192, 128, 128,  # Color Cone
        192, 64, 0  # Bump
    ]
    mask_pil = Image.fromarray(mask.astype(np.uint8)).convert('P')
    mask_pil.putpalette(palette)
    return mask_pil


def save_comparison_vis(pred_mask, gt_mask, save_path):
    pred_vis = colorize_mask(pred_mask)
    gt_vis = colorize_mask(gt_mask)

    # Concat horizontally
    w, h = pred_vis.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(pred_vis.convert('RGB'), (0, 0))
    combined.paste(gt_vis.convert('RGB'), (w, 0))

    combined.save(save_path)


def inference_single(model, v, i):
    # 【修复重点】处理模型返回的元组
    out = model(v, i)
    if isinstance(out, tuple):
        logits = out[0]  # 取第一个返回值 Seg Logits
    else:
        logits = out
    return logits


def test():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found: {CKPT_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Build Model
    print(f"🚀 Loading model from {CKPT_PATH}...")
    SAM_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM_CKPT = "../checkpoints/sam2.1_hiera_large.pt"  # 仅用于初始化结构

    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES)

    # Load Weights
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Weights loaded successfully (Partial/Strict=False).")

    model.to(DEVICE)
    model.eval()

    # 2. Dataset
    test_ds = MSRSTestDataset(TEST_DIRS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    global_evaluator = Evaluator(NUM_CLASSES)

    print(f"📂 Visualizations (GT vs Pred) will be saved to: {OUTPUT_DIR}")
    print(f"🚀 Start testing on {len(test_ds)} images (TTA={USE_TTA})...")

    total_image_miou = 0.0
    total_image_pa = 0.0
    valid_samples = 0

    with torch.no_grad():
        for v, i_img, l, fname in tqdm(test_loader):
            v, i_img = v.to(DEVICE), i_img.to(DEVICE)
            l = l.numpy()[0]
            fname = fname[0]

            # --- Inference ---
            logits = inference_single(model, v, i_img)

            if USE_TTA:
                # Flip TTA
                v_flip = torch.flip(v, dims=[3])
                i_flip = torch.flip(i_img, dims=[3])
                logits_flip = inference_single(model, v_flip, i_flip)
                logits_flip = torch.flip(logits_flip, dims=[3])
                logits = (logits + logits_flip) / 2.0

            logits = F.interpolate(logits, size=(480, 640), mode='bilinear', align_corners=False)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            # 计算指标
            single_eval = Evaluator(NUM_CLASSES)
            single_eval.add_batch(l, pred)
            s_miou, s_pa, _ = single_eval.get_miou_and_pa()

            total_image_miou += s_miou
            total_image_pa += s_pa
            valid_samples += 1
            global_evaluator.add_batch(l, pred)

            # 保存结果
            save_path = os.path.join(OUTPUT_DIR, fname)
            save_comparison_vis(pred, l, save_path)

    # --- 最终打印 ---
    g_miou, g_pa, class_iou = global_evaluator.get_miou_and_pa()
    class_pa = global_evaluator.get_class_pa()

    print("\n" + "=" * 60)
    print(f"📊 Final Test Results (TTA={USE_TTA})")
    print(f"   Global Mean IoU:    {g_miou * 100:.2f}%")
    print(f"   Global Pixel Acc:   {g_pa * 100:.2f}%")
    print("-" * 60)
    print(f"{'Class Name':<15} | {'IoU (%)':<10} | {'PA (%)':<10}")
    print("-" * 60)
    for i, name in enumerate(CLASSES):
        print(f"{name:<15} | {class_iou[i] * 100:.2f}%     | {class_pa[i] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    test()