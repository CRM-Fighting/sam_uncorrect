import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from utils.custom_losses import BinarySUMLoss, StandardSegLoss

# --- 配置 ---
EXP_NAME = "Exp_DynamicFusion_Final"
ENTROPY_ROOT = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/entropy_maps"

TRAIN_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/Segmentation_labels'
}
# ★★★ 修正验证集路径 ★★★
VAL_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/val/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/val/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/val/Segmentation_labels'
}
UNCERTAINTY_DIRS = {
    'train': "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map/train",
    # 验证集不需要
}

SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"
BATCH_SIZE = 1
EPOCHS = 50
NUM_CLASSES = 9


# --- 评估工具 ---
class SegEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def add_batch(self, preds, labels):
        mask = (labels >= 0) & (labels < self.num_classes)
        self.confusion_matrix += np.bincount(
            self.num_classes * labels[mask].astype(int) + preds[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def get_metrics(self):
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        pixel_acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        return {"mIoU": miou, "PA": pixel_acc}


# --- Dataset ---
class MSRSDataset(Dataset):
    def __init__(self, dirs, uncertainty_root=None, entropy_root=None, is_train=True):
        self.vis = dirs['vi']
        self.ir = dirs['ir']
        self.lbl = dirs['label']
        self.uncertainty_root = uncertainty_root
        self.entropy_dirs = {}
        # 验证/测试时不加载熵图
        if entropy_root and is_train:
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                self.entropy_dirs[stage] = os.path.join(entropy_root, 'train', stage)
        self.files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        v = torch.from_numpy(
            np.array(Image.open(os.path.join(self.vis, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0,
                                                                                                               1) / 255.0
        i_img = torch.from_numpy(
            np.array(Image.open(os.path.join(self.ir, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0,
                                                                                                              1) / 255.0
        l = torch.from_numpy(np.array(Image.open(os.path.join(self.lbl, n)).resize((640, 480), Image.NEAREST))).long()

        s = torch.zeros((1, 480, 640))
        if self.uncertainty_root:
            sp = os.path.join(self.uncertainty_root, n.replace('.png', '.npy'))
            if os.path.exists(sp):
                s = torch.from_numpy(np.load(sp)).float()
                if len(s.shape) == 2: s = s.unsqueeze(0)
                if s.shape[-1] != 640: s = F.interpolate(s.unsqueeze(0), (480, 640), mode='bilinear').squeeze(0)

        entropy_maps = []
        if self.entropy_dirs:
            npy_name = n.replace('.png', '.npy')
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                path = os.path.join(self.entropy_dirs[stage], npy_name)
                if os.path.exists(path):
                    em = torch.from_numpy(np.load(path).astype(np.float32))
                    if em.ndim == 2: em = em.unsqueeze(0)
                    entropy_maps.append(em)
                else:
                    entropy_maps.append(torch.zeros(1, 1, 1))

        return v, i_img, l, s, entropy_maps


def train():
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)

    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scaler = GradScaler()
    crit_seg = StandardSegLoss(NUM_CLASSES)
    crit_sam = BinarySUMLoss(theta=0.6)

    # 1. 训练集 (带 Entropy, Uncertainty)
    train_dl = DataLoader(MSRSDataset(TRAIN_DIRS, UNCERTAINTY_DIRS['train'], entropy_root=ENTROPY_ROOT, is_train=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # 2. 验证集 (不带 Entropy, Uncertainty, 读 val 文件夹)
    val_dl = DataLoader(MSRSDataset(VAL_DIRS, uncertainty_root=None, entropy_root=None, is_train=False),
                        batch_size=1, shuffle=False, num_workers=4)

    best_miou = 0.0

    for ep in range(EPOCHS):
        # === 训练 ===
        model.train()
        pbar = tqdm(train_dl, desc=f"Ep {ep + 1} Train")
        for v, i_img, l, s, entropy_maps in pbar:
            v, i_img, l, s = v.cuda(), i_img.cuda(), l.cuda(), s.cuda()
            e_maps_cuda = [em.cuda() for em in entropy_maps] if entropy_maps else None

            with autocast():
                # 传入 GT 供 Aux Loss 和 Fusion Loss 使用
                seg_out, sam_preds, moe_loss, fusion_loss = model(
                    v, i_img, gt_semantic=l, gt_entropy_maps=e_maps_cuda
                )
                l_main = crit_seg(seg_out, l)
                l_aux = (crit_sam(sam_preds['rgb_s4'], l, s) + crit_sam(sam_preds['ir_s4'], l, s)) / 2.0
                loss = l_main + 0.5 * l_aux + 0.5 * fusion_loss + 0.01 * moe_loss

            scaler.scale(loss).backward()
            scaler.step(opt);
            scaler.update();
            opt.zero_grad()
            pbar.set_postfix(
                {'Main': f"{l_main.item():.2f}", 'Aux': f"{l_aux.item():.2f}", 'Fus': f"{fusion_loss.item():.2f}"})

        # === 验证 ===
        model.eval()
        evaluator = SegEvaluator(NUM_CLASSES)
        print(f"⏳ Validating Ep {ep + 1}...")

        with torch.no_grad():
            for v, i_img, l, _, _ in tqdm(val_dl, desc="Val"):
                v, i_img, l = v.cuda(), i_img.cuda(), l.cpu().numpy()
                # 验证时：传入 None, 不计算 Aux/Fusion Loss, Agent 独立工作
                logits, _, _, _ = model(v, i_img, gt_semantic=None, gt_entropy_maps=None)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                evaluator.add_batch(pred, l)

        metrics = evaluator.get_metrics()
        cur_miou = metrics['mIoU']
        print(f"📊 Ep {ep + 1} Val mIoU: {cur_miou * 100:.2f}% | PA: {metrics['PA'] * 100:.2f}%")

        torch.save(model.state_dict(), f"{ckpt_dir}/last_model.pth")
        if cur_miou > best_miou:
            best_miou = cur_miou
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pth")
            print(f"🏆 New Best Saved! mIoU: {best_miou * 100:.2f}%")


if __name__ == "__main__": train()