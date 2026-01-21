import os
import warnings
import sys
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# 过滤烦人的 FutureWarning
warnings.filterwarnings("ignore")

# --- 防止显存碎片化 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.transforms.functional as TF

# 引入高级调度器
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# --- 导入模型 ---
from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from utils.custom_losses import BinarySUMLoss, StandardSegLoss

# --- 配置区域 ---
EXP_NAME = "publication_v2_cross_mixer_final_logs_fixed_1_20"

# 路径配置
ENTROPY_ROOT = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/entropy_maps_add"

TRAIN_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/Segmentation_labels'
}
VAL_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/val/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/val/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/val/Segmentation_labels'
}
UNCERTAINTY_ROOT_TRAIN = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map/train"

SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"

BATCH_SIZE = 2
ACCUM_STEPS = 8
EPOCHS = 100
NUM_CLASSES = 9


# --- 基础工具 ---
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- 评估工具 ---
class SegEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self): self.confusion_matrix.fill(0)

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

        self.entropy_vis_dirs = {}
        self.entropy_ir_dirs = {}
        self.is_train = is_train

        if entropy_root:
            sub = 'train' if is_train else 'val'
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                self.entropy_vis_dirs[stage] = os.path.join(entropy_root, sub, 'vi', stage)
                self.entropy_ir_dirs[stage] = os.path.join(entropy_root, sub, 'ir', stage)

        raw_files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])
        self.files = []
        for f in raw_files:
            if os.path.exists(os.path.join(self.ir, f)) and os.path.exists(os.path.join(self.lbl, f)):
                self.files.append(f)

        print(f"✅ Loaded {len(self.files)} samples (Train={is_train})")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def _load_gray_as_tensor(self, path):
        if not os.path.exists(path): return torch.zeros((1, 15, 20))
        arr = np.load(path)
        t = torch.from_numpy(arr).float()
        if t.ndim == 2: t = t.unsqueeze(0)
        return t

    def _load_uncertainty(self, root, sub_dir, name):
        if not root: return torch.zeros((1, 480, 640))
        path = os.path.join(root, sub_dir, name.replace('.png', '.npy'))
        try:
            arr = np.load(path)
            tensor = torch.from_numpy(arr).float()
            if len(tensor.shape) == 2: tensor = tensor.unsqueeze(0)
            return tensor
        except:
            return torch.zeros((1, 480, 640))

    def robust_augment(self, v_img, i_img, lbl, s_vi, s_ir, ent_sum, ent_vis, ent_ir):
        if torch.rand(1) > 0.5:
            v_img = TF.hflip(v_img)
            i_img = TF.hflip(i_img)
            lbl = TF.hflip(lbl)
            s_vi = TF.hflip(s_vi)
            s_ir = TF.hflip(s_ir)
            ent_sum = [TF.hflip(e) for e in ent_sum]
            ent_vis = [TF.hflip(e) for e in ent_vis]
            ent_ir = [TF.hflip(e) for e in ent_ir]

        if torch.rand(1) > 0.5:
            v_img = TF.vflip(v_img)
            i_img = TF.vflip(i_img)
            lbl = TF.vflip(lbl)
            s_vi = TF.vflip(s_vi)
            s_ir = TF.vflip(s_ir)
            ent_sum = [TF.vflip(e) for e in ent_sum]
            ent_vis = [TF.vflip(e) for e in ent_vis]
            ent_ir = [TF.vflip(e) for e in ent_ir]

        scale = random.uniform(0.75, 1.25)
        target_h, target_w = int(480 * scale), int(640 * scale)

        def _resize_list(l, mode):
            return [TF.resize(x, [target_h, target_w], interpolation=mode) for x in l]

        v_img = TF.resize(v_img, [target_h, target_w], interpolation=Image.BILINEAR)
        i_img = TF.resize(i_img, [target_h, target_w], interpolation=Image.BILINEAR)
        lbl = TF.resize(lbl, [target_h, target_w], interpolation=Image.NEAREST)
        s_vi = TF.resize(s_vi, [target_h, target_w], interpolation=Image.BILINEAR)
        s_ir = TF.resize(s_ir, [target_h, target_w], interpolation=Image.BILINEAR)

        ent_sum = _resize_list(ent_sum, Image.NEAREST)
        ent_vis = _resize_list(ent_vis, Image.NEAREST)
        ent_ir = _resize_list(ent_ir, Image.NEAREST)

        crop_h, crop_w = 480, 640
        padded_h = max(target_h, crop_h)
        padded_w = max(target_w, crop_w)

        if padded_h > target_h or padded_w > target_w:
            pad_fn = transforms.Pad((0, 0, padded_w - target_w, padded_h - target_h))
            v_img = pad_fn(v_img)
            i_img = pad_fn(i_img)
            lbl = pad_fn(lbl)
            s_vi = pad_fn(s_vi)
            s_ir = pad_fn(s_ir)
            ent_sum = [pad_fn(e) for e in ent_sum]
            ent_vis = [pad_fn(e) for e in ent_vis]
            ent_ir = [pad_fn(e) for e in ent_ir]

        i, j, h, w = transforms.RandomCrop.get_params(v_img, output_size=(crop_h, crop_w))

        v_img = TF.crop(v_img, i, j, h, w)
        i_img = TF.crop(i_img, i, j, h, w)
        lbl = TF.crop(lbl, i, j, h, w)
        s_vi = TF.crop(s_vi, i, j, h, w)
        s_ir = TF.crop(s_ir, i, j, h, w)

        def _crop_list(l):
            return [TF.crop(x, i, j, h, w) for x in l]

        ent_sum = _crop_list(ent_sum)
        ent_vis = _crop_list(ent_vis)
        ent_ir = _crop_list(ent_ir)

        return v_img, i_img, lbl, s_vi, s_ir, ent_sum, ent_vis, ent_ir

    def __getitem__(self, i):
        n = self.files[i]
        v_img = Image.open(os.path.join(self.vis, n)).convert('RGB')
        i_img = Image.open(os.path.join(self.ir, n)).convert('RGB')
        lbl = Image.open(os.path.join(self.lbl, n))

        s_vi = self._load_uncertainty(self.uncertainty_root, 'vi', n)
        s_ir = self._load_uncertainty(self.uncertainty_root, 'ir', n)

        list_vis, list_ir, list_sum = [], [], []
        if self.entropy_vis_dirs:
            npy_name = n.replace('.png', '.npy')
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                p_vis = os.path.join(self.entropy_vis_dirs[stage], npy_name)
                p_ir = os.path.join(self.entropy_ir_dirs[stage], npy_name)
                t_vis = self._load_gray_as_tensor(p_vis)
                t_ir = self._load_gray_as_tensor(p_ir)
                t_sum = t_vis + t_ir
                list_vis.append(t_vis)
                list_ir.append(t_ir)
                list_sum.append(t_sum)

        if self.is_train:
            v_img, i_img, lbl, s_vi, s_ir, list_sum, list_vis, list_ir = self.robust_augment(
                v_img, i_img, lbl, s_vi, s_ir, list_sum, list_vis, list_ir
            )
            target_sizes = [[120, 160], [60, 80], [30, 40], [15, 20]]
            for idx, size in enumerate(target_sizes):
                list_sum[idx] = TF.resize(list_sum[idx], size, interpolation=TF.InterpolationMode.NEAREST)
                list_vis[idx] = TF.resize(list_vis[idx], size, interpolation=TF.InterpolationMode.NEAREST)
                list_ir[idx] = TF.resize(list_ir[idx], size, interpolation=TF.InterpolationMode.NEAREST)
        else:
            v_img = v_img.resize((640, 480), Image.BILINEAR)
            i_img = i_img.resize((640, 480), Image.BILINEAR)
            lbl = lbl.resize((640, 480), Image.NEAREST)
            if s_vi.shape[-2:] != (480, 640): s_vi = F.interpolate(s_vi.unsqueeze(0), (480, 640)).squeeze(0)
            if s_ir.shape[-2:] != (480, 640): s_ir = F.interpolate(s_ir.unsqueeze(0), (480, 640)).squeeze(0)

        v_tensor = TF.to_tensor(v_img)
        v = self.normalize(v_tensor)
        i_tensor = TF.to_tensor(i_img)
        i_new = self.normalize(i_tensor)
        l = torch.from_numpy(np.array(lbl)).long()

        return v, i_new, l, s_vi, s_ir, list_sum, list_vis, list_ir


def train():
    setup_seed(42)
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"🚀 Experiment: {EXP_NAME}")

    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    # ★★★ [新增] 打印模型参数量 ★★★
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters:     {total_params / 1e6:.2f} M")
    print(f"   Trainable Parameters: {trainable_params / 1e6:.2f} M")
    print("-" * 30)

    high_lr_params = []
    low_lr_params = []
    high_keywords = ["shared_moe_layers", "fusion_layers", "segformer_head", "sam_proj_s4"]
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(k in name for k in high_keywords):
            high_lr_params.append(param)
        else:
            low_lr_params.append(param)

    opt = optim.AdamW([
        {'params': high_lr_params, 'lr': 0.0002},
        {'params': low_lr_params, 'lr': 0.0001}
    ], weight_decay=1e-4)

    train_dataset = MSRSDataset(TRAIN_DIRS, UNCERTAINTY_ROOT_TRAIN, ENTROPY_ROOT, is_train=True)
    steps_per_epoch = len(DataLoader(train_dataset, batch_size=BATCH_SIZE)) // ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(total_steps * 0.05)

    scheduler = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    scaler = GradScaler()
    crit_seg = StandardSegLoss(NUM_CLASSES)
    crit_sam = BinarySUMLoss(theta=0.6)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,
                          worker_init_fn=worker_init_fn)
    val_dl = DataLoader(MSRSDataset(VAL_DIRS, None, ENTROPY_ROOT, is_train=False), batch_size=1, shuffle=False,
                        num_workers=4, worker_init_fn=worker_init_fn)

    evaluator = SegEvaluator(NUM_CLASSES)
    train_evaluator = SegEvaluator(NUM_CLASSES)
    best_miou = 0.0

    for ep in range(EPOCHS):
        model.train()
        train_evaluator.reset()
        print(f"\n=== Epoch {ep + 1}/{EPOCHS} | LR: {opt.param_groups[0]['lr']:.2e} ===")
        pbar = tqdm(train_dl, desc="Train")
        metrics = {'Seg': 0.0, 'Aux': 0.0, 'Fus': 0.0, 'Moe': 0.0}

        for step, (v, i_img, l, s_vi, s_ir, l_sum, l_vis, l_ir) in enumerate(pbar):
            v, i_img, l = v.cuda(), i_img.cuda(), l.cuda()
            s_vi, s_ir = s_vi.cuda(), s_ir.cuda()

            e_sum = [e.cuda() for e in l_sum]
            e_vis = [e.cuda() for e in l_vis]
            e_ir = [e.cuda() for e in l_ir]

            with autocast():
                seg_out, sam_preds, moe_loss, fusion_loss = model(
                    vis=v, ir=i_img, gt_semantic=l,
                    gt_entropy_maps=e_sum,
                    gt_entropy_vis=e_vis,
                    gt_entropy_ir=e_ir
                )

                l_main = crit_seg(seg_out, l)
                l_aux = (crit_sam(sam_preds['rgb_s4'], l, s_vi) + crit_sam(sam_preds['ir_s4'], l, s_ir)) / 2.0
                loss = l_main + 0.5 * l_aux + 0.5 * fusion_loss + 0.02 * moe_loss
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            metrics['Seg'] += l_main.item()
            metrics['Aux'] += l_aux.item()
            metrics['Fus'] += fusion_loss.item()
            metrics['Moe'] += moe_loss.item()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()

            with torch.no_grad():
                train_evaluator.add_batch(torch.argmax(seg_out, 1).cpu().numpy(), l.cpu().numpy())

            pbar.set_postfix(
                {'Seg': f"{l_main.item():.3f}", 'Aux': f"{l_aux.item():.3f}", 'Fus': f"{fusion_loss.item():.3f}",
                 'Moe': f"{moe_loss.item():.3f}"})

        model.eval()
        evaluator.reset()
        val_loss_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for v, i_img, l, _, _, l_sum, l_vis, l_ir in tqdm(val_dl, desc="Val"):
                v, i_img, l = v.cuda(), i_img.cuda(), l.cuda()
                e_sum = [e.cuda() for e in l_sum]
                e_vis = [e.cuda() for e in l_vis]
                e_ir = [e.cuda() for e in l_ir]

                seg_out, _, _, _ = model(
                    vis=v, ir=i_img, gt_semantic=l,
                    gt_entropy_maps=e_sum,
                    gt_entropy_vis=e_vis,
                    gt_entropy_ir=e_ir
                )
                loss_val = crit_seg(seg_out, l)
                val_loss_total += loss_val.item()
                val_steps += 1
                evaluator.add_batch(torch.argmax(seg_out, 1).cpu().numpy(), l.cpu().numpy())

        avg_val = val_loss_total / max(val_steps, 1)
        res = evaluator.get_metrics()
        train_res = train_evaluator.get_metrics()

        print(f"📊 Summary Ep {ep + 1}:")
        print(
            f"   Train Loss -> Seg: {metrics['Seg'] / len(train_dl):.4f} | Aux: {metrics['Aux'] / len(train_dl):.4f} | Fus: {metrics['Fus'] / len(train_dl):.4f} | Moe: {metrics['Moe'] / len(train_dl):.4f}")
        print(f"   Train Metric -> mIoU: {train_res['mIoU'] * 100:.2f}% | PA: {train_res['PA'] * 100:.2f}%")
        print(
            f"   Val Metric   -> mIoU: {res['mIoU'] * 100:.2f}% | PA: {res['PA'] * 100:.2f}% | Val Loss: {avg_val:.4f}")

        if res['mIoU'] > best_miou:
            best_miou = res['mIoU']
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pth")
            print("🏆 Saved Best!")


if __name__ == "__main__":
    train()