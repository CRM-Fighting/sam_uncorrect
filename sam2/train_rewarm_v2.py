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
torch.backends.cuda.enable_flash_sdp(False)
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
from utils.custom_losses import BinarySUMLoss  # StandardSegLoss 被我们替换了

# --- 配置区域 ---
EXP_NAME = "1_27日23点权重"  # 更新实验名

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

# 保持 Batch=1 配合 Accumulation Steps 防止 OOM
BATCH_SIZE = 1
ACCUM_STEPS = 16
EPOCHS = 100
NUM_CLASSES = 9


# --- [改进] 自定义带 Label Smoothing 的 Loss ---
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=255)

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


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
        # ImageNet 归一化
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    # 手动转 Tensor 助手函数 (保留你的修复)
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

    def _load_gray_as_tensor(self, path):
        if not os.path.exists(path): return torch.zeros((1, 480, 640))
        arr = np.load(path)
        t = torch.tensor(arr).float()  # 保留修复
        if t.ndim == 2: t = t.unsqueeze(0)
        if t.shape[-2:] != (480, 640):
            t = F.interpolate(t.unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False).squeeze(0)
        return t

    def _load_uncertainty(self, root, sub_dir, name):
        if not root: return torch.zeros((1, 480, 640))
        path = os.path.join(root, sub_dir, name.replace('.png', '.npy'))
        try:
            arr = np.load(path)
            tensor = torch.tensor(arr).float()  # 保留修复
            if len(tensor.shape) == 2: tensor = tensor.unsqueeze(0)
            return tensor
        except:
            return torch.zeros((1, 480, 640))

    def robust_augment(self, v_img, i_img, lbl, s_vi, s_ir, ent_sum, ent_vis, ent_ir):
        # 1. 随机水平翻转
        if random.random() > 0.5:
            v_img = TF.hflip(v_img)
            i_img = TF.hflip(i_img)
            lbl = TF.hflip(lbl)
            s_vi = TF.hflip(s_vi)
            s_ir = TF.hflip(s_ir)
            ent_sum = [TF.hflip(e) for e in ent_sum]
            ent_vis = [TF.hflip(e) for e in ent_vis]
            ent_ir = [TF.hflip(e) for e in ent_ir]

        # 2. 随机缩放裁剪 (0.75 ~ 1.0)
        do_crop = random.random() > 0.5
        target_size = (480, 640)

        if do_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                v_img, scale=(0.75, 1.0), ratio=(0.75, 1.33)
            )
        else:
            w, h = v_img.size
            i, j = 0, 0

        v_img = TF.resized_crop(v_img, i, j, h, w, target_size, Image.BILINEAR)
        i_img = TF.resized_crop(i_img, i, j, h, w, target_size, Image.BILINEAR)
        lbl = TF.resized_crop(lbl, i, j, h, w, target_size, Image.NEAREST)
        s_vi = TF.resized_crop(s_vi, i, j, h, w, target_size, Image.BILINEAR)
        s_ir = TF.resized_crop(s_ir, i, j, h, w, target_size, Image.BILINEAR)

        def _apply_list(l):
            return [TF.resized_crop(x, i, j, h, w, target_size, Image.BILINEAR) for x in l]

        ent_sum = _apply_list(ent_sum)
        ent_vis = _apply_list(ent_vis)
        ent_ir = _apply_list(ent_ir)

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
                list_sum[idx] = TF.resize(list_sum[idx], size, interpolation=TF.InterpolationMode.BILINEAR)
                list_vis[idx] = TF.resize(list_vis[idx], size, interpolation=TF.InterpolationMode.BILINEAR)
                list_ir[idx] = TF.resize(list_ir[idx], size, interpolation=TF.InterpolationMode.BILINEAR)
        else:
            v_img = v_img.resize((640, 480), Image.BILINEAR)
            i_img = i_img.resize((640, 480), Image.BILINEAR)
            lbl = lbl.resize((640, 480), Image.NEAREST)
            if s_vi.shape[-2:] != (480, 640): s_vi = F.interpolate(s_vi.unsqueeze(0), (480, 640),
                                                                   mode='bilinear').squeeze(0)
            if s_ir.shape[-2:] != (480, 640): s_ir = F.interpolate(s_ir.unsqueeze(0), (480, 640),
                                                                   mode='bilinear').squeeze(0)
            target_sizes = [[120, 160], [60, 80], [30, 40], [15, 20]]
            for idx, size in enumerate(target_sizes):
                list_sum[idx] = TF.resize(list_sum[idx], size, interpolation=TF.InterpolationMode.BILINEAR)
                list_vis[idx] = TF.resize(list_vis[idx], size, interpolation=TF.InterpolationMode.BILINEAR)
                list_ir[idx] = TF.resize(list_ir[idx], size, interpolation=TF.InterpolationMode.BILINEAR)

        v_tensor = self._safe_to_tensor(v_img)
        v = self.normalize(v_tensor)
        i_tensor = self._safe_to_tensor(i_img)
        i_new = self.normalize(i_tensor)
        l = torch.tensor(np.array(lbl)).long()

        return v, i_new, l, s_vi, s_ir, list_sum, list_vis, list_ir


def train():
    setup_seed(42)
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"🚀 Experiment: {EXP_NAME}")

    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters:     {total_params / 1e6:.2f} M")
    print(f"   Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # [核心改进] 显式参数分组，确保新模块 (Refinement, Injector) 都有高学习率
    high_lr_params = []
    low_lr_params = []
    # 增加 context_refine 和 injector 到高学习率组
    high_keywords = ["shared_moe_layers", "fusion_layers", "segformer_head", "sam_proj_s4",
                     "context_refine", "injector", "alpha_net"]

    print("\n🔍 Parameter Grouping Check:")
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(k in name for k in high_keywords):
            high_lr_params.append(param)
            # 抽样打印，确认分组正确
            if "context_refine" in name and random.random() < 0.1:
                print(f"   [High LR] {name}")
        else:
            low_lr_params.append(param)
    print("-" * 30)

    # [核心改进] 权重衰减 (Weight Decay) 提高到 0.01，防止过拟合
    opt = optim.AdamW([
        {'params': high_lr_params, 'lr': 0.0002},
        {'params': low_lr_params, 'lr': 0.0001}
    ], weight_decay=0.01)  # <--- 修改这里：从 1e-4 改为 0.01

    train_dataset = MSRSDataset(TRAIN_DIRS, UNCERTAINTY_ROOT_TRAIN, ENTROPY_ROOT, is_train=True)
    steps_per_epoch = len(DataLoader(train_dataset, batch_size=BATCH_SIZE)) // ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(total_steps * 0.05)

    scheduler = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
    ], milestones=[warmup_steps])

    scaler = GradScaler()

    # [核心改进] 使用带 Label Smoothing 的 Loss
    crit_seg = SoftCrossEntropyLoss(NUM_CLASSES, smoothing=0.1)
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
                opt.zero_grad(set_to_none=True)
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