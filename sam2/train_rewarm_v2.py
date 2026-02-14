import os
import warnings
import sys
import random
import shutil
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import _LRScheduler

from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from utils.custom_losses import BinarySUMLoss, StandardSegLoss

# ==============================================================================
# 配置区域
# ==============================================================================
EXP_NAME = "2月6日17点权重"
EXP_ROOT = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/checkpoints/"

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

BATCH_SIZE = 1
ACCUM_STEPS = 16
EPOCHS = 200
NUM_CLASSES = 9
CLASS_WEIGHTS = [1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 2.5, 2.5, 2.0]


def setup_experiment():
    base_dir = os.path.join(EXP_ROOT, EXP_NAME)
    dirs = {
        "root": base_dir,
        "ckpt_best": os.path.join(base_dir, "checkpoints", "best"),
        "ckpt_reg": os.path.join(base_dir, "checkpoints", "regular"),
        "logs": os.path.join(base_dir, "logs"),
        "tb": os.path.join(base_dir, "logs", "tensorboard"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    if os.path.exists(SAM_CFG):
        shutil.copy(SAM_CFG, os.path.join(base_dir, "config.yaml"))
    log_file = os.path.join(dirs["logs"], "train.log")
    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info(f"🚀 Experiment Setup Completed: {base_dir}")
    return dirs


class PolyLRScheduler:
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, warmup_iters=1000):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_iters = warmup_iters

    def step(self, cur_iter):
        if cur_iter < self.warmup_iters:
            lr = self.base_lr * (cur_iter / self.warmup_iters)
        else:
            lr = self.base_lr * (
                        (1 - (cur_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power)
        lr = max(lr, 1e-7)
        self.optimizer.param_groups[0]['lr'] = lr * 2.0
        self.optimizer.param_groups[1]['lr'] = lr


# 【重点修改】强力数据增强类
class RobustAugmentation:
    def __init__(self, crop_size=(480, 640)):
        self.crop_h, self.crop_w = crop_size
        # 颜色抖动：亮度、对比度、饱和度、色相
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        # 高斯模糊
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    def __call__(self, v_img, i_img, lbl, s_vi, s_ir, ent_sum, ent_vis, ent_ir):
        # 1. 颜色增强 (仅针对可见光) - 50% 概率
        if random.random() > 0.5:
            v_img = self.color_jitter(v_img)

        # 2. 红外增强 (仅调整亮度和对比度，不调颜色) - 50% 概率
        if random.random() > 0.5:
            # 手动调整，避免色相变化
            i_img = TF.adjust_brightness(i_img, random.uniform(0.7, 1.3))
            i_img = TF.adjust_contrast(i_img, random.uniform(0.7, 1.3))

        # 3. 高斯模糊 (模拟运动模糊/对焦不准) - 20% 概率
        if random.random() < 0.2:
            v_img = self.gaussian_blur(v_img)
            i_img = self.gaussian_blur(i_img)

        # 4. 几何增强 (翻转)
        if random.random() > 0.5:
            v_img = TF.hflip(v_img)
            i_img = TF.hflip(i_img)
            lbl = TF.hflip(lbl)
            s_vi = TF.hflip(s_vi)
            s_ir = TF.hflip(s_ir)
            ent_sum = [TF.hflip(e) for e in ent_sum]
            ent_vis = [TF.hflip(e) for e in ent_vis]
            ent_ir = [TF.hflip(e) for e in ent_ir]

        # 5. 随机缩放 + 裁剪
        scale = random.uniform(0.75, 1.5)
        w, h = v_img.size
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = max(new_w, self.crop_w)
        new_h = max(new_h, self.crop_h)
        v_img = TF.resize(v_img, (new_h, new_w), interpolation=Image.BILINEAR)
        i_img = TF.resize(i_img, (new_h, new_w), interpolation=Image.BILINEAR)
        s_vi = TF.resize(s_vi.unsqueeze(0), (new_h, new_w), interpolation=Image.BILINEAR).squeeze(0)
        s_ir = TF.resize(s_ir.unsqueeze(0), (new_h, new_w), interpolation=Image.BILINEAR).squeeze(0)
        lbl = TF.resize(lbl, (new_h, new_w), interpolation=Image.NEAREST)

        def resize_list(l, mode=Image.BILINEAR):
            return [TF.resize(x.unsqueeze(0) if x.ndim == 2 else x, (new_h, new_w), interpolation=mode) for x in l]

        ent_sum = resize_list(ent_sum)
        ent_vis = resize_list(ent_vis)
        ent_ir = resize_list(ent_ir)

        i, j, h, w = transforms.RandomCrop.get_params(v_img, output_size=(self.crop_h, self.crop_w))
        v_img = TF.crop(v_img, i, j, h, w)
        i_img = TF.crop(i_img, i, j, h, w)
        lbl = TF.crop(lbl, i, j, h, w)
        s_vi = TF.crop(s_vi, i, j, h, w)
        s_ir = TF.crop(s_ir, i, j, h, w)
        ent_sum = [TF.crop(x, i, j, h, w) for x in ent_sum]
        ent_vis = [TF.crop(x, i, j, h, w) for x in ent_vis]
        ent_ir = [TF.crop(x, i, j, h, w) for x in ent_ir]

        return v_img, i_img, lbl, s_vi, s_ir, ent_sum, ent_vis, ent_ir


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
        return {"mIoU": miou, "PA": pixel_acc}, iou


class MSRSDataset(Dataset):
    def __init__(self, dirs, uncertainty_root=None, entropy_root=None, is_train=True):
        self.vis = dirs['vi']
        self.ir = dirs['ir']
        self.lbl = dirs['label']
        self.uncertainty_root = uncertainty_root
        self.entropy_vis_dirs = {}
        self.entropy_ir_dirs = {}
        self.is_train = is_train
        self.augmentor = RobustAugmentation(crop_size=(480, 640))
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
        logging.info(f"✅ Loaded {len(self.files)} samples (Train={is_train})")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

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
        t = torch.tensor(arr).float()
        if t.ndim == 2: t = t.unsqueeze(0)
        if t.shape[-2:] != (480, 640): t = F.interpolate(t.unsqueeze(0), size=(480, 640), mode='bilinear',
                                                         align_corners=False).squeeze(0)
        return t

    def _load_uncertainty(self, root, sub_dir, name):
        if not root: return torch.zeros((1, 480, 640))
        path = os.path.join(root, sub_dir, name.replace('.png', '.npy'))
        try:
            arr = np.load(path)
            tensor = torch.tensor(arr).float()
            if len(tensor.shape) == 2: tensor = tensor.unsqueeze(0)
            return tensor
        except:
            return torch.zeros((1, 480, 640))

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
            v_img, i_img, lbl, s_vi, s_ir, list_sum, list_vis, list_ir = self.augmentor(v_img, i_img, lbl, s_vi, s_ir,
                                                                                        list_sum, list_vis, list_ir)
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
    exp_dirs = setup_experiment()

    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    high_lr_params = []
    low_lr_params = []
    high_keywords = ["shared_moe_layers", "fusion_layers", "segformer_head", "sam_proj_s4",
                     "context_refine", "injector", "alpha_net", "aux_head", "detail_proj", "rect_ir", "rect_vis"]

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(k in name for k in high_keywords):
            high_lr_params.append(param)
        else:
            low_lr_params.append(param)

    base_lr_low = 0.00005
    opt = optim.AdamW([
        {'params': high_lr_params, 'lr': base_lr_low * 2.0},
        {'params': low_lr_params, 'lr': base_lr_low}
    ], weight_decay=0.02)  # 建议这里也可以稍微改大一点到 0.05，不过先保持不动也行

    train_dataset = MSRSDataset(TRAIN_DIRS, UNCERTAINTY_ROOT_TRAIN, ENTROPY_ROOT, is_train=True)
    steps_per_epoch = len(DataLoader(train_dataset, batch_size=BATCH_SIZE)) // ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    poly_scheduler = PolyLRScheduler(opt, base_lr=base_lr_low, max_iters=total_steps, power=0.9, warmup_iters=1500)
    scaler = GradScaler()

    crit_seg = StandardSegLoss(NUM_CLASSES, class_weights=CLASS_WEIGHTS)
    crit_sam = BinarySUMLoss(theta=0.6)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,
                          worker_init_fn=worker_init_fn)
    val_dl = DataLoader(MSRSDataset(VAL_DIRS, None, ENTROPY_ROOT, is_train=False), batch_size=1, shuffle=False,
                        num_workers=4, worker_init_fn=worker_init_fn)

    evaluator = SegEvaluator(NUM_CLASSES)
    train_evaluator = SegEvaluator(NUM_CLASSES)

    best_guardrail_iou = 0.0
    best_ckpts_history = []
    global_step = 0

    for ep in range(EPOCHS):
        model.train()
        train_evaluator.reset()
        current_lr = opt.param_groups[1]['lr']
        logging.info(f"\n=== Epoch {ep + 1}/{EPOCHS} | LR: {current_lr:.2e} ===")

        pbar = tqdm(train_dl, desc="Train")
        metrics = {'SegTotal': 0.0, 'CE': 0.0, 'Dice': 0.0, 'Edge': 0.0,
                   'Aux': 0.0, 'Fus': 0.0, 'Moe': 0.0, 'FusAux': 0.0}

        for step, (v, i_img, l, s_vi, s_ir, l_sum, l_vis, l_ir) in enumerate(pbar):
            v, i_img, l = v.cuda(), i_img.cuda(), l.cuda()
            s_vi, s_ir = s_vi.cuda(), s_ir.cuda()
            e_sum = [e.cuda() for e in l_sum]
            e_vis = [e.cuda() for e in l_vis]
            e_ir = [e.cuda() for e in l_ir]

            if random.random() < 0.2:
                train_e_sum = None;
                train_e_vis = None;
                train_e_ir = None
            else:
                train_e_sum = e_sum;
                train_e_vis = e_vis;
                train_e_ir = e_ir

            with autocast():
                # 接收 pred_edge
                seg_out, sam_preds, moe_loss, fusion_loss, aux_logits, pred_edge = model(
                    vis=v, ir=i_img, gt_semantic=l,
                    gt_entropy_maps=train_e_sum,
                    gt_entropy_vis=train_e_vis,
                    gt_entropy_ir=train_e_ir
                )

                l_main, l_stats = crit_seg(seg_out, l, pred_edge=pred_edge)

                l_aux = (crit_sam(sam_preds['rgb_s4'], l, s_vi) + crit_sam(sam_preds['ir_s4'], l, s_ir)) / 2.0

                if aux_logits is not None:
                    aux_logits_up = F.interpolate(aux_logits, size=l.shape[-2:], mode='bilinear', align_corners=False)
                    loss_fusion_aux, _ = crit_seg(aux_logits_up, l)
                else:
                    loss_fusion_aux = torch.tensor(0.0).cuda()

                # 【重点修改】降低辅助Loss权重，突出主任务
                # 0.5 -> 0.2
                loss = l_main + 0.2 * l_aux + 0.2 * fusion_loss + 0.02 * moe_loss + 0.2 * loss_fusion_aux
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            metrics['SegTotal'] += l_main.item()
            metrics['CE'] += l_stats['CE']
            metrics['Dice'] += l_stats['Dice']
            metrics['Edge'] += l_stats['Edge']
            metrics['Aux'] += l_aux.item()
            metrics['Fus'] += fusion_loss.item()
            metrics['Moe'] += moe_loss.item()
            metrics['FusAux'] += loss_fusion_aux.item()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                poly_scheduler.step(global_step)

            with torch.no_grad():
                train_evaluator.add_batch(torch.argmax(seg_out, 1).cpu().numpy(), l.cpu().numpy())

            pbar.set_postfix({
                'CE': f"{l_stats['CE']:.3f}",
                'Dice': f"{l_stats['Dice']:.3f}",
                'Edge': f"{l_stats['Edge']:.3f}"
            })

        # --- Validation ---
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

                seg_out, _, _, _, _, _ = model(
                    vis=v, ir=i_img, gt_semantic=l,
                    gt_entropy_maps=e_sum, gt_entropy_vis=e_vis, gt_entropy_ir=e_ir
                )
                loss_val, _ = crit_seg(seg_out, l)
                val_loss_total += loss_val.item()
                val_steps += 1
                evaluator.add_batch(torch.argmax(seg_out, 1).cpu().numpy(), l.cpu().numpy())

        avg_val = val_loss_total / max(val_steps, 1)
        res, class_iou = evaluator.get_metrics()
        train_res, _ = train_evaluator.get_metrics()
        current_guardrail_iou = class_iou[6]

        n_batches = len(train_dl)
        logging.info(f"📊 Losses Ep {ep + 1}: "
                     f"SegTotal={metrics['SegTotal'] / n_batches:.3f} | "
                     f"CE={metrics['CE'] / n_batches:.3f} | "
                     f"Dice={metrics['Dice'] / n_batches:.3f} | "
                     f"Edge={metrics['Edge'] / n_batches:.3f} | "
                     f"Aux={metrics['Aux'] / n_batches:.3f} | "
                     f"Fus={metrics['Fus'] / n_batches:.3f}")

        logging.info(f"   Train mIoU: {train_res['mIoU'] * 100:.2f}% | Val mIoU: {res['mIoU'] * 100:.2f}%")
        logging.info(f"   🛡️ Guardrail IoU: {current_guardrail_iou * 100:.2f}%")

        # 保存逻辑
        ckpt_root_dir = exp_dirs["root"]
        ckpt_best_dir = exp_dirs["ckpt_best"]
        ckpt_reg_dir = exp_dirs["ckpt_reg"]
        latest_path = os.path.join(exp_dirs["root"], "checkpoints", "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if (ep + 1) % 10 == 0:
            reg_path = os.path.join(ckpt_reg_dir, f"checkpoint_epoch_{ep + 1}.pth")
            torch.save(model.state_dict(), reg_path)

        current_miou = res['mIoU']
        current_ckpt_path = os.path.join(ckpt_best_dir, f"best_model_epoch_{ep + 1}_miou_{current_miou:.4f}.pth")

        if len(best_ckpts_history) < 5 or current_miou > best_ckpts_history[-1][0]:
            torch.save(model.state_dict(), current_ckpt_path)
            best_ckpts_history.append((current_miou, ep + 1, current_ckpt_path))
            best_ckpts_history.sort(key=lambda x: x[0], reverse=True)
            logging.info(f"🏆 Saved Top-5 Best Model: {current_ckpt_path}")
            if len(best_ckpts_history) > 5:
                worst_ckpt = best_ckpts_history.pop()
                worst_path = worst_ckpt[2]
                if os.path.exists(worst_path): os.remove(worst_path)

        if current_guardrail_iou > best_guardrail_iou:
            best_guardrail_iou = current_guardrail_iou
            gr_path = os.path.join(exp_dirs["root"], "checkpoints", "best_guardrail.pth")
            torch.save(model.state_dict(), gr_path)
            logging.info(f"🛡️ Saved Best Guardrail Model: {current_guardrail_iou * 100:.2f}%")


if __name__ == "__main__":
    train()