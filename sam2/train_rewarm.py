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

# ★★★ 新增：引入高级调度器 ★★★
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# --- 导入模型 (保持您的路径不变) ---
from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from utils.custom_losses import BinarySUMLoss, StandardSegLoss

# --- 配置区域 ---
EXP_NAME = "visual_correct_opt_v1_1_15"  # 建议改个名区分实验

# 路径配置 (请确保这些路径与您的环境一致)
ENTROPY_ROOT = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/entropy_maps"

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
EPOCHS = 50  # 保持 50 轮，配合 Warmup 和高 LR 足够收敛
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

    def reset(self):
        self.confusion_matrix.fill(0)

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


# --- Dataset (包含之前的 Resize 修复) ---
class MSRSDataset(Dataset):
    def __init__(self, dirs, uncertainty_root=None, entropy_root=None, is_train=True):
        self.vis = dirs['vi']
        self.ir = dirs['ir']
        self.lbl = dirs['label']
        self.uncertainty_root = uncertainty_root
        self.entropy_dirs = {}
        self.is_train = is_train

        if entropy_root and is_train:
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                self.entropy_dirs[stage] = os.path.join(entropy_root, 'train', stage)

        raw_files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])

        self.files = []
        # 初始化时做一次完整性检查
        print(f"Checking data integrity for {'Train' if is_train else 'Val'}...")
        for f in tqdm(raw_files, desc="Filtering"):
            # 基础检查
            if not os.path.exists(os.path.join(self.ir, f)): continue
            if not os.path.exists(os.path.join(self.lbl, f)): continue

            npy_name = f.replace('.png', '.npy')
            # 训练集额外检查
            if is_train:
                if self.uncertainty_root:
                    path_vi = os.path.join(self.uncertainty_root, 'vi', npy_name)
                    path_ir = os.path.join(self.uncertainty_root, 'ir', npy_name)
                    if not (os.path.exists(path_vi) and os.path.exists(path_ir)): continue

                if self.entropy_dirs:
                    missing_stage = False
                    for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                        if not os.path.exists(os.path.join(self.entropy_dirs[stage], npy_name)):
                            missing_stage = True
                            break
                    if missing_stage: continue

            self.files.append(f)

        print(f"✅ Matched Samples: {len(self.files)} / {len(raw_files)}")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

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

    def __getitem__(self, i):
        n = self.files[i]
        v_img = Image.open(os.path.join(self.vis, n)).convert('RGB')
        i_img_pil = Image.open(os.path.join(self.ir, n)).convert('RGB')
        lbl_pil = Image.open(os.path.join(self.lbl, n))

        # 读取不确定性
        s_vi = self._load_uncertainty(self.uncertainty_root, 'vi', n)
        s_ir = self._load_uncertainty(self.uncertainty_root, 'ir', n)

        entropy_maps_list = []
        if self.is_train and self.entropy_dirs:
            npy_name = n.replace('.png', '.npy')
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                target_path = os.path.join(self.entropy_dirs[stage], npy_name)
                try:
                    if not os.path.exists(target_path):
                        raise FileNotFoundError(f"File does not exist: {target_path}")

                    em_numpy = np.load(target_path).astype(np.float32)
                    em = torch.from_numpy(em_numpy)
                    if em.ndim == 2: em = em.unsqueeze(0)

                    # [修复] 强制 Resize 到与图片一致 (480, 640)，防止尺寸不匹配
                    # 注意：如果您的 .npy 已经是 Hiera 特征层对应的小尺寸，请注释掉这行 Resize！
                    # 根据您之前的说法“熵图都是已经生成好的对应尺寸”，这里我【注释掉】Resize。
                    # em = F.interpolate(em.unsqueeze(0), size=(480, 640), mode='nearest').squeeze(0)

                    entropy_maps_list.append(em)

                except Exception as e:
                    print(f"\n[💥 Error Loading Entropy Map] {n} - {stage}")
                    raise e

        if self.is_train and torch.rand(1).item() > 0.5:
            v_img = TF.hflip(v_img)
            i_img_pil = TF.hflip(i_img_pil)
            lbl_pil = TF.hflip(lbl_pil)
            s_vi = TF.hflip(s_vi)
            s_ir = TF.hflip(s_ir)
            entropy_maps_list = [TF.hflip(em) for em in entropy_maps_list]

        v_img = v_img.resize((640, 480), Image.BILINEAR)
        i_img_pil = i_img_pil.resize((640, 480), Image.BILINEAR)
        lbl_pil = lbl_pil.resize((640, 480), Image.NEAREST)

        if s_vi.shape[-1] != 640: s_vi = F.interpolate(s_vi.unsqueeze(0), (480, 640), mode='bilinear').squeeze(0)
        if s_ir.shape[-1] != 640: s_ir = F.interpolate(s_ir.unsqueeze(0), (480, 640), mode='bilinear').squeeze(0)

        v_tensor = torch.from_numpy(np.array(v_img)).float().permute(2, 0, 1) / 255.0
        v = self.normalize(v_tensor)
        i_tensor = torch.from_numpy(np.array(i_img_pil)).float().permute(2, 0, 1) / 255.0
        i_img = self.normalize(i_tensor)
        l = torch.from_numpy(np.array(lbl_pil)).long()

        return v, i_img, l, s_vi, s_ir, entropy_maps_list


def train():
    setup_seed(42)
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"🚀 Experiment: {EXP_NAME}")

    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    # --- 【优化 1：修正参数分组】 ---
    # 逻辑：新增的层 (MoE, Fusion, Head) 需要大学习率，冻结层/微调层用小学习率

    high_lr_params = []  # 存放 MoE, Fusion, Head, Projection
    low_lr_params = []  # 存放 Backbone 中可能解冻的层

    # 定义高学习率模块的关键词
    high_lr_keywords = [
        "shared_moe_layers",  # MoE (包含 Router 和 Expert)
        "fusion_layers",  # Fusion (包含 Agent, Mixer)
        "segformer_head",  # 主分类头
        "sam_proj_s4"  # 辅助头投影
    ]

    print("\n🔧 Parameter Grouping:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 判断是否属于高学习率组
        is_high = any(k in name for k in high_lr_keywords)

        if is_high:
            high_lr_params.append(param)
        else:
            low_lr_params.append(param)
            print(f"  [Low LR / Finetune] {name}")

    # 优化器配置：High LR = 5e-4, Low LR = 1e-4
    opt = optim.AdamW([
        {'params': high_lr_params, 'lr': 0.0005},
        {'params': low_lr_params, 'lr': 0.0001}
    ], weight_decay=1e-4)

    # --- 【优化 2：引入 Warmup + Step-wise Scheduler】 ---
    # 计算总更新步数 (Total Steps)
    # 必须基于 DataLoader 长度和 ACCUM_STEPS 计算
    train_dataset = MSRSDataset(TRAIN_DIRS, UNCERTAINTY_ROOT_TRAIN, ENTROPY_ROOT, is_train=True)
    steps_per_epoch = len(DataLoader(train_dataset, batch_size=BATCH_SIZE)) // ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(total_steps * 0.05)  # 5% 的步数用于热身

    print(f"📅 Schedule: Total Steps={total_steps}, Warmup Steps={warmup_steps}")

    # 定义组合调度器 (Linear Warmup -> Cosine Annealing)
    # 1. 前 5% 步：从 1% LR 线性增加到 100% LR
    scheduler_warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    # 2. 剩余步：余弦退火到微小值
    scheduler_cosine = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
    # 3. 串联
    scheduler = SequentialLR(opt, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

    crit_seg = StandardSegLoss(NUM_CLASSES)
    crit_sam = BinarySUMLoss(theta=0.6)
    scaler = GradScaler()

    train_dl = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, worker_init_fn=worker_init_fn
    )
    val_dl = DataLoader(
        MSRSDataset(VAL_DIRS, None, None, is_train=False),
        batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn
    )

    evaluator = SegEvaluator(NUM_CLASSES)
    best_miou = 0.0

    for ep in range(EPOCHS):
        model.train()
        curr_lr = opt.param_groups[0]['lr']
        print(f"\n=== Epoch {ep + 1}/{EPOCHS} | LR(High): {curr_lr:.2e} ===")

        pbar = tqdm(train_dl, desc="Train")
        opt.zero_grad()

        metrics = {'Seg': 0.0, 'Aux': 0.0, 'Fus': 0.0, 'Moe': 0.0}

        for step, (v, i_img, l, s_vi, s_ir, entropy_maps) in enumerate(pbar):
            v, i_img, l = v.cuda(), i_img.cuda(), l.cuda()
            s_vi, s_ir = s_vi.cuda(), s_ir.cuda()
            e_maps_cuda = [em.cuda() for em in entropy_maps]

            with autocast():
                # 显式传入 gt_entropy_maps
                seg_out, sam_preds, moe_loss, fusion_loss = model(
                    vis=v,
                    ir=i_img,
                    gt_semantic=l,
                    gt_entropy_maps=e_maps_cuda
                )

                l_main = crit_seg(seg_out, l)
                l_rgb = crit_sam(sam_preds['rgb_s4'], l, s_vi)
                l_ir = crit_sam(sam_preds['ir_s4'], l, s_ir)
                l_aux = (l_rgb + l_ir) / 2.0

                # 调整权重：稍微降低 MoE Loss 权重 (0.05 -> 0.02)
                loss = l_main + 0.5 * l_aux + 0.5 * fusion_loss + 0.02 * moe_loss
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            metrics['Seg'] += l_main.item()
            metrics['Aux'] += l_aux.item()
            metrics['Fus'] += fusion_loss.item()
            metrics['Moe'] += moe_loss.item()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                # ★★★ 关键修改：Step-wise Scheduler 更新 ★★★
                scheduler.step()

            pbar.set_postfix({
                'Seg': f"{l_main.item():.3f}",
                'Aux': f"{l_aux.item():.3f}",
                'Fus': f"{fusion_loss.item():.3f}",
                'Moe': f"{moe_loss.item():.3f}"
            })

        # 移除原有的 Epoch 级 scheduler.step()
        torch.cuda.empty_cache()

        # === 验证 ===
        model.eval()
        evaluator.reset()
        val_loss_total = 0.0
        val_steps = 0

        with torch.no_grad():
            for v, i_img, l, _, _, _ in tqdm(val_dl, desc="Val"):
                v, i_img, l = v.cuda(), i_img.cuda(), l.cuda()
                # 验证时不传 entropy maps -> 触发 Agent 推理模式
                seg_out, _, _, _ = model(vis=v, ir=i_img, gt_semantic=l)
                loss_val = crit_seg(seg_out, l)
                val_loss_total += loss_val.item()
                val_steps += 1
                evaluator.add_batch(torch.argmax(seg_out, 1).cpu().numpy(), l.cpu().numpy())

        avg_val_loss = val_loss_total / max(val_steps, 1)
        res = evaluator.get_metrics()

        steps = len(train_dl)
        print(f"📊 Summary Ep {ep + 1}:")
        print(
            f"   Train Loss -> Seg: {metrics['Seg'] / steps:.4f} | Aux: {metrics['Aux'] / steps:.4f} | Fus: {metrics['Fus'] / steps:.4f} | Moe: {metrics['Moe'] / steps:.4f}")
        print(
            f"   Val Metric -> mIoU: {res['mIoU'] * 100:.2f}% | PA: {res['PA'] * 100:.2f}% | Val Loss: {avg_val_loss:.4f}")

        if res['mIoU'] > best_miou:
            best_miou = res['mIoU']
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pth")
            print(f"🏆 New Best Saved! mIoU: {best_miou * 100:.2f}%")


if __name__ == "__main__":
    train()