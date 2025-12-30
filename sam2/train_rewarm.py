import os

# ★★★ 防止显存碎片化 ★★★
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# 引入你的模型组件
from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from utils.custom_losses import BinarySUMLoss, StandardSegLoss

# --- 配置 ---
EXP_NAME = "Exp_Rewarm_LR_Optimization"
# ★★★ 请确认这个路径指向您 Ep 40+ 的最佳权重 ★★★
PRETRAINED_PATH = "checkpoints/Exp_DualStream_Accum8_Scheduler_12_28/best_model.pth"

# 路径配置
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
UNCERTAINTY_DIRS = {'train': "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map/train"}

SAM_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_small.pt"

BATCH_SIZE = 2
ACCUM_STEPS = 8
EPOCHS = 30
NUM_CLASSES = 9


# --- 1. 补全缺失的工具函数 ---
def setup_seed(seed=42):
    """固定全局随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    """保证 DataLoader 多线程读取时种子不同，避免数据重复"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- 2. 评估工具 ---
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


# --- 3. 数据集定义 ---
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
        self.files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        v_img = Image.open(os.path.join(self.vis, n)).convert('RGB')
        i_img_pil = Image.open(os.path.join(self.ir, n)).convert('RGB')
        lbl_pil = Image.open(os.path.join(self.lbl, n))

        s_tensor = torch.zeros((1, 480, 640))
        if self.uncertainty_root:
            sp = os.path.join(self.uncertainty_root, n.replace('.png', '.npy'))
            if os.path.exists(sp):
                s_np = np.load(sp)
                s_tensor = torch.from_numpy(s_np).float()
                if len(s_tensor.shape) == 2: s_tensor = s_tensor.unsqueeze(0)

        entropy_maps_list = []
        if self.entropy_dirs:
            npy_name = n.replace('.png', '.npy')
            for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                path = os.path.join(self.entropy_dirs[stage], npy_name)
                if os.path.exists(path):
                    em = torch.from_numpy(np.load(path).astype(np.float32))
                    if em.ndim == 2: em = em.unsqueeze(0)
                    entropy_maps_list.append(em)
                else:
                    entropy_maps_list.append(torch.zeros(1, 1, 1))

        if self.is_train:
            if torch.rand(1).item() > 0.5:
                v_img = TF.hflip(v_img)
                i_img_pil = TF.hflip(i_img_pil)
                lbl_pil = TF.hflip(lbl_pil)
                s_tensor = TF.hflip(s_tensor)
                entropy_maps_list = [TF.hflip(em) for em in entropy_maps_list]

        v_img = v_img.resize((640, 480), Image.BILINEAR)
        i_img_pil = i_img_pil.resize((640, 480), Image.BILINEAR)
        lbl_pil = lbl_pil.resize((640, 480), Image.NEAREST)

        if s_tensor.shape[-1] != 640:
            s_tensor = F.interpolate(s_tensor.unsqueeze(0), (480, 640), mode='bilinear').squeeze(0)

        v_tensor = torch.from_numpy(np.array(v_img)).float().permute(2, 0, 1) / 255.0
        v = self.normalize(v_tensor)
        i_tensor = torch.from_numpy(np.array(i_img_pil)).float().permute(2, 0, 1) / 255.0
        i_img = self.normalize(i_tensor)
        l = torch.from_numpy(np.array(lbl_pil)).long()

        return v, i_img, l, s_tensor, entropy_maps_list


def train():
    setup_seed(42)
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"🚀 Experiment: {EXP_NAME} (LR Rewarm)")
    print(f"📥 Loading weights from: {PRETRAINED_PATH}")

    # 1. 初始化模型结构
    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=NUM_CLASSES).cuda()

    # 2. 加载最佳权重
    if os.path.exists(PRETRAINED_PATH):
        state_dict = torch.load(PRETRAINED_PATH)
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"✅ Weights Loaded! {msg}")
    else:
        raise FileNotFoundError(f"找不到权重文件: {PRETRAINED_PATH}")

    # 3. 设置优化器 (LR 重置策略：5e-5 / 5e-4)
    router_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'router' in name:
            router_params.append(param)
        else:
            other_params.append(param)

    opt = optim.AdamW([
        {'params': other_params, 'lr': 5e-5},
        {'params': router_params, 'lr': 5e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)

    scaler = GradScaler()
    crit_seg = StandardSegLoss(NUM_CLASSES)
    crit_sam = BinarySUMLoss(theta=0.6)

    # DataLoader
    train_dl = DataLoader(
        MSRSDataset(TRAIN_DIRS, UNCERTAINTY_DIRS['train'], ENTROPY_ROOT, is_train=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn
    )
    val_dl = DataLoader(
        MSRSDataset(VAL_DIRS, None, None, is_train=False),
        batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn
    )

    evaluator = SegEvaluator(NUM_CLASSES)
    best_miou = 0.0

    # 训练循环
    for ep in range(EPOCHS):
        model.train()
        pbar = tqdm(train_dl, desc=f"Ep {ep + 1} Rewarm")
        opt.zero_grad()

        avg_metrics = {'seg': 0, 'aux': 0, 'fus': 0}
        steps = 0

        for step, (v, i_img, l, s, entropy_maps) in enumerate(pbar):
            v, i_img, l, s = v.cuda(), i_img.cuda(), l.cuda(), s.cuda()
            e_maps_cuda = [em.cuda() for em in entropy_maps] if entropy_maps else None

            with autocast('cuda'):
                seg_out, sam_preds, moe_loss, fusion_loss = model(v, i_img, l, e_maps_cuda)
                l_main = crit_seg(seg_out, l)
                l_aux = (crit_sam(sam_preds['rgb_s4'], l, s) + crit_sam(sam_preds['ir_s4'], l, s)) / 2.0

                loss = 0.5 * l_main + 0.5 * l_aux + 0.5 * fusion_loss + 0.05 * moe_loss
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            avg_metrics['seg'] += l_main.item()
            steps += 1

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            pbar.set_postfix({'Seg': f"{l_main.item():.3f}", 'LR': f"{opt.param_groups[0]['lr']:.6f}"})

        scheduler.step()
        torch.cuda.empty_cache()

        # === 验证 (包含 Loss 计算) ===
        model.eval()
        evaluator.reset()
        val_loss_total = 0.0
        val_steps = 0
        print(f"⏳ Validating Ep {ep + 1}...")

        with torch.no_grad():
            for v, i_img, l, s, _ in tqdm(val_dl, desc="Val"):
                v, i_img, l, s = v.cuda(), i_img.cuda(), l.cuda(), s.cuda()

                # 为了计算 Loss，传入 gt_semantic
                seg_out, sam_preds, moe_loss, fusion_loss = model(v, i_img, gt_semantic=l)

                # 计算 Val Loss
                l_main = crit_seg(seg_out, l)
                l_aux = torch.tensor(0.0, device=v.device)
                if sam_preds:
                    l_aux = (crit_sam(sam_preds['rgb_s4'], l, s) + crit_sam(sam_preds['ir_s4'], l, s)) / 2.0

                loss = 0.5 * l_main + 0.5 * l_aux + 0.5 * fusion_loss + 0.05 * moe_loss
                val_loss_total += loss.item()
                val_steps += 1

                pred = torch.argmax(seg_out, dim=1).cpu().numpy()
                evaluator.add_batch(pred, l.cpu().numpy())

        avg_val_loss = val_loss_total / max(val_steps, 1)
        metrics = evaluator.get_metrics()
        cur_miou = metrics['mIoU']

        # 打印详细结果
        print(f"📊 Rewarm Ep {ep + 1}: mIoU={cur_miou * 100:.2f}% | Val Loss={avg_val_loss:.4f}")

        if cur_miou > best_miou:
            best_miou = cur_miou
            torch.save(model.state_dict(), f"{ckpt_dir}/best_rewarm_model.pth")
            print(f"🏆 New Best Saved! mIoU: {best_miou * 100:.2f}%")


if __name__ == "__main__":
    train()