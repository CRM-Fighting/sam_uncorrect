import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from prettytable import PrettyTable
from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from torchvision import transforms # 记得导入
EXP_NAME = "Exp_DynamicFusion_Final"  # 可根据需要修改为对应实验名
# 修改为新的最佳权重路径
CHECKPOINT_PATH = "/home/mmsys/disk/MCL/MultiModal_Project/sam2/checkpoints/Exp_MultiTask_SUM_Stage4Only/best_model.pth"
TEST_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'
}
SAVE_DIR = f"results/{EXP_NAME}_vis"  # 结果保存路径可根据EXP_NAME自动调整
SAM_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_small.pt"
NUM_CLASSES = 9
CLASS_NAMES = ['BackG', 'Car', 'Person', 'Bike', 'Curve', 'CarStop', 'Guardrail', 'Cone', 'Bump']
PALETTE = np.array([
    [0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192],
    [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]
], dtype=np.uint8)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dirs):
        self.vis = dirs['vi']
        self.ir = dirs['ir']
        self.lbl = dirs['label']
        self.files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])
        # ✅ 添加与训练一致的归一化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        # 读取并归一化 RGB
        v_tensor = torch.from_numpy(np.array(Image.open(os.path.join(self.vis, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0, 1) / 255.0
        v = self.normalize(v_tensor) # ✅ 应用归一化

        # 读取并归一化 IR
        i_tensor = torch.from_numpy(np.array(Image.open(os.path.join(self.ir, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0, 1) / 255.0
        i_img = self.normalize(i_tensor) # ✅ 应用归一化

        l = torch.from_numpy(np.array(Image.open(os.path.join(self.lbl, n)).resize((640, 480), Image.NEAREST))).long()
        return v, i_img, l, n

def colorize(mask):
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(NUM_CLASSES): img[mask == i] = PALETTE[i]
    return img

def calculate_iou_per_image(pred, label):
    iou_list = []
    for c in range(NUM_CLASSES):
        if ((label == c).sum() == 0) and ((pred == c).sum() == 0): continue
        u = ((pred == c) | (label == c)).sum()
        if u == 0: continue
        iou_list.append(((pred == c) & (label == c)).sum() / u)
    return sum(iou_list) / len(iou_list) if iou_list else 0.0

def test():
    print(f"🔄 Loading weights: {CHECKPOINT_PATH}")
    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, NUM_CLASSES).cuda()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    dl = DataLoader(TestDataset(TEST_DIRS), batch_size=1, shuffle=False)
    global_hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    image_miou_list, image_names = [], []
    os.makedirs(SAVE_DIR, exist_ok=True)

    with torch.no_grad():
        for v, i_img, l, fname in tqdm(dl, desc="Testing"):
            v, i_img = v.cuda(), i_img.cuda()
            logits, _, _, _ = model(v, i_img, gt_semantic=None, gt_entropy_maps=None)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            label = l.numpy()[0]

            mask = (label >= 0) & (label < NUM_CLASSES)
            global_hist += np.bincount(NUM_CLASSES * label[mask].astype(int) + pred[mask], minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            image_miou_list.append(calculate_iou_per_image(pred, label))
            image_names.append(fname[0])

            vis = np.vstack([
                np.hstack([(v[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), (i_img[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)]),
                np.hstack([colorize(label), colorize(pred)])
            ])
            cv2.imwrite(os.path.join(SAVE_DIR, fname[0]), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # Metrics
    intersection = np.diag(global_hist)
    union = global_hist.sum(1) + global_hist.sum(0) - intersection
    iou_per_class = intersection / (union + 1e-10)
    miou = np.nanmean(iou_per_class)
    pixel_acc = np.diag(global_hist).sum() / global_hist.sum()

    print("\n" + "="*50)
    print("📢 FINAL TEST REPORT")
    t = PrettyTable(["Class", "IoU (%)"])
    for i, score in enumerate(iou_per_class): t.add_row([CLASS_NAMES[i], f"{score*100:.2f}"])
    print(t)
    print(f"✅ Global mIoU: {miou*100:.2f}% | Pixel Acc: {pixel_acc*100:.2f}%")
    best_idx, worst_idx = np.argmax(image_miou_list), np.argmin(image_miou_list)
    print(f"🖼️  Best Image : {image_names[best_idx]} ({image_miou_list[best_idx]*100:.2f}%)")
    print(f"🖼️  Worst Image: {image_names[worst_idx]} ({image_miou_list[worst_idx]*100:.2f}%)")
    print(f"   Avg Img mIoU: {np.mean(image_miou_list)*100:.2f}%")
    print("="*50)

if __name__ == "__main__": test()