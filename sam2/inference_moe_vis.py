import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam_moe import MultiModalSegFormerMoE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 实例化新模型
    base_sam = build_sam2("sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt", device=device)
    model = MultiModalSegFormerMoE(base_sam, [96, 192, 384, 768], num_classes=9)

    # 2. 加载权重
    ckpt_path = "checkpoints_moe/best_moe_model.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded MoE model.")
    else:
        print("Model not found!")
        return

    model.to(device)
    model.eval()

    # 3. 随便找张图测试
    # 请替换为你实际存在的图片路径
    test_vis = "sam2/data/MSRS/test/vis/00101D.png"
    test_ir = "sam2/data/MSRS/test/ir/00101D.png"

    vis_img = Image.open(test_vis).convert('RGB').resize((640, 480))
    ir_img = Image.open(test_ir).convert('RGB').resize((640, 480))

    vis_t = torch.from_numpy(np.array(vis_img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    ir_t = torch.from_numpy(np.array(ir_img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        preds, _ = model(vis_t.to(device), ir_t.to(device))
        mask = torch.argmax(preds, dim=1).squeeze().cpu().numpy()

    # 简单的可视化保存
    plt.imshow(mask)
    plt.savefig("moe_prediction.png")
    print("Result saved to moe_prediction.png")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()