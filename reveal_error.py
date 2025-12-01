import sys
import os

# 1. 强制添加路径 (根据你的 debug 结果)
project_path = r"F:\sam2-main"
if project_path not in sys.path:
    sys.path.insert(0, project_path)

print(f"当前路径: {sys.path[0]}")
print("正在尝试裸导入 SAM2...")

# 2. 直接导入，不加保护，让它报错！
import sam2
from sam2.build_sam import build_sam2

print("🎉 奇迹！导入成功了！")