import sys
import os
import platform
from pathlib import Path

print("=" * 60)
print("🔍 SAM2 环境深度诊断程序")
print("=" * 60)

# 1. 打印当前运行位置
current_dir = os.getcwd()
script_path = Path(__file__).resolve()
print(f"📂 当前工作目录: {current_dir}")
print(f"📜 脚本所在位置: {script_path}")

# 2. 检查 sam2 文件夹是否存在
sam2_folder = script_path.parent / "sam2"
if sam2_folder.exists():
    print(f"✅ 发现 sam2 源代码文件夹: {sam2_folder}")
else:
    print(f"❌ [严重] 在当前目录下找不到 'sam2' 文件夹！")
    print(f"   请务必将此脚本放在 'sam2-main' 根目录下运行。")

# 3. 打印系统路径 (sys.path)
print("\n🛣️ Python 系统路径 (sys.path):")
for p in sys.path:
    print(f"   - {p}")

# 4. 尝试导入关键依赖 (hydra)
print("\n🔍 正在检查依赖库 'hydra-core' ...")
try:
    import hydra

    print(f"✅ hydra 导入成功 (版本: {hydra.__version__})")
except ImportError as e:
    print(f"❌ [缺失] 无法导入 hydra。报错: {e}")
    print("👉 请在终端运行: pip install hydra-core --upgrade")

# 5. 尝试导入 SAM2 并捕获详细错误
print("\n🔍 正在尝试导入 SAM2 ...")
try:
    # 尝试把当前目录加入路径，防止 Python 找不到
    if str(script_path.parent) not in sys.path:
        sys.path.insert(0, str(script_path.parent))

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    print("🎉🎉🎉 SAM2 导入成功！环境配置正确。")

except Exception as e:
    print("\n❌❌❌ SAM2 导入失败！详细报错堆栈如下：")
    print("-" * 40)
    import traceback

    traceback.print_exc()
    print("-" * 40)

    # 智能分析错误
    err_str = str(e)
    if "No module named 'hydra'" in err_str:
        print("\n💡 解决方案: 你的环境缺少 hydra-core。请运行:")
        print("   pip install hydra-core")
    elif "No module named 'sam2'" in err_str:
        print("\n💡 解决方案: Python 找不到 sam2 文件夹。")
        print("   确保此脚本 debug_env.py 是直接放在 sam2-main 文件夹下的。")
    elif "DLL load failed" in err_str:
        print("\n💡 解决方案: 这是一个 Windows 常见错误。")
        print("   可能是 CUDA 扩展编译失败，或者缺少 PyTorch 的 C++ 依赖。")
        print("   尝试重新安装 torch，或者忽略编译错误重新安装 sam2：")
        print("   set SAM2_BUILD_ALLOW_ERRORS=1 && pip install -e .")

print("\n诊断结束。")