#!/usr/bin/env python3
"""
GPU环境检测脚本
GPU Environment validation script
"""

import torch
import sys
import subprocess
import logging
from pathlib import Path
import importlib.util

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("🐍 Python版本检查")
    print(f"Python版本: {sys.version}")

    if sys.version_info >= (3, 8):
        print("✅ Python版本满足要求 (>= 3.8)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.8")
        return False

def check_cuda_availability():
    """检查CUDA可用性"""
    print("\n" + "=" * 50)
    print("🚀 CUDA环境检查")

    # 检查PyTorch CUDA支持
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        return True
    else:
        print("❌ CUDA不可用")
        return False

def check_gpu_memory():
    """检查GPU内存"""
    print("\n" + "=" * 50)
    print("💾 GPU内存检查")

    if not torch.cuda.is_available():
        print("❌ 无法检查GPU内存：CUDA不可用")
        return False

    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = total_memory - reserved

            print(f"GPU {i}:")
            print(f"  总显存: {total_memory:.1f}GB")
            print(f"  已分配: {allocated:.1f}GB")
            print(f"  已预留: {reserved:.1f}GB")
            print(f"  可用: {free:.1f}GB")

            # 检查是否满足训练要求
            if total_memory >= 8.0:
                print(f"  ✅ 显存充足 (>= 8GB)")
            else:
                print(f"  ⚠️  显存可能不足，建议 >= 8GB")

        return True
    except Exception as e:
        print(f"❌ 检查GPU内存时出错: {e}")
        return False

def test_gpu_operations():
    """测试GPU基本操作"""
    print("\n" + "=" * 50)
    print("🧪 GPU操作测试")

    if not torch.cuda.is_available():
        print("❌ 跳过GPU测试：CUDA不可用")
        return False

    try:
        # 测试张量操作
        device = torch.device("cuda")
        print(f"使用设备: {device}")

        # 创建测试张量
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # 矩阵乘法测试
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()  # 等待GPU操作完成
        end_time = time.time()

        print(f"矩阵乘法测试: ✅ 完成 ({end_time - start_time:.3f}s)")

        # 测试混合精度
        with torch.cuda.amp.autocast():
            z_amp = torch.mm(x.half(), y.half())
        print("混合精度测试: ✅ 完成")

        # 清理显存
        del x, y, z, z_amp
        torch.cuda.empty_cache()
        print("显存清理: ✅ 完成")

        return True

    except Exception as e:
        print(f"❌ GPU操作测试失败: {e}")
        return False

def check_required_packages():
    """检查必需包"""
    print("\n" + "=" * 50)
    print("📦 依赖包检查")

    required_packages = [
        'torch',
        'transformers',
        'peft',
        'datasets',
        'accelerate',
        'bitsandbytes',
    ]

    optional_packages = [
        'flash_attn',
        'deepspeed',
        'xformers',
    ]

    all_good = True

    # 检查必需包
    for package in required_packages:
        try:
            if package == 'flash_attn':
                import flash_attn
                version = flash_attn.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: 未安装")
            all_good = False

    # 检查可选包
    print("\n可选包:")
    for package in optional_packages:
        try:
            if package == 'flash_attn':
                import flash_attn
                version = flash_attn.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"⚠️  {package}: 未安装 (可选)")

    return all_good

def check_nvidia_smi():
    """检查nvidia-smi"""
    print("\n" + "=" * 50)
    print("🔧 NVIDIA驱动检查")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi可用")
            # 提取关键信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"驱动版本: {line.split('Driver Version: ')[1].split()[0]}")
                if 'CUDA Version' in line:
                    print(f"CUDA版本: {line.split('CUDA Version: ')[1].split()[0]}")
            return True
        else:
            print("❌ nvidia-smi不可用")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi未找到，请检查NVIDIA驱动安装")
        return False

def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 50)
    print("⚡ 性能基准测试")

    if not torch.cuda.is_available():
        print("❌ 跳过性能测试：CUDA不可用")
        return False

    try:
        device = torch.device("cuda")

        # 测试不同精度的性能
        sizes = [512, 1024, 2048]
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        print("矩阵乘法性能测试:")
        print("大小\t\tFP32\t\tFP16\t\tBF16")
        print("-" * 50)

        for size in sizes:
            times = []
            for dtype in dtypes:
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    times.append("不支持")
                    continue

                x = torch.randn(size, size, device=device, dtype=dtype)
                y = torch.randn(size, size, device=device, dtype=dtype)

                # 预热
                for _ in range(5):
                    _ = torch.mm(x, y)
                torch.cuda.synchronize()

                # 计时
                import time
                start_time = time.time()
                for _ in range(10):
                    _ = torch.mm(x, y)
                torch.cuda.synchronize()
                end_time = time.time()

                avg_time = (end_time - start_time) / 10 * 1000  # ms
                times.append(f"{avg_time:.1f}ms")

                del x, y
                torch.cuda.empty_cache()

            print(f"{size}x{size}\t\t{times[0]}\t\t{times[1]}\t\t{times[2]}")

        return True

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 GPU训练环境检测开始")
    print("=" * 50)

    checks = [
        ("Python版本", check_python_version),
        ("NVIDIA驱动", check_nvidia_smi),
        ("CUDA环境", check_cuda_availability),
        ("GPU内存", check_gpu_memory),
        ("依赖包", check_required_packages),
        ("GPU操作", test_gpu_operations),
        ("性能基准", performance_benchmark),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name}检查时出错: {e}")
            results[name] = False

    # 总结
    print("\n" + "=" * 50)
    print("📊 检查结果总结")
    print("=" * 50)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有检查通过！GPU训练环境就绪")
    else:
        print("\n⚠️  部分检查未通过，请查看上述详细信息")
        print("\n建议:")
        if not results.get("CUDA环境", True):
            print("- 安装CUDA工具包和兼容的PyTorch版本")
        if not results.get("依赖包", True):
            print("- 运行: pip install -r requirements-gpu.txt")
        if not results.get("GPU内存", True):
            print("- 考虑使用量化或减少batch size")

    return all_passed

if __name__ == "__main__":
    main()