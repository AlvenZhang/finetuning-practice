#!/usr/bin/env python3
"""
GPU 性能基准测试脚本
检查 WSL GPU 性能是否正常
"""

import torch
import time
import numpy as np
from datetime import datetime

def test_gpu_availability():
    """检查 GPU 可用性"""
    print("=" * 60)
    print("🔍 GPU 可用性检查")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！请检查 CUDA 安装")
        return False

    print(f"✅ CUDA 可用")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch CUDA 版本: {torch.cuda.get_arch_list()}")
    return True


def get_gpu_memory_info():
    """获取 GPU 内存信息"""
    print("\n" + "=" * 60)
    print("💾 GPU 内存信息")
    print("=" * 60)

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)

    print(f"总内存: {total_memory / 1024**3:.2f} GB")
    print(f"已分配: {allocated_memory / 1024**3:.2f} GB")
    print(f"已缓存: {cached_memory / 1024**3:.2f} GB")
    print(f"可用内存: {(total_memory - allocated_memory) / 1024**3:.2f} GB")


def benchmark_matrix_multiplication(size=4096, num_iterations=100):
    """矩阵乘法基准测试"""
    print("\n" + "=" * 60)
    print(f"⚡ 矩阵乘法性能测试 ({size}x{size} x {num_iterations} 次)")
    print("=" * 60)

    device = torch.device("cuda:0")

    # 预热
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(10):
        _ = torch.mm(a, b)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        c = torch.mm(a, b)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    total_ops = 2 * size ** 3 * num_iterations  # 2 * n^3 乘加操作
    gflops = total_ops / (elapsed_time * 1e9)

    print(f"⏱️  总时间: {elapsed_time:.3f} 秒")
    print(f"📊 性能: {gflops:.2f} GFLOPS")
    print(f"🚀 每次迭代: {elapsed_time/num_iterations*1000:.2f} ms")

    # 性能参考
    expected_gflops = {
        "RTX 4060": 15000,  # 理论峰值
        "RTX 3060": 13000,
        "GTX 1660": 5000,
    }

    current_gpu = torch.cuda.get_device_name(0)
    print(f"\n📈 性能评估:")
    if gflops > 10000:
        print(f"✅ 性能优秀 ({gflops:.0f} GFLOPS)")
    elif gflops > 5000:
        print(f"⚠️  性能一般 ({gflops:.0f} GFLOPS)")
    else:
        print(f"❌ 性能较差 ({gflops:.0f} GFLOPS) - 可能存在问题")

    return gflops


def benchmark_memory_transfer(size=1024*1024*100):  # 100M 元素
    """内存传输速度测试"""
    print("\n" + "=" * 60)
    print(f"🔄 内存传输速度测试 ({size/1e6:.1f}M 元素)")
    print("=" * 60)

    device = torch.device("cuda:0")

    # CPU -> GPU
    cpu_tensor = torch.randn(size)
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        gpu_tensor = cpu_tensor.to(device)

    torch.cuda.synchronize()
    cpu_to_gpu_time = (time.time() - start_time) / 10

    # GPU -> CPU
    start_time = time.time()

    for _ in range(10):
        cpu_tensor_back = gpu_tensor.to('cpu')

    torch.cuda.synchronize()
    gpu_to_cpu_time = (time.time() - start_time) / 10

    data_size_gb = size * 4 / 1e9  # float32 = 4 bytes

    print(f"CPU -> GPU: {cpu_to_gpu_time*1000:.2f} ms ({data_size_gb/cpu_to_gpu_time:.2f} GB/s)")
    print(f"GPU -> CPU: {gpu_to_cpu_time*1000:.2f} ms ({data_size_gb/gpu_to_cpu_time:.2f} GB/s)")

    # 性能评估
    print(f"\n📈 传输速度评估:")
    if cpu_to_gpu_time < 0.01:  # < 10ms
        print("✅ 传输速度优秀")
    elif cpu_to_gpu_time < 0.05:  # < 50ms
        print("⚠️  传输速度一般")
    else:
        print("❌ 传输速度较慢 - 可能存在 WSL/Windows GPU 驱动问题")


def benchmark_pytorch_operations():
    """PyTorch 常见操作性能测试"""
    print("\n" + "=" * 60)
    print("🔥 PyTorch 操作性能测试")
    print("=" * 60)

    device = torch.device("cuda:0")
    batch_size = 32
    seq_length = 512
    hidden_dim = 768

    # 模拟 Transformer 层计算
    x = torch.randn(batch_size, seq_length, hidden_dim, device=device)

    # 预热
    for _ in range(10):
        y = torch.matmul(x, x.transpose(-2, -1))

    torch.cuda.synchronize()
    start_time = time.time()

    num_iterations = 100
    for _ in range(num_iterations):
        y = torch.matmul(x, x.transpose(-2, -1))

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(f"注意力机制计算 ({batch_size}x{seq_length}x{hidden_dim}):")
    print(f"⏱️  {elapsed_time/num_iterations*1000:.2f} ms/次")
    print(f"🚀 {num_iterations/elapsed_time:.1f} 次/秒")


def check_wsl_performance_issues():
    """检查 WSL 特定的性能问题"""
    print("\n" + "=" * 60)
    print("🪟 WSL 性能问题检查")
    print("=" * 60)

    import subprocess
    import os

    # 检查 WSL 版本
    try:
        result = subprocess.run(['wsl', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ WSL 版本信息:")
            print(result.stdout)
    except:
        print("⚠️  无法获取 WSL 版本信息")

    # 检查是否在 WSL2 中运行
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
            if 'microsoft' in version.lower():
                print("✅ 正在 WSL 环境中运行")
                if 'wsl2' in version.lower() or '2' in version:
                    print("✅ WSL2 模式 (推荐)")
                else:
                    print("⚠️  可能是 WSL1，GPU 支持有限")
    except:
        print("❌ 无法确定 WSL 版本")

    # 检查 Windows GPU 驱动
    print("\n建议在 Windows 中检查:")
    print("1. 运行 'nvidia-smi' 查看驱动版本")
    print("2. 访问 https://www.nvidia.com/Download/index.aspx 更新驱动")
    print("3. 确保安装了 WSL2 支持的最新驱动")


def compare_with_native_windows():
    """建议与原生 Windows 性能对比"""
    print("\n" + "=" * 60)
    print("📊 性能对比建议")
    print("=" * 60)

    print("要准确评估 WSL 性能损失，建议:")
    print("1. 在 Windows 原生环境中运行相同测试")
    print("2. 对比相同操作的时间差异")
    print("3. WSL2 通常有 5-15% 的性能损失是正常的")
    print("4. 如果损失超过 20%，可能存在配置问题")


def main():
    """主测试流程"""
    print("🚀 开始 GPU 性能基准测试")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not test_gpu_availability():
        return

    get_gpu_memory_info()

    # 运行各种基准测试
    try:
        gflops = benchmark_matrix_multiplication()
        benchmark_memory_transfer()
        benchmark_pytorch_operations()
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return

    check_wsl_performance_issues()
    compare_with_native_windows()

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
    print("\n💡 建议:")
    print("1. 如果性能显著低于预期，考虑更新 NVIDIA 驱动")
    print("2. 确保使用 WSL2 而非 WSL1")
    print("3. 检查 Windows 电源计划设置为高性能")
    print("4. 关闭后台程序以释放 GPU 资源")


if __name__ == "__main__":
    main()