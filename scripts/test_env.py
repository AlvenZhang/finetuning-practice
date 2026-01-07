#!/usr/bin/env python3
"""
环境测试脚本
Test environment setup for LLM fine-tuning
"""

import sys
import platform
import subprocess
from pathlib import Path

def test_python_version():
    """测试Python版本"""
    print("🐍 Python版本检查")
    version = sys.version_info
    print(f"   版本: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("   ❌ Python版本过低，需要 >= 3.8")
        return False

def test_system_info():
    """显示系统信息"""
    print("\\n💻 系统信息")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   架构: {platform.machine()}")
    print(f"   处理器: {platform.processor()}")

    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   内存: {memory.total / (1024**3):.1f} GB")
        print(f"   可用内存: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("   ⚠️  无法获取内存信息 (psutil未安装)")

def test_pytorch():
    """测试PyTorch安装"""
    print("\\n🔥 PyTorch检查")
    try:
        import torch
        print(f"   版本: {torch.__version__}")

        # 检查设备支持
        if torch.cuda.is_available():
            print("   ✅ CUDA可用")
            print(f"      CUDA版本: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"      GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        elif torch.backends.mps.is_available():
            print("   ✅ Apple Silicon MPS可用")
            print("      适用于M系列芯片")
        else:
            print("   ⚠️  仅CPU可用")

        return True
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def test_transformers():
    """测试Transformers库"""
    print("\\n🤗 Transformers检查")
    try:
        import transformers
        print(f"   版本: {transformers.__version__}")

        # 测试基本功能
        from transformers import AutoTokenizer
        print("   ✅ 基本功能正常")
        return True
    except ImportError:
        print("   ❌ Transformers未安装")
        return False

def test_peft():
    """测试PEFT库"""
    print("\\n🔧 PEFT检查")
    try:
        import peft
        print(f"   版本: {peft.__version__}")
        print("   ✅ LoRA支持可用")
        return True
    except ImportError:
        print("   ❌ PEFT未安装")
        return False

def test_datasets():
    """测试Datasets库"""
    print("\\n📊 Datasets检查")
    try:
        import datasets
        print(f"   版本: {datasets.__version__}")
        print("   ✅ 数据集处理可用")
        return True
    except ImportError:
        print("   ❌ Datasets未安装")
        return False

def test_mlx():
    """测试MLX库 (Apple Silicon)"""
    print("\\n🍎 MLX检查 (Apple Silicon优化)")
    try:
        import mlx
        import mlx.core as mx
        print(f"   MLX版本: {mx.__version__}")

        # 测试基本操作
        x = mx.array([1, 2, 3])
        y = mx.array([4, 5, 6])
        z = x + y
        print("   ✅ MLX基本操作正常")

        try:
            import mlx_lm
            print("   ✅ MLX-LM可用")
        except ImportError:
            print("   ⚠️  MLX-LM未安装")

        return True
    except ImportError:
        print("   ❌ MLX未安装 (仅Apple Silicon需要)")
        return False

def test_optional_packages():
    """测试可选包"""
    print("\\n📦 可选包检查")

    packages = {
        'wandb': 'Weights & Biases实验跟踪',
        'matplotlib': '图表绘制',
        'seaborn': '数据可视化',
        'jupyter': 'Jupyter Notebook',
        'evaluate': '评估指标',
        'rouge_score': 'ROUGE评估',
        'sacrebleu': 'BLEU评估'
    }

    results = {}
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package}: {description}")
            results[package] = True
        except ImportError:
            print(f"   ⚠️  {package}: {description} (未安装)")
            results[package] = False

    return results

def test_huggingface_login():
    """测试Hugging Face登录"""
    print("\\n🤗 Hugging Face登录检查")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"   ✅ 已登录: {user_info['name']}")
        return True
    except Exception:
        print("   ⚠️  未登录或token无效")
        print("      运行: huggingface-cli login")
        return False

def test_project_structure():
    """测试项目结构"""
    print("\\n📁 项目结构检查")

    required_dirs = [
        'config', 'data/raw', 'data/processed', 'src',
        'scripts', 'models', 'experiments'
    ]

    required_files = [
        'requirements.txt', 'README.md',
        'config/model_config.yaml',
        'config/lora_config.yaml',
        'config/eval_config.yaml'
    ]

    all_good = True

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (缺失)")
            all_good = False

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (缺失)")
            all_good = False

    return all_good

def run_installation_test():
    """运行安装测试"""
    print("\\n🧪 运行简单安装测试")

    try:
        # 测试基本导入
        print("   测试基本导入...")
        import torch
        import transformers
        import peft
        import datasets

        # 测试分词器加载
        print("   测试分词器加载...")
        from transformers import AutoTokenizer

        # 使用一个小模型测试
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)

        print("   ✅ 基本功能测试通过")
        return True

    except Exception as e:
        print(f"   ❌ 安装测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 LLM微调环境检查")
    print("=" * 60)

    results = {}

    # 基本检查
    results['python'] = test_python_version()
    test_system_info()
    results['pytorch'] = test_pytorch()
    results['transformers'] = test_transformers()
    results['peft'] = test_peft()
    results['datasets'] = test_datasets()

    # Apple Silicon特定
    if platform.machine() in ['arm64', 'aarch64']:
        results['mlx'] = test_mlx()
    else:
        results['mlx'] = True  # 非Apple Silicon不需要

    # 可选包
    optional_results = test_optional_packages()

    # 其他检查
    results['hf_login'] = test_huggingface_login()
    results['project_structure'] = test_project_structure()
    results['installation_test'] = run_installation_test()

    # 总结
    print("\\n" + "=" * 60)
    print("📋 检查总结")

    required_checks = ['python', 'pytorch', 'transformers', 'peft', 'datasets', 'mlx']
    passed = sum(results.get(check, False) for check in required_checks)
    total = len(required_checks)

    print(f"\\n必需组件: {passed}/{total} 通过")

    if passed == total:
        print("✅ 环境配置完整，可以开始微调！")

        # 提供下一步建议
        print("\\n🎯 下一步:")
        print("1. 下载数据: python data/download_data.py")
        print("2. 下载模型: python scripts/download_model.py")
        print("3. 开始训练: python scripts/train.py")

        return 0
    else:
        print("❌ 环境配置不完整，请安装缺失组件")
        print("\\n💡 安装命令:")
        print("pip install -r requirements.txt")

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)