#!/usr/bin/env python3
"""
测试本地模型和数据加载
Test local model and data loading
"""

import sys
from pathlib import Path
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.model_loader import ModelLoader
import json

def test_local_model():
    """测试本地模型加载"""
    print("=== 测试本地模型加载 ===\n")

    try:
        # 使用配置文件加载模型
        loader = ModelLoader("config/gpu_model_config.yaml")

        print("📂 加载配置文件: config/gpu_model_config.yaml")
        print(f"📦 模型路径: {loader.model_config['model']['name']}")
        print(f"📝 Tokenizer路径: {loader.model_config['tokenizer']['name']}\n")

        # 加载tokenizer
        print("🔤 加载tokenizer...")
        tokenizer = loader.load_tokenizer()
        print(f"✅ Tokenizer加载成功！")
        print(f"   词汇表大小: {len(tokenizer)}")
        print(f"   PAD token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}\n")

        # 加载模型
        print("🤖 加载模型...")
        model = loader.load_model()
        print(f"✅ 模型加载成功！")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   设备: {next(model.parameters()).device}")
        print(f"   数据类型: {next(model.parameters()).dtype}\n")

        # 显存使用
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU显存使用:")
            print(f"   已分配: {mem_allocated:.2f} GB")
            print(f"   已预留: {mem_reserved:.2f} GB")
            print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_data():
    """测试本地数据加载"""
    print("\n=== 测试本地数据加载 ===\n")

    data_path = Path("data/raw/alpaca_data_cleaned.json")

    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return False

    try:
        print(f"📂 读取数据文件: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 数据加载成功！")
        print(f"   数据类型: {type(data)}")

        if isinstance(data, list):
            print(f"   样本数量: {len(data)}")
            if len(data) > 0:
                print(f"   第一个样本的键: {list(data[0].keys())}")
                print(f"\n   示例样本:")
                print(f"   指令: {data[0].get('instruction', 'N/A')[:100]}...")
                if 'input' in data[0]:
                    print(f"   输入: {data[0]['input'][:50] if data[0]['input'] else '(空)'}...")
                print(f"   输出: {data[0].get('output', 'N/A')[:100]}...")
        elif isinstance(data, dict):
            print(f"   数据键: {list(data.keys())}")

        print(f"\n   文件大小: {data_path.stat().st_size / 1024**2:.1f} MB\n")

        return True

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始测试本地环境...\n")
    print("=" * 60)

    # 测试GPU
    print("\n=== GPU信息 ===\n")
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"   显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   PyTorch版本: {torch.__version__}\n")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练\n")

    # 测试模型
    model_ok = test_local_model()

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 已清理GPU缓存\n")

    # 测试数据
    data_ok = test_local_data()

    # 总结
    print("\n" + "=" * 60)
    print("\n📊 测试总结:")
    print(f"   {'✅' if model_ok else '❌'} 模型加载")
    print(f"   {'✅' if data_ok else '❌'} 数据加载")

    if model_ok and data_ok:
        print("\n🎉 所有测试通过！可以开始训练了！")
        print("\n💡 下一步:")
        print("   运行训练: python scripts/train.py --model_config config/gpu_model_config.yaml --lora_config config/gpu_lora_config.yaml --device cuda")
        return 0
    else:
        print("\n⚠️  请检查上述错误并修复后再训练")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
