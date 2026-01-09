#!/usr/bin/env python3
"""
快速测试本地模型和数据
Quick test for local model and data
"""

import sys
from pathlib import Path
import json

def test_data():
    """测试数据文件"""
    print("=== 测试数据文件 ===\n")

    data_path = Path("data/raw/alpaca_data_cleaned.json")

    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return False

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 数据加载成功！")
        print(f"   样本数量: {len(data)}")
        print(f"   文件大小: {data_path.stat().st_size / 1024**2:.1f} MB")

        if len(data) > 0:
            sample = data[0]
            print(f"   样本格式: {list(sample.keys())}")
            print(f"\n   示例:")
            print(f"   指令: {sample.get('instruction', 'N/A')[:80]}...")
            if 'input' in sample:
                inp = sample['input']
                print(f"   输入: {inp[:50] if inp else '(空)'}...")
            print(f"   输出: {sample.get('output', 'N/A')[:80]}...")

        return True

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_model_files():
    """测试模型文件"""
    print("\n=== 测试模型文件 ===\n")

    model_path = Path("models")

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json"
    ]

    print(f"📂 检查模型目录: {model_path}")

    all_ok = True
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024
            print(f"✅ {file}: {size:.1f} KB")
        else:
            print(f"❌ {file}: 不存在")
            all_ok = False

    # 检查模型权重文件
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        print(f"\n📦 模型权重文件:")
        total_size = 0
        for file in safetensors_files:
            size = file.stat().st_size / 1024**3
            total_size += size
            print(f"   {file.name}: {size:.2f} GB")
        print(f"   总计: {total_size:.2f} GB")

    return all_ok

def main():
    print("🚀 快速测试本地环境\n")

    # 测试数据
    data_ok = test_data()

    # 测试模型文件
    model_ok = test_model_files()

    # 总结
    print("\n" + "="*60)
    print("\n📊 测试结果:")
    print(f"   {'✅' if data_ok else '❌'} 数据文件")
    print(f"   {'✅' if model_ok else '❌'} 模型文件")

    if data_ok and model_ok:
        print("\n🎉 文件检查通过！")
        print("\n💡 下一步:")
        print("   1. 测试GPU环境: python scripts/test_gpu_env.py")
        print("   2. 开始训练: python scripts/train.py --model_config config/gpu_model_config.yaml --lora_config config/gpu_lora_config.yaml --device cuda")
        return 0
    else:
        print("\n⚠️  请检查上述问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
