#!/usr/bin/env python3
"""
测试 Qwen 模型加载
Test Qwen model loading
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.model_loader import ModelLoader

def test_qwen_loading():
    """测试Qwen模型加载"""
    print("=== 测试 Qwen 模型加载 ===")

    try:
        # 检查模型文件是否存在
        model_path = "./models/base/qwen2.5-3b-instruct"
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            return False

        print(f"✅ 模型路径存在: {model_path}")

        # 检查关键文件
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]

        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"✅ {file} 存在")
            else:
                print(f"❌ {file} 不存在")
                return False

        # 初始化模型加载器
        config_path = "./config/qwen_model_config.yaml"
        loader = ModelLoader(config_path)

        print("\\n=== 加载分词器 ===")
        tokenizer = loader.load_tokenizer(model_path)
        print(f"✅ 分词器加载成功")
        print(f"   词汇表大小: {len(tokenizer)}")
        print(f"   特殊token: bos={tokenizer.bos_token}, eos={tokenizer.eos_token}, pad={tokenizer.pad_token}")

        # 测试分词
        test_text = "你好，世界！Hello, World!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   测试文本: {test_text}")
        print(f"   编码结果: {tokens[:10]}... (显示前10个)")
        print(f"   解码结果: {decoded}")

        print("\\n=== 加载模型 ===")
        print("⚠️  注意: 加载3B模型可能需要1-2分钟时间...")

        # 尝试加载模型（可能会因为内存不足而失败）
        try:
            model = loader.load_model(model_path)
            print("✅ 模型加载成功")

            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   总参数量: {total_params:,}")
            print(f"   模型类型: {type(model).__name__}")
            print(f"   设备: {next(model.parameters()).device}")

            # 测试简单推理
            print("\\n=== 测试推理 ===")
            model.eval()
            with torch.no_grad():
                input_ids = tokenizer.encode("你好", return_tensors="pt")
                if torch.backends.mps.is_available():
                    input_ids = input_ids.to("mps")
                    model = model.to("mps")

                outputs = model(input_ids)
                print("✅ 推理测试成功")
                print(f"   输出形状: {outputs.logits.shape}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 这可能是因为内存不足。Qwen 3B需要约6-8GB内存。")
            print("💡 建议:")
            print("   1. 关闭其他占用内存的程序")
            print("   2. 使用量化加载 (load_in_4bit=True)")
            print("   3. 检查模型文件是否完整")
            return False

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Qwen 模型加载测试")
    print("=" * 50)

    # 检查环境
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")

    # 检查内存
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon (MPS) 环境")
    elif torch.cuda.is_available():
        print(f"🚀 CUDA环境: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 CPU环境")

    print("\\n" + "=" * 50)

    # 运行测试
    success = test_qwen_loading()

    if success:
        print("\\n🎉 所有测试通过！")
        print("\\n📝 下一步:")
        print("1. 运行训练: python scripts/train.py --model_config config/qwen_model_config.yaml --lora_config config/qwen_lora_config.yaml")
        print("2. 监控内存使用情况")
        print("3. 如果内存不足，考虑调整批次大小或使用量化")
    else:
        print("\\n❌ 测试失败，请检查上述错误信息")

if __name__ == "__main__":
    main()