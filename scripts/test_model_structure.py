#!/usr/bin/env python3
"""
测试新的模型目录结构
Test the new model directory structure
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

def test_model_structure():
    """测试模型目录结构和文件完整性"""

    print("=== 测试新的模型目录结构 ===\n")

    # 定义预期的目录结构
    base_models_dir = Path("models/base")
    qwen_model_dir = base_models_dir / "qwen2.5-3b-instruct"

    # 检查基础目录
    print("1. 检查目录结构:")
    if base_models_dir.exists():
        print(f"✅ 基础目录存在: {base_models_dir}")
    else:
        print(f"❌ 基础目录不存在: {base_models_dir}")
        return False

    if qwen_model_dir.exists():
        print(f"✅ Qwen模型目录存在: {qwen_model_dir}")
    else:
        print(f"❌ Qwen模型目录不存在: {qwen_model_dir}")
        return False

    # 检查必需的模型文件
    print("\n2. 检查模型文件:")
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"
    ]

    missing_files = []
    for file in required_files:
        file_path = qwen_model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {file} 缺失")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ 发现 {len(missing_files)} 个缺失文件")
        return False

    # 检查配置文件更新
    print("\n3. 检查配置文件更新:")
    config_files = {
        "config/qwen_model_config.yaml": "qwen2.5-3b-instruct",
        "examples/tokenizer_usage.py": "qwen2.5-3b-instruct",
        "scripts/test_qwen_loading.py": "qwen2.5-3b-instruct"
    }

    for config_file, expected_path in config_files.items():
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if expected_path in content:
                    print(f"✅ {config_file} 路径已更新")
                else:
                    print(f"❌ {config_file} 路径未更新")
                    return False
        else:
            print(f"❌ {config_file} 不存在")
            return False

    # 测试tokenizer加载
    print("\n4. 测试tokenizer加载:")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(qwen_model_dir))
        print(f"✅ Tokenizer加载成功")
        print(f"   词汇表大小: {len(tokenizer):,}")
        print(f"   特殊token: {tokenizer.special_tokens_map}")

        # 测试编码/解码
        test_text = "你好，这是一个测试。Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   编码测试: {len(tokens)} tokens")
        print(f"   解码测试: {'✅' if decoded == test_text else '❌'}")

    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return False

    # 测试模型配置加载
    print("\n5. 测试模型配置加载:")
    try:
        config = AutoConfig.from_pretrained(str(qwen_model_dir))
        print(f"✅ 模型配置加载成功")
        print(f"   模型类型: {config.model_type}")
        print(f"   词汇表大小: {config.vocab_size:,}")
        print(f"   隐藏层大小: {config.hidden_size}")
        print(f"   注意力头数: {config.num_attention_heads}")
        print(f"   层数: {config.num_hidden_layers}")

    except Exception as e:
        print(f"❌ 模型配置加载失败: {e}")
        return False

    # 检查目录大小
    print("\n6. 检查目录大小:")
    total_size = 0
    for file_path in qwen_model_dir.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    total_size_gb = total_size / (1024 ** 3)
    print(f"✅ 模型总大小: {total_size_gb:.2f}GB")

    if total_size_gb < 5 or total_size_gb > 8:
        print("⚠️  模型大小异常，预期应该在5-8GB之间")
        return False

    return True

def test_directory_permissions():
    """测试目录权限"""
    print("\n=== 测试目录权限 ===")

    test_dirs = [
        "models/base",
        "models/base/qwen2.5-3b-instruct",
        "models/checkpoints",
        "models/final"
    ]

    for dir_path in test_dirs:
        path = Path(dir_path)
        if path.exists():
            if os.access(path, os.R_OK):
                print(f"✅ {dir_path} 可读")
            else:
                print(f"❌ {dir_path} 不可读")
                return False

            if os.access(path, os.W_OK):
                print(f"✅ {dir_path} 可写")
            else:
                print(f"❌ {dir_path} 不可写")
                return False
        else:
            # 创建缺失的目录
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ 创建目录: {dir_path}")
            except Exception as e:
                print(f"❌ 无法创建目录 {dir_path}: {e}")
                return False

    return True

def show_directory_tree():
    """显示目录结构树"""
    print("\n=== 当前目录结构 ===")

    base_dir = Path("models/base")
    if not base_dir.exists():
        print("❌ models/base 目录不存在")
        return

    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return

        items = sorted(directory.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "

            if item.is_dir():
                print(f"{prefix}{current_prefix}{item.name}/")
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"{prefix}{current_prefix}{item.name} ({size_mb:.1f}MB)")

    print("models/base/")
    print_tree(base_dir)

def main():
    """主函数"""
    print("模型目录结构测试")
    print("=" * 50)

    # 显示当前目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python版本: {sys.version}")

    # 运行测试
    structure_ok = test_model_structure()
    permissions_ok = test_directory_permissions()

    # 显示目录树
    show_directory_tree()

    # 总结结果
    print("\n" + "=" * 50)
    if structure_ok and permissions_ok:
        print("🎉 所有测试通过！新的模型目录结构工作正常。")
        print("\n📝 下一步:")
        print("1. 测试训练: python scripts/train.py --model_config config/qwen_model_config.yaml")
        print("2. 添加新模型时使用: mkdir models/base/new-model-name")
        print("3. 查看文档: cat models/base/README.md")
    else:
        print("❌ 测试失败，请检查上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()