#!/usr/bin/env python3
"""
模型下载脚本
Download base model for fine-tuning
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import logging
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.logging import setup_logging

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载预训练模型")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="模型名称"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/base",
        help="模型保存目录"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/model_config_qwen.yaml",
        help="模型配置文件"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face访问token"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载"
    )

    parser.add_argument(
        "--tokenizer_only",
        action="store_true",
        help="仅下载分词器"
    )

    parser.add_argument(
        "--model_only",
        action="store_true",
        help="仅下载模型"
    )

    return parser.parse_args()

def load_config(config_file: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"配置文件不存在: {config_file}")
        return {}
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return {}

def check_model_access(model_name: str, token: Optional[str] = None) -> bool:
    """检查模型访问权限"""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo_info = api.repo_info(model_name)

        # 检查是否为私有仓库
        if hasattr(repo_info, 'private') and repo_info.private:
            if not token:
                print(f"❌ 模型 {model_name} 需要访问token")
                print("请使用 --token 参数或运行 huggingface-cli login")
                return False

        return True

    except Exception as e:
        logging.warning(f"检查模型访问权限时出错: {e}")
        return True  # 假设可以访问，让下载过程自己处理错误

def download_tokenizer(
    model_name: str,
    output_dir: Path,
    token: Optional[str] = None,
    force: bool = False
) -> bool:
    """下载分词器"""
    tokenizer_dir = output_dir / "tokenizer"

    if tokenizer_dir.exists() and not force:
        print(f"✅ 分词器已存在: {tokenizer_dir}")
        return True

    try:
        print(f"📥 下载分词器: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )

        # 保存分词器
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)

        print(f"✅ 分词器下载完成: {tokenizer_dir}")
        print(f"   词汇表大小: {len(tokenizer)}")

        return True

    except Exception as e:
        print(f"❌ 分词器下载失败: {e}")
        logging.error(f"分词器下载失败: {e}", exc_info=True)
        return False

def download_model(
    model_name: str,
    output_dir: Path,
    token: Optional[str] = None,
    force: bool = False
) -> bool:
    """下载模型"""
    model_dir = output_dir / "model"

    if model_dir.exists() and not force:
        print(f"✅ 模型已存在: {model_dir}")
        return True

    try:
        print(f"📥 下载模型: {model_name}")
        print("⚠️  注意: 模型下载可能需要较长时间...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            torch_dtype="auto",  # 自动选择数据类型
            low_cpu_mem_usage=True  # 降低内存使用
        )

        # 保存模型
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型下载完成: {model_dir}")
        print(f"   参数数量: {total_params:,}")
        print(f"   模型大小: {total_params * 4 / (1024**3):.2f} GB (FP32)")

        return True

    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        logging.error(f"模型下载失败: {e}", exc_info=True)
        return False

def create_download_info(
    model_name: str,
    output_dir: Path,
    tokenizer_success: bool,
    model_success: bool
):
    """创建下载信息文件"""
    from datetime import datetime

    download_info = {
        "model_name": model_name,
        "download_time": datetime.now().isoformat(),
        "tokenizer_downloaded": tokenizer_success,
        "model_downloaded": model_success,
        "output_directory": str(output_dir)
    }

    info_file = output_dir / "download_info.yaml"
    with open(info_file, 'w', encoding='utf-8') as f:
        yaml.dump(download_info, f, default_flow_style=False)

    print(f"📋 下载信息已保存: {info_file}")

def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    print("🤖 LLM模型下载工具")
    print("=" * 50)

    # 从配置文件获取模型名称（如果使用默认值）
    if args.model_name == "Qwen/Qwen2.5-3B-Instruct":  # 默认值
        config = load_config(args.config_file)
        if config and 'model' in config and 'name' in config['model']:
            args.model_name = config['model']['name']

    print(f"📦 模型: {args.model_name}")
    print(f"📁 输出目录: {args.output_dir}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查访问权限
    if not check_model_access(args.model_name, args.token):
        return 1

    # 确定下载内容
    download_tokenizer_flag = not args.model_only
    download_model_flag = not args.tokenizer_only

    success_count = 0
    total_tasks = sum([download_tokenizer_flag, download_model_flag])

    tokenizer_success = True
    model_success = True

    try:
        # 下载分词器
        if download_tokenizer_flag:
            print("\\n" + "="*30)
            print("📝 下载分词器")
            print("="*30)

            tokenizer_success = download_tokenizer(
                args.model_name,
                output_dir,
                args.token,
                args.force
            )

            if tokenizer_success:
                success_count += 1

        # 下载模型
        if download_model_flag:
            print("\\n" + "="*30)
            print("🧠 下载模型")
            print("="*30)

            model_success = download_model(
                args.model_name,
                output_dir,
                args.token,
                args.force
            )

            if model_success:
                success_count += 1

        # 创建下载信息
        create_download_info(
            args.model_name,
            output_dir,
            tokenizer_success,
            model_success
        )

        # 总结
        print("\\n" + "="*50)
        if success_count == total_tasks:
            print("🎉 所有下载任务完成！")
            print(f"📁 文件位置: {output_dir}")

            # 显示目录结构
            print("\\n📂 目录结构:")
            for item in sorted(output_dir.rglob("*")):
                if item.is_file():
                    relative_path = item.relative_to(output_dir)
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"   {relative_path} ({size_mb:.1f} MB)")

            return 0
        else:
            print(f"⚠️  部分任务失败: {success_count}/{total_tasks}")
            return 1

    except KeyboardInterrupt:
        print("\\n⏸️  下载被用户中断")
        return 0

    except Exception as e:
        print(f"\\n❌ 下载过程中出错: {e}")
        logger.error(f"下载过程中出错: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)