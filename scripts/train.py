#!/usr/bin/env python3
"""
主训练脚本
Main training script for LLM fine-tuning with LoRA
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import logging
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import LoRATrainer
from src.model.model_loader import ModelLoader
from src.data.dataset import create_dataloaders
from src.utils.logging import setup_logging, MemoryLogger

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM LoRA微调训练脚本")

    # 基础参数
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/qwen_model_config.yaml",
        help="模型配置文件路径"
    )

    parser.add_argument(
        "--lora_config",
        type=str,
        default="config/qwen_lora_config.yaml",
        help="LoRA配置文件路径"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="输出目录"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="实验名称"
    )

    # 数据参数
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/processed/alpaca_train.json",
        help="训练数据路径"
    )

    parser.add_argument(
        "--val_data",
        type=str,
        default="data/processed/alpaca_validation.json",
        help="验证数据路径"
    )

    # 训练参数覆盖
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批次大小"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="训练轮次"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="最大序列长度"
    )

    # LoRA参数覆盖
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank"
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha"
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=None,
        help="LoRA dropout"
    )

    # 其他参数
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="禁用Weights & Biases跟踪"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )

    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="强制使用CPU"
    )

    return parser.parse_args()

def load_and_override_config(config_path: str, overrides: dict) -> dict:
    """加载配置并应用覆盖"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 应用覆盖
    for key, value in overrides.items():
        if value is not None:
            # 支持嵌套键 (如 training.learning_rate)
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    return config

def check_data_files(train_path: str, val_path: str) -> bool:
    """检查数据文件是否存在"""
    train_exists = Path(train_path).exists()
    val_exists = Path(val_path).exists()

    if not train_exists:
        print(f"❌ 训练数据文件不存在: {train_path}")
        print("请先运行: python data/download_data.py")

    if not val_exists:
        print(f"❌ 验证数据文件不存在: {val_path}")
        print("请先运行: python data/download_data.py")

    return train_exists and val_exists

def setup_device(force_cpu: bool = False) -> str:
    """设置计算设备"""
    if force_cpu:
        device = "cpu"
        print("🖥️  强制使用CPU")
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name()
        print(f"🚀 使用GPU: {gpu_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 使用Apple Silicon MPS")
    else:
        device = "cpu"
        print("🖥️  使用CPU")

    return device

def print_system_info():
    """打印系统信息"""
    import psutil
    import platform

    print("\\n=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f} GB)")

    if torch.backends.mps.is_available():
        print("Apple Silicon MPS: 可用")

    print(f"PyTorch版本: {torch.__version__}")

def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志级别
    log_level = "DEBUG" if args.debug else "INFO"

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    setup_logging(
        log_file=output_dir / "training.log",
        level=log_level
    )

    logger = logging.getLogger(__name__)
    memory_logger = MemoryLogger()

    logger.info("开始LLM LoRA微调训练")

    # 打印系统信息
    print_system_info()

    # 设置设备
    device = setup_device(args.force_cpu)

    # 检查数据文件
    if not check_data_files(args.train_data, args.val_data):
        return 1

    try:
        # 记录初始内存
        memory_logger.log_memory_usage("开始")

        # 准备配置覆盖
        config_overrides = {}

        # 训练参数覆盖
        if args.learning_rate is not None:
            config_overrides['training.learning_rate'] = args.learning_rate
        if args.batch_size is not None:
            config_overrides['training.per_device_train_batch_size'] = args.batch_size
            config_overrides['training.per_device_eval_batch_size'] = args.batch_size
        if args.num_epochs is not None:
            config_overrides['training.num_train_epochs'] = args.num_epochs
        if args.max_length is not None:
            config_overrides['data.max_length'] = args.max_length

        # 应用配置覆盖
        if config_overrides:
            logger.info(f"应用配置覆盖: {config_overrides}")

        # LoRA参数覆盖
        lora_overrides = {}
        if args.lora_r is not None:
            lora_overrides['lora.r'] = args.lora_r
        if args.lora_alpha is not None:
            lora_overrides['lora.lora_alpha'] = args.lora_alpha
        if args.lora_dropout is not None:
            lora_overrides['lora.lora_dropout'] = args.lora_dropout

        # 禁用wandb（如果指定）
        if args.no_wandb:
            os.environ["WANDB_DISABLED"] = "true"

        # 初始化训练器
        logger.info("初始化训练器...")
        trainer = LoRATrainer(
            model_config_path=args.model_config,
            lora_config_path=args.lora_config,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )

        # 应用配置覆盖
        if config_overrides:
            for key, value in config_overrides.items():
                keys = key.split('.')
                current = trainer.model_config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value

        if lora_overrides:
            for key, value in lora_overrides.items():
                keys = key.split('.')
                current = trainer.lora_config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value

        # 设置模型和分词器
        logger.info("加载模型和分词器...")
        trainer.setup_model_and_tokenizer()
        memory_logger.log_memory_usage("模型加载后")

        # 创建数据加载器
        logger.info("创建数据加载器...")
        train_loader, val_loader = create_dataloaders(
            train_path=args.train_data,
            val_path=args.val_data,
            tokenizer=trainer.tokenizer,
            batch_size=trainer.model_config['training']['per_device_train_batch_size'],
            max_length=trainer.model_config['data']['max_length']
        )

        trainer.setup_data_loaders(train_loader, val_loader)
        memory_logger.log_memory_usage("数据加载器创建后")

        # 从检查点恢复（如果指定）
        if args.resume_from_checkpoint:
            logger.info(f"从检查点恢复: {args.resume_from_checkpoint}")
            trainer.resume_from_checkpoint(args.resume_from_checkpoint)

        # 开始训练
        logger.info("开始训练...")
        print("\\n🚀 开始训练...")
        print(f"📁 输出目录: {args.output_dir}")
        print(f"📊 实验名称: {trainer.experiment_name}")

        trainer.train()

        # 训练完成
        logger.info("训练完成！")
        print("\\n✅ 训练完成！")

        # 最终内存使用
        memory_logger.log_memory_usage("训练完成")

        return 0

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        print("\\n⏸️  训练被用户中断")
        return 0

    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
        print(f"\\n❌ 训练失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)