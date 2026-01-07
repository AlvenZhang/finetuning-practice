#!/usr/bin/env python3
"""
数据下载和预处理脚本
Download and preprocess Alpaca dataset for instruction tuning
"""

import os
import json
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, List, Any
import argparse
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaDataProcessor:
    """Alpaca数据集下载和处理器"""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """初始化数据处理器"""
        self.config = self.load_config(config_path)
        self.data_config = self.config['data']
        self.tokenizer = None

        # 创建必要目录
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"无法加载配置文件 {config_path}: {e}")
            # 返回默认配置
            return {
                'data': {
                    'dataset_name': 'tatsu-lab/alpaca',
                    'max_length': 512,
                    'validation_split': 0.1,
                    'test_split': 0.1,
                    'instruction_template': "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n{output}",
                    'prompt_template': "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n"
                },
                'model': {
                    'name': 'meta-llama/Llama-3.2-3B-Instruct'
                }
            }

    def download_dataset(self) -> None:
        """加载Alpaca数据集（从本地parquet文件或HuggingFace）"""
        logger.info("开始加载Alpaca数据集...")

        # 检查是否存在本地parquet文件
        parquet_file = self.raw_dir / "train-00000-of-00001-a09b74b3ef9c3b56.parquet"

        try:
            if parquet_file.exists():
                logger.info(f"发现本地parquet文件，直接加载: {parquet_file}")

                # 读取parquet文件
                import pandas as pd
                df = pd.read_parquet(parquet_file)

                # 转换为项目期望的格式
                data_list = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="处理本地数据"):
                    data_list.append({
                        'instruction': row.get('instruction', ''),
                        'input': row.get('input', ''),
                        'output': row.get('output', '')
                    })

                logger.info(f"从本地parquet文件加载了 {len(data_list)} 条样本")

            else:
                logger.info("未找到本地parquet文件，从HuggingFace下载...")

                # 下载数据集
                dataset = load_dataset(self.data_config['dataset_name'])

                # 转换为更易处理的格式
                data_list = []
                for item in tqdm(dataset['train'], desc="处理下载数据"):
                    data_list.append({
                        'instruction': item.get('instruction', ''),
                        'input': item.get('input', ''),
                        'output': item.get('output', '')
                    })

                logger.info(f"从HuggingFace下载了 {len(data_list)} 条样本")

            # 保存原始数据为JSON（如果不存在）
            raw_file = self.raw_dir / "alpaca_raw.json"
            if not raw_file.exists():
                with open(raw_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)
                logger.info(f"原始数据已保存到: {raw_file}")

            logger.info(f"数据集大小: {len(data_list)} 条样本")
            return data_list

        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise

    def load_tokenizer(self) -> None:
        """加载分词器"""
        try:
            model_name = self.config['model']['name']
            logger.info(f"加载分词器: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("分词器加载成功")

        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise

    def format_instruction(self, instruction: str, input_text: str, output: str = "") -> str:
        """格式化指令数据"""
        template = self.data_config['instruction_template']

        return template.format(
            instruction=instruction,
            input=input_text if input_text else "",
            output=output
        )

    def format_prompt(self, instruction: str, input_text: str) -> str:
        """格式化提示（用于推理）"""
        template = self.data_config['prompt_template']

        return template.format(
            instruction=instruction,
            input=input_text if input_text else ""
        )

    def process_data(self, data_list: List[Dict[str, str]]) -> Dict[str, List[Dict]]:
        """处理和分割数据"""
        logger.info("开始处理数据...")

        if self.tokenizer is None:
            self.load_tokenizer()

        processed_data = []
        max_length = self.data_config['max_length']

        for item in tqdm(data_list, desc="格式化数据"):
            instruction = item['instruction']
            input_text = item['input']
            output = item['output']

            # 格式化完整文本（用于训练）
            full_text = self.format_instruction(instruction, input_text, output)
            prompt_text = self.format_prompt(instruction, input_text)

            # 分词并检查长度
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

            # 跳过过长的样本
            if len(full_tokens) > max_length:
                continue

            processed_data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output,
                'full_text': full_text,
                'prompt_text': prompt_text,
                'full_tokens_length': len(full_tokens),
                'prompt_tokens_length': len(prompt_tokens)
            })

        logger.info(f"处理完成，保留 {len(processed_data)} 条有效样本")

        # 数据集分割
        val_split = self.data_config['validation_split']
        test_split = self.data_config['test_split']

        total_size = len(processed_data)
        val_size = int(total_size * val_split)
        test_size = int(total_size * test_split)
        train_size = total_size - val_size - test_size

        # 随机打乱并分割
        import random
        random.seed(42)  # 确保可复现
        random.shuffle(processed_data)

        splits = {
            'train': processed_data[:train_size],
            'validation': processed_data[train_size:train_size + val_size],
            'test': processed_data[train_size + val_size:]
        }

        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(splits['train'])} 条")
        logger.info(f"  验证集: {len(splits['validation'])} 条")
        logger.info(f"  测试集: {len(splits['test'])} 条")

        return splits

    def save_processed_data(self, splits: Dict[str, List[Dict]]) -> None:
        """保存处理后的数据"""
        logger.info("保存处理后的数据...")

        for split_name, data in splits.items():
            # 保存JSON格式
            json_file = self.processed_dir / f"alpaca_{split_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 保存CSV格式（可选）
            csv_file = self.processed_dir / f"alpaca_{split_name}.csv"
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False, encoding='utf-8')

            logger.info(f"{split_name}集已保存到: {json_file}")

    def generate_statistics(self, splits: Dict[str, List[Dict]]) -> None:
        """生成数据统计信息"""
        logger.info("生成数据统计信息...")

        stats = {
            'dataset_info': {
                'source': self.data_config['dataset_name'],
                'total_samples': sum(len(data) for data in splits.values()),
                'max_length': self.data_config['max_length']
            },
            'splits': {}
        }

        for split_name, data in splits.items():
            lengths = [item['full_tokens_length'] for item in data]

            split_stats = {
                'count': len(data),
                'avg_length': sum(lengths) / len(lengths) if lengths else 0,
                'min_length': min(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0
            }

            stats['splits'][split_name] = split_stats

        # 保存统计信息
        stats_file = self.processed_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"统计信息已保存到: {stats_file}")

        # 打印统计信息
        print("\\n=== 数据集统计信息 ===")
        print(f"数据源: {stats['dataset_info']['source']}")
        print(f"总样本数: {stats['dataset_info']['total_samples']}")
        print(f"最大长度限制: {stats['dataset_info']['max_length']}")
        print("\\n各分割统计:")
        for split_name, split_stats in stats['splits'].items():
            print(f"  {split_name}:")
            print(f"    样本数: {split_stats['count']}")
            print(f"    平均长度: {split_stats['avg_length']:.1f}")
            print(f"    长度范围: {split_stats['min_length']}-{split_stats['max_length']}")

    def run(self) -> None:
        """运行完整的数据处理流程"""
        try:
            # 1. 下载数据集
            data_list = self.download_dataset()

            # 2. 处理和分割数据
            splits = self.process_data(data_list)

            # 3. 保存处理后的数据
            self.save_processed_data(splits)

            # 4. 生成统计信息
            self.generate_statistics(splits)

            logger.info("数据处理流程完成！")

        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载和处理Alpaca数据集")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config_qwen.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载和处理数据"
    )

    args = parser.parse_args()

    # 检查是否已经存在处理后的数据
    processed_dir = Path("data/processed")
    if not args.force and processed_dir.exists():
        existing_files = list(processed_dir.glob("alpaca_*.json"))
        if len(existing_files) >= 3:  # train, val, test
            print("检测到已存在的处理后数据。使用 --force 参数强制重新处理。")
            return

    # 运行数据处理器
    processor = AlpacaDataProcessor(args.config)
    processor.run()

if __name__ == "__main__":
    main()