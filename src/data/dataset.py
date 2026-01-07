"""
数据集类定义
Dataset classes for instruction tuning format
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class InstructionDataset(Dataset):
    """
    指令微调数据集类 - 已优化用于速度和内存效率

    速度优化特性:
    - 激进的文本长度限制（指令150字符，输入80字符，输出100字符）
    - 简化的模板格式（减少token数量）
    - 更高效的文本处理逻辑
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None,
        response_template: Optional[str] = None
    ):
        """
        初始化指令数据集

        Args:
            data_path: 数据文件路径 (JSON格式)
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示模板
            response_template: 响应模板
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 默认模板
        self.prompt_template = prompt_template or "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n"
        self.response_template = response_template or "{output}"

        # 加载数据
        self.data = self._load_data()

        logger.info(f"加载了 {len(self.data)} 条训练样本")

    def _load_data(self) -> List[Dict]:
        """加载JSON数据文件"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("数据文件应包含样本列表")

            return data

        except Exception as e:
            logger.error(f"加载数据文件失败 {self.data_path}: {e}")
            raise

    def _format_prompt(self, instruction: str, input_text: str = "") -> str:
        """格式化提示文本"""
        return self.prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else ""
        )

    def _format_response(self, output: str) -> str:
        """格式化响应文本"""
        return self.response_template.format(output=output)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个训练样本 - 速度优化版本"""
        item = self.data[idx]

        # 获取字段并进行激进的长度限制（速度优化）
        instruction = item.get('instruction', '').strip()[:150]  # 限制指令长度至150字符
        input_text = item.get('input', '').strip()[:80]          # 限制输入长度至80字符
        output = item.get('output', '').strip()[:100]            # 限制输出长度至100字符

        # 使用简化模板格式，减少token数量（速度优化）
        if input_text:
            prompt = f"指令: {instruction}\n输入: {input_text}\n回答: "
            full_text = f"{prompt}{output}"
        else:
            prompt = f"指令: {instruction}\n回答: "
            full_text = f"{prompt}{output}"

        # 分词
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )

        full_tokens = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )

        # 创建标签（只对响应部分计算损失）
        prompt_length = len(prompt_tokens)
        labels = [-100] * prompt_length + full_tokens[prompt_length:]

        # 填充到最大长度
        labels += [-100] * (self.max_length - len(labels))
        full_tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(full_tokens))

        # 截断到最大长度
        labels = labels[:self.max_length]
        full_tokens = full_tokens[:self.max_length]

        return {
            'input_ids': torch.tensor(full_tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if token != self.tokenizer.pad_token_id else 0 for token in full_tokens],
                dtype=torch.long
            )
        }

class EvaluationDataset(Dataset):
    """评估数据集类"""

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ):
        """
        初始化评估数据集

        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示模板
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n"

        # 加载数据
        self.data = self._load_data()

        logger.info(f"加载了 {len(self.data)} 条评估样本")

    def _load_data(self) -> List[Dict]:
        """加载数据文件"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"加载评估数据失败 {self.data_path}: {e}")
            raise

    def _format_prompt(self, instruction: str, input_text: str = "") -> str:
        """格式化提示文本"""
        return self.prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else ""
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """获取评估样本"""
        item = self.data[idx]

        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        expected_output = item.get('output', '')

        prompt = self._format_prompt(instruction, input_text)

        # 分词提示文本
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length - 100  # 为生成留出空间
        )

        return {
            'prompt': prompt,
            'expected_output': expected_output,
            'instruction': instruction,
            'input': input_text,
            'prompt_tokens': torch.tensor(prompt_tokens, dtype=torch.long)
        }

class DataCollator:
    """数据整理器"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        # 获取批次中的所有键
        keys = batch[0].keys()

        # 为每个键创建批次张量
        batch_dict = {}
        for key in keys:
            if key in ['input_ids', 'labels', 'attention_mask']:
                # 堆叠张量
                batch_dict[key] = torch.stack([item[key] for item in batch])
            else:
                # 其他数据直接列表
                batch_dict[key] = [item[key] for item in batch]

        return batch_dict

def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    max_length: int = 512,
    num_workers: int = 0,
    **kwargs
):
    """
    创建训练和验证数据加载器

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载工作进程数
        **kwargs: 其他参数

    Returns:
        训练和验证数据加载器
    """
    from torch.utils.data import DataLoader

    # 创建数据集
    train_dataset = InstructionDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length,
        **kwargs
    )

    val_dataset = InstructionDataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        **kwargs
    )

    # 创建数据整理器
    data_collator = DataCollator(tokenizer, max_length)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=False  # M3 Pro上禁用pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=False
    )

    return train_loader, val_loader