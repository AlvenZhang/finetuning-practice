"""
模型加载工具
Model loading utilities for LLM fine-tuning
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel
from typing import Optional, Dict, Any, Tuple, Union
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ModelLoader:
    """模型加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型加载器

        Args:
            config_path: 模型配置文件路径
        """
        self.config_path = config_path
        self.model_config = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载模型配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.model_config = config
            logger.info(f"模型配置加载成功: {config_path}")

            return config

        except Exception as e:
            logger.error(f"加载模型配置失败 {config_path}: {e}")
            # 使用默认配置
            self.model_config = self.get_default_config()
            return self.model_config

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认模型配置"""
        return {
            'model': {
                'name': 'meta-llama/Llama-3.2-3B-Instruct',
                'torch_dtype': 'float16',
                'device_map': 'auto',
                'trust_remote_code': True,
                'use_cache': False
            },
            'tokenizer': {
                'name': 'meta-llama/Llama-3.2-3B-Instruct',
                'padding_side': 'left',
                'truncation_side': 'left',
                'add_eos_token': True,
                'add_bos_token': True
            }
        }

    def load_tokenizer(
        self,
        model_name: Optional[str] = None,
        **kwargs
    ) -> PreTrainedTokenizer:
        """
        加载分词器

        Args:
            model_name: 模型名称
            **kwargs: 其他参数

        Returns:
            分词器
        """
        try:
            # 获取模型名称
            if model_name is None:
                if self.model_config is None:
                    self.model_config = self.get_default_config()
                model_name = self.model_config['tokenizer']['name']

            logger.info(f"加载分词器: {model_name}")

            # 合并配置
            tokenizer_config = {}
            if self.model_config:
                tokenizer_config.update(self.model_config.get('tokenizer', {}))
            tokenizer_config.update(kwargs)

            # 移除非AutoTokenizer参数
            auto_tokenizer_kwargs = {
                k: v for k, v in tokenizer_config.items()
                if k in ['trust_remote_code', 'revision', 'use_fast', 'cache_dir']
            }

            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **auto_tokenizer_kwargs
            )

            # 设置特殊配置
            if tokenizer_config.get('padding_side'):
                tokenizer.padding_side = tokenizer_config['padding_side']

            if tokenizer_config.get('truncation_side'):
                tokenizer.truncation_side = tokenizer_config['truncation_side']

            # 设置pad_token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})

            # 添加特殊token
            if tokenizer_config.get('add_eos_token') and tokenizer.eos_token:
                if tokenizer.eos_token not in tokenizer.get_vocab():
                    tokenizer.add_special_tokens({'eos_token': tokenizer.eos_token})

            if tokenizer_config.get('add_bos_token') and tokenizer.bos_token:
                if tokenizer.bos_token not in tokenizer.get_vocab():
                    tokenizer.add_special_tokens({'bos_token': tokenizer.bos_token})

            logger.info(f"分词器加载成功，词汇表大小: {len(tokenizer)}")

            return tokenizer

        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise

    def load_model(
        self,
        model_name: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ) -> PreTrainedModel:
        """
        加载预训练模型

        Args:
            model_name: 模型名称
            load_in_4bit: 是否使用4bit量化
            load_in_8bit: 是否使用8bit量化
            **kwargs: 其他参数

        Returns:
            预训练模型
        """
        try:
            # 获取模型名称
            if model_name is None:
                if self.model_config is None:
                    self.model_config = self.get_default_config()
                model_name = self.model_config['model']['name']

            logger.info(f"加载模型: {model_name}")

            # 合并配置
            model_config = {}
            if self.model_config:
                model_config.update(self.model_config.get('model', {}))
            model_config.update(kwargs)

            # 移除不是AutoModelForCausalLM参数的字段
            invalid_keys = ['name', 'model_type']
            for key in invalid_keys:
                model_config.pop(key, None)

            # 处理torch_dtype
            if 'torch_dtype' in model_config:
                dtype_str = model_config['torch_dtype']
                if isinstance(dtype_str, str):
                    dtype_mapping = {
                        'float16': torch.float16,
                        'float32': torch.float32,
                        'bfloat16': torch.bfloat16,
                        'auto': 'auto'
                    }
                    model_config['torch_dtype'] = dtype_mapping.get(dtype_str, torch.float16)

            # 量化配置
            quantization_config = None
            if load_in_4bit or load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_config['quantization_config'] = quantization_config

            # Apple Silicon优化
            if torch.backends.mps.is_available():
                logger.info("检测到Apple Silicon，启用MPS优化")
                # 对于M系列芯片，使用CPU或MPS
                if 'device_map' in model_config and model_config['device_map'] == 'auto':
                    model_config['device_map'] = None  # 让PyTorch自动选择

            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_config
            )

            # 禁用缓存（训练时）
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False

            logger.info("模型加载成功")

            # 打印模型信息
            self._print_model_info(model)

            return model

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def load_model_and_tokenizer(
        self,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        同时加载模型和分词器

        Args:
            model_name: 模型名称
            **kwargs: 其他参数

        Returns:
            (模型, 分词器)元组
        """
        tokenizer = self.load_tokenizer(model_name, **kwargs)
        model = self.load_model(model_name, **kwargs)

        # 调整模型词汇表大小以匹配分词器
        if len(tokenizer) != model.config.vocab_size:
            logger.info(f"调整模型词汇表大小: {model.config.vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def load_peft_model(
        self,
        base_model_name: str,
        peft_model_path: Union[str, Path],
        **kwargs
    ) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """
        加载PEFT微调模型

        Args:
            base_model_name: 基础模型名称
            peft_model_path: PEFT模型路径
            **kwargs: 其他参数

        Returns:
            (PEFT模型, 分词器)元组
        """
        try:
            logger.info(f"加载PEFT模型: {peft_model_path}")

            # 加载基础模型和分词器
            base_model, tokenizer = self.load_model_and_tokenizer(
                base_model_name, **kwargs
            )

            # 加载PEFT模型
            peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

            logger.info("PEFT模型加载成功")

            return peft_model, tokenizer

        except Exception as e:
            logger.error(f"加载PEFT模型失败: {e}")
            raise

    def _print_model_info(self, model: PreTrainedModel) -> None:
        """打印模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\\n=== 模型信息 ===")
        print(f"模型类型: {type(model).__name__}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        if total_params > 0:
            print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")

        # 内存使用估算
        param_size = total_params * 4  # 假设float32
        print(f"模型大小估算: {param_size / 1024**3:.2f} GB (FP32)")
        print(f"模型大小估算: {param_size / 2 / 1024**3:.2f} GB (FP16)")

    def prepare_model_for_training(
        self,
        model: PreTrainedModel,
        use_gradient_checkpointing: bool = True
    ) -> PreTrainedModel:
        """
        为训练准备模型（不处理梯度检查点，由trainer负责）

        Args:
            model: 预训练模型
            use_gradient_checkpointing: 是否使用梯度检查点（兼容性参数，实际由trainer处理）

        Returns:
            准备好的模型
        """
        # 注意：梯度检查点的启用现在由trainer负责，确保与PEFT兼容
        if use_gradient_checkpointing:
            logger.info("梯度检查点将由trainer启用以确保PEFT兼容性")

        # 确保模型处于训练模式
        model.train()

        # 禁用缓存
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

        return model

def load_model_for_training(
    model_name: str,
    config_path: Optional[str] = None,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    便捷函数：加载用于训练的模型

    Args:
        model_name: 模型名称
        config_path: 配置文件路径
        **kwargs: 其他参数

    Returns:
        (模型, 分词器)元组
    """
    loader = ModelLoader(config_path)
    model, tokenizer = loader.load_model_and_tokenizer(model_name, **kwargs)
    model = loader.prepare_model_for_training(model)

    return model, tokenizer

def load_model_for_inference(
    model_name: str,
    peft_path: Optional[Union[str, Path]] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizer]:
    """
    便捷函数：加载用于推理的模型

    Args:
        model_name: 模型名称
        peft_path: PEFT模型路径（可选）
        config_path: 配置文件路径
        **kwargs: 其他参数

    Returns:
        (模型, 分词器)元组
    """
    loader = ModelLoader(config_path)

    if peft_path:
        model, tokenizer = loader.load_peft_model(model_name, peft_path, **kwargs)
    else:
        model, tokenizer = loader.load_model_and_tokenizer(model_name, **kwargs)

    # 设置推理模式
    model.eval()

    return model, tokenizer