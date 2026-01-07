"""
LoRA (Low-Rank Adaptation) 实现和配置管理
LoRA utilities for parameter-efficient fine-tuning
"""

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
from transformers import PreTrainedModel
from typing import Dict, List, Optional, Union, Any
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LoRAManager:
    """LoRA配置和管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化LoRA管理器

        Args:
            config_path: LoRA配置文件路径
        """
        self.config_path = config_path
        self.lora_config = None
        self.peft_config = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载LoRA配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.lora_config = config.get('lora', {})
            logger.info(f"LoRA配置加载成功: {config_path}")

            return config

        except Exception as e:
            logger.error(f"加载LoRA配置失败 {config_path}: {e}")
            # 使用默认配置
            self.lora_config = self.get_default_config()
            return {'lora': self.lora_config}

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认LoRA配置"""
        return {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            'bias': 'none',
            'task_type': 'CAUSAL_LM',
            'inference_mode': False,
            'fan_in_fan_out': False,
            'init_lora_weights': True
        }

    def create_peft_config(self) -> LoraConfig:
        """创建PEFT LoRA配置"""
        if self.lora_config is None:
            self.lora_config = self.get_default_config()

        # 转换task_type字符串为枚举
        task_type_mapping = {
            'CAUSAL_LM': TaskType.CAUSAL_LM,
            'SEQ_CLS': TaskType.SEQ_CLS,
            'TOKEN_CLS': TaskType.TOKEN_CLS,
            'SEQ_2_SEQ_LM': TaskType.SEQ_2_SEQ_LM
        }

        task_type = task_type_mapping.get(
            self.lora_config.get('task_type', 'CAUSAL_LM'),
            TaskType.CAUSAL_LM
        )

        self.peft_config = LoraConfig(
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('lora_alpha', 32),
            lora_dropout=self.lora_config.get('lora_dropout', 0.1),
            target_modules=self.lora_config.get('target_modules', [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]),
            bias=self.lora_config.get('bias', 'none'),
            task_type=task_type,
            inference_mode=self.lora_config.get('inference_mode', False),
            fan_in_fan_out=self.lora_config.get('fan_in_fan_out', False),
            init_lora_weights=self.lora_config.get('init_lora_weights', True)
        )

        return self.peft_config

    def apply_lora(self, model: PreTrainedModel) -> PeftModel:
        """将LoRA应用到模型"""
        try:
            if self.peft_config is None:
                self.create_peft_config()

            # 应用LoRA
            peft_model = get_peft_model(model, self.peft_config)

            # 打印模型信息
            self.print_model_info(peft_model)

            logger.info("LoRA应用成功")
            return peft_model

        except Exception as e:
            logger.error(f"应用LoRA失败: {e}")
            raise

    def print_model_info(self, model: Union[PreTrainedModel, PeftModel]) -> None:
        """打印模型参数信息"""
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            # 手动计算参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")

    def save_lora_weights(self, model: PeftModel, save_path: Union[str, Path]) -> None:
        """保存LoRA权重"""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            # 保存LoRA权重
            model.save_pretrained(save_path)

            # 保存配置
            config_path = save_path / "lora_config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump({'lora': self.lora_config}, f, default_flow_style=False)

            logger.info(f"LoRA权重已保存到: {save_path}")

        except Exception as e:
            logger.error(f"保存LoRA权重失败: {e}")
            raise

    def load_lora_weights(
        self,
        base_model: PreTrainedModel,
        lora_path: Union[str, Path]
    ) -> PeftModel:
        """加载LoRA权重"""
        try:
            lora_path = Path(lora_path)

            # 加载PEFT模型
            peft_model = PeftModel.from_pretrained(base_model, lora_path)

            logger.info(f"LoRA权重加载成功: {lora_path}")
            return peft_model

        except Exception as e:
            logger.error(f"加载LoRA权重失败: {e}")
            raise

    def merge_and_save(
        self,
        peft_model: PeftModel,
        save_path: Union[str, Path]
    ) -> None:
        """合并LoRA权重到基础模型并保存"""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            # 合并LoRA权重
            merged_model = peft_model.merge_and_unload()

            # 保存合并后的模型
            merged_model.save_pretrained(save_path)

            logger.info(f"合并后的模型已保存到: {save_path}")

        except Exception as e:
            logger.error(f"合并和保存模型失败: {e}")
            raise

class LoRAOptimizer:
    """LoRA优化器配置"""

    @staticmethod
    def get_optimizer_params(
        model: PeftModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        获取LoRA优化器参数配置

        Args:
            model: PEFT模型
            learning_rate: 学习率
            weight_decay: 权重衰减

        Returns:
            优化器参数组
        """
        # 分离LoRA参数和其他参数
        lora_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                else:
                    other_params.append(param)

        # 为LoRA参数使用不同的学习率
        optimizer_params = [
            {
                'params': lora_params,
                'lr': learning_rate,
                'weight_decay': weight_decay
            }
        ]

        # 如果有其他可训练参数，使用较小的学习率
        if other_params:
            optimizer_params.append({
                'params': other_params,
                'lr': learning_rate * 0.1,
                'weight_decay': weight_decay
            })

        return optimizer_params

class LoRAAnalyzer:
    """LoRA分析工具"""

    @staticmethod
    def analyze_model_parameters(model: Union[PreTrainedModel, PeftModel]) -> Dict[str, Any]:
        """分析模型参数分布"""
        analysis = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'frozen_parameters': 0,
            'lora_parameters': 0,
            'parameter_groups': {}
        }

        for name, param in model.named_parameters():
            param_count = param.numel()
            analysis['total_parameters'] += param_count

            if param.requires_grad:
                analysis['trainable_parameters'] += param_count
                if 'lora' in name.lower():
                    analysis['lora_parameters'] += param_count
            else:
                analysis['frozen_parameters'] += param_count

            # 按模块分组
            module_name = name.split('.')[0] if '.' in name else name
            if module_name not in analysis['parameter_groups']:
                analysis['parameter_groups'][module_name] = {
                    'total': 0,
                    'trainable': 0,
                    'lora': 0
                }

            analysis['parameter_groups'][module_name]['total'] += param_count
            if param.requires_grad:
                analysis['parameter_groups'][module_name]['trainable'] += param_count
                if 'lora' in name.lower():
                    analysis['parameter_groups'][module_name]['lora'] += param_count

        # 计算比例
        total = analysis['total_parameters']
        if total > 0:
            analysis['trainable_percentage'] = analysis['trainable_parameters'] / total * 100
            analysis['lora_percentage'] = analysis['lora_parameters'] / total * 100

        return analysis

    @staticmethod
    def print_analysis(analysis: Dict[str, Any]) -> None:
        """打印分析结果"""
        print("\\n=== LoRA模型参数分析 ===")
        print(f"总参数量: {analysis['total_parameters']:,}")
        print(f"可训练参数量: {analysis['trainable_parameters']:,} ({analysis.get('trainable_percentage', 0):.2f}%)")
        print(f"冻结参数量: {analysis['frozen_parameters']:,}")
        print(f"LoRA参数量: {analysis['lora_parameters']:,} ({analysis.get('lora_percentage', 0):.4f}%)")

        print("\\n=== 模块参数分布 ===")
        for module, stats in analysis['parameter_groups'].items():
            if stats['trainable'] > 0:
                print(f"{module}: {stats['trainable']:,} 可训练 / {stats['total']:,} 总计")

def create_lora_model(
    base_model: PreTrainedModel,
    config_path: Optional[str] = None,
    **lora_kwargs
) -> PeftModel:
    """
    便捷函数：创建LoRA模型

    Args:
        base_model: 基础模型
        config_path: LoRA配置文件路径
        **lora_kwargs: LoRA参数覆盖

    Returns:
        PEFT模型
    """
    manager = LoRAManager(config_path)

    # 应用参数覆盖
    if lora_kwargs:
        if manager.lora_config is None:
            manager.lora_config = manager.get_default_config()
        manager.lora_config.update(lora_kwargs)

    return manager.apply_lora(base_model)