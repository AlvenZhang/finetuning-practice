"""
核心训练逻辑
Main training loop with LoRA integration and memory optimization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
import wandb
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from tqdm import tqdm
import time
import psutil

from ..model.lora_utils import LoRAManager
from ..model.model_loader import ModelLoader
from ..utils.logging import setup_logging
from ..utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

class LoRATrainer:
    """LoRA微调训练器"""

    def __init__(
        self,
        model_config_path: str = "config/model_config.yaml",
        lora_config_path: str = "config/lora_config.yaml",
        output_dir: str = "models/checkpoints",
        experiment_name: Optional[str] = None
    ):
        """
        初始化LoRA训练器

        Args:
            model_config_path: 模型配置文件路径
            lora_config_path: LoRA配置文件路径
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.model_config_path = model_config_path
        self.lora_config_path = lora_config_path
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"lora_training_{int(time.time())}"

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.model_config = self._load_config(model_config_path)
        self.lora_config = self._load_config(lora_config_path)

        # 初始化组件
        self.model_loader = ModelLoader(model_config_path)
        self.lora_manager = LoRAManager(lora_config_path)
        self.checkpoint_manager = CheckpointManager(self.output_dir)

        # 模型和数据
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None

        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')

        # 设置日志
        setup_logging(self.output_dir / "training.log")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
            raise

    def setup_model_and_tokenizer(self) -> None:
        """按正确顺序设置模型和分词器"""
        logger.info("设置模型和分词器...")

        # 步骤1：加载基础模型和分词器
        model_name = self.model_config['model']['name']
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
            model_name
        )

        # 步骤2：应用LoRA（PEFT会正确管理参数）
        self.model = self.lora_manager.apply_lora(self.model)

        # 步骤3：准备训练（不启用梯度检查点）
        self.model = self.model_loader.prepare_model_for_training(
            self.model, use_gradient_checkpointing=False
        )

        # 步骤4：设置设备
        self._setup_device()

        # 步骤5：启用梯度检查点（在设备设置后）
        self._enable_gradient_checkpointing()

        # 步骤6：验证梯度配置（不覆盖PEFT设置）
        self._verify_gradient_configuration()

        logger.info("模型和分词器设置完成")

    def _verify_gradient_configuration(self) -> None:
        """验证模型梯度配置（信任PEFT，不强制覆盖）"""
        # 统计参数状态
        total_params = 0
        trainable_params = 0
        lora_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                trainable_params += 1
                if 'lora' in name.lower():
                    lora_params += 1

        # 验证LoRA参数存在且可训练
        if lora_params == 0:
            raise ValueError("未找到LoRA参数！检查LoRA配置。")

        if trainable_params == 0:
            raise ValueError("未找到可训练参数！")

        logger.info(f"参数验证通过 - 总数: {total_params}, 可训练: {trainable_params}, LoRA: {lora_params}")

        # 确保PEFT梯度检查点兼容性
        if hasattr(self.model, 'gradient_checkpointing') and self.model.gradient_checkpointing:
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
                logger.info("已启用PEFT输入梯度需求")

    def _enable_gradient_checkpointing(self) -> None:
        """启用梯度检查点（与PEFT兼容）"""
        training_config = self.model_config.get('training', {})

        if training_config.get('gradient_checkpointing', False):
            # 启用梯度检查点
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点")

            # 为PEFT启用输入梯度需求
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()
                logger.info("已为PEFT启用输入梯度需求")
        else:
            logger.info("梯度检查点未启用")

    def _setup_device(self) -> None:
        """设置模型设备"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("使用Apple Silicon MPS设备")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("使用CUDA设备")
        else:
            device = torch.device("cpu")
            logger.info("使用CPU设备")

        # 确保模型在正确的设备上
        if not hasattr(self.model, 'device') or self.model.device != device:
            self.model = self.model.to(device)
            logger.info(f"模型已移动到设备: {device}")

    def setup_data_loaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """设置数据加载器"""
        self.train_loader = train_loader
        self.val_loader = val_loader

        logger.info(f"训练集大小: {len(train_loader.dataset)}")
        logger.info(f"验证集大小: {len(val_loader.dataset)}")

    def setup_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, Any]:
        """设置优化器和学习率调度器"""
        training_config = self.model_config['training']

        # 确保数值参数为正确类型（修复YAML字符串问题）
        learning_rate = float(training_config['learning_rate'])
        weight_decay = float(training_config['weight_decay'])
        adam_beta1 = float(training_config['adam_beta1'])
        adam_beta2 = float(training_config['adam_beta2'])
        adam_epsilon = float(training_config['adam_epsilon'])

        # 获取LoRA优化器参数
        from ..model.lora_utils import LoRAOptimizer
        optimizer_params = LoRAOptimizer.get_optimizer_params(
            self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # 创建优化器
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay
        )

        # 计算总训练步数
        num_epochs = int(training_config['num_train_epochs'])
        gradient_accumulation_steps = int(training_config['gradient_accumulation_steps'])
        steps_per_epoch = len(self.train_loader) // gradient_accumulation_steps
        total_steps = num_epochs * steps_per_epoch

        # 创建学习率调度器
        # 优先使用warmup_steps，如果为0则使用warmup_ratio
        if 'warmup_steps' in training_config and training_config['warmup_steps'] > 0:
            warmup_steps = int(training_config['warmup_steps'])
        else:
            warmup_ratio = float(training_config.get('warmup_ratio', 0.0))
            warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"优化器设置完成，总步数: {total_steps}, 预热步数: {warmup_steps}")

        return optimizer, scheduler

    def setup_wandb(self) -> None:
        """设置Weights & Biases跟踪"""
        try:
            wandb.init(
                project="llm-finetuning",
                name=self.experiment_name,
                config={
                    **self.model_config,
                    **self.lora_config
                },
                tags=["llama-3.2-3B", "lora", "alpaca", "m3-pro"]
            )
            logger.info("Wandb初始化成功")
        except Exception as e:
            logger.warning(f"Wandb初始化失败: {e}")

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        training_config = self.model_config['training']

        total_loss = 0.0
        gradient_accumulation_steps = int(training_config['gradient_accumulation_steps'])
        max_grad_norm = float(training_config['max_grad_norm'])

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        )

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            try:
                # 获取模型设备
                device = next(self.model.parameters()).device

                # 移动数据到设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 如果使用梯度检查点，确保输入张量需要梯度
                if (hasattr(self.model, 'gradient_checkpointing') and
                    self.model.gradient_checkpointing and
                    'input_ids' in batch):
                    batch['input_ids'].requires_grad_(True)

                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss

                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"检测到无效损失值: {loss}, 跳过此步骤")
                    continue

                # 缩放损失（梯度累积）
                loss = loss / gradient_accumulation_steps

                # 反向传播
                loss.backward()

                total_loss += loss.item()

                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    # 优化器步骤
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    self.current_step += 1

                    # 更新进度条
                    avg_loss = total_loss / (step + 1)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                    # 记录到wandb
                    logging_steps = int(training_config['logging_steps'])
                    if step % logging_steps == 0:
                        self._log_metrics({
                            'train/loss': avg_loss,
                            'train/learning_rate': scheduler.get_last_lr()[0],
                            'train/step': self.current_step,
                            'train/memory_usage': self._get_memory_usage()
                        })

                    # 检查点保存
                    save_steps = int(training_config['save_steps'])
                    if (self.current_step % save_steps == 0 and
                        self.current_step > 0):
                        self._save_checkpoint(optimizer, scheduler)

                    # 验证
                    eval_steps = int(training_config['eval_steps'])
                    if (self.current_step % eval_steps == 0 and
                        self.current_step > 0):
                        val_metrics = self.evaluate()
                        self._log_metrics(val_metrics)

                        # 保存最佳模型
                        if val_metrics['eval/loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['eval/loss']
                            self._save_best_model()

                        self.model.train()  # 返回训练模式

            except Exception as e:
                logger.error(f"训练步骤 {step} 出错: {e}")
                continue

        avg_loss = total_loss / len(self.train_loader)
        return {'train/epoch_loss': avg_loss}

    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                try:
                    # 获取模型设备
                    device = next(self.model.parameters()).device

                    # 移动数据到设备
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # 前向传播
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    total_loss += loss.item()
                    total_samples += 1

                except Exception as e:
                    logger.warning(f"评估步骤出错: {e}")
                    continue

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

        return {
            'eval/loss': avg_loss,
            'eval/perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }

    def train(self) -> None:
        """主训练循环"""
        logger.info("开始训练...")

        # 设置优化器和调度器
        optimizer, scheduler = self.setup_optimizer_and_scheduler()

        # 设置wandb
        self.setup_wandb()

        training_config = self.model_config['training']
        num_epochs = int(training_config['num_train_epochs'])

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch

                logger.info(f"开始第 {epoch + 1}/{num_epochs} 个epoch")

                # 训练一个epoch
                train_metrics = self.train_epoch(optimizer, scheduler, epoch)

                # 记录epoch指标
                self._log_metrics({
                    **train_metrics,
                    'train/epoch': epoch
                })

                # Epoch结束时评估
                val_metrics = self.evaluate()
                self._log_metrics(val_metrics)

                # 保存epoch检查点
                self._save_checkpoint(optimizer, scheduler, is_epoch_end=True)

                logger.info(f"Epoch {epoch + 1} 完成 - "
                          f"训练损失: {train_metrics['train/epoch_loss']:.4f}, "
                          f"验证损失: {val_metrics['eval/loss']:.4f}")

            # 训练完成
            logger.info("训练完成！")

            # 保存最终模型
            self._save_final_model()

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            self._save_checkpoint(optimizer, scheduler, is_interrupted=True)

        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise

        finally:
            # 清理
            if wandb.run:
                wandb.finish()

    def _save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        is_epoch_end: bool = False,
        is_interrupted: bool = False
    ) -> None:
        """保存检查点"""
        try:
            checkpoint_name = f"checkpoint-{self.current_step}"
            if is_epoch_end:
                checkpoint_name = f"checkpoint-epoch-{self.current_epoch}"
            if is_interrupted:
                checkpoint_name = "checkpoint-interrupted"

            checkpoint_dir = self.output_dir / checkpoint_name

            # 保存LoRA权重
            self.lora_manager.save_lora_weights(self.model, checkpoint_dir)

            # 保存训练状态
            training_state = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'best_val_loss': self.best_val_loss,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'model_config': self.model_config,
                'lora_config': self.lora_config
            }

            with open(checkpoint_dir / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2, default=str)

            logger.info(f"检查点已保存: {checkpoint_dir}")

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def _save_best_model(self) -> None:
        """保存最佳模型"""
        try:
            best_model_dir = self.output_dir / "best_model"
            self.lora_manager.save_lora_weights(self.model, best_model_dir)
            logger.info(f"最佳模型已保存: {best_model_dir}")
        except Exception as e:
            logger.error(f"保存最佳模型失败: {e}")

    def _save_final_model(self) -> None:
        """保存最终模型"""
        try:
            final_model_dir = self.output_dir / "final_model"
            self.lora_manager.save_lora_weights(self.model, final_model_dir)

            # 可选：合并LoRA权重
            merged_model_dir = self.output_dir / "merged_model"
            self.lora_manager.merge_and_save(self.model, merged_model_dir)

            logger.info(f"最终模型已保存: {final_model_dir}")
            logger.info(f"合并模型已保存: {merged_model_dir}")

        except Exception as e:
            logger.error(f"保存最终模型失败: {e}")

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """记录指标"""
        try:
            if wandb.run:
                wandb.log(metrics)
        except Exception as e:
            logger.warning(f"记录wandb指标失败: {e}")

        # 本地日志
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")

    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """从检查点恢复训练"""
        try:
            checkpoint_path = Path(checkpoint_path)

            # 加载训练状态
            with open(checkpoint_path / "training_state.json", 'r') as f:
                training_state = json.load(f)

            self.current_epoch = training_state['epoch']
            self.current_step = training_state['step']
            self.best_val_loss = training_state['best_val_loss']

            # 加载LoRA权重
            self.model = self.lora_manager.load_lora_weights(
                self.model, checkpoint_path
            )

            logger.info(f"从检查点恢复: {checkpoint_path}")

        except Exception as e:
            logger.error(f"恢复检查点失败: {e}")
            raise