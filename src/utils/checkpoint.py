"""
检查点管理工具
Checkpoint management utilities
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """检查点管理器"""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 3,
        auto_cleanup: bool = True
    ):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点目录
            max_checkpoints: 最大保留检查点数量
            auto_cleanup: 是否自动清理旧检查点
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup

        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 检查点列表文件
        self.checkpoint_list_file = self.checkpoint_dir / "checkpoints.json"

        # 加载现有检查点列表
        self.checkpoint_list = self._load_checkpoint_list()

    def _load_checkpoint_list(self) -> List[Dict[str, Any]]:
        """加载检查点列表"""
        if self.checkpoint_list_file.exists():
            try:
                with open(self.checkpoint_list_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载检查点列表失败: {e}")

        return []

    def _save_checkpoint_list(self) -> None:
        """保存检查点列表"""
        try:
            with open(self.checkpoint_list_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_list, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存检查点列表失败: {e}")

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """
        保存检查点

        Args:
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            scheduler_state: 调度器状态字典
            epoch: 当前轮次
            step: 当前步数
            metrics: 当前指标
            checkpoint_name: 检查点名称

        Returns:
            检查点路径
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-step-{step}"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        try:
            # 创建检查点目录
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # 保存模型状态
            model_file = checkpoint_path / "pytorch_model.bin"
            torch.save(model_state, model_file)

            # 保存优化器状态
            if optimizer_state is not None:
                optimizer_file = checkpoint_path / "optimizer.bin"
                torch.save(optimizer_state, optimizer_file)

            # 保存调度器状态
            if scheduler_state is not None:
                scheduler_file = checkpoint_path / "scheduler.bin"
                torch.save(scheduler_state, scheduler_file)

            # 保存训练信息
            training_info = {
                "epoch": epoch,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {}
            }

            info_file = checkpoint_path / "training_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)

            # 更新检查点列表
            checkpoint_info = {
                "name": checkpoint_name,
                "path": str(checkpoint_path),
                "epoch": epoch,
                "step": step,
                "timestamp": training_info["timestamp"],
                "metrics": metrics or {}
            }

            self.checkpoint_list.append(checkpoint_info)
            self._save_checkpoint_list()

            logger.info(f"检查点已保存: {checkpoint_path}")

            # 自动清理
            if self.auto_cleanup:
                self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            raise

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径

        Returns:
            检查点数据
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

        try:
            checkpoint_data = {}

            # 加载模型状态
            model_file = checkpoint_path / "pytorch_model.bin"
            if model_file.exists():
                checkpoint_data["model_state"] = torch.load(model_file, map_location="cpu")

            # 加载优化器状态
            optimizer_file = checkpoint_path / "optimizer.bin"
            if optimizer_file.exists():
                checkpoint_data["optimizer_state"] = torch.load(optimizer_file, map_location="cpu")

            # 加载调度器状态
            scheduler_file = checkpoint_path / "scheduler.bin"
            if scheduler_file.exists():
                checkpoint_data["scheduler_state"] = torch.load(scheduler_file, map_location="cpu")

            # 加载训练信息
            info_file = checkpoint_path / "training_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    checkpoint_data["training_info"] = json.load(f)

            logger.info(f"检查点已加载: {checkpoint_path}")

            return checkpoint_data

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise

    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新检查点路径"""
        if not self.checkpoint_list:
            return None

        # 按步数排序，获取最新的
        latest = max(self.checkpoint_list, key=lambda x: x["step"])
        return Path(latest["path"])

    def get_best_checkpoint(self, metric_name: str = "eval_loss", lower_is_better: bool = True) -> Optional[Path]:
        """
        获取最佳检查点路径

        Args:
            metric_name: 指标名称
            lower_is_better: 是否越小越好

        Returns:
            最佳检查点路径
        """
        if not self.checkpoint_list:
            return None

        # 过滤包含指定指标的检查点
        valid_checkpoints = [
            cp for cp in self.checkpoint_list
            if metric_name in cp.get("metrics", {})
        ]

        if not valid_checkpoints:
            return None

        # 找到最佳检查点
        if lower_is_better:
            best = min(valid_checkpoints, key=lambda x: x["metrics"][metric_name])
        else:
            best = max(valid_checkpoints, key=lambda x: x["metrics"][metric_name])

        return Path(best["path"])

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        return self.checkpoint_list.copy()

    def _cleanup_old_checkpoints(self) -> None:
        """清理旧检查点"""
        if len(self.checkpoint_list) <= self.max_checkpoints:
            return

        # 按步数排序
        sorted_checkpoints = sorted(self.checkpoint_list, key=lambda x: x["step"])

        # 删除最旧的检查点
        to_remove = sorted_checkpoints[:-self.max_checkpoints]

        for checkpoint in to_remove:
            try:
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"已删除旧检查点: {checkpoint_path}")

                # 从列表中移除
                self.checkpoint_list.remove(checkpoint)

            except Exception as e:
                logger.warning(f"删除检查点失败 {checkpoint['path']}: {e}")

        # 更新检查点列表
        self._save_checkpoint_list()

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        删除指定检查点

        Args:
            checkpoint_name: 检查点名称

        Returns:
            是否删除成功
        """
        checkpoint_info = None
        for cp in self.checkpoint_list:
            if cp["name"] == checkpoint_name:
                checkpoint_info = cp
                break

        if checkpoint_info is None:
            logger.warning(f"检查点不存在: {checkpoint_name}")
            return False

        try:
            checkpoint_path = Path(checkpoint_info["path"])
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)

            # 从列表中移除
            self.checkpoint_list.remove(checkpoint_info)
            self._save_checkpoint_list()

            logger.info(f"检查点已删除: {checkpoint_name}")
            return True

        except Exception as e:
            logger.error(f"删除检查点失败: {e}")
            return False

    def export_checkpoint_summary(self) -> Dict[str, Any]:
        """导出检查点摘要"""
        summary = {
            "total_checkpoints": len(self.checkpoint_list),
            "checkpoint_dir": str(self.checkpoint_dir),
            "max_checkpoints": self.max_checkpoints,
            "checkpoints": []
        }

        for cp in sorted(self.checkpoint_list, key=lambda x: x["step"]):
            cp_summary = {
                "name": cp["name"],
                "epoch": cp["epoch"],
                "step": cp["step"],
                "timestamp": cp["timestamp"],
                "metrics": cp["metrics"]
            }
            summary["checkpoints"].append(cp_summary)

        return summary

def save_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    checkpoint_name: str,
    **kwargs
) -> Path:
    """
    便捷函数：保存模型检查点

    Args:
        model: 模型
        checkpoint_dir: 检查点目录
        checkpoint_name: 检查点名称
        **kwargs: 其他参数

    Returns:
        检查点路径
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.save_checkpoint(
        model_state=model.state_dict(),
        checkpoint_name=checkpoint_name,
        **kwargs
    )

def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path]
) -> torch.nn.Module:
    """
    便捷函数：加载模型检查点

    Args:
        model: 模型
        checkpoint_path: 检查点路径

    Returns:
        加载后的模型
    """
    manager = CheckpointManager(Path(checkpoint_path).parent)
    checkpoint_data = manager.load_checkpoint(checkpoint_path)

    if "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])

    return model