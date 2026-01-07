"""
日志配置工具
Logging configuration utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import json
from datetime import datetime

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> None:
    """
    设置日志配置

    Args:
        log_file: 日志文件路径
        level: 日志级别
        format_string: 自定义格式字符串
    """
    # 默认格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 日志级别映射
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    log_level = level_mapping.get(level.upper(), logging.INFO)

    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=[]
    )

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: Union[str, Path]):
        """
        初始化训练日志器

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建指标日志文件
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.events_file = self.log_dir / "events.jsonl"

        # 初始化日志器
        self.logger = logging.getLogger("TrainingLogger")

    def log_metrics(self, metrics: dict, step: int, epoch: int) -> None:
        """
        记录训练指标

        Args:
            metrics: 指标字典
            step: 训练步数
            epoch: 训练轮次
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics
        }

        # 写入文件
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\\n')

    def log_event(self, event_type: str, message: str, **kwargs) -> None:
        """
        记录训练事件

        Args:
            event_type: 事件类型
            message: 事件消息
            **kwargs: 其他事件数据
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            **kwargs
        }

        # 写入文件
        with open(self.events_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\\n')

        # 同时记录到标准日志
        self.logger.info(f"[{event_type}] {message}")

    def load_metrics(self) -> list:
        """加载历史指标"""
        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        return metrics

    def load_events(self) -> list:
        """加载历史事件"""
        if not self.events_file.exists():
            return []

        events = []
        with open(self.events_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        return events

class MemoryLogger:
    """内存使用监控日志器"""

    def __init__(self):
        self.logger = logging.getLogger("MemoryLogger")

    def log_memory_usage(self, stage: str = "unknown") -> None:
        """记录当前内存使用情况"""
        try:
            import psutil
            import torch

            # 系统内存
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            memory_percent = memory.percent

            # GPU内存（如果可用）
            gpu_memory_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
                gpu_memory_info = f", GPU: {gpu_memory:.2f}GB (max: {gpu_memory_max:.2f}GB)"

            # MPS内存（Apple Silicon）
            mps_memory_info = ""
            if torch.backends.mps.is_available():
                try:
                    mps_memory = torch.mps.current_allocated_memory() / (1024**3)
                    mps_memory_info = f", MPS: {mps_memory:.2f}GB"
                except:
                    pass

            self.logger.info(
                f"[{stage}] Memory usage: {memory_gb:.2f}GB ({memory_percent:.1f}%)"
                f"{gpu_memory_info}{mps_memory_info}"
            )

        except Exception as e:
            self.logger.warning(f"无法获取内存使用信息: {e}")

def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器

    Args:
        name: 日志器名称

    Returns:
        日志器实例
    """
    return logging.getLogger(name)