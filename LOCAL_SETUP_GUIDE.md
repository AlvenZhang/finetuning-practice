# 本地模型和数据训练指南

## 📋 环境配置

### ✅ 当前状态
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **模型**: Qwen2.5-3B-Instruct (本地下载，5.75 GB)
- **数据**: Alpaca数据集 (51,760样本，42.3 MB)
- **配置**: RTX 4060优化的GPU配置

## 🚀 快速开始

### 方法1：使用训练脚本（推荐）

```bash
# 直接运行训练脚本
./train_local.sh
```

### 方法2：手动命令

```bash
# 基础训练
python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda

# 自定义参数训练
python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda \
    --learning_rate 1.5e-4 \
    --batch_size 2 \
    --num_epochs 3 \
    --output_dir models/my_checkpoints
```

## 📊 配置说明

### RTX 4060 优化配置 (`config/gpu_model_config.yaml`)

- **模型路径**: `./models` (本地)
- **数据类型**: bfloat16 (GPU数值稳定性更好)
- **批次大小**: 2 (每GPU)
- **梯度累积**: 32步 (有效批次 = 2×32 = 64)
- **序列长度**: 1024 tokens
- **学习率**: 1.5e-4 (Qwen推荐值)
- **Flash Attention 2**: 启用 (加速训练)
- **预估显存**: ~7.3GB (RTX 4060安全范围)

### LoRA 配置 (`config/gpu_lora_config.yaml`)

- **Rank (r)**: 16 (平衡性能和显存)
- **Alpha**: 32 (2×r)
- **Dropout**: 0.1
- **目标模块**: 所有Qwen2线性层 (7个模块)
- **可训练参数**: ~4.2M (0.14%总参数)

## 📈 预期性能

- **训练时间**: 2-3小时 (3个epoch)
- **显存使用**: 6-7GB VRAM
- **训练速度**: 30-50 tokens/秒
- **性能提升**: 15-25% (指令跟随任务)

## 🔍 验证步骤

### 1. 测试GPU环境
```bash
python scripts/test_gpu_env.py
```

### 2. 快速文件检查
```bash
python scripts/quick_test.py
```

### 3. 完整环境测试（包含模型加载）
```bash
python scripts/test_local_setup.py
```

## 📝 训练监控

### 实时监控GPU使用
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志
```bash
tail -f models/qwen_local_checkpoints/training.log
```

### 检查检查点
```bash
ls -lh models/qwen_local_checkpoints/
```

## ⚙️ 高级选项

### 调整批次大小（如果显存不足）
```bash
# 减小批次大小到1
python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda \
    --batch_size 1
```

### 调整序列长度（如果OOM）
```bash
# 减小序列长度到512
python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda \
    --max_length 512
```

### 从检查点恢复训练
```bash
python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda \
    --resume_from_checkpoint models/qwen_local_checkpoints/checkpoint-500
```

## 🎯 训练后使用

### 运行推理
```bash
python scripts/inference.py \
    --model_path models/qwen_local_checkpoints/final/ \
    --device cuda
```

### 评估模型
```bash
python scripts/evaluate.py \
    --model_path models/qwen_local_checkpoints/final/ \
    --data_path data/raw/alpaca_data_cleaned.json \
    --device cuda
```

## 🔧 故障排除

### 显存不足 (OOM)
- 减小 `batch_size` 到 1
- 减小 `max_length` 到 512
- 减小 LoRA `rank` 到 8

### 训练速度慢
- 检查 Flash Attention 2 是否启用
- 确保 `dataloader_num_workers` > 0
- 检查 GPU 是否正确识别

### 模型加载失败
- 确认模型文件完整: `python scripts/quick_test.py`
- 检查 `config.json` 和 `tokenizer.json` 存在
- 验证模型路径配置正确

## 📂 文件结构

```
finetuning-practice/
├── models/                          # 本地模型
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-00001-of-00002.safetensors
│   └── model-00002-of-00002.safetensors
├── data/
│   └── raw/
│       └── alpaca_data_cleaned.json # 本地数据
├── config/
│   ├── gpu_model_config.yaml        # RTX 4060模型配置
│   └── gpu_lora_config.yaml         # LoRA配置
├── scripts/
│   ├── train.py                     # 训练脚本
│   ├── quick_test.py                # 快速测试
│   └── test_local_setup.py          # 完整测试
└── train_local.sh                   # 一键训练脚本
```

## 🎉 开始训练

一切就绪！运行以下命令开始训练：

```bash
./train_local.sh
```

或者直接使用 Python：

```bash
python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda
```

祝训练顺利！🚀
