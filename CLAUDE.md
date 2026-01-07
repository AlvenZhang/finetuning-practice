# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM fine-tuning project supporting both NVIDIA GPU and Apple Silicon environments. It implements parameter-efficient fine-tuning of Qwen2.5-3B using LoRA (Low-Rank Adaptation) technique for instruction-following tasks with the Alpaca dataset.

**Key Constraints:**
- **GPU (RTX 4060)**: 8GB VRAM limit, batch_size=2, accumulation_steps=32
- **Apple Silicon**: ~14GB usable (18GB total), single-threaded data loading
- Effective batch size achieved via gradient accumulation for both environments

## Common Development Commands

### Training

#### GPU Training (NVIDIA RTX 4060)
```bash
# Quick start with RTX 4060 optimized settings
./scripts/train_gpu.sh

# RTX 4060 training with custom parameters
./scripts/train_gpu.sh --batch_size 2 --lora_rank 16 --learning_rate 1.5e-4

# GPU training with Qwen-specific configs
python scripts/train.py --model_config config/gpu_model_config.yaml --lora_config config/gpu_lora_config.yaml --device cuda

# RTX 4060 optimized config (simplified)
python scripts/train.py --model_config config/qwen_rtx4060_config.yaml --device cuda

# Resume GPU training from checkpoint
python scripts/train.py --model_config config/gpu_model_config.yaml --resume_from_checkpoint models/qwen_gpu_checkpoints/checkpoint-1000
```

#### Apple Silicon Training (MPS)
```bash
# Start training with Apple Silicon optimization
python scripts/train.py --model_config config/model_config.yaml --lora_config config/lora_config.yaml --device mps

# Train with custom parameters (Apple Silicon)
python scripts/train.py --lora_r 16 --lora_alpha 32 --learning_rate 1e-4 --device mps

# Resume from checkpoint (Apple Silicon)
python scripts/train.py --resume_from_checkpoint models/checkpoints/checkpoint-1000 --device mps
```

### Setup and Data Preparation

#### GPU Environment Setup
```bash
# GPU environment validation
python scripts/test_gpu_env.py  # Comprehensive GPU environment check

# Install GPU dependencies
pip install -r requirements-gpu.txt

# Download data and models
python data/download_data.py          # Download Alpaca dataset
python scripts/download_model.py     # Download Qwen2.5-3B model

# Interactive inference (GPU)
python scripts/inference.py --model_path models/gpu_checkpoints/final/ --device cuda
```

#### Apple Silicon Environment Setup
```bash
# Apple Silicon environment validation
python scripts/test_env.py  # Validate MPS environment and dependencies

# Install Apple Silicon dependencies
pip install -r requirements.txt

# Interactive inference (Apple Silicon)
python scripts/inference.py --model_path models/final/ --device mps
```

### Development Workflow

#### GPU Development
```bash
# Install GPU dependencies
pip install -r requirements-gpu.txt

# Activate virtual environment (if needed)
source .venv/bin/activate

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Monitor GPU memory with Python
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory: {torch.cuda.memory_allocated(i)/1024**3:.1f}GB / {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB')
"
```

#### Apple Silicon Development
```bash
# Install Apple Silicon dependencies
pip install -r requirements.txt

# Activate virtual environment (if needed)
source .venv/bin/activate

# Monitor memory usage during training
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

### Core Components

**ModelLoader** (`src/model/model_loader.py`):
- Loads Qwen2.5-3B models and tokenizers from HuggingFace
- Handles quantization and device optimization (CUDA/MPS)
- Configures gradient checkpointing for memory efficiency

**LoRAManager** (`src/model/lora_utils.py`):
- Creates PEFT LoRA configurations for Qwen2 architecture
- Manages target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- GPU default: rank=16, alpha=32, dropout=0.1

**InstructionDataset** (`src/data/dataset.py`):
- Custom PyTorch Dataset for instruction-following format
- Qwen template: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>`
- Implements label masking (loss computed only on response tokens)

**LoRATrainer** (`src/training/trainer.py`):
- Main training orchestrator extending HuggingFace Trainer
- Integrates checkpoint management and memory monitoring
- Handles LoRA-specific training configurations

### Data Flow
1. Raw Alpaca data → InstructionDataset → DataLoader
2. Model loading → LoRA wrapping → Training
3. Checkpoints saved every 500 steps to `models/checkpoints/`
4. Final model saved to `models/final/`

## Configuration System

All configurations are YAML-based in the `config/` directory:

### GPU Configuration Files

#### gpu_model_config.yaml (RTX 4060 Optimized)
- **Model**: Qwen2.5-3B-Instruct with bfloat16
- **Training**: lr=1.5e-4, 3 epochs, cosine scheduler with 5% warmup
- **Memory**: gradient_checkpointing=true, batch_size=2, accumulation=32
- **Data**: max_length=1024, 10% validation split, Qwen chat format
- **GPU Features**: Flash Attention 2, fused optimizer
- **Memory Management**: pin_memory=true, num_workers=2

#### gpu_lora_config.yaml (RTX 4060 LoRA Settings)
- **LoRA params**: rank=16, alpha=32, dropout=0.1
- **Target modules**: All Qwen2 linear layers (7 modules)
- **Memory estimation**: ~7.3GB total GPU memory usage (RTX 4060 safe)
- **Performance**: 2-3x speed improvement over Apple Silicon

### Apple Silicon Configuration Files

#### qwen_model_config.yaml (Apple Silicon MPS)
- **Model**: Qwen2.5-3B-Instruct with bfloat16
- **Training**: lr=1e-4, 3 epochs, cosine scheduler with 3% warmup
- **Memory**: gradient_checkpointing=true, batch_size=2, accumulation=32
- **Data**: max_length=2048, 10% validation split, Qwen chat format
- **MPS Optimization**: MLX enabled, single-threaded data loading

#### qwen_lora_config.yaml (Apple Silicon LoRA)
- **LoRA params**: rank=16, alpha=32, dropout=0.1
- **Target modules**: All Qwen2 linear layers (7 modules)
- **Memory estimation**: ~10GB total usage

### eval_config.yaml
- **Metrics**: BLEU, ROUGE, BERTScore, perplexity
- **Evaluation**: Every 500 steps during training
- **Tracking**: Weights & Biases integration

## Key Development Guidelines

### Memory Management
- Always monitor memory usage when modifying batch sizes or model parameters
- Test memory-intensive changes with `scripts/test_env.py` first
- Use gradient accumulation instead of increasing batch_size
- Keep `dataloader_num_workers: 0` to prevent memory fragmentation

### Configuration Changes
- Modify hyperparameters in YAML files rather than hardcoding
- Test LoRA parameter changes in `lora_config.yaml` experiments section first
- Always validate config changes with a short training run

### Adding New Features
- **New evaluation metrics**: Extend `src/evaluation/` (currently placeholder)
- **Custom data formats**: Modify `InstructionDataset.format_example()` method
- **Different models**: Update `model_config.yaml` and ensure compatibility with LoRA target modules
- **New optimizers**: Add to `training` section in `model_config.yaml`

### Debugging Common Issues
- **OOM errors**: Reduce max_length, batch_size, or LoRA rank
- **Slow training**: Verify MLX is enabled and FP16 is working
- **Model loading failures**: Check HuggingFace token with `huggingface-cli whoami`
- **Data loading issues**: Ensure Alpaca dataset is downloaded to `data/raw/`

## File Locations for Common Tasks

- **Modify training hyperparameters**: `config/model_config.yaml:19-57`
- **Adjust LoRA settings**: `config/lora_config.yaml:4-12`
- **Change data preprocessing**: `src/data/dataset.py:45-80`
- **Extend training logic**: `src/training/trainer.py:35-120`
- **Add CLI arguments**: `scripts/train.py:24-100`
- **Modify model loading**: `src/model/model_loader.py:25-85`

## Performance Expectations

### GPU Performance (RTX 4060)
- **Training time**: 2-3 hours for 3 epochs (Qwen2.5-3B)
- **Memory usage**: Peak ~7.3GB GPU memory (safe for 8GB VRAM)
- **Batch size**: 2 per device (RTX 4060 optimized)
- **Sequence length**: 1024 tokens
- **LoRA rank**: 16 (balanced for RTX 4060)
- **Improvement**: 15-25% over baseline on instruction-following tasks
- **Inference speed**: 35-50 tokens/second
- **Speed improvement**: 2-3x faster than Apple Silicon

### Apple Silicon Performance (MPS)
- **Training time**: 3-4 hours for 3 epochs (Qwen2.5-3B)
- **Memory usage**: Peak ~10GB (within 18GB limit)
- **Batch size**: 2 per device (Qwen optimized)
- **Sequence length**: 2048 tokens (Qwen supports longer sequences)
- **LoRA rank**: 16 (memory optimized)
- **Improvement**: 15-25% over baseline on instruction-following tasks
- **Inference speed**: 20-30 tokens/second on M3 Pro

## Alternative Model Support

### Qwen Models
The project includes specialized configurations for Qwen models:
- **Qwen configs**: `config/qwen_model_config.yaml` and `config/qwen_lora_config.yaml`
- **Local model support**: Use `./models/base/qwen2.5-3b-instruct` for locally downloaded models
- **Test script**: `python scripts/test_qwen_loading.py` to validate Qwen model loading

```bash
# Train with Qwen model
python scripts/train.py \
  --model_config config/qwen_model_config.yaml \
  --lora_config config/qwen_lora_config.yaml
```

### Model Loading Fixes
- **Fixed parameter filtering**: ModelLoader now filters invalid parameters (`name`, `model_type`)
- **Local path support**: Models can be loaded from local directories
- **Format compatibility**: Supports both HuggingFace and local model formats

## Data Format Guidelines

### Supported Formats
The project supports multiple data formats (see `docs/dataset_formats_guide.md`):

1. **Alpaca Format** (Default):
   ```
   ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}
   ```

2. **Qwen Chat Format**:
   ```
   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>
   ```

3. **ShareGPT Format**: Multi-turn conversations with `{"from": "human/gpt", "value": "..."}`

4. **OpenAI Chat Format**: Standard API format with system/user/assistant roles

### Format Selection Guidelines
- **Alpaca**: Best for single-turn instructions, memory-efficient
- **Qwen**: Native format for Qwen models, better performance
- **ShareGPT**: Multi-turn conversations, chat systems
- **OpenAI**: Production APIs, enterprise applications

## Advanced Configuration

### Memory Optimization Strategies
- **Gradient accumulation**: `batch_size=1` × `accumulation_steps=64` = effective batch size 64
- **Sequence length limits**: Max 1024 tokens for RTX 4060, 2048 for Apple Silicon
- **Single-threaded loading**: `dataloader_num_workers: 0` prevents memory fragmentation
- **Gradient checkpointing**: Trade computation for memory savings

### LoRA Parameter Tuning
- **Rank (r)**: Controls parameter count (8, 16, 32)
- **Alpha**: Scaling factor, typically 2×r
- **Target modules**: All attention and MLP linear layers
- **Dropout**: 0.05-0.1 for stability

### Training Strategies
- **Learning rates**: 1.5e-4 for RTX 4060, 1e-4 for Apple Silicon (Qwen is sensitive to LR)
- **Warmup ratio**: 3-5% of total steps
- **Scheduler**: Cosine annealing with warmup
- **Evaluation**: Every 500 steps with checkpoint saving

## Important Notes

### General
- This project uses Chinese documentation in README.md but code comments are bilingual
- Checkpoint resumption is supported and recommended for long training runs
- WandB integration requires separate login: `wandb login`
- Model requires HuggingFace access token for Qwen2.5 downloads
- **Model path flexibility**: Supports both HuggingFace model names and local paths

### GPU Training (NVIDIA CUDA)
- **Environment validation**: Always run `python scripts/test_gpu_env.py` before training
- **Memory requirements**: Minimum 8GB GPU memory, 16GB+ recommended
- **Flash Attention 2**: Requires compatible GPU (compute capability 7.5+)
- **CUDA version**: Requires CUDA 11.8+ and compatible PyTorch
- **Dependencies**: Install with `pip install -r requirements-gpu.txt`
- **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES` to specify GPUs
- **Performance monitoring**: Use `nvidia-smi` or `watch -n 1 nvidia-smi`
- **Checkpoints**: Saved to `models/gpu_checkpoints/` by default

### Apple Silicon Training (MPS)
- **MLX optimization**: Enabled by default for Apple Silicon
- **Configuration validation**: Always test with `scripts/test_env.py` before training
- **Memory monitoring**: Critical for 18GB limit - use `psutil` for tracking
- **Dependencies**: Install with `pip install -r requirements.txt`
- **Checkpoints**: Saved to `models/checkpoints/` by default

### Device Selection Priority
1. **GPU (CUDA)**: Fastest, supports larger models and batch sizes
2. **Apple Silicon (MPS)**: Good performance on M-series Macs
3. **CPU**: Fallback option, significantly slower