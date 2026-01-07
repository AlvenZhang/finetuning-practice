# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM fine-tuning project optimized for Apple Silicon (MacBook Pro M3 Pro with 18GB RAM). It implements parameter-efficient fine-tuning of Llama 3.2-3B using LoRA (Low-Rank Adaptation) technique for instruction-following tasks with the Alpaca dataset.

**Key Constraints:**
- Memory limit: ~14GB usable (18GB total with 4GB reserved for system)
- Single-threaded data loading (`dataloader_num_workers: 0`) to prevent memory issues
- Effective batch size achieved via gradient accumulation (batch_size=1, accumulation_steps=64)

## Common Development Commands

### Training
```bash
# Start training with default configuration
python scripts/train.py

# Train with custom parameters
python scripts/train.py --lora_r 32 --lora_alpha 64 --learning_rate 2e-4

# Resume from checkpoint
python scripts/train.py --resume_from_checkpoint models/checkpoints/checkpoint-1000

# Train with specific config files
python scripts/train.py --model_config config/model_config.yaml --lora_config config/lora_config.yaml
```

### Setup and Data Preparation
```bash
# Environment setup
python scripts/test_env.py  # Validate environment and dependencies

# Download data and models
python data/download_data.py          # Download Alpaca dataset
python scripts/download_model.py     # Download Llama 3.2-3B model

# Interactive inference
python scripts/inference.py --model_path models/final/
```

### Development Workflow
```bash
# Install dependencies
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
- Loads Llama models and tokenizers from HuggingFace
- Handles quantization and Apple Silicon MPS optimization
- Configures gradient checkpointing for memory efficiency

**LoRAManager** (`src/model/lora_utils.py`):
- Creates PEFT LoRA configurations
- Manages target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Default: rank=16, alpha=32, dropout=0.1

**InstructionDataset** (`src/data/dataset.py`):
- Custom PyTorch Dataset for instruction-following format
- Template: `### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}`
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

### model_config.yaml
- **Model**: Llama 3.2-3B-Instruct with FP16
- **Training**: lr=1e-4, 3 epochs, cosine scheduler with 3% warmup
- **Memory**: gradient_checkpointing=true, batch_size=1, accumulation=64
- **Data**: max_length=512, 10% validation split

### lora_config.yaml
- **LoRA params**: rank=16, alpha=32, dropout=0.1
- **Target modules**: All linear layers in attention and MLP
- **Memory estimation**: ~11GB total usage

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

- **Training time**: 4-6 hours for 3 epochs
- **Memory usage**: Peak ~14GB (within 18GB limit)
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
- **Sequence length limits**: Max 512 tokens for Llama, 2048 for Qwen
- **Single-threaded loading**: `dataloader_num_workers: 0` prevents memory fragmentation
- **Gradient checkpointing**: Trade computation for memory savings

### LoRA Parameter Tuning
- **Rank (r)**: Controls parameter count (8, 16, 32)
- **Alpha**: Scaling factor, typically 2×r
- **Target modules**: All attention and MLP linear layers
- **Dropout**: 0.05-0.1 for stability

### Training Strategies
- **Learning rates**: 1e-4 for Llama, 5e-5 for Qwen (more sensitive)
- **Warmup ratio**: 3-5% of total steps
- **Scheduler**: Cosine annealing with warmup
- **Evaluation**: Every 500 steps with checkpoint saving

## Important Notes

- This project uses Chinese documentation in README.md but code comments are bilingual
- MLX optimization is enabled by default for Apple Silicon
- Checkpoint resumption is supported and recommended for long training runs
- WandB integration requires separate login: `wandb login`
- Model requires HuggingFace access token for Llama 3.2 downloads
- **Model path flexibility**: Supports both HuggingFace model names and local paths
- **Configuration validation**: Always test with `scripts/test_env.py` before training
- **Memory monitoring**: Critical for 18GB limit - use `psutil` for tracking