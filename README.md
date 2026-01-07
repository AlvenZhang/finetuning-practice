# 🚀 LLM微调项目 - MacBook Pro M3 Pro

一个专为MacBook Pro M3 Pro (18GB内存) 优化的大语言模型微调实验项目，使用LoRA技术进行参数高效的指令微调。

## 📋 项目概述

本项目旨在通过实践学习大语言模型微调的核心概念和技术，特别针对Apple Silicon硬件进行了优化。我们将使用Llama 3.2-3B模型和Alpaca数据集进行指令跟随任务的微调。

### 🎯 学习目标
- 理解参数高效微调(LoRA)技术
- 掌握指令微调和对齐技术
- 学习全面的模型评估方法
- 掌握Apple Silicon优化技术

### 🔧 技术栈
- **模型**: Llama 3.2-3B Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **数据集**: Alpaca-GPT4 (52K样本)
- **优化框架**: MLX (Apple Silicon优化)
- **任务类型**: 指令跟随/对话补全

## 🏗️ 项目结构

```
/Users/xifeng/project/finetuning-0106/
├── README.md                     # 项目文档
├── requirements.txt              # Python依赖
├── config/                       # 配置文件
│   ├── model_config.yaml         # 模型和训练配置
│   ├── lora_config.yaml          # LoRA专用设置
│   └── eval_config.yaml          # 评估设置
├── data/                         # 数据集管理
│   ├── raw/                      # 原始数据集
│   ├── processed/                # 清洗格式化数据
│   └── download_data.py          # 数据下载脚本
├── src/                          # 源代码
│   ├── data/                     # 数据处理模块
│   ├── model/                    # 模型相关模块
│   ├── training/                 # 训练模块
│   ├── evaluation/               # 评估模块
│   └── utils/                    # 工具模块
├── scripts/                      # 可执行脚本
├── notebooks/                    # Jupyter分析笔记本
├── experiments/                  # 实验跟踪
├── models/                       # 模型存储
└── docs/                         # 文档
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目 (如果从git)
git clone <repository-url>
cd finetuning-0106

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置Hugging Face Token

```bash
# 登录Hugging Face (需要访问Llama模型)
huggingface-cli login
```

### 3. 下载数据和模型

```bash
# 下载Alpaca数据集
python data/download_data.py

# 下载基础模型
python scripts/download_model.py
```

### 4. 开始训练

```bash
# 使用默认配置开始训练
python scripts/train.py

# 或者使用自定义配置
python scripts/train.py --config config/model_config.yaml
```

## ⚙️ 配置说明

### 模型配置 (model_config.yaml)
- 模型选择和加载设置
- 训练超参数
- 数据处理配置
- 路径设置

### LoRA配置 (lora_config.yaml)
- LoRA参数设置 (rank=16, alpha=32)
- 目标模块配置
- 内存优化设置

### 评估配置 (eval_config.yaml)
- 自动化评估指标
- 人工评估设置
- 实验跟踪配置

## 📊 预期成果

### 性能指标
- **指令跟随改进**: 15-25% vs 基线
- **训练时间**: 4-6小时
- **内存使用**: 峰值~14GB
- **推理速度**: 20-30 tokens/秒

### 输出文件
- 训练好的LoRA适配器
- 评估报告和指标
- 训练日志和可视化
- 样本输出对比

## 🔧 内存优化策略

针对M3 Pro 18GB内存的优化措施：

1. **LoRA微调**: 减少99%+可训练参数
2. **混合精度**: FP16训练节省内存
3. **梯度累积**: 小批次实现大有效批大小
4. **梯度检查点**: 用计算换内存
5. **MLX优化**: 原生Apple Silicon加速

## 📈 实验跟踪

### Weights & Biases
```python
# 配置wandb
wandb.init(
    project="llm-finetuning",
    tags=["llama-3.2-3B", "lora", "alpaca", "m3-pro"]
)
```

### 本地日志
- 训练日志: `experiments/logs/`
- 模型检查点: `models/checkpoints/`
- 评估结果: `experiments/results/`

## 📚 使用指南

### 训练脚本
```bash
# 基础训练
python scripts/train.py

# 自定义LoRA参数
python scripts/train.py --lora_r 32 --lora_alpha 64

# 恢复训练
python scripts/train.py --resume_from_checkpoint models/checkpoints/checkpoint-1000
```

### 评估脚本
```bash
# 评估训练好的模型
python scripts/evaluate.py --model_path models/final/

# 对比评估
python scripts/evaluate.py --compare_baseline
```

### 推理脚本
```bash
# 交互式推理
python scripts/inference.py --model_path models/final/

# 批量推理
python scripts/inference.py --input_file test_prompts.txt --output_file results.txt
```

## 🛠️ 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 增加gradient_accumulation_steps
   - 降低max_length

2. **训练速度慢**
   - 确保启用MLX优化
   - 检查FP16设置
   - 验证gradient_checkpointing配置

3. **模型加载失败**
   - 检查Hugging Face token
   - 验证模型路径
   - 确认网络连接

### 性能监控
```bash
# 监控内存使用
python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available/1024**3:.1f}GB')
"

# 监控GPU使用 (如果适用)
python -c "
import torch
if torch.backends.mps.is_available():
    print('MPS (Metal Performance Shaders) available')
"
```

## 📖 学习资源

### 推荐阅读
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Alpaca论文](https://arxiv.org/abs/2303.16199)
- [MLX文档](https://ml-explore.github.io/mlx/build/html/index.html)

### 相关项目
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

- Hugging Face团队提供的transformers库
- Apple的MLX团队
- Stanford Alpaca项目
- Meta的Llama模型

---

**Happy Fine-tuning! 🎉**

如有问题或建议，请创建Issue或联系项目维护者。