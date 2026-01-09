#!/bin/bash
# 本地模型和数据训练脚本
# Training script for local model and data on RTX 4060

echo "🚀 开始本地模型训练"
echo "======================================"
echo ""
echo "📋 配置信息:"
echo "   模型路径: ./models (本地)"
echo "   数据路径: ./data/raw/alpaca_data_cleaned.json"
echo "   GPU: NVIDIA RTX 4060 (8GB)"
echo "   配置文件: config/gpu_model_config.yaml"
echo ""

# 检查GPU状态
echo "🔍 检查GPU状态..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "📦 激活虚拟环境..."
    source venv/bin/activate
fi

# 运行训练
echo "🏋️ 开始训练..."
echo ""

python scripts/train.py \
    --model_config config/gpu_model_config.yaml \
    --lora_config config/gpu_lora_config.yaml \
    --device cuda \
    --train_data data/raw/alpaca_data_cleaned.json \
    --val_data data/raw/alpaca_data_cleaned.json \
    --output_dir models/qwen_local_checkpoints \
    --experiment_name qwen_local_training

echo ""
echo "✅ 训练完成！"
echo ""
echo "📊 查看训练日志:"
echo "   cat models/qwen_local_checkpoints/training.log"
echo ""
echo "🎯 使用训练好的模型:"
echo "   python scripts/inference.py --model_path models/qwen_local_checkpoints/final/ --device cuda"
