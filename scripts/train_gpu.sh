#!/bin/bash

# GPU训练启动脚本
# GPU Training Launch Script

set -e  # 遇到错误时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查CUDA环境
check_cuda() {
    print_info "检查CUDA环境..."

    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi未找到，请确保安装了NVIDIA驱动"
        exit 1
    fi

    if ! python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        print_error "PyTorch CUDA不可用，请检查安装"
        exit 1
    fi

    print_success "CUDA环境检查通过"
}

# 检查GPU内存
check_gpu_memory() {
    print_info "检查GPU内存..."

    # 获取GPU信息
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits)

    echo "可用GPU:"
    echo "$gpu_info" | while IFS=', ' read -r name total used free; do
        echo "  - $name: ${total}MB总内存, ${free}MB可用"

        # 检查是否有足够内存（至少8GB）
        if [ "$total" -lt 8000 ]; then
            print_warning "GPU内存可能不足 (<8GB)，建议使用量化或减少batch size"
        fi
    done
}

# 设置环境变量
setup_env() {
    print_info "设置环境变量..."

    # CUDA设备
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    print_info "使用GPU: $CUDA_VISIBLE_DEVICES"

    # PyTorch优化
    export TORCH_CUDNN_V8_API_ENABLED=1
    export TORCH_CUDNN_ALLOW_TF32=1

    # 内存优化
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

    # 多线程优化
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    print_success "环境变量设置完成"
}

# 解析命令行参数
BATCH_SIZE=2
GRADIENT_ACCUMULATION=32
LEARNING_RATE=1.5e-4
MAX_LENGTH=1024
LORA_RANK=16
EPOCHS=3
MODEL_CONFIG="config/gpu_model_config.yaml"
LORA_CONFIG="config/gpu_lora_config.yaml"
OUTPUT_DIR="models/qwen_gpu_checkpoints"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --lora_config)
            LORA_CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "GPU训练脚本使用说明:"
            echo "  --batch_size          每设备批次大小 (默认: 4)"
            echo "  --gradient_accumulation 梯度累积步数 (默认: 16)"
            echo "  --learning_rate       学习率 (默认: 2e-4)"
            echo "  --max_length          最大序列长度 (默认: 1024)"
            echo "  --lora_rank           LoRA rank (默认: 32)"
            echo "  --epochs              训练轮数 (默认: 3)"
            echo "  --model_config        模型配置文件 (默认: config/gpu_model_config.yaml)"
            echo "  --lora_config         LoRA配置文件 (默认: config/gpu_lora_config.yaml)"
            echo "  --output_dir          输出目录 (默认: models/gpu_checkpoints)"
            echo ""
            echo "示例:"
            echo "  ./scripts/train_gpu.sh --batch_size 8 --lora_rank 64"
            echo "  ./scripts/train_gpu.sh --learning_rate 1e-4 --max_length 512"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 主函数
main() {
    print_info "开始GPU训练..."
    print_info "======================================"

    # 环境检查
    check_cuda
    check_gpu_memory
    setup_env

    # 显示配置
    print_info "训练配置:"
    echo "  批次大小: $BATCH_SIZE"
    echo "  梯度累积: $GRADIENT_ACCUMULATION"
    echo "  有效批次: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
    echo "  学习率: $LEARNING_RATE"
    echo "  序列长度: $MAX_LENGTH"
    echo "  LoRA rank: $LORA_RANK"
    echo "  训练轮数: $EPOCHS"
    echo "  模型配置: $MODEL_CONFIG"
    echo "  LoRA配置: $LORA_CONFIG"
    echo "  输出目录: $OUTPUT_DIR"

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"

    # 启动训练
    print_info "启动训练进程..."

    python scripts/train.py \
        --model_config "$MODEL_CONFIG" \
        --lora_config "$LORA_CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda \
        --gpu_id 0 \
        --learning_rate "$LEARNING_RATE" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
        --max_length "$MAX_LENGTH" \
        --lora_r "$LORA_RANK" \
        --num_train_epochs "$EPOCHS" \
        --experiment_name "gpu_training_$(date +%Y%m%d_%H%M%S)"

    if [ $? -eq 0 ]; then
        print_success "训练完成！"
        print_info "检查点保存在: $OUTPUT_DIR"
    else
        print_error "训练失败！"
        exit 1
    fi
}

# 运行主函数
main "$@"