#!/usr/bin/env python3
"""
推理脚本
Inference script for fine-tuned models
"""

import sys
import argparse
import torch
from pathlib import Path
import json
import yaml
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.model_loader import load_model_for_inference
from src.utils.logging import setup_logging

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM推理脚本")

    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="基础模型名称"
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA权重路径"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/model_config.yaml",
        help="模型配置文件"
    )

    # 推理参数
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成token数"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p采样"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k采样"
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="启用采样"
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="重复惩罚"
    )

    # 输入输出
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="单个推理提示"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="输入文件路径"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式"
    )

    # 其他参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备"
    )

    return parser.parse_args()

class InferenceEngine:
    """推理引擎"""

    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        config_file: Optional[str] = None,
        device: str = "auto",
        **generation_kwargs
    ):
        """
        初始化推理引擎

        Args:
            model_name: 基础模型名称
            lora_path: LoRA权重路径
            config_file: 配置文件路径
            device: 计算设备
            **generation_kwargs: 生成参数
        """
        self.model_name = model_name
        self.lora_path = lora_path
        self.config_file = config_file
        self.device = self._setup_device(device)
        self.generation_kwargs = generation_kwargs

        # 加载模型和分词器
        print(f"🤖 加载模型: {model_name}")
        if lora_path:
            print(f"🔧 加载LoRA权重: {lora_path}")

        self.model, self.tokenizer = load_model_for_inference(
            model_name=model_name,
            peft_path=lora_path,
            config_path=config_file
        )

        # 移动到设备
        if self.device != "auto":
            self.model = self.model.to(self.device)

        print(f"✅ 模型加载完成，设备: {self.model.device}")

        # 设置默认提示模板
        self.prompt_template = "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n"

    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """格式化提示"""
        return self.prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else ""
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p采样
            top_k: Top-k采样
            do_sample: 是否采样
            repetition_penalty: 重复惩罚
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        try:
            # 分词
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            )

            # 移动到设备
            if self.device != "auto":
                inputs = inputs.to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],  # 只取新生成的部分
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return ""

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 1,
        **generation_kwargs
    ) -> List[str]:
        """批量生成"""
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []

            for prompt in batch_prompts:
                result = self.generate(prompt, **generation_kwargs)
                batch_results.append(result)

            results.extend(batch_results)

            # 显示进度
            progress = min(i + batch_size, len(prompts))
            print(f"📊 进度: {progress}/{len(prompts)}")

        return results

def interactive_mode(engine: InferenceEngine):
    """交互式模式"""
    print("\\n🎯 交互式推理模式")
    print("输入指令，按回车生成回复，输入 'quit' 退出")
    print("-" * 50)

    while True:
        try:
            # 获取用户输入
            instruction = input("\\n💬 指令: ").strip()

            if instruction.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            if not instruction:
                continue

            # 可选的输入内容
            input_text = input("📝 输入内容 (可选): ").strip()

            # 格式化提示
            prompt = engine.format_prompt(instruction, input_text)

            # 生成回复
            print("\\n🤖 生成中...")
            response = engine.generate(prompt, **engine.generation_kwargs)

            # 显示结果
            print("\\n✨ 回复:")
            print(response)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def load_prompts_from_file(file_path: str) -> List[Dict[str, str]]:
    """从文件加载提示"""
    file_path = Path(file_path)

    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    elif file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = [{"instruction": line.strip()} for line in lines if line.strip()]
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    return data

def save_results(results: List[Dict], output_file: str):
    """保存结果到文件"""
    output_path = Path(output_file)

    if output_path.suffix == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif output_path.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    else:
        # 默认保存为JSON
        output_path = output_path.with_suffix('.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 结果已保存到: {output_path}")

def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    setup_logging(level="INFO")

    print("🚀 LLM推理工具")
    print("=" * 50)

    try:
        # 准备生成参数
        generation_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'do_sample': args.do_sample,
            'repetition_penalty': args.repetition_penalty
        }

        # 初始化推理引擎
        engine = InferenceEngine(
            model_name=args.model_name,
            lora_path=args.lora_path,
            config_file=args.config_file,
            device=args.device,
            **generation_kwargs
        )

        # 交互式模式
        if args.interactive:
            interactive_mode(engine)
            return 0

        # 单个提示推理
        if args.prompt:
            print(f"\\n💬 提示: {args.prompt}")
            prompt = engine.format_prompt(args.prompt)
            response = engine.generate(prompt)
            print(f"\\n✨ 回复:\\n{response}")
            return 0

        # 文件批处理
        if args.input_file:
            print(f"\\n📁 加载输入文件: {args.input_file}")
            prompts_data = load_prompts_from_file(args.input_file)

            results = []
            for i, item in enumerate(prompts_data):
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')

                print(f"\\n处理 {i+1}/{len(prompts_data)}: {instruction[:50]}...")

                prompt = engine.format_prompt(instruction, input_text)
                response = engine.generate(prompt)

                result = {
                    'instruction': instruction,
                    'input': input_text,
                    'response': response
                }
                results.append(result)

            # 保存结果
            if args.output_file:
                save_results(results, args.output_file)
            else:
                # 打印结果
                for i, result in enumerate(results):
                    print(f"\\n=== 结果 {i+1} ===")
                    print(f"指令: {result['instruction']}")
                    if result['input']:
                        print(f"输入: {result['input']}")
                    print(f"回复: {result['response']}")

            return 0

        # 如果没有指定任何输入，显示帮助
        print("❌ 请指定输入方式:")
        print("  --prompt '指令'        : 单个推理")
        print("  --input_file 文件路径   : 批量推理")
        print("  --interactive         : 交互式模式")
        return 1

    except KeyboardInterrupt:
        print("\\n👋 推理被用户中断")
        return 0
    except Exception as e:
        print(f"\\n❌ 推理过程中出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)