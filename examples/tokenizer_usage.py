#!/usr/bin/env python3
"""
Qwen2.5 Tokenizer使用示例
演示如何使用本地模型的tokenizer进行文本编码和解码
"""

from transformers import AutoTokenizer

def demonstrate_tokenizer():
    """演示tokenizer的基本功能"""

    # 加载本地tokenizer
    model_path = "models/base/qwen2.5-3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("=== Qwen2.5 Tokenizer 使用示例 ===\n")

    # 1. 基本编码和解码
    print("1. 基本文本处理:")
    text = "你好，我是一个AI助手。"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"原始文本: {text}")
    print(f"编码结果: {tokens}")
    print(f"解码结果: {decoded}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print()

    # 2. 特殊token展示
    print("2. 特殊token:")
    special_tokens = {
        "BOS": tokenizer.bos_token,
        "EOS": tokenizer.eos_token,
        "PAD": tokenizer.pad_token,
        "UNK": tokenizer.unk_token
    }

    for name, token in special_tokens.items():
        if token:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{name} token: '{token}' (ID: {token_id})")
    print()

    # 3. 对话格式处理
    print("3. 对话格式处理:")
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请解释什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
    ]

    # 使用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    print("格式化的对话:")
    print(formatted_text)
    print()

    # 4. 批量处理
    print("4. 批量处理:")
    texts = [
        "第一个句子",
        "这是第二个更长的句子，用于演示不同长度的处理",
        "第三个句子"
    ]

    # 批量编码，自动填充到相同长度
    encoded_batch = tokenizer(
        texts,
        padding=True,           # 自动填充
        truncation=True,        # 自动截断
        max_length=50,          # 最大长度
        return_tensors="pt"     # 返回PyTorch张量
    )

    print("批量编码结果:")
    print(f"输入IDs形状: {encoded_batch['input_ids'].shape}")
    print(f"注意力掩码形状: {encoded_batch['attention_mask'].shape}")
    print()

    # 5. 子词分析
    print("5. 子词分析:")
    complex_word = "人工智能技术"
    tokens = tokenizer.tokenize(complex_word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"复杂词汇: {complex_word}")
    print(f"子词分解: {tokens}")
    print(f"对应ID: {token_ids}")
    print()

    # 6. 模型配置信息
    print("6. Tokenizer配置信息:")
    print(f"模型最大长度: {tokenizer.model_max_length}")
    print(f"是否添加BOS: {getattr(tokenizer, 'add_bos_token', 'N/A')}")
    print(f"是否添加EOS: {getattr(tokenizer, 'add_eos_token', 'N/A')}")
    print(f"填充方向: {tokenizer.padding_side}")
    print(f"截断方向: {tokenizer.truncation_side}")

if __name__ == "__main__":
    try:
        demonstrate_tokenizer()
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已安装transformers库: pip install transformers")
        print("并且模型文件位于正确路径: models/base/qwen2.5-3b-instruct/")