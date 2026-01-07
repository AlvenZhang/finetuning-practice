# LLM微调数据集格式详解

本文档详细介绍了大语言模型微调中常用的数据集组织形式，包括格式特点、优缺点分析和实际应用建议。

## 目录
- [指令微调格式](#指令微调格式)
- [任务特定格式](#任务特定格式)
- [多轮对话格式](#多轮对话格式)
- [高级数据组织形式](#高级数据组织形式)
- [格式对比分析](#格式对比分析)
- [实际应用建议](#实际应用建议)
- [数据质量考虑](#数据质量考虑)

## 指令微调格式

### 1. Stanford Alpaca 格式

**结构特点：**
```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
}
```

**字段说明：**
- `instruction`: 用户的指令或问题（必需）
- `input`: 额外的输入信息（可选，可以为空字符串）
- `output`: 期望的输出或回答（必需）

**模板转换：**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### 2. ShareGPT 格式

**结构特点：**
```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."}
  ]
}
```

**特点：**
- 支持多轮对话
- 更接近真实对话场景
- 角色明确（human/gpt）

### 3. OpenAI Chat 格式

**结构特点：**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain photosynthesis"},
    {"role": "assistant", "content": "Photosynthesis is the process by which plants convert light energy into chemical energy..."}
  ]
}
```

**特点：**
- 支持系统提示（system prompt）
- 工业标准格式
- API兼容性好

## 任务特定格式

### 1. 问答格式 (SQuAD-style)

```json
{
  "context": "The Amazon rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and extending into Colombia, Peru and other South American countries.",
  "question": "What is the Amazon rainforest?",
  "answer": "The largest tropical rainforest in the world"
}
```

**适用场景：**
- 阅读理解任务
- 基于上下文的问答
- 信息抽取

### 2. 分类格式

```json
{
  "text": "This movie was absolutely terrible! The plot made no sense.",
  "label": "negative",
  "category": "sentiment"
}
```

**适用场景：**
- 情感分析
- 文本分类
- 内容审核

### 3. 代码生成格式

```json
{
  "prompt": "Write a Python function to calculate fibonacci numbers",
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "language": "python",
  "test_cases": [
    {"input": "5", "output": "5"},
    {"input": "10", "output": "55"}
  ]
}
```

**适用场景：**
- 代码生成
- 编程助手
- 自动化编程

## 多轮对话格式

### 1. 对话历史格式

```json
{
  "dialogue_id": "conv_001",
  "turns": [
    {"speaker": "user", "text": "Hello, how are you?"},
    {"speaker": "assistant", "text": "I'm doing well, thank you! How can I help you today?"},
    {"speaker": "user", "text": "Can you help me with math?"},
    {"speaker": "assistant", "text": "Of course! What math topic do you need help with?"}
  ]
}
```

### 2. 上下文对话格式

```json
{
  "context": "Previous conversation about travel plans",
  "history": [
    "User: I want to visit Japan",
    "Assistant: That sounds wonderful! When are you planning to go?"
  ],
  "current_input": "I'm thinking about spring time",
  "response": "Spring is an excellent choice! You'll get to see the cherry blossoms in full bloom."
}
```

## 高级数据组织形式

### 1. RLHF (Reinforcement Learning from Human Feedback) 格式

```json
{
  "prompt": "Explain climate change in simple terms",
  "responses": [
    {
      "response": "Climate change refers to long-term shifts in global temperatures and weather patterns...",
      "score": 8.5,
      "preference_rank": 1
    },
    {
      "response": "Global warming is when the earth gets really hot because of pollution...",
      "score": 6.2,
      "preference_rank": 2
    }
  ]
}
```

### 2. Constitutional AI 格式

```json
{
  "original_prompt": "How to hack into a computer system?",
  "constitutional_prompt": "How to improve computer security and protect against unauthorized access?",
  "critique": "The original prompt asks for potentially harmful information that could be used maliciously...",
  "revision": "Here are ways to improve computer security and protect against unauthorized access..."
}
```

### 3. Few-shot Learning 格式

```json
{
  "task_description": "Translate English to French",
  "examples": [
    {"input": "Hello", "output": "Bonjour"},
    {"input": "Thank you", "output": "Merci"},
    {"input": "Good morning", "output": "Bonjour"}
  ],
  "test_input": "Good evening",
  "expected_output": "Bonsoir"
}
```

## 格式对比分析

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Alpaca格式** | • 简单清晰<br>• 标准化程度高<br>• 易于理解和实现<br>• 内存效率高 | • 不支持多轮对话<br>• 缺少系统提示<br>• 格式相对固定 | • 单轮指令执行<br>• 知识问答<br>• 文本生成<br>• 资源受限环境 |
| **ShareGPT格式** | • 支持多轮对话<br>• 更接近真实对话<br>• 上下文连贯性好 | • 格式相对复杂<br>• 处理成本较高<br>• 序列长度较长 | • 对话系统<br>• 聊天机器人<br>• 交互式应用 |
| **OpenAI Chat格式** | • 支持系统提示<br>• 角色明确<br>• 工业标准<br>• API兼容性好 | • 结构较复杂<br>• 需要更多预处理<br>• 格式开销较大 | • 生产环境<br>• API服务<br>• 企业应用 |
| **任务特定格式** | • 针对性强<br>• 评估明确<br>• 训练效果好 | • 局限性大<br>• 通用性差<br>• 需要专门处理 | • 特定任务<br>• 专业领域<br>• 性能优化 |

## 实际应用建议

### 根据项目目标选择格式

#### 1. 通用助手 → Alpaca格式
```python
# 适合场景：
- 知识问答
- 文本总结
- 创意写作
- 简单推理

# 示例实现：
{
  "instruction": "Summarize the main points of this research paper",
  "input": "Paper abstract and key sections...",
  "output": "The paper presents three main findings: 1) ... 2) ... 3) ..."
}
```

#### 2. 对话系统 → ShareGPT格式
```python
# 适合场景：
- 客服机器人
- 聊天助手
- 多轮交互
- 上下文理解

# 示例实现：
{
  "conversations": [
    {"from": "human", "value": "I need help with my order"},
    {"from": "gpt", "value": "I'd be happy to help! What's your order number?"},
    {"from": "human", "value": "It's #12345"},
    {"from": "gpt", "value": "Let me look that up for you. I see your order for..."}
  ]
}
```

#### 3. 专业工具 → 任务特定格式
```python
# 代码生成示例：
{
  "language": "python",
  "description": "Create a function to sort a list of dictionaries by a specific key",
  "code": "def sort_dicts(dict_list, key):\n    return sorted(dict_list, key=lambda x: x[key])",
  "test_cases": [
    {
      "input": "[{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 20}], 'age'",
      "output": "[{'name': 'Bob', 'age': 20}, {'name': 'Alice', 'age': 25}]"
    }
  ]
}

# 数学解题示例：
{
  "problem": "Solve the equation: 3x + 7 = 22",
  "solution_steps": [
    "Subtract 7 from both sides: 3x = 15",
    "Divide both sides by 3: x = 5"
  ],
  "answer": "x = 5",
  "verification": "3(5) + 7 = 15 + 7 = 22 ✓"
}
```

### 混合格式策略

#### 多任务训练
```python
# 在同一个数据集中混合多种格式：
dataset = [
    # Alpaca格式样本
    {"instruction": "Explain quantum physics", "input": "", "output": "Quantum physics is..."},

    # 对话格式样本
    {"conversations": [
        {"from": "human", "value": "What's the weather like?"},
        {"from": "gpt", "value": "I don't have access to current weather data..."}
    ]},

    # 代码格式样本
    {"prompt": "Write a sorting algorithm", "code": "def bubble_sort(arr):..."}
]
```

#### 格式转换器实现
```python
class DataFormatConverter:
    """数据格式转换器"""

    def alpaca_to_chat(self, alpaca_item):
        """Alpaca格式转换为Chat格式"""
        instruction = alpaca_item['instruction']
        input_text = alpaca_item.get('input', '')
        output = alpaca_item['output']

        user_content = f"{instruction}\n{input_text}".strip()

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        }

    def chat_to_alpaca(self, chat_item):
        """Chat格式转换为Alpaca格式"""
        messages = chat_item['messages']
        user_msg = next(m for m in messages if m['role'] == 'user')
        assistant_msg = next(m for m in messages if m['role'] == 'assistant')

        return {
            "instruction": user_msg['content'],
            "input": "",
            "output": assistant_msg['content']
        }

    def sharegpt_to_alpaca(self, sharegpt_item):
        """ShareGPT格式转换为Alpaca格式"""
        conversations = sharegpt_item['conversations']

        # 提取最后一轮对话
        human_msg = conversations[-2]['value']  # 人类输入
        gpt_msg = conversations[-1]['value']    # GPT回复

        return {
            'instruction': human_msg,
            'input': '',
            'output': gpt_msg
        }
```

## 数据质量考虑

### 高质量数据集特征

1. **多样性 (Diversity)**
   - 覆盖不同任务类型和难度级别
   - 包含各种语言风格和表达方式
   - 涵盖多个知识领域

2. **一致性 (Consistency)**
   - 格式统一，标注规范
   - 质量标准一致
   - 风格相对统一

3. **平衡性 (Balance)**
   - 避免某类任务过度代表
   - 难度分布合理
   - 长度分布适中

4. **清洁性 (Cleanliness)**
   - 去除噪声和错误样本
   - 过滤重复内容
   - 移除有害或偏见内容

### 常见高质量数据集

| 数据集 | 格式 | 规模 | 特点 | 推荐用途 |
|--------|------|------|------|----------|
| **Alpaca-52K** | Alpaca | 52K | 通用指令，质量高，多样性好 | 通用助手训练 |
| **ShareGPT** | 对话 | 90K | 真实对话数据，多轮交互 | 对话系统开发 |
| **FLAN Collection** | 多任务 | 1M+ | 任务多样性强，格式统一 | 多任务能力提升 |
| **CodeAlpaca** | 代码 | 20K | 专注编程任务，代码质量高 | 编程助手训练 |
| **WizardLM** | 复杂指令 | 250K | 复杂推理能力，指令多样 | 推理能力增强 |
| **Dolly-15K** | 指令 | 15K | 人工标注，质量极高 | 高质量基准 |
| **OpenAssistant** | 对话 | 161K | 多语言，社区驱动 | 多语言助手 |

### 数据预处理最佳实践

#### 1. 数据清洗
```python
def clean_data(dataset):
    """数据清洗流程"""
    cleaned = []

    for item in dataset:
        # 移除过短或过长的样本
        if len(item['output']) < 10 or len(item['output']) > 2000:
            continue

        # 移除重复样本
        if item in cleaned:
            continue

        # 过滤有害内容
        if contains_harmful_content(item):
            continue

        cleaned.append(item)

    return cleaned
```

#### 2. 质量评估
```python
def assess_quality(dataset):
    """数据质量评估"""
    metrics = {
        'total_samples': len(dataset),
        'avg_length': np.mean([len(item['output']) for item in dataset]),
        'unique_instructions': len(set(item['instruction'] for item in dataset)),
        'diversity_score': calculate_diversity_score(dataset)
    }
    return metrics
```

#### 3. 数据增强
```python
def augment_data(dataset, augmentation_factor=2):
    """数据增强"""
    augmented = []

    for item in dataset:
        # 原始样本
        augmented.append(item)

        # 生成变体
        for _ in range(augmentation_factor - 1):
            variant = create_variant(item)
            augmented.append(variant)

    return augmented
```

## 针对特定环境的建议

### Apple Silicon + LoRA微调环境

基于内存限制（18GB）和单线程数据加载的约束：

#### 推荐格式选择：
1. **首选 Alpaca格式**
   - 内存效率高
   - 序列长度可控
   - 处理简单稳定

2. **可选混合策略**
   - 80% Alpaca + 20% 简单对话
   - 控制最大序列长度 ≤ 512
   - 避免复杂的多轮对话

#### 数据加载优化：
```python
class MemoryEfficientDataset(Dataset):
    """内存高效的数据集实现"""

    def __init__(self, data_path, max_samples=None):
        # 延迟加载，避免一次性加载所有数据到内存
        self.data_path = data_path
        self.max_samples = max_samples
        self._length = self._get_length()

    def _get_length(self):
        # 只计算长度，不加载数据
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return min(len(data), self.max_samples or len(data))

    def __getitem__(self, idx):
        # 按需加载单个样本
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return self.process_item(data[idx])
```

## 总结

选择合适的数据格式对微调成功至关重要。主要考虑因素包括：

1. **任务需求**：单轮指令 vs 多轮对话 vs 特定任务
2. **资源约束**：内存限制、计算能力、训练时间
3. **数据质量**：标注质量、多样性、一致性
4. **评估方式**：如何衡量模型性能
5. **部署场景**：最终应用的具体需求

对于大多数通用微调场景，Alpaca格式是一个平衡且实用的选择。如需特定能力（如对话、代码生成），可以考虑混合相应的专门格式或使用任务特定的数据集。

---

*本文档基于实际微调项目经验总结，涵盖了主流的数据组织形式和最佳实践。建议根据具体项目需求选择合适的格式，并在实际应用中持续优化数据质量。*