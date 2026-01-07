# 强化学习快速实践项目

通过4个独立的小案例，40分钟体验完整的强化学习算法谱系！

## 🚀 快速开始

### 1. 安装依赖 (2分钟)
```bash
pip install -r requirements.txt
```

### 2. 配置中文字体（可选）
如果图表中文显示为方框，运行字体配置：
```bash
python setup_fonts.py
```

### 3. 运行案例 (38分钟)

#### 案例1：多臂赌博机 (5分钟看效果)
```bash
python case1_bandit.py
```
**学习目标**：理解探索vs利用的根本权衡

#### 案例2：网格世界Q-Learning (10分钟看效果)
```bash
python case2_gridworld.py
```
**学习目标**：理解Q-Learning算法和价值函数学习过程

#### 案例3：CartPole DQN (15分钟看效果)
```bash
python case3_cartpole.py
```
**学习目标**：体验深度强化学习，理解神经网络在RL中的作用

#### 案例4：算法对比 (10分钟看对比)
```bash
python case4_comparison.py
```
**学习目标**：直观对比不同算法性能，理解算法选择考虑因素

## 📊 预期学习效果

- **案例1**：掌握RL问题基本设定，看到不同策略的明显差异
- **案例2**：理解价值函数如何学习，策略如何从价值函数中提取
- **案例3**：体验深度学习解决复杂控制任务，理解DQN关键技术
- **案例4**：建立对RL算法全景的认识，掌握算法选择原则

## 🎯 核心概念回顾

### 强化学习基本要素
- **Agent（智能体）**：做决策的实体
- **Environment（环境）**：Agent交互的世界
- **State（状态）**：环境的当前情况
- **Action（动作）**：Agent可执行的操作
- **Reward（奖励）**：环境对Agent行为的反馈
- **Policy（策略）**：Agent选择动作的规则

### 核心算法类型
1. **多臂赌博机**：最简单的RL问题，理解探索vs利用
2. **Q-Learning**：表格式价值方法，适用于小状态空间
3. **DQN**：深度Q网络，使用神经网络处理高维状态

### 关键技术
- **探索策略**：ε-贪心、UCB等
- **价值函数**：评估状态或状态-动作对的好坏
- **经验回放**：存储和重用历史经验
- **目标网络**：稳定深度RL训练

## 🔧 技术细节

### 环境要求
- Python 3.8+
- 支持CUDA GPU加速（可选）
- 支持Apple Silicon MPS（可选）

### 文件说明
- `case1_bandit.py`：多臂赌博机完整实现
- `case2_gridworld.py`：网格世界和Q-Learning实现
- `case3_cartpole.py`：CartPole环境和DQN实现
- `case4_comparison.py`：算法性能对比实验
- `results/`：保存生成的图片和模型

## 📈 扩展学习

完成这4个案例后，你可以：
1. 调整超参数，观察对学习效果的影响
2. 尝试不同的环境和奖励设置
3. 实现更高级的算法（Double DQN、PPO等）
4. 应用到实际问题中

## 🔧 故障排除

### 中文字体显示问题
如果图表中的中文显示为方框：
1. 运行 `python setup_fonts.py` 自动配置字体
2. 或者手动安装中文字体：
   - **macOS**: 系统自带Arial Unicode MS
   - **Windows**: 安装SimHei或微软雅黑
   - **Linux**: 安装 `sudo apt-get install fonts-wqy-microhei`

### 依赖安装问题
如果pip安装失败：
```bash
# 使用conda安装
conda install numpy matplotlib gymnasium pytorch seaborn pandas -c pytorch -c conda-forge

# 或者降级matplotlib版本
pip install matplotlib==3.7.0
```

### 运行错误
- **ModuleNotFoundError**: 确保已安装所有依赖
- **GPU相关错误**: DQN案例会自动切换到CPU，不影响学习效果
- **内存不足**: 可以减少训练回合数或批次大小

## 🤝 贡献

欢迎提出改进建议和问题！这个项目专注于快速理解和实践，如果你有更好的教学案例想法，请随时分享。