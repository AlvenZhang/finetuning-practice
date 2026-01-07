#!/usr/bin/env python3
"""
案例3：CartPole DQN - 体验深度强化学习

这是深度强化学习的经典入门问题：
- 平衡杆子任务：通过左右移动小车来保持杆子直立
- 使用深度Q网络(DQN)处理连续状态空间
- 体验经验回放和目标网络等关键技术

运行时间：约15分钟看到学习效果
学习目标：
1. 理解深度学习在RL中的作用
2. 体验DQN的关键技术：经验回放、目标网络
3. 观察神经网络学习复杂控制任务的过程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from typing import List, Tuple, Optional
import time

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  使用设备: {device}")

class DQN(nn.Module):
    """深度Q网络

    使用全连接神经网络来逼近Q函数
    输入：状态 (4维：位置, 速度, 角度, 角速度)
    输出：每个动作的Q值 (2维：左移, 右移)
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        """初始化网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DQN, self).__init__()

        # 定义网络层
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        print(f"🧠 创建DQN网络: {state_dim} → {hidden_dim} → {hidden_dim} → {action_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入状态 [batch_size, state_dim]

        Returns:
            Q值 [batch_size, action_dim]
        """
        return self.network(x)

class ReplayBuffer:
    """经验回放缓冲区

    存储和采样历史经验，打破数据相关性，提高样本效率
    """

    def __init__(self, capacity: int = 10000):
        """初始化缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

        print(f"💾 创建经验回放缓冲区，容量: {capacity}")

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """存储一个经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """随机采样一批经验

        Args:
            batch_size: 批次大小

        Returns:
            状态、动作、奖励、下一状态、结束标志的批次
        """
        batch = random.sample(self.buffer, batch_size)

        # 分离各个组件
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # 转换为张量
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """DQN智能体

    实现DQN算法的核心逻辑，包括经验回放和目标网络
    """

    def __init__(self,
                 state_dim: int = 4,
                 action_dim: int = 2,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        """初始化DQN Agent

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 创建主网络和目标网络
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)

        # 初始化目标网络（复制主网络参数）
        self.update_target_network()

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)

        # 训练计数器
        self.train_step = 0

        print(f"🤖 创建DQN Agent")
        print(f"参数: lr={learning_rate}, γ={gamma}, ε={epsilon}→{epsilon_min}")
        print(f"批次大小: {batch_size}, 目标网络更新频率: {target_update_freq}")

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            选择的动作
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        """训练网络

        Returns:
            损失值（如果进行了训练）
        """
        # 检查缓冲区是否有足够的经验
        if len(self.memory) < self.batch_size:
            return None

        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 更新训练计数器
        self.train_step += 1

        # 定期更新目标网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """更新目标网络（复制主网络参数）"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
        print(f"💾 模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        print(f"📂 模型已从 {filepath} 加载")

def run_training(episodes: int = 1000, max_steps: int = 500) -> Tuple[DQNAgent, List[float], List[float]]:
    """运行DQN训练

    Args:
        episodes: 训练回合数
        max_steps: 每回合最大步数

    Returns:
        训练好的Agent、奖励历史、损失历史
    """
    print(f"\n🚀 开始DQN训练")
    print(f"参数：{episodes}个回合，每回合最多{max_steps}步")

    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"🎮 CartPole环境: 状态维度={state_dim}, 动作维度={action_dim}")

    # 创建Agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    # 记录训练数据
    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilon_history = []

    # 用于实时显示的变量
    recent_rewards = deque(maxlen=100)  # 最近100回合的奖励
    best_avg_reward = -float('inf')

    print(f"\n📊 开始训练...")
    start_time = time.time()

    for episode in range(episodes):
        # 重置环境
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []

        while steps < max_steps:
            # 选择动作
            action = agent.choose_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)

            # 训练网络
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

            # 更新状态和统计
            state = next_state
            total_reward += reward
            steps += 1

            # 检查结束条件
            if done:
                break

        # 记录回合数据
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        recent_rewards.append(total_reward)
        epsilon_history.append(agent.epsilon)

        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0)

        # 计算最近100回合的平均奖励
        avg_reward = np.mean(recent_rewards)

        # 定期打印进度
        if (episode + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            print(f"回合 {episode+1:4d}: 奖励={total_reward:6.1f}, "
                  f"平均奖励={avg_reward:6.1f}, 步数={steps:3d}, "
                  f"ε={agent.epsilon:.3f}, 时间={elapsed_time:.1f}s")

        # 保存最佳模型
        if avg_reward > best_avg_reward and len(recent_rewards) >= 100:
            best_avg_reward = avg_reward
            agent.save_model('/Users/xifeng/project/finetuning-0106/experiments/RL/results/models/best_cartpole_dqn.pth')

        # 早停条件：连续100回合平均奖励 >= 475（接近最大值500）
        if len(recent_rewards) >= 100 and avg_reward >= 475:
            print(f"\n🎉 提前达到目标！平均奖励: {avg_reward:.1f}")
            break

    # 保存最终模型
    agent.save_model('/Users/xifeng/project/finetuning-0106/experiments/RL/results/models/final_cartpole_dqn.pth')

    total_time = time.time() - start_time
    print(f"\n✅ 训练完成！")
    print(f"总时间: {total_time:.1f}秒")
    print(f"最终探索率: {agent.epsilon:.3f}")
    print(f"最佳平均奖励: {best_avg_reward:.1f}")

    env.close()
    return agent, episode_rewards, losses, epsilon_history

def test_agent(agent: DQNAgent, n_tests: int = 5, render: bool = False) -> None:
    """测试训练好的Agent

    Args:
        agent: 训练好的Agent
        n_tests: 测试次数
        render: 是否渲染环境
    """
    print(f"\n🧪 测试训练好的Agent ({n_tests}次测试)")

    # 创建环境（可选择渲染）
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')

    test_rewards = []

    for test in range(n_tests):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        print(f"\n测试 {test+1}:")

        while True:
            # 使用贪心策略（不探索）
            action = agent.choose_action(state, training=False)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            state = next_state

            # 可选：显示状态信息
            if not render and steps % 50 == 0:
                print(f"  步骤 {steps}: 位置={state[0]:.3f}, 角度={state[2]:.3f}")

            if done:
                break

            # 防止无限循环
            if steps >= 500:
                break

        test_rewards.append(total_reward)
        print(f"  总奖励: {total_reward}, 步数: {steps}")

        if render:
            time.sleep(1)  # 暂停以便观察

    # 统计结果
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)

    print(f"\n📈 测试结果:")
    print(f"平均奖励: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"最佳表现: {max(test_rewards):.1f}")
    print(f"最差表现: {min(test_rewards):.1f}")

    # CartPole的评价标准
    if avg_reward >= 475:
        print("🏆 优秀！达到了CartPole的解决标准（平均奖励≥475）")
    elif avg_reward >= 400:
        print("👍 良好！接近解决标准")
    elif avg_reward >= 200:
        print("📈 不错！有明显学习效果")
    else:
        print("📚 还需要更多训练")

    env.close()

def visualize_training_results(episode_rewards: List[float],
                             losses: List[float],
                             epsilon_history: List[float]) -> None:
    """可视化训练结果"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN训练过程分析', fontsize=16, fontweight='bold')

    episodes = range(len(episode_rewards))

    # 1. 回合奖励
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, alpha=0.6, linewidth=0.8, color='blue')

    # 添加滑动平均
    window_size = 50
    if len(episode_rewards) > window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg,
                'r-', linewidth=2, label=f'滑动平均({window_size})')
        ax1.legend()

    # 添加目标线
    ax1.axhline(y=475, color='green', linestyle='--', alpha=0.7, label='目标(475)')
    ax1.legend()

    ax1.set_xlabel('回合')
    ax1.set_ylabel('总奖励')
    ax1.set_title('回合奖励变化')
    ax1.grid(True, alpha=0.3)

    # 2. 损失变化
    ax2 = axes[0, 1]
    if losses and max(losses) > 0:  # 确保有有效的损失数据
        ax2.plot(episodes, losses, alpha=0.6, linewidth=0.8, color='orange')

        # 添加滑动平均
        if len(losses) > window_size:
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(losses)), moving_avg,
                    'r-', linewidth=2, label=f'滑动平均({window_size})')
            ax2.legend()
    else:
        ax2.text(0.5, 0.5, '暂无损失数据', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)

    ax2.set_xlabel('回合')
    ax2.set_ylabel('损失')
    ax2.set_title('训练损失变化')
    ax2.grid(True, alpha=0.3)

    # 3. 探索率衰减
    ax3 = axes[1, 0]
    ax3.plot(episodes, epsilon_history, color='purple', linewidth=2)
    ax3.set_xlabel('回合')
    ax3.set_ylabel('探索率 (ε)')
    ax3.set_title('探索率衰减')
    ax3.grid(True, alpha=0.3)

    # 4. 性能分布
    ax4 = axes[1, 1]
    ax4.hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax4.axvline(np.mean(episode_rewards), color='red', linestyle='--',
               label=f'平均值: {np.mean(episode_rewards):.1f}')
    ax4.axvline(475, color='green', linestyle='--', alpha=0.7, label='目标: 475')
    ax4.legend()
    ax4.set_xlabel('总奖励')
    ax4.set_ylabel('频次')
    ax4.set_title('奖励分布')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig('/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case3_training_results.png',
                dpi=300, bbox_inches='tight')
    print(f"\n📊 训练结果图已保存到: results/plots/case3_training_results.png")

    plt.show()

def compare_with_random_policy() -> None:
    """与随机策略对比"""
    print(f"\n🎲 与随机策略对比")

    env = gym.make('CartPole-v1')
    n_tests = 10

    # 测试随机策略
    random_rewards = []
    for _ in range(n_tests):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = env.action_space.sample()  # 随机动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        random_rewards.append(total_reward)

    random_avg = np.mean(random_rewards)
    random_std = np.std(random_rewards)

    print(f"随机策略平均奖励: {random_avg:.1f} ± {random_std:.1f}")

    # 如果有训练好的模型，进行对比
    try:
        agent = DQNAgent()
        agent.load_model('/Users/xifeng/project/finetuning-0106/experiments/RL/results/models/best_cartpole_dqn.pth')

        # 测试训练好的策略
        dqn_rewards = []
        for _ in range(n_tests):
            state, _ = env.reset()
            total_reward = 0

            while True:
                action = agent.choose_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            dqn_rewards.append(total_reward)

        dqn_avg = np.mean(dqn_rewards)
        dqn_std = np.std(dqn_rewards)

        print(f"DQN策略平均奖励: {dqn_avg:.1f} ± {dqn_std:.1f}")
        print(f"性能提升: {dqn_avg - random_avg:.1f} (+{(dqn_avg/random_avg-1)*100:.1f}%)")

        # 可视化对比
        plt.figure(figsize=(10, 6))

        x = ['随机策略', 'DQN策略']
        means = [random_avg, dqn_avg]
        stds = [random_std, dqn_std]

        bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                      color=['red', 'blue'], edgecolor='black')

        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.ylabel('平均奖励')
        plt.title('策略性能对比')
        plt.grid(True, alpha=0.3)

        # 添加目标线
        plt.axhline(y=475, color='green', linestyle='--', alpha=0.7, label='目标(475)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case3_policy_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    except FileNotFoundError:
        print("未找到训练好的模型，请先完成训练")

    env.close()

def main():
    """主函数"""
    print("🤖 CartPole DQN：深度强化学习入门")
    print("=" * 45)

    # 运行训练
    agent, episode_rewards, losses, epsilon_history = run_training(
        episodes=1000, max_steps=500
    )

    # 可视化训练结果
    visualize_training_results(episode_rewards, losses, epsilon_history)

    # 测试Agent
    test_agent(agent, n_tests=5, render=False)

    # 与随机策略对比
    compare_with_random_policy()

    # 询问是否观看实时演示
    try:
        user_input = input("\n是否要观看实时演示？(y/n): ").strip().lower()
        if user_input == 'y':
            print("🎬 启动实时演示...")
            test_agent(agent, n_tests=3, render=True)
    except KeyboardInterrupt:
        print("\n用户取消")

    print("\n✅ 案例3完成！")
    print("🎓 你学到了：")
    print("  • 神经网络如何逼近复杂的价值函数")
    print("  • 经验回放如何提高样本效率和训练稳定性")
    print("  • 目标网络如何解决训练不稳定问题")
    print("  • DQN相比随机策略的巨大性能提升")
    print("\n➡️  下一步：运行 python case4_comparison.py 对比所有算法")

if __name__ == "__main__":
    main()