#!/usr/bin/env python3
"""
案例1：多臂赌博机 - 理解探索vs利用权衡

这是强化学习中最基础的问题：
- 有10个拉杆（臂），每个有不同的奖励分布
- Agent需要学习哪个臂的期望奖励最高
- 核心挑战：探索新选择 vs 利用已知最好的选择

运行时间：约5分钟看到明显效果
学习目标：
1. 理解探索vs利用的根本权衡
2. 对比不同策略的性能差异
3. 理解强化学习的基本设定
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MultiArmBandit:
    """多臂赌博机环境

    每个臂有不同的奖励分布（正态分布），Agent需要找到最优臂
    """

    def __init__(self, n_arms: int = 10, seed: int = 42):
        """初始化赌博机

        Args:
            n_arms: 臂的数量
            seed: 随机种子
        """
        np.random.seed(seed)
        self.n_arms = n_arms

        # 每个臂的真实期望奖励（Agent不知道）
        self.true_values = np.random.normal(0, 1, n_arms)
        self.optimal_arm = np.argmax(self.true_values)

        print(f"🎰 创建了{n_arms}臂赌博机")
        print(f"真实最优臂: {self.optimal_arm} (期望奖励: {self.true_values[self.optimal_arm]:.3f})")
        print(f"所有臂期望奖励: {[f'{v:.2f}' for v in self.true_values]}")

    def pull(self, arm: int) -> float:
        """拉动指定的臂，返回奖励

        Args:
            arm: 要拉动的臂编号

        Returns:
            从该臂获得的奖励（加了噪声）
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"臂编号必须在0-{self.n_arms-1}之间")

        # 返回真实期望值 + 噪声
        reward = np.random.normal(self.true_values[arm], 1)
        return reward

class EpsilonGreedyAgent:
    """ε-贪心策略Agent

    以ε的概率随机探索，以(1-ε)的概率选择当前估计最好的臂
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """初始化Agent

        Args:
            n_arms: 臂的数量
            epsilon: 探索概率
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)  # 每个臂的价值估计
        self.action_counts = np.zeros(n_arms)  # 每个臂被选择的次数

    def choose_action(self) -> int:
        """根据ε-贪心策略选择动作"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.n_arms)
        else:
            # 利用：选择当前估计最好的臂
            return np.argmax(self.q_values)

    def update(self, action: int, reward: float):
        """更新价值估计"""
        self.action_counts[action] += 1
        # 增量更新：Q(a) = Q(a) + α[R - Q(a)]，其中α = 1/N(a)
        alpha = 1.0 / self.action_counts[action]
        # reward - self.q_values[action]是预测误差，正误差表示实际奖励好于预期，表明低估了该动作价值，需要提高self.q_values[action]。反之则相反
        self.q_values[action] += alpha * (reward - self.q_values[action])

class UCBAgent:
    """Upper Confidence Bound (UCB) 策略Agent

    选择具有最高上置信界的臂：Q(a) + c*sqrt(ln(t)/N(a))
    自动平衡探索和利用
    """

    def __init__(self, n_arms: int, c: float = 2.0):
        """初始化Agent

        Args:
            n_arms: 臂的数量
            c: 置信度参数，控制探索程度
        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.t = 0  # 总时间步

    def choose_action(self) -> int:
        """根据UCB策略选择动作"""
        self.t += 1

        # 如果有臂还没被选过，先选择它们
        for a in range(self.n_arms):
            if self.action_counts[a] == 0:
                return a

        # 计算UCB值：Q(a) + c*sqrt(ln(t)/N(a))
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.t) / self.action_counts
        )
        return np.argmax(ucb_values)

    def update(self, action: int, reward: float):
        """更新价值估计"""
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

class GreedyAgent:
    """纯贪心策略Agent（不探索）

    总是选择当前估计最好的臂，作为对比基线
    """

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)

    def choose_action(self) -> int:
        """选择当前估计最好的臂"""
        # 如果有臂还没被选过，随机选一个
        if np.min(self.action_counts) == 0:
            unselected = np.where(self.action_counts == 0)[0]
            return np.random.choice(unselected)

        return np.argmax(self.q_values)

    def update(self, action: int, reward: float):
        """更新价值估计"""
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

def run_experiment(n_steps: int = 2000, n_runs: int = 10) -> None:
    """运行多臂赌博机实验

    Args:
        n_steps: 每次运行的步数
        n_runs: 运行次数（用于平均）
    """
    print(f"\n🚀 开始多臂赌博机实验")
    print(f"参数：{n_steps}步，{n_runs}次运行平均")

    # 创建环境
    bandit = MultiArmBandit(n_arms=10, seed=42)

    # 创建不同策略的Agent
    agents = {
        'ε-贪心 (ε=0.1)': lambda: EpsilonGreedyAgent(10, epsilon=0.1),
        'ε-贪心 (ε=0.01)': lambda: EpsilonGreedyAgent(10, epsilon=0.01),
        'UCB (c=2.0)': lambda: UCBAgent(10, c=2.0),
        '纯贪心': lambda: GreedyAgent(10)
    }

    # 存储结果
    results = {}

    # 为每个策略运行实验
    for agent_name, agent_factory in agents.items():
        print(f"\n📊 测试策略: {agent_name}")

        # 多次运行求平均
        all_rewards = np.zeros((n_runs, n_steps))
        all_optimal_actions = np.zeros((n_runs, n_steps))

        for run in range(n_runs):
            # 创建新的Agent和环境
            agent = agent_factory()
            run_bandit = MultiArmBandit(n_arms=10, seed=42+run)

            rewards = []
            optimal_actions = []

            for step in range(n_steps):
                # Agent选择动作
                action = agent.choose_action()

                # 环境返回奖励
                reward = run_bandit.pull(action)

                # Agent更新知识
                agent.update(action, reward)

                # 记录结果
                rewards.append(reward)
                # 用于记录是否选择了最优的动作，用于评估算法的性能
                optimal_actions.append(1 if action == run_bandit.optimal_arm else 0)

            all_rewards[run] = rewards
            all_optimal_actions[run] = optimal_actions

        # 计算平均结果
        # 每步的平均奖励，评估学习进度和最终性能
        avg_rewards = np.mean(all_rewards, axis=0)
        # 每步的最优动作选择率，评估决策准确性和学习效率
        avg_optimal_rate = np.mean(all_optimal_actions, axis=0)

        results[agent_name] = {
            'rewards': avg_rewards,
            'optimal_rate': avg_optimal_rate,
            'cumulative_reward': np.cumsum(avg_rewards),
            'final_reward': np.mean(avg_rewards[-100:]),  # 最后100步的平均奖励
            'final_optimal_rate': np.mean(avg_optimal_rate[-100:])  # 最后100步的最优动作率
        }

        print(f"  最终平均奖励: {results[agent_name]['final_reward']:.3f}")
        print(f"  最终最优动作率: {results[agent_name]['final_optimal_rate']:.1%}")

    # 可视化结果
    visualize_results(results, bandit.true_values, n_steps)

    # 打印总结
    print_summary(results)

def visualize_results(results: dict, true_values: np.ndarray, n_steps: int) -> None:
    """可视化实验结果"""

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('多臂赌博机实验结果对比', fontsize=16, fontweight='bold')

    # 1. 累积奖励对比
    ax1 = axes[0, 0]
    for agent_name, data in results.items():
        ax1.plot(data['cumulative_reward'], label=agent_name, linewidth=2)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('累积奖励')
    ax1.set_title('累积奖励对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 平均奖励对比（滑动窗口）
    ax2 = axes[0, 1]
    window_size = 100
    for agent_name, data in results.items():
        # 计算滑动平均
        rewards = data['rewards']
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        x = np.arange(window_size-1, len(rewards))
        ax2.plot(x, moving_avg, label=agent_name, linewidth=2)

    # 添加最优期望奖励线
    optimal_reward = np.max(true_values)
    ax2.axhline(y=optimal_reward, color='red', linestyle='--',
                label=f'最优期望奖励 ({optimal_reward:.3f})', alpha=0.7)

    ax2.set_xlabel('时间步')
    ax2.set_ylabel('平均奖励')
    ax2.set_title(f'平均奖励对比 (滑动窗口={window_size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 最优动作选择率
    ax3 = axes[1, 0]
    for agent_name, data in results.items():
        # 计算滑动平均的最优动作率
        optimal_rate = data['optimal_rate']
        moving_avg = np.convolve(optimal_rate, np.ones(window_size)/window_size, mode='valid')
        x = np.arange(window_size-1, len(optimal_rate))
        ax3.plot(x, moving_avg, label=agent_name, linewidth=2)

    ax3.set_xlabel('时间步')
    ax3.set_ylabel('最优动作选择率')
    ax3.set_title(f'最优动作选择率 (滑动窗口={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # 4. 最终性能对比（柱状图）
    ax4 = axes[1, 1]
    agent_names = list(results.keys())
    final_rewards = [results[name]['final_reward'] for name in agent_names]
    final_optimal_rates = [results[name]['final_optimal_rate'] for name in agent_names]

    x = np.arange(len(agent_names))
    width = 0.35

    bars1 = ax4.bar(x - width/2, final_rewards, width, label='平均奖励', alpha=0.8)
    bars2 = ax4.bar(x + width/2, final_optimal_rates, width, label='最优动作率', alpha=0.8)

    # 添加最优奖励参考线
    ax4.axhline(y=optimal_reward, color='red', linestyle='--', alpha=0.7)

    ax4.set_xlabel('策略')
    ax4.set_ylabel('性能')
    ax4.set_title('最终性能对比 (最后100步平均)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(agent_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, value in zip(bars1, final_rewards):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars2, final_optimal_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图片
    plt.savefig('/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case1_bandit_results.png',
                dpi=300, bbox_inches='tight')
    print(f"\n📊 结果图已保存到: results/plots/case1_bandit_results.png")

    plt.show()

def print_summary(results: dict) -> None:
    """打印实验总结"""
    print("\n" + "="*60)
    print("🎯 实验总结：探索vs利用权衡")
    print("="*60)

    # 按最终奖励排序
    sorted_results = sorted(results.items(),
                           key=lambda x: x[1]['final_reward'],
                           reverse=True)

    print(f"{'策略':<20} {'最终奖励':<12} {'最优动作率':<12} {'总累积奖励':<15}")
    print("-" * 60)

    for agent_name, data in sorted_results:
        final_reward = data['final_reward']
        final_optimal_rate = data['final_optimal_rate']
        total_reward = data['cumulative_reward'][-1]

        print(f"{agent_name:<20} {final_reward:>8.3f}    {final_optimal_rate:>8.1%}    {total_reward:>10.1f}")

    print("\n💡 关键洞察：")
    print("1. 纯贪心策略可能陷入次优解（局部最优）")
    print("2. 适度探索（ε=0.1）通常比过度探索（ε=0.01）效果更好")
    print("3. UCB策略能自适应地平衡探索和利用")
    print("4. 探索的价值在长期运行中体现得更明显")

    print("\n🔍 策略特点：")
    print("• ε-贪心：简单有效，但探索是随机的")
    print("• UCB：智能探索，优先选择不确定性高的选项")
    print("• 纯贪心：收敛快但容易陷入局部最优")

def interactive_demo() -> None:
    """交互式演示，让用户体验不同策略"""
    print("\n🎮 交互式体验：你来选择策略参数！")

    try:
        epsilon = float(input("请输入ε-贪心的探索率 (建议0.01-0.2): "))
        c = float(input("请输入UCB的置信度参数 (建议1.0-3.0): "))
        n_steps = int(input("请输入实验步数 (建议1000-5000): "))

        print(f"\n🔧 使用参数：ε={epsilon}, c={c}, 步数={n_steps}")

        # 运行自定义实验
        bandit = MultiArmBandit(n_arms=10, seed=42)

        agents = {
            f'你的ε-贪心 (ε={epsilon})': EpsilonGreedyAgent(10, epsilon=epsilon),
            f'你的UCB (c={c})': UCBAgent(10, c=c),
            '基线ε-贪心 (ε=0.1)': EpsilonGreedyAgent(10, epsilon=0.1),
            '基线UCB (c=2.0)': UCBAgent(10, c=2.0)
        }

        print(f"\n🏃 运行{n_steps}步实验...")

        for agent_name, agent in agents.items():
            total_reward = 0
            optimal_actions = 0

            for step in range(n_steps):
                action = agent.choose_action()
                reward = bandit.pull(action)
                agent.update(action, reward)

                total_reward += reward
                if action == bandit.optimal_arm:
                    optimal_actions += 1

            avg_reward = total_reward / n_steps
            optimal_rate = optimal_actions / n_steps

            print(f"{agent_name}: 平均奖励={avg_reward:.3f}, 最优率={optimal_rate:.1%}")

    except (ValueError, KeyboardInterrupt):
        print("输入无效或用户取消，跳过交互式演示")

def main():
    """主函数"""
    print("🎰 多臂赌博机：强化学习入门案例")
    print("=" * 50)

    # 运行标准实验
    run_experiment(n_steps=2000, n_runs=10)

    # 交互式演示
    # try:
    #     user_input = input("\n是否要尝试交互式演示？(y/n): ").strip().lower()
    #     if user_input == 'y':
    #         interactive_demo()
    # except KeyboardInterrupt:
    #     print("\n用户取消")

    print("\n✅ 案例1完成！")
    print("🎓 你学到了：")
    print("  • 探索vs利用是强化学习的核心权衡")
    print("  • 不同策略有不同的探索方式")
    print("  • 长期性能往往需要短期的探索代价")
    print("\n➡️  下一步：运行 python case2_gridworld.py 学习Q-Learning算法")

if __name__ == "__main__":
    main()