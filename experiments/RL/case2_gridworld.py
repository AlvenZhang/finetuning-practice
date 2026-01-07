#!/usr/bin/env python3
"""
案例2：网格世界Q-Learning - 理解价值函数和策略学习

这是强化学习中的经典问题：
- 5x5网格世界，Agent从起点走到终点
- 学习最优路径，避开障碍物
- 使用Q-Learning算法学习动作价值函数

运行时间：约10分钟看到学习过程
学习目标：
1. 理解Q-Learning算法的核心思想
2. 观察价值函数的学习过程
3. 理解策略如何从价值函数中提取
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import time

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GridWorld:
    """5x5网格世界环境

    Agent需要从起点(0,0)走到终点(4,4)，避开障碍物
    """

    def __init__(self, size: int = 5):
        """初始化网格世界

        Args:
            size: 网格大小
        """
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)

        # 设置障碍物位置
        self.obstacles = {(1, 1), (1, 2), (2, 1), (3, 3)}

        # 动作：上、下、左、右
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_names = ['上', '下', '左', '右']
        self.n_actions = len(self.actions)

        # 当前状态
        self.current_pos = self.start_pos

        print(f"🗺️  创建{size}x{size}网格世界")
        print(f"起点: {self.start_pos}, 终点: {self.goal_pos}")
        print(f"障碍物: {self.obstacles}")

    def reset(self) -> Tuple[int, int]:
        """重置环境到起始状态"""
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """执行动作

        Args:
            action: 动作编号 (0:上, 1:下, 2:左, 3:右)

        Returns:
            下一个状态, 奖励, 是否结束
        """
        # 计算下一个位置
        dx, dy = self.actions[action]
        next_x = self.current_pos[0] + dx
        next_y = self.current_pos[1] + dy
        next_pos = (next_x, next_y)

        # 检查边界
        if (next_x < 0 or next_x >= self.size or
            next_y < 0 or next_y >= self.size):
            # 撞墙，位置不变，给予负奖励
            reward = -0.1
            done = False
            return self.current_pos, reward, done

        # 检查障碍物
        if next_pos in self.obstacles:
            # 撞到障碍物，位置不变，给予负奖励
            reward = -0.5
            done = False
            return self.current_pos, reward, done

        # 正常移动
        self.current_pos = next_pos

        # 计算奖励
        if next_pos == self.goal_pos:
            reward = 10.0  # 到达目标，大奖励
            done = True
        else:
            reward = -0.01  # 每步小惩罚，鼓励找最短路径
            done = False

        return self.current_pos, reward, done

    def get_state_id(self, pos: Tuple[int, int]) -> int:
        """将二维坐标转换为状态ID"""
        return pos[1] * self.size + pos[0]

    def get_pos_from_id(self, state_id: int) -> Tuple[int, int]:
        """将状态ID转换为二维坐标"""
        x = state_id % self.size
        y = state_id // self.size
        return (x, y)

    def visualize(self, q_table: np.ndarray = None, policy: np.ndarray = None):
        """可视化网格世界

        Args:
            q_table: Q表，用于显示价值函数
            policy: 策略，用于显示最优动作
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 环境布局
        ax1 = axes[0]
        grid = np.zeros((self.size, self.size))

        # 设置障碍物
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1

        # 设置起点和终点
        grid[self.start_pos[1], self.start_pos[0]] = 0.5
        grid[self.goal_pos[1], self.goal_pos[0]] = 1

        im1 = ax1.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1)
        ax1.set_title('环境布局', fontsize=14, fontweight='bold')

        # 添加文字标注
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) == self.start_pos:
                    ax1.text(x, y, 'S', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='blue')
                elif (x, y) == self.goal_pos:
                    ax1.text(x, y, 'G', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='red')
                elif (x, y) in self.obstacles:
                    ax1.text(x, y, '■', ha='center', va='center',
                            fontsize=12, color='black')

        ax1.set_xticks(range(self.size))
        ax1.set_yticks(range(self.size))
        ax1.grid(True, alpha=0.3)

        # 2. 状态价值函数 (如果提供了Q表)
        ax2 = axes[1]
        if q_table is not None:
            # 计算状态价值：V(s) = max_a Q(s,a)
            state_values = np.zeros((self.size, self.size))
            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) not in self.obstacles:
                        state_id = self.get_state_id((x, y))
                        state_values[y, x] = np.max(q_table[state_id])
                    else:
                        state_values[y, x] = np.nan

            im2 = ax2.imshow(state_values, cmap='viridis')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            ax2.set_title('状态价值函数 V(s)', fontsize=14, fontweight='bold')

            # 添加数值标注
            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) not in self.obstacles:
                        value = state_values[y, x]
                        ax2.text(x, y, f'{value:.2f}', ha='center', va='center',
                                fontsize=10, color='white', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '等待Q表数据...', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('状态价值函数', fontsize=14)

        ax2.set_xticks(range(self.size))
        ax2.set_yticks(range(self.size))
        ax2.grid(True, alpha=0.3)

        # 3. 最优策略 (如果提供了策略)
        ax3 = axes[2]
        if policy is not None:
            # 创建策略可视化
            policy_grid = np.zeros((self.size, self.size))
            arrow_symbols = ['↑', '↓', '←', '→']

            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) not in self.obstacles and (x, y) != self.goal_pos:
                        state_id = self.get_state_id((x, y))
                        best_action = policy[state_id]
                        ax3.text(x, y, arrow_symbols[best_action], ha='center', va='center',
                                fontsize=16, fontweight='bold', color='blue')

            # 设置背景
            background = np.ones((self.size, self.size))
            for obs in self.obstacles:
                background[obs[1], obs[0]] = 0.5

            ax3.imshow(background, cmap='gray', alpha=0.3)
            ax3.set_title('最优策略 π*(s)', fontsize=14, fontweight='bold')

            # 标注起点和终点
            ax3.text(self.start_pos[0], self.start_pos[1], 'S',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='green', bbox=dict(boxstyle='circle', facecolor='white'))
            ax3.text(self.goal_pos[0], self.goal_pos[1], 'G',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='red', bbox=dict(boxstyle='circle', facecolor='white'))
        else:
            ax3.text(0.5, 0.5, '等待策略数据...', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('最优策略', fontsize=14)

        ax3.set_xticks(range(self.size))
        ax3.set_yticks(range(self.size))
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

class QLearningAgent:
    """Q-Learning算法实现

    学习动作价值函数Q(s,a)，并从中提取最优策略
    """

    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        """初始化Q-Learning Agent

        Args:
            n_states: 状态数量
            n_actions: 动作数量
            learning_rate: 学习率α
            discount_factor: 折扣因子γ
            epsilon: 探索率ε
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # 初始化Q表
        self.q_table = np.zeros((n_states, n_actions))

        print(f"🤖 创建Q-Learning Agent")
        print(f"参数: α={learning_rate}, γ={discount_factor}, ε={epsilon}")

    def choose_action(self, state: int, training: bool = True) -> int:
        """根据ε-贪心策略选择动作

        Args:
            state: 当前状态
            training: 是否在训练（训练时使用ε-贪心，测试时使用贪心）

        Returns:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """更新Q表

        Q-Learning更新规则：
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束

        Returns:
            TD误差（用于监控学习进度）
        """
        # 计算目标值
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # 计算TD误差
        td_error = target - self.q_table[state, action]

        # 更新Q值
        self.q_table[state, action] += self.lr * td_error

        return abs(td_error)

    def get_policy(self) -> np.ndarray:
        """从Q表提取贪心策略

        Returns:
            策略数组，policy[s] = argmax_a Q(s,a)
        """
        return np.argmax(self.q_table, axis=1)

    def decay_epsilon(self, decay_rate: float = 0.995):
        """衰减探索率"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)

def run_training(episodes: int = 1000, visualize_every: int = 200) -> Tuple[QLearningAgent, List[float]]:
    """运行Q-Learning训练

    Args:
        episodes: 训练回合数
        visualize_every: 每隔多少回合可视化一次

    Returns:
        训练好的Agent和奖励历史
    """
    print(f"\n🚀 开始Q-Learning训练")
    print(f"参数：{episodes}个回合")

    # 创建环境和Agent
    env = GridWorld(size=5)
    n_states = env.size * env.size
    agent = QLearningAgent(n_states, env.n_actions,
                          learning_rate=0.1,
                          discount_factor=0.95,
                          epsilon=0.3)  # 开始时较高的探索率

    # 记录训练数据
    episode_rewards = []
    episode_lengths = []
    td_errors = []

    print(f"\n📊 开始训练...")

    for episode in range(episodes):
        # 重置环境
        pos = env.reset()
        state = env.get_state_id(pos)
        total_reward = 0
        steps = 0
        episode_td_errors = []

        while True:
            # 选择动作
            action = agent.choose_action(state)

            # 执行动作
            next_pos, reward, done = env.step(action)
            next_state = env.get_state_id(next_pos)

            # 更新Q表
            td_error = agent.update(state, action, reward, next_state, done)

            # 记录数据
            total_reward += reward
            steps += 1
            episode_td_errors.append(td_error)

            # 更新状态
            state = next_state

            # 检查结束条件
            if done or steps > 100:  # 防止无限循环
                break

        # 记录回合数据
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        td_errors.append(np.mean(episode_td_errors))

        # 衰减探索率
        agent.decay_epsilon(0.995)

        # 定期打印进度和可视化
        if (episode + 1) % visualize_every == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            current_epsilon = agent.epsilon

            print(f"回合 {episode+1:4d}: 平均奖励={avg_reward:6.2f}, "
                  f"平均步数={avg_length:5.1f}, ε={current_epsilon:.3f}")

            # 可视化当前状态
            if episode >= visualize_every - 1:  # 从第一次可视化开始
                policy = agent.get_policy()
                fig = env.visualize(agent.q_table, policy)
                fig.suptitle(f'Q-Learning训练进度 - 回合 {episode+1}', fontsize=16, fontweight='bold')

                # 保存图片
                plt.savefig(f'/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case2_episode_{episode+1}.png',
                           dpi=150, bbox_inches='tight')
                plt.show()
                time.sleep(1)  # 短暂暂停以便观察

    print(f"\n✅ 训练完成！")
    print(f"最终探索率: {agent.epsilon:.3f}")

    return agent, episode_rewards, episode_lengths, td_errors

def test_agent(agent: QLearningAgent, env: GridWorld, n_tests: int = 5) -> None:
    """测试训练好的Agent

    Args:
        agent: 训练好的Agent
        env: 环境
        n_tests: 测试次数
    """
    print(f"\n🧪 测试训练好的Agent ({n_tests}次测试)")

    success_count = 0
    total_steps = []

    for test in range(n_tests):
        pos = env.reset()
        state = env.get_state_id(pos)
        path = [pos]
        steps = 0

        print(f"\n测试 {test+1}: 起点{pos} → 终点{env.goal_pos}")

        while True:
            # 使用贪心策略（不探索）
            action = agent.choose_action(state, training=False)
            action_name = env.action_names[action]

            # 执行动作
            next_pos, reward, done = env.step(action)
            next_state = env.get_state_id(next_pos)

            path.append(next_pos)
            steps += 1

            print(f"  步骤{steps}: {env.current_pos} --{action_name}--> {next_pos} (奖励: {reward:.2f})")

            # 更新状态
            state = next_state

            # 检查结束条件
            if done:
                print(f"  ✅ 成功到达目标！总步数: {steps}")
                success_count += 1
                total_steps.append(steps)
                break
            elif steps > 50:  # 防止无限循环
                print(f"  ❌ 超过最大步数限制")
                break

        # 显示路径
        print(f"  路径: {' → '.join(map(str, path))}")

    # 统计结果
    success_rate = success_count / n_tests
    avg_steps = np.mean(total_steps) if total_steps else 0

    print(f"\n📈 测试结果:")
    print(f"成功率: {success_rate:.1%} ({success_count}/{n_tests})")
    if total_steps:
        print(f"平均步数: {avg_steps:.1f}")
        print(f"最少步数: {min(total_steps)}")

def visualize_learning_curves(episode_rewards: List[float],
                            episode_lengths: List[float],
                            td_errors: List[float]) -> None:
    """可视化学习曲线"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning学习过程分析', fontsize=16, fontweight='bold')

    episodes = range(len(episode_rewards))

    # 1. 回合奖励
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, alpha=0.6, linewidth=0.8)
    # 添加滑动平均
    window_size = 50
    if len(episode_rewards) > window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg,
                'r-', linewidth=2, label=f'滑动平均({window_size})')
        ax1.legend()

    ax1.set_xlabel('回合')
    ax1.set_ylabel('总奖励')
    ax1.set_title('回合奖励变化')
    ax1.grid(True, alpha=0.3)

    # 2. 回合长度
    ax2 = axes[0, 1]
    ax2.plot(episodes, episode_lengths, alpha=0.6, linewidth=0.8)
    # 添加滑动平均
    if len(episode_lengths) > window_size:
        moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(episode_lengths)), moving_avg,
                'r-', linewidth=2, label=f'滑动平均({window_size})')
        ax2.legend()

    ax2.set_xlabel('回合')
    ax2.set_ylabel('步数')
    ax2.set_title('回合长度变化')
    ax2.grid(True, alpha=0.3)

    # 3. TD误差
    ax3 = axes[1, 0]
    ax3.plot(episodes, td_errors, alpha=0.6, linewidth=0.8)
    # 添加滑动平均
    if len(td_errors) > window_size:
        moving_avg = np.convolve(td_errors, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(window_size-1, len(td_errors)), moving_avg,
                'r-', linewidth=2, label=f'滑动平均({window_size})')
        ax3.legend()

    ax3.set_xlabel('回合')
    ax3.set_ylabel('平均TD误差')
    ax3.set_title('学习进度 (TD误差)')
    ax3.grid(True, alpha=0.3)

    # 4. 奖励分布直方图
    ax4 = axes[1, 1]
    ax4.hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(episode_rewards), color='red', linestyle='--',
               label=f'平均值: {np.mean(episode_rewards):.2f}')
    ax4.legend()
    ax4.set_xlabel('总奖励')
    ax4.set_ylabel('频次')
    ax4.set_title('奖励分布')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig('/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case2_learning_curves.png',
                dpi=300, bbox_inches='tight')
    print(f"\n📊 学习曲线已保存到: results/plots/case2_learning_curves.png")

    plt.show()

def compare_parameters() -> None:
    """对比不同参数设置的效果"""
    print(f"\n🔬 参数对比实验")

    # 不同的参数设置
    configs = [
        {'lr': 0.05, 'epsilon': 0.1, 'name': '保守学习'},
        {'lr': 0.1, 'epsilon': 0.1, 'name': '标准设置'},
        {'lr': 0.2, 'epsilon': 0.1, 'name': '激进学习'},
        {'lr': 0.1, 'epsilon': 0.3, 'name': '高探索'},
    ]

    env = GridWorld(size=5)
    n_states = env.size * env.size
    episodes = 500

    results = {}

    for config in configs:
        print(f"\n测试配置: {config['name']} (α={config['lr']}, ε={config['epsilon']})")

        agent = QLearningAgent(n_states, env.n_actions,
                              learning_rate=config['lr'],
                              epsilon=config['epsilon'])

        episode_rewards = []

        for episode in range(episodes):
            pos = env.reset()
            state = env.get_state_id(pos)
            total_reward = 0
            steps = 0

            while True:
                action = agent.choose_action(state)
                next_pos, reward, done = env.step(action)
                next_state = env.get_state_id(next_pos)

                agent.update(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1
                state = next_state

                if done or steps > 100:
                    break

            episode_rewards.append(total_reward)

        results[config['name']] = episode_rewards
        final_avg = np.mean(episode_rewards[-50:])
        print(f"  最终50回合平均奖励: {final_avg:.2f}")

    # 可视化对比结果
    plt.figure(figsize=(12, 6))

    for name, rewards in results.items():
        # 计算滑动平均
        window_size = 50
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        episodes_range = range(window_size-1, len(rewards))
        plt.plot(episodes_range, moving_avg, linewidth=2, label=name)

    plt.xlabel('回合')
    plt.ylabel('平均奖励')
    plt.title('不同参数设置的学习效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig('/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case2_parameter_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🗺️  网格世界Q-Learning：理解价值函数学习")
    print("=" * 55)

    # 运行训练
    agent, episode_rewards, episode_lengths, td_errors = run_training(
        episodes=1000, visualize_every=200
    )

    # 可视化最终结果
    print(f"\n📊 可视化最终学习结果")
    env = GridWorld(size=5)
    policy = agent.get_policy()
    fig = env.visualize(agent.q_table, policy)
    fig.suptitle('最终Q-Learning结果', fontsize=16, fontweight='bold')

    # 保存最终结果
    plt.savefig('/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/case2_final_result.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # 测试Agent
    test_agent(agent, env, n_tests=3)

    # 可视化学习曲线
    visualize_learning_curves(episode_rewards, episode_lengths, td_errors)

    # 参数对比实验
    # try:
    #     user_input = input("\n是否要进行参数对比实验？(y/n): ").strip().lower()
    #     if user_input == 'y':
    #         compare_parameters()
    # except KeyboardInterrupt:
    #     print("\n用户取消")

    print("\n✅ 案例2完成！")
    print("🎓 你学到了：")
    print("  • Q-Learning如何通过TD学习更新价值函数")
    print("  • 策略如何从价值函数中提取")
    print("  • 探索率和学习率对学习效果的影响")
    print("  • 价值函数的可视化帮助理解算法行为")
    print("\n➡️  下一步：运行 python case3_cartpole.py 体验深度强化学习")

if __name__ == "__main__":
    main()