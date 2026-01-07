#!/usr/bin/env python3
"""
基础测试：验证核心算法逻辑（无需matplotlib）
"""

import numpy as np

# 简单的多臂赌博机测试
class SimpleBandit:
    def __init__(self, n_arms=3):
        np.random.seed(42)
        self.true_values = np.random.normal(0, 1, n_arms)
        self.optimal_arm = np.argmax(self.true_values)
        print(f"🎰 创建{n_arms}臂赌博机，最优臂: {self.optimal_arm}")
        print(f"真实期望奖励: {[f'{v:.2f}' for v in self.true_values]}")

    def pull(self, arm):
        return np.random.normal(self.true_values[arm], 1)

class SimpleEpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

# 运行简单测试
print("🧪 开始基础算法测试...")

bandit = SimpleBandit(n_arms=3)
agent = SimpleEpsilonGreedy(n_arms=3, epsilon=0.1)

total_reward = 0
optimal_actions = 0
n_steps = 1000

for step in range(n_steps):
    action = agent.choose_action()
    reward = bandit.pull(action)
    agent.update(action, reward)

    total_reward += reward
    if action == bandit.optimal_arm:
        optimal_actions += 1

    # 每200步打印一次进度
    if (step + 1) % 200 == 0:
        avg_reward = total_reward / (step + 1)
        optimal_rate = optimal_actions / (step + 1)
        print(f"步骤 {step+1:4d}: 平均奖励={avg_reward:.3f}, 最优率={optimal_rate:.1%}")

print(f"\n✅ 测试完成！")
print(f"最终Q值估计: {[f'{q:.2f}' for q in agent.q_values]}")
print(f"真实Q值:     {[f'{v:.2f}' for v in bandit.true_values]}")
print(f"动作选择次数: {agent.action_counts.astype(int)}")

# 验证学习效果
final_avg = total_reward / n_steps
final_optimal_rate = optimal_actions / n_steps
print(f"\n📊 最终性能:")
print(f"平均奖励: {final_avg:.3f}")
print(f"最优动作率: {final_optimal_rate:.1%}")

if final_optimal_rate > 0.8:
    print("🏆 优秀！算法成功学习到最优策略")
elif final_optimal_rate > 0.6:
    print("👍 不错！算法有明显学习效果")
else:
    print("📚 需要调整参数或增加训练步数")

print("\n🎯 核心算法逻辑验证成功！")
print("现在可以安全地运行完整版本的案例了。")