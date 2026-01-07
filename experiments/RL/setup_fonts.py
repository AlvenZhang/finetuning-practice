#!/usr/bin/env python3
"""
字体配置脚本 - 解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def setup_chinese_fonts():
    """设置中文字体显示"""

    # 获取系统类型
    system = platform.system()

    print(f"🖥️  检测到系统: {system}")

    # 根据不同系统设置字体
    if system == "Darwin":  # macOS
        # macOS常见中文字体
        fonts = [
            'Arial Unicode MS',    # macOS默认
            'PingFang SC',        # 苹方
            'Hiragino Sans GB',   # 冬青黑体
            'STHeiti',            # 华文黑体
            'SimHei'              # 黑体
        ]
    elif system == "Windows":  # Windows
        fonts = [
            'SimHei',             # 黑体
            'Microsoft YaHei',    # 微软雅黑
            'KaiTi',              # 楷体
            'SimSun'              # 宋体
        ]
    else:  # Linux
        fonts = [
            'DejaVu Sans',        # Linux默认
            'WenQuanYi Micro Hei', # 文泉驿微米黑
            'Noto Sans CJK SC',   # 思源黑体
            'SimHei'              # 黑体
        ]

    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    print("🔍 检查可用的中文字体...")
    found_fonts = []
    for font in fonts:
        if font in available_fonts:
            found_fonts.append(font)
            print(f"  ✅ 找到字体: {font}")
        else:
            print(f"  ❌ 未找到字体: {font}")

    if found_fonts:
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = found_fonts + ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"\n✅ 字体配置成功！使用字体: {found_fonts[0]}")

        # 测试中文显示
        test_chinese_display()

    else:
        print("\n⚠️  未找到合适的中文字体，将使用英文标签")
        # 提供英文替代方案
        return False

    return True

def test_chinese_display():
    """测试中文显示效果"""
    import numpy as np

    print("\n🧪 测试中文字体显示...")

    try:
        # 创建简单测试图
        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        ax.plot(x, y, label='正弦波')
        ax.set_xlabel('时间')
        ax.set_ylabel('幅度')
        ax.set_title('中文字体测试')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存测试图片
        test_path = '/Users/xifeng/project/finetuning-0106/experiments/RL/results/plots/font_test.png'
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 中文字体测试成功！测试图片保存到: {test_path}")

    except Exception as e:
        print(f"❌ 中文字体测试失败: {e}")

def get_english_labels():
    """如果中文字体不可用，返回英文标签映射"""
    return {
        '累积奖励': 'Cumulative Reward',
        '时间步': 'Time Step',
        '平均奖励': 'Average Reward',
        '最优动作选择率': 'Optimal Action Rate',
        '最终性能对比': 'Final Performance Comparison',
        '策略': 'Strategy',
        '性能': 'Performance',
        '多臂赌博机实验结果对比': 'Multi-Armed Bandit Results Comparison',
        '环境布局': 'Environment Layout',
        '状态价值函数': 'State Value Function V(s)',
        '最优策略': 'Optimal Policy π*(s)',
        '回合': 'Episode',
        '总奖励': 'Total Reward',
        '步数': 'Steps',
        '网格世界': 'GridWorld',
        '学习曲线': 'Learning Curve',
        '回合长度': 'Episode Length',
        '强化学习算法综合对比分析': 'Comprehensive RL Algorithm Comparison'
    }

if __name__ == "__main__":
    setup_chinese_fonts()