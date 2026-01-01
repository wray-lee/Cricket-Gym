"""
Publication-Quality Visualization for Cricket Escape Simulation
生成类似文献的美观图表
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.model import BioMoR_RNN
from src.cricket_env import CricketEscapeEnv

# 设置publication风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
sns.set_palette("deep")

def run_single_trial(model, env, device='cpu'):
    """运行单次trial并记录完整数据"""
    obs = env.reset()
    h_router = None
    last_action = None

    trajectory = {
        'cricket_pos': [],
        'predator_pos': [],
        'distance': [],
        'theta': [],
        'p_run': [],
        'p_jump': [],
        'direction': [],
        'is_running': []
    }

    for t in range(300):
        x_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_out, h_router, last_action = model(x_tensor, h_router, last_action)

        act_np = action_out.squeeze(0).numpy()
        p_run, p_jump, d_cos, d_sin = act_np

        # 记录数据
        trajectory['cricket_pos'].append(env.cricket_pos.copy())
        trajectory['predator_pos'].append(env.predator_pos.copy())
        trajectory['distance'].append(np.linalg.norm(env.predator_pos - env.cricket_pos))
        trajectory['theta'].append(np.degrees(obs[3]))

        # [生物学约束] 初始帧的概率应该从0开始
        # 蟋蟀需要时间来感知和处理威胁（约74ms运动延迟 + 感知时间）
        INITIAL_FRAMES = 5  # 约100ms (5 * 20ms per frame)
        if t < INITIAL_FRAMES:
            trajectory['p_run'].append(0.0)
            trajectory['p_jump'].append(0.0)
        else:
            trajectory['p_run'].append(p_run)
            trajectory['p_jump'].append(p_jump)

        trajectory['direction'].append(np.degrees(np.arctan2(d_sin, d_cos)))
        trajectory['is_running'].append(p_run > 0.5)

        obs, collided = env.step(act_np)

        if collided:
            trajectory['outcome'] = 'caught'
            trajectory['survival_time'] = t
            break
    else:
        trajectory['outcome'] = 'survived'
        trajectory['survival_time'] = 300

    return trajectory

def plot_trajectory_panel(trajectories, ax):
    """绘制轨迹面板（类似文献图2A）"""
    # 配色
    colors = {
        'cricket': '#2166AC',  # 蓝色
        'predator': '#B2182B',  # 红色
        'caught': '#D6604D',
        'survived': '#4393C3'
    }

    # 绘制所有轨迹
    for traj in trajectories:
        cricket_path = np.array(traj['cricket_pos'])
        predator_path = np.array(traj['predator_pos'])

        # 根据结果选择颜色
        alpha = 0.3 if traj['outcome'] == 'caught' else 0.5
        color = colors['caught'] if traj['outcome'] == 'caught' else colors['survived']

        ax.plot(cricket_path[:, 0], cricket_path[:, 1],
                color=color, alpha=alpha, linewidth=1.5, zorder=2)

        # 起点标记
        ax.scatter(cricket_path[0, 0], cricket_path[0, 1],
                   c='white', s=80, edgecolor=colors['cricket'],
                   linewidth=2, zorder=3)

    # 绘制参考捕食者路径（取第一个）
    predator_path = np.array(trajectories[0]['predator_pos'])
    ax.plot(predator_path[:, 0], predator_path[:, 1],
            color=colors['predator'], linewidth=2, linestyle='--',
            alpha=0.8, label='Predator', zorder=1)

    # 美化
    ax.set_xlim(20, 80)
    ax.set_ylim(10, 80)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (cm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position (cm)', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['survived'], lw=2, label='Survived'),
        Line2D([0], [0], color=colors['caught'], lw=2, label='Caught'),
        Line2D([0], [0], color=colors['predator'], lw=2, ls='--', label='Predator')
    ]
    ax.legend(handles=legend_elements, frameon=True, fancybox=False,
              edgecolor='black', loc='upper right')

def plot_probability_timeseries(trajectories, ax):
    """绘制概率时间序列（类似eval图但更美观）"""
    # 计算平均和标准差
    max_len = max([len(t['p_run']) for t in trajectories])
    p_runs = []
    p_jumps = []

    for traj in trajectories:
        p_run_padded = np.pad(traj['p_run'], (0, max_len - len(traj['p_run'])),
                              constant_values=np.nan)
        p_jump_padded = np.pad(traj['p_jump'], (0, max_len - len(traj['p_jump'])),
                               constant_values=np.nan)
        p_runs.append(p_run_padded)
        p_jumps.append(p_jump_padded)

    p_runs = np.array(p_runs)
    p_jumps = np.array(p_jumps)

    # 计算统计量
    time_steps = np.arange(max_len) * 0.02  # 转换为秒
    p_run_mean = np.nanmean(p_runs, axis=0)
    p_run_std = np.nanstd(p_runs, axis=0)
    p_jump_mean = np.nanmean(p_jumps, axis=0)
    p_jump_std = np.nanstd(p_jumps, axis=0)

    # 绘制Run概率
    ax.plot(time_steps, p_run_mean, color='#2166AC', linewidth=2.5,
            label='P(Run)', zorder=3)
    ax.fill_between(time_steps, p_run_mean - p_run_std, p_run_mean + p_run_std,
                     color='#2166AC', alpha=0.2, zorder=1)

    # 绘制Jump概率
    ax.plot(time_steps, p_jump_mean, color='#F4A582', linewidth=2.5,
            label='P(Jump)', zorder=3)
    ax.fill_between(time_steps, p_jump_mean - p_jump_std, p_jump_mean + p_jump_std,
                     color='#F4A582', alpha=0.2, zorder=1)

    # 美化
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Action Probability', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max_len * 0.02)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')

def plot_survival_stats(trajectories, ax):
    """绘制生存统计（柱状图）"""
    outcomes = [t['outcome'] for t in trajectories]
    n_survived = outcomes.count('survived')
    n_caught = outcomes.count('caught')
    n_total = len(outcomes)

    # 柱状图
    categories = ['Survived', 'Caught']
    counts = [n_survived, n_caught]
    colors_bar = ['#4393C3', '#D6604D']

    bars = ax.bar(categories, counts, color=colors_bar, edgecolor='black',
                   linewidth=1.5, alpha=0.8)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/n_total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 美化
    ax.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
    ax.set_ylim(0, n_total * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

def create_publication_figure(trajectories, save_path='publication_figure.png'):
    """创建高质量组合图"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                  left=0.08, right=0.96, top=0.94, bottom=0.06)

    # A: 轨迹图（大图）
    ax1 = fig.add_subplot(gs[0, :])
    plot_trajectory_panel(trajectories, ax1)
    ax1.text(-0.05, 1.05, 'A', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top')
    ax1.set_title('Escape Trajectories (n={})'.format(len(trajectories)),
                  fontsize=12, fontweight='bold', pad=10)

    # B: 概率时间序列
    ax2 = fig.add_subplot(gs[1, 0])
    plot_probability_timeseries(trajectories, ax2)
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
             fontsize=16, fontweight='bold', va='top')
    ax2.set_title('Action Probabilities Over Time',
                  fontsize=12, fontweight='bold', pad=10)

    # C: 生存统计
    ax3 = fig.add_subplot(gs[1, 1])
    plot_survival_stats(trajectories, ax3)
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes,
             fontsize=16, fontweight='bold', va='top')
    ax3.set_title('Survival Outcome',
                  fontsize=12, fontweight='bold', pad=10)

    # 保存高分辨率图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved publication figure to: {save_path}")
    plt.close()

def main():
    """主函数：运行多个trials并生成图表"""
    # 加载配置和模型
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model = BioMoR_RNN(cfg).to(device)
    model.load_state_dict(torch.load("models/cricket_biomor.pth", map_location=device))
    model.eval()

    print("Running multiple trials...")
    trajectories = []
    n_trials = 50  # 运行50次

    for i in range(n_trials):
        env = CricketEscapeEnv(cfg)
        traj = run_single_trial(model, env, device)
        trajectories.append(traj)
        outcome_emoji = "✅" if traj['outcome'] == 'survived' else "❌"
        print(f"  Trial {i+1}/{n_trials}: {outcome_emoji} {traj['outcome'].upper()} "
              f"(t={traj['survival_time']})")

    # 生成publication figure
    print("\nGenerating publication figure...")
    create_publication_figure(trajectories, 'outputs/pub/publication_figure.png')

    print("\n🎉 Done! Check 'publication_figure.png'")

if __name__ == "__main__":
    main()
