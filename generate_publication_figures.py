"""
Publication Figures Generator - Updated for new project structure
适配新项目结构：models/cricket_biomor.pth, outputs/pub/
修复：增大散点可见性，修复response time显示
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy import stats
import torch
import yaml
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.model import BioMoR_RNN

# Publication style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0

# 输出目录
OUTPUT_DIR = Path("outputs/pub")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
MODEL_PATH = "models/cricket_biomor.pth"

def create_audio_wind_input(cfg, isi_ms):
    """创建Audio-Wind输入"""
    dt = cfg['simulation']['dt']
    seq_len = int(cfg['simulation']['episode_length_ms'] / 1000 / dt)
    x = np.zeros((seq_len, 6), dtype=np.float32)

    wind_idx = int(1.0 / dt)
    wind_dur = int(0.2 / dt)

    # 添加微小wind方向变异（±10度）来模拟真实实验的自然变异性
    angle_noise = np.random.uniform(-10, 10)  # degrees
    wind_angle_rad = np.deg2rad(angle_noise)
    wind_cos = np.cos(wind_angle_rad)
    wind_sin = np.sin(wind_angle_rad)

    x[wind_idx:wind_idx+wind_dur, 0] = wind_cos
    x[wind_idx:wind_idx+wind_dur, 1] = wind_sin

    if isi_ms >= 0:
        audio_idx = wind_idx - int(isi_ms / 1000 / dt)
        if audio_idx >= 0:
            audio_dur = int(0.2 / dt)
            x[audio_idx:audio_idx+audio_dur, 2] = 1.0

    return x, wind_idx

def create_visual_input(cfg, lv_ms):
    """创建Visual looming输入"""
    dt = cfg['simulation']['dt']
    seq_len = int(cfg['simulation']['episode_length_ms'] / 1000 / dt)
    x = np.zeros((seq_len, 6), dtype=np.float32)

    lv_sec = lv_ms / 1000.0
    t_start = -0.5
    t_collision = 2.0

    time_axis = np.arange(seq_len) * dt + t_start
    t_remain = t_collision - time_axis
    t_remain = np.clip(t_remain, 0.001, None)

    theta = 2 * np.arctan(lv_sec / t_remain)
    d_theta = (2 * lv_sec) / (t_remain**2 + lv_sec**2)
    d_theta = np.clip(d_theta, 0, 4.0)

    theta[time_axis >= t_collision] = 0
    d_theta[time_axis >= t_collision] = 0

    x[:, 3] = theta
    x[:, 4] = d_theta
    x[:, 5] = 1.0

    vis_threshold = np.deg2rad(cfg['visual']['looming_threshold_deg'])
    trigger_indices = np.where(theta > vis_threshold)[0]

    if len(trigger_indices) > 0:
        trigger_idx = trigger_indices[0]
        motor_delay_steps = int(cfg['visual']['motor_delay_ms'] / 1000 / dt)
        trigger_idx = min(trigger_idx + motor_delay_steps, seq_len - 1)
        return x, trigger_idx
    else:
        return x, None

def analyze_audio_wind(model, cfg, device='cpu'):
    """Audio-Wind ISI分析"""
    print("\n=== Audio-Wind ISI Analysis ===")

    patterns = [
        ('Pattern1', 0),
        ('Pattern2', 200),
        ('Pattern3', 1000),
        ('Pattern4', 2000)
    ]

    n_trials = 150  # 增加样本数

    results = {
        'patterns': [],
        'directions': [],
        'jumps': [],
        'response_times': [],
        'isis': []
    }

    for pattern_name, isi in patterns:
        print(f"  {pattern_name} (ISI={isi}ms) - {n_trials} trials...")

        dirs, jumps, rts = [], [], []

        for _ in range(n_trials):
            x, wind_idx = create_audio_wind_input(cfg, isi)

            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                y_pred, _ = model(x_t)

                window = slice(wind_idx, min(wind_idx + 5, len(y_pred[0])))

                dir_deg = np.degrees(np.arctan2(
                    y_pred[0, window, 3].mean().item(),
                    y_pred[0, window, 2].mean().item()
                ))
                dirs.append(dir_deg)

                jump_prob = y_pred[0, window, 1].mean().item()
                jumps.append(jump_prob)

                run_seq = y_pred[0, wind_idx:, 0].numpy()
                jump_seq = y_pred[0, wind_idx:, 1].numpy()
                response_mask = (run_seq > 0.5) | (jump_seq > 0.5)

                if np.any(response_mask):
                    first_response = np.where(response_mask)[0][0]
                    rt_ms = first_response * cfg['simulation']['dt'] * 1000
                    rts.append(rt_ms)
                else:
                    rts.append(np.nan)

        results['patterns'].append(pattern_name)
        results['directions'].append(dirs)
        results['jumps'].append(jumps)
        results['response_times'].append(rts)
        results['isis'].append(isi)

    return results

def analyze_visual_lv(model, cfg, device='cpu'):
    """Visual l/v分析"""
    print("\n=== Visual l/v Analysis ===")

    lv_values = [20, 40, 60, 80, 100, 120, 140]
    n_trials = 150  # 增加样本数

    results = {
        'lv_values': lv_values,
        'response_prob': [],
        'jump_prob': [],
        'directions': [],
        'response_times': []
    }

    for lv in lv_values:
        print(f"  l/v = {lv}ms - {n_trials} trials...")

        responses, jumps, dirs, rts = [], [], [], []

        for _ in range(n_trials):
            x, trigger_idx = create_visual_input(cfg, lv)

            if trigger_idx is None:
                responses.append(0)
                jumps.append(0)
                dirs.append(np.nan)
                rts.append(np.nan)
                continue

            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                y_pred, _ = model(x_t)

                run_seq = y_pred[0, trigger_idx:, 0].numpy()
                jump_seq = y_pred[0, trigger_idx:, 1].numpy()
                response_mask = (run_seq > 0.5) | (jump_seq > 0.5)

                has_response = np.any(response_mask)
                responses.append(1 if has_response else 0)

                if has_response:
                    first_response = np.where(response_mask)[0][0]
                    absolute_idx = trigger_idx + first_response

                    window = slice(absolute_idx, min(absolute_idx + 5, len(y_pred[0])))

                    jumps.append(y_pred[0, window, 1].mean().item())

                    dir_deg = np.degrees(np.arctan2(
                        y_pred[0, window, 3].mean().item(),
                        y_pred[0, window, 2].mean().item()
                    ))
                    dirs.append(dir_deg)

                    rt_ms = first_response * cfg['simulation']['dt'] * 1000
                    rts.append(rt_ms)
                else:
                    jumps.append(0)
                    dirs.append(np.nan)
                    rts.append(np.nan)

        results['response_prob'].append(responses)
        results['jump_prob'].append(jumps)
        results['directions'].append(dirs)
        results['response_times'].append(rts)

    return results

def plot_audio_wind_figures(results, filename='fig_audio_wind.png'):
    """绘制Audio-Wind图 - 修正散点可见性"""
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, left=0.07, right=0.97)

    patterns = results['patterns']
    isis = results['isis']

    # Panel A: Movement Direction
    ax1 = fig.add_subplot(gs[0])

    for i, pattern in enumerate(patterns):
        dirs = results['directions'][i]
        x_jitter = np.random.normal(i, 0.15, len(dirs))  # 增大jitter
        ax1.scatter(x_jitter, dirs, c='#92C5DE', s=30, alpha=0.5, zorder=1)  # 增大size和alpha

        mean_dir = np.mean(dirs)
        sem_dir = stats.sem(dirs)
        ax1.errorbar([i], [mean_dir], yerr=[sem_dir],
                    fmt='D', color='#2166AC', markersize=11,
                    markeredgecolor='black', markeredgewidth=1.5,
                    capsize=6, capthick=2.5, linewidth=2, zorder=3,
                    elinewidth=2.5)

    ax1.set_xticks(range(len(patterns)))
    ax1.set_xticklabels(patterns)
    ax1.set_ylabel('Movement Direction (deg)', fontweight='bold')
    ax1.axhline(180, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Backward')
    ax1.set_ylim(60, 200)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=False)
    ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes,
            fontsize=14, fontweight='bold')

    # Panel B: Jump Probability
    ax2 = fig.add_subplot(gs[1])

    for i, pattern in enumerate(patterns):
        jumps = results['jumps'][i]
        x_jitter = np.random.normal(i, 0.15, len(jumps))
        ax2.scatter(x_jitter, jumps, c='#F4A582', s=30, alpha=0.5, zorder=1)

        mean_jump = np.mean(jumps)
        sem_jump = stats.sem(jumps)
        ax2.errorbar([i], [mean_jump], yerr=[sem_jump],
                    fmt='D', color='#B2182B', markersize=11,
                    markeredgecolor='black', markeredgewidth=1.5,
                    capsize=6, capthick=2.5, linewidth=2, zorder=3,
                    elinewidth=2.5)

    ax2.set_xticks(range(len(patterns)))
    ax2.set_xticklabels(patterns)
    ax2.set_ylabel('Jumping Prob.', fontweight='bold')
    ax2.set_ylim(0, 0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes,
            fontsize=14, fontweight='bold')

    # Panel C: ISI Schematic
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    y_positions = [3.5, 2.5, 1.5, 0.5]

    for i, (pattern, isi) in enumerate(zip(patterns, isis)):
        y = y_positions[i]

        ax3.text(-0.15, y, pattern, ha='right', va='center', fontsize=9, fontweight='bold')

        if isi == 0:
            audio_x = 1.0
        else:
            audio_x = 1.0 - isi/1000.0

        ax3.add_patch(patches.Rectangle((audio_x, y-0.15), 0.2, 0.3,
                                       facecolor='#2166AC', edgecolor='black', linewidth=1))
        ax3.add_patch(patches.Rectangle((1.0, y-0.15), 0.2, 0.3,
                                       facecolor='black', edgecolor='black', linewidth=1))

        if isi > 0:
            ax3.annotate('', xy=(1.0, y+0.35), xytext=(audio_x+0.2, y+0.35),
                        arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
            ax3.text((1.0 + audio_x+0.2)/2, y+0.45, f'{isi} ms',
                    ha='center', fontsize=8)
        else:
            ax3.text(1.1, y+0.35, '0 ms', ha='center', fontsize=8)

    ax3.set_xlim(-0.3, 2.0)
    ax3.set_ylim(0, 4.2)
    ax3.text(0.5, 4.0, 'Tone', ha='center', fontsize=9, color='#2166AC', fontweight='bold')
    ax3.text(1.1, 4.0, 'Air puff', ha='center', fontsize=9, fontweight='bold')
    ax3.text(-0.12, 1.05, 'C', transform=ax3.transAxes,
            fontsize=14, fontweight='bold')

    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_visual_lv_figures(results, filename='fig_visual_lv.png'):
    """绘制Visual l/v图 - 修正散点可见性和response time"""
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, left=0.07, right=0.97)

    lv_values = results['lv_values']

    # Panel A: Response Probability
    ax1 = fig.add_subplot(gs[0])

    resp_means = [np.mean(r) for r in results['response_prob']]
    resp_sems = [stats.sem(r) for r in results['response_prob']]

    for i, lv in enumerate(lv_values):
        y_vals = results['response_prob'][i]
        x_jitter = np.random.normal(lv, 3.0, len(y_vals))  # 增大jitter
        ax1.scatter(x_jitter, y_vals, c='lightgray', s=20, alpha=0.45, zorder=1)  # 增大size和alpha

    ax1.bar(lv_values, resp_means, width=10, color='#4393C3',
            alpha=0.8, edgecolor='black', linewidth=1.2, zorder=2)
    ax1.errorbar(lv_values, resp_means, yerr=resp_sems,
                 fmt='none', ecolor='black', capsize=4, linewidth=1.5, zorder=3)

    ax1.set_xlabel('l / v (ms)', fontweight='bold')
    ax1.set_ylabel('Response Probability', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes,
            fontsize=14, fontweight='bold')

    # Panel B: Jump Probability
    ax2 = fig.add_subplot(gs[1])

    jump_means, jump_sems = [], []
    for i, lv in enumerate(lv_values):
        jp = [j for j in results['jump_prob'][i] if j > 0]
        if len(jp) > 2:
            jump_means.append(np.mean(jp))
            jump_sems.append(stats.sem(jp))
            x_jitter = np.random.normal(lv, 3.0, len(jp))
            ax2.scatter(x_jitter, jp, c='lightgray', s=20, alpha=0.45, zorder=1)
        else:
            jump_means.append(0)
            jump_sems.append(0)

    ax2.bar(lv_values, jump_means, width=10, color='#F4A582',
            alpha=0.8, edgecolor='black', linewidth=1.2, zorder=2)
    ax2.errorbar(lv_values, jump_means, yerr=jump_sems,
                 fmt='none', ecolor='black', capsize=4, linewidth=1.5, zorder=3)

    ax2.set_xlabel('l / v (ms)', fontweight='bold')
    ax2.set_ylabel('Jumping Prob.', fontweight='bold')
    ax2.set_ylim(0, 0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes,
            fontsize=14, fontweight='bold')

    # Panel C: Response Time - 修复版，不过滤0
    ax3 = fig.add_subplot(gs[2])

    rt_means, rt_sems, valid_lv = [], [], []

    print("\nResponse Time Debug:")
    for i, lv in enumerate(lv_values):
        rt_vals = [t for t in results['response_times'][i] if not np.isnan(t)]  # 只移除NaN，保留0
        print(f"  l/v={lv}ms: {len(rt_vals)} valid RTs, range=[{min(rt_vals) if rt_vals else 'N/A'}–{max(rt_vals) if rt_vals else 'N/A'}]")

        if len(rt_vals) > 5:
            rt_means.append(np.mean(rt_vals))
            rt_sems.append(stats.sem(rt_vals))
            valid_lv.append(lv)

            x_jitter = np.random.normal(lv, 3.0, len(rt_vals))
            ax3.scatter(x_jitter, rt_vals, c='#D6604D', s=20, alpha=0.45, zorder=1)

    if len(valid_lv) > 0:
        ax3.errorbar(valid_lv, rt_means, yerr=rt_sems,
                     marker='o', color='#2166AC', linewidth=2.5,
                     markersize=8, capsize=4, zorder=2)

        if len(valid_lv) > 2:
            z = np.polyfit(valid_lv, rt_means, 1)
            p = np.poly1d(z)
            lv_fit = np.linspace(min(valid_lv), max(valid_lv), 100)
            ax3.plot(lv_fit, p(lv_fit), 'r--', alpha=0.7, linewidth=2, zorder=3)
            ax3.text(0.5, 0.95, f'y = {z[0]:.2f}x + {z[1]:.1f}',
                    transform=ax3.transAxes, fontsize=9, color='red')

    ax3.set_xlabel('l / v (ms)', fontweight='bold')
    ax3.set_ylabel('Response Time (ms)', fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.2)
    ax3.text(-0.12, 1.05, 'C', transform=ax3.transAxes,
            fontsize=14, fontweight='bold')

    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("🔬 Generating Publication Figures")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Output: {OUTPUT_DIR}/")
    print(f"   Sample size: 150 trials per condition\n")

    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("   Please train the model first using: python main.py")
        return

    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model = BioMoR_RNN(cfg).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Audio-Wind分析
    audio_results = analyze_audio_wind(model, cfg, device)
    plot_audio_wind_figures(audio_results)

    # Visual分析
    visual_results = analyze_visual_lv(model, cfg, device)
    plot_visual_lv_figures(visual_results)

    print("\n🎉 All Figures Generated!")
    print(f"   - {OUTPUT_DIR}/fig_audio_wind.png")
    print(f"   - {OUTPUT_DIR}/fig_visual_lv.png")

if __name__ == "__main__":
    main()
