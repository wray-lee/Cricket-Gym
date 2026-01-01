import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.data_generator import CricketDataGenerator
from src.model import BioMoR_RNN

class NeuroEvaluator:
    def __init__(self, config, model_path, device="cpu"):
        self.cfg = config
        self.device = device

        # 1. Initialize Model
        self.model = BioMoR_RNN(config).to(device)

        # 2. Load Weights
        if os.path.exists(model_path):
            print(f"[Eval] Loading weights from: {model_path}")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"[Error] Load failed: {e}")
                print("[Info] Retrying with strict=False...")
                self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        else:
            raise FileNotFoundError(f"[Fatal] Weights not found at {model_path}")

        self.model.eval()
        self.gen = CricketDataGenerator(config)
        self.dt = config["simulation"]["dt"]

        # Config params
        self.vis_threshold_deg = config["visual"]["looming_threshold_deg"]
        self.motor_delay_ms = config["visual"]["motor_delay_ms"]

    def analyze_trial(self, trial_type="visual"):
        print(f"[Eval] Generating {trial_type} trial...")

        # 1. Generate Data
        x_np, y_gt = self.gen.generate_trial(trial_type)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Forward Pass
        with torch.no_grad():
            y_pred_logits, router_state = self.model(x)

        y_pred = y_pred_logits.squeeze(0).cpu().numpy() # [Seq, 4]

        # LAL Activity
        if router_state.dim() == 3:
            router_activity = router_state.squeeze(0).cpu().numpy()
        else:
            router_activity = np.zeros((len(x_np), self.cfg["model"]["hidden_dim"]))
        lal_activity = np.linalg.norm(router_activity, axis=1)

        # Angles
        pred_angles = np.degrees(np.arctan2(y_pred[:, 3], y_pred[:, 2]))
        gt_angles = np.degrees(np.arctan2(y_gt[:, 3], y_gt[:, 2]))

        # Masking for Angles
        action_mask = (y_gt[:, 0] + y_gt[:, 1]) > 0.01
        pred_angles[~action_mask] = np.nan
        gt_angles[~action_mask] = np.nan

        # 3. Dynamic Plotting Configuration
        time = np.arange(len(x_np)) * self.dt * 1000  # ms

        # 定义不同模式下要画的图层
        if trial_type == "visual":
            # Visual 模式：不画 Wind/Audio
            plots_to_draw = ["Visual", "LAL", "Action", "Direction"]
            figsize = (10, 12)
        elif trial_type == "audio_wind":
            # AudioWind 模式：不画 Visual
            plots_to_draw = ["Wind", "Audio", "LAL", "Action", "Direction"]
            figsize = (10, 14)
        else:
            # Mixed/Conflict: 全画
            plots_to_draw = ["Visual", "Wind", "Audio", "LAL", "Action", "Direction"]
            figsize = (10, 16)

        fig, axes = plt.subplots(len(plots_to_draw), 1, figsize=figsize, sharex=True)
        if len(plots_to_draw) == 1: axes = [axes]

        # 遍历绘制
        for i, plot_name in enumerate(plots_to_draw):
            ax = axes[i]

            # === Visual Input ===
            if plot_name == "Visual":
                ax.plot(time, x_np[:, 3], label="Visual Theta (rad)", color="blue", linewidth=2)

                # Visual 特有的阈值线和触发线
                threshold_rad = np.radians(self.vis_threshold_deg)
                ax.axhline(y=threshold_rad, color='red', linestyle='--', label=f"Thresh {self.vis_threshold_deg}°")

                # 计算物理触发时刻
                trigs = np.where(x_np[:, 3] > threshold_rad)[0]
                if len(trigs) > 0:
                    t_trig = time[trigs[0]]
                    ax.axvline(x=t_trig, color='red', linestyle=':', label="Trigger")

                ax.set_ylabel("Visual")
                ax.legend(loc="upper left")
                ax.set_title(f"Visual Input (Threshold: {self.vis_threshold_deg}°)")

            # === Wind Input ===
            elif plot_name == "Wind":
                ax.plot(time, x_np[:, 0], label="Wind Cos", color="cyan", linewidth=2)
                ax.set_ylabel("Wind")
                ax.legend(loc="upper left")
                ax.set_title("Wind Input (Reflex)")

            # === Audio Input ===
            elif plot_name == "Audio":
                ax.plot(time, x_np[:, 2], label="Audio Amp", color="magenta", linewidth=2)
                ax.set_ylabel("Audio")
                ax.legend(loc="upper left")
                ax.set_title("Audio Input")

            # === LAL Activity ===
            elif plot_name == "LAL":
                ax.plot(time, lal_activity, color="purple", linewidth=2, label="LAL Energy")
                ax.fill_between(time, 0, lal_activity, color="purple", alpha=0.3)
                ax.set_ylabel("Neural State")
                ax.legend(loc="upper left")
                ax.set_title("Internal State (LAL Router)")

            # === Action Probabilities ===
            elif plot_name == "Action":
                # GT
                ax.plot(time, y_gt[:, 0], color="green", linestyle=":", alpha=0.5, label="GT Run")
                ax.plot(time, y_gt[:, 1], color="orange", linestyle=":", alpha=0.5, label="GT Jump")

                # Pred - 模型输出已经是概率，不需要再sigmoid
                # [修复] 之前错误地应用了双重sigmoid导致显示总是0.5
                pred_run = y_pred[:, 0]  # 已经是sigmoid后的概率
                pred_jump = y_pred[:, 1]  # 已经是sigmoid后的概率

                ax.plot(time, pred_run, color="green", linewidth=2, label="Pred Run")
                ax.plot(time, pred_jump, color="orange", linewidth=2, label="Pred Jump")

                if trial_type == "visual":
                    threshold_rad = np.radians(self.vis_threshold_deg)
                    trigs = np.where(x_np[:, 3] > threshold_rad)[0]
                    if len(trigs) > 0:
                        ax.axvline(x=time[trigs[0]], color='red', linestyle=':', alpha=0.5)

                ax.set_ylabel("Probability")
                ax.legend(loc="upper left", ncol=2, fontsize='small') # 双列图例
                ax.set_title("Action Output (Run vs Jump)")

            # === Direction ===
            elif plot_name == "Direction":
                ax.plot(time, gt_angles, label="GT Angle", color="gray", linestyle="--")
                ax.plot(time, pred_angles, label="Pred Angle", color="red")
                ax.set_ylabel("Deg")
                ax.set_ylim(0, 180)
                ax.legend(loc="upper left")
                ax.set_title("Direction Control")

            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (ms)")
        plt.tight_layout()
        save_name = f"outputs/eval/eval_{trial_type}.png"
        plt.savefig(save_name)
        print(f"[Result] Saved clean visualization to {save_name}")