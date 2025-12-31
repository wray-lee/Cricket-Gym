import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_generator import CricketDataGenerator
from src.model import BioMoR_RNN


class NeuroEvaluator:
    def __init__(self, config, model_path, device="cpu"):
        self.cfg = config
        self.device = device

        # Load BioMoR
        self.model = BioMoR_RNN(config).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.gen = CricketDataGenerator(config)
        self.dt = config["simulation"]["dt"]

    def analyze_trial(self, trial_type="audio_wind"):
        """
        Runs a single trial and visualizes:
        1. Sensory Input (Stimuli)
        2. Internal Neural State (LAL Router Activity)
        3. Motor Output (Action Probabilities)
        4. [NEW] Direction Control (Angle)
        """
        # 1. Generate Data
        x_np, y_gt = self.gen.generate_trial(trial_type)
        x = (
            torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        )  # Add batch dim

        # 2. Forward Pass
        with torch.no_grad():
            # BioMoR returns (Action, Router_State)
            y_pred_logits, router_state = self.model(x)

        y_pred_probs = y_pred_logits.clone()
        y_pred_probs[:, :, :2] = torch.sigmoid(y_pred_logits[:, :, :2])

        # Convert to numpy
        y_pred = y_pred_logits.squeeze(0).cpu().numpy()
        router_activity = router_state.squeeze(0).cpu().numpy()

        # Calculate Router "Energy" (L2 Norm of hidden state)
        lal_activity = np.linalg.norm(router_activity, axis=1)

        # --- [新增] 计算角度 (Phase 2) ---
        # y_pred indices: 0=P_run, 1=P_jump, 2=Cos, 3=Sin
        pred_cos = y_pred[:, 2]
        pred_sin = y_pred[:, 3]
        # 使用 arctan2 将 sin/cos 转换为度数 (-180 到 180)
        pred_angles = np.degrees(np.arctan2(pred_sin, pred_cos))

        gt_cos = y_gt[:, 2]
        gt_sin = y_gt[:, 3]
        gt_angles = np.degrees(np.arctan2(gt_sin, gt_cos))

        # 过滤/掩码: 只有在有动作发生时才显示角度，避免显示噪音
        # 使用 Ground Truth 的动作概率和来判断
        action_mask = (y_gt[:, 0] + y_gt[:, 1]) > 0.01
        pred_angles[~action_mask] = np.nan
        gt_angles[~action_mask] = np.nan

        # 3. Plotting (Physiological Dashboard)
        time = np.arange(len(x_np)) * self.dt * 1000  # ms

        # [修改] 改为 5 行子图
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

        # A. Stimuli (Context: Visual/Audio)
        ax = axes[0]
        ax.plot(time, x_np[:, 3], label="Visual Theta", color="blue")
        ax.plot(time, x_np[:, 2], label="Audio Amp", color="magenta")
        ax.set_ylabel("Context Input")
        ax.legend(loc="upper right")
        ax.set_title(f"Trial Analysis (Phase 2): {trial_type}")

        # B. Stimuli (Reflex Trigger: Wind)
        ax = axes[1]
        ax.plot(time, x_np[:, 0], label="Wind Cos", color="cyan")
        ax.set_ylabel("Reflex Input")
        ax.legend(loc="upper right")

        # C. Internal State (LAL Activity)
        ax = axes[2]
        ax.plot(
            time,
            lal_activity,
            color="purple",
            linewidth=2,
            label="LAL Router Activity (Norm)",
        )
        ax.fill_between(time, 0, lal_activity, color="purple", alpha=0.3)
        ax.set_ylabel("Neural State")
        ax.legend(loc="upper right")

        # D. Behavior (Probabilities)
        ax = axes[3]
        ax.plot(time, y_pred[:, 0], label="P_Run (Pred)", color="green", linestyle="-")
        ax.plot(
            time, y_pred[:, 1], label="P_Jump (Pred)", color="orange", linestyle="-"
        )
        ax.plot(
            time, y_gt[:, 0], color="green", linestyle=":", alpha=0.5, label="GT"
        )  # 简化 label
        ax.plot(time, y_gt[:, 1], color="orange", linestyle=":", alpha=0.5)
        ax.set_ylabel("Action Prob")
        ax.legend(loc="upper right")

        # E. [新增] Behavior (Direction Control)
        ax = axes[4]
        # 绘制 Ground Truth 目标线 (虚线)
        ax.plot(
            time,
            gt_angles,
            label="Target Angle (GT)",
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        # 绘制 模型预测 (实线)
        ax.plot(time, pred_angles, label="Pred Angle", color="red", linewidth=2)

        # 辅助参考线 (Control vs Primed)
        ax.axhline(
            y=100, color="blue", linestyle=":", alpha=0.3, label="~100° (Control)"
        )
        ax.axhline(
            y=125, color="magenta", linestyle=":", alpha=0.3, label="~125° (Primed)"
        )

        ax.set_ylabel("Direction (deg)")
        ax.set_ylim(0, 180)  # 限制显示范围，通常蟋蟀逃跑在 90-180 之间
        ax.set_xlabel("Time (ms)")
        ax.legend(loc="upper right")

        plt.tight_layout()
        save_name = f"eval_{trial_type}.png"
        plt.savefig(save_name)
        print(f"[Result] Saved visualization to {save_name}")
