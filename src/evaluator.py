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
        2. Internal Neural State (LAL Router Activity) -> The "Hidden State"
        3. Motor Output (Behavior)
        """
        # 1. Generate Data
        x_np, y_gt = self.gen.generate_trial(trial_type)
        x = (
            torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        )  # Add batch dim

        # 2. Forward Pass
        with torch.no_grad():
            # BioMoR returns (Action, Router_State)
            y_pred, router_state = self.model(x)

        # Convert to numpy
        y_pred = y_pred.squeeze(0).cpu().numpy()
        router_activity = router_state.squeeze(0).cpu().numpy()

        # Calculate Router "Energy" (L2 Norm of hidden state) representing population firing rate
        lal_activity = np.linalg.norm(router_activity, axis=1)

        # 3. Plotting (Physiological Dashboard)
        time = np.arange(len(x_np)) * self.dt * 1000  # ms

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # A. Stimuli (Context: Visual/Audio)
        ax = axes[0]
        ax.plot(time, x_np[:, 3], label="Visual Theta", color="blue")
        ax.plot(time, x_np[:, 2], label="Audio Amp", color="magenta")
        ax.set_ylabel("Context Input")
        ax.legend(loc="upper right")
        ax.set_title(f"Trial Analysis: {trial_type}")

        # B. Stimuli (Reflex Trigger: Wind)
        ax = axes[1]
        ax.plot(time, x_np[:, 0], label="Wind Cos", color="cyan")
        ax.set_ylabel("Reflex Input")
        ax.legend(loc="upper right")

        # C. Internal State (The "Science" - LAL Activity)
        ax = axes[2]
        # Plot the norm of the hidden state
        ax.plot(
            time,
            lal_activity,
            color="purple",
            linewidth=2,
            label="LAL Router Activity (Norm)",
        )
        ax.fill_between(time, 0, lal_activity, color="purple", alpha=0.3)
        ax.set_ylabel("Neural State (a.u.)")
        ax.legend(loc="upper right")
        # Mark the "Reverberation" / Memory Window
        ax.text(
            time[len(time) // 2],
            max(lal_activity) * 0.8,
            "Persistent State?",
            ha="center",
            color="purple",
        )

        # D. Behavior (Output)
        ax = axes[3]
        ax.plot(time, y_pred[:, 0], label="P_Run (Pred)", color="green", linestyle="-")
        ax.plot(
            time, y_pred[:, 1], label="P_Jump (Pred)", color="orange", linestyle="-"
        )
        # Optional: Plot GT dashed
        ax.plot(time, y_gt[:, 0], color="green", linestyle=":", alpha=0.5)
        ax.plot(time, y_gt[:, 1], color="orange", linestyle=":", alpha=0.5)
        ax.set_ylabel("Action Prob")
        ax.set_xlabel("Time (ms)")
        ax.legend(loc="upper right")

        plt.tight_layout()
        save_name = f"eval_{trial_type}.png"
        plt.savefig(save_name)
        print(f"[Result] Saved visualization to {save_name}")
