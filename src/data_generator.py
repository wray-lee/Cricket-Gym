import numpy as np
import math
from typing import Dict, Tuple, List


class CricketDataGenerator:
    """
    Synthetic Data Generator for Cricket Multi-Sensory Integration.
    Based on constraints from Takeuchi (2025) and Lu et al. (2023).
    """

    def __init__(self, config: Dict):
        self.cfg = config
        self.dt = config["simulation"]["dt"]  # seconds
        self.seq_len = int(config["simulation"]["episode_length_ms"] / 1000 / self.dt)

        # Inputs/Outputs Config
        self.input_dim = config["model"]["input_dim"]  # 6
        self.output_dim = config["model"]["output_dim"]  # 4

        # Biological Constraints
        self.vis_threshold = np.deg2rad(config["visual"]["looming_threshold_deg"])
        self.vis_delay_steps = int(config["visual"]["motor_delay_ms"] / 1000 / self.dt)
        self.audio_win_start = config["audio"]["coincidence_window_start"] / 1000  # sec
        self.audio_win_end = config["audio"]["coincidence_window_end"] / 1000  # sec

    def _compute_looming_trajectory(
        self, l_v_ratio: float, collision_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes theta and d_theta for a looming object.
        Theta(t) = 2 * atan( (l/v) / (t_collision - t) )
        """
        time_axis = np.linspace(0, self.seq_len * self.dt, self.seq_len)
        # Avoid division by zero by clipping time slightly before collision
        epsilon = 0.05

        # Distance proxy: t_remain
        t_remain = collision_time - time_axis
        t_remain = np.clip(t_remain, epsilon, None)

        # Calculate Visual Angle (theta)
        # l_v_ratio is half-size/velocity roughly, simplified as the scale factor
        theta = 2 * np.arctan(l_v_ratio / t_remain)

        # Calculate Angular Velocity (d_theta)
        # Derivative of 2*atan(C/t) -> 2 * (1/(1+(C/t)^2)) * (C/t^2) = 2C / (t^2 + C^2)
        d_theta = (2 * l_v_ratio) / (t_remain**2 + l_v_ratio**2)

        # Zero out after collision
        mask = time_axis > collision_time
        theta[mask] = 0
        d_theta[mask] = 0

        return theta, d_theta

    def generate_trial(
        self, trial_type: str = "mixed"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a single trial (Sequence).
        Args:
            trial_type: 'visual', 'audio_wind', 'mixed', 'conflict'
        Returns:
            x_t: (seq_len, 6)
            y_t: (seq_len, 4)
        """
        # Initialize containers
        x_t = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        y_t = np.zeros((self.seq_len, self.output_dim), dtype=np.float32)

        # --- 1. Generate Stimuli Params ---
        # Randomize collision time (e.g., between 1.5s and 2.5s)
        t_collision = np.random.uniform(1.5, 2.5)
        # Randomize l/v ratio (approaching speed proxy)
        l_v = np.random.uniform(0.1, 0.4)

        # Stimuli Timings
        wind_time_idx = np.random.randint(
            int(0.5 * self.seq_len), int(0.9 * self.seq_len)
        )
        # Audio potentially happens before wind
        audio_offset_sec = np.random.uniform(0.5, 2.5)
        audio_time_idx = wind_time_idx - int(audio_offset_sec / self.dt)

        if trial_type == "empty":
            # 全零输入，全零输出
            x_t = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
            y_t = np.zeros((self.seq_len, self.output_dim), dtype=np.float32)
            # 即使在 empty 试次，也可以保留极微弱的噪音以增加鲁棒性（可选）
            return x_t, y_t

        has_visual = trial_type in ["visual", "mixed", "conflict"]
        has_audio_wind = trial_type in ["audio_wind", "mixed", "conflict"]

        # --- 2. Build Input Stream (x_t) ---

        # [0, 1] Wind Vector (Assume wind comes from direction 0 for simplicity, so cos=1)
        if has_audio_wind:
            # Wind duration 200ms
            wind_dur_steps = int(0.2 / self.dt)
            x_t[wind_time_idx : wind_time_idx + wind_dur_steps, 0] = 1.0  # Cos
            x_t[wind_time_idx : wind_time_idx + wind_dur_steps, 1] = 0.0  # Sin

        # [2] Audio Amp (Pulse)
        if has_audio_wind and audio_time_idx > 0:
            audio_dur_steps = int(0.05 / self.dt)  # 50ms beep
            x_t[audio_time_idx : audio_time_idx + audio_dur_steps, 2] = 1.0

        # [3, 4, 5] Visual Theta, dTheta, On/Off
        if has_visual:
            theta, d_theta = self._compute_looming_trajectory(l_v, t_collision)
            x_t[:, 3] = theta
            x_t[:, 4] = d_theta
            x_t[:, 5] = 1.0  # Visual ON

        # --- 3. Build Ground Truth Labels (y_t) ---
        # Logic: We define specific events that trigger specific reactions

        # A. Visual Reaction (Takeuchi 2025)
        if has_visual:
            # Find first index where theta > threshold
            trigger_indices = np.where(x_t[:, 3] > self.vis_threshold)[0]
            if len(trigger_indices) > 0:
                trig_idx = trigger_indices[0]
                reaction_idx = trig_idx + self.vis_delay_steps

                if reaction_idx < self.seq_len:
                    # Visual almost always triggers Run (Jump < 2%)
                    # We simulate a burst of activity (e.g., 100ms duration)
                    act_dur = int(0.1 / self.dt)
                    end_idx = min(reaction_idx + act_dur, self.seq_len)

                    # Target: Run=1, Jump=0 (Softened: Run=0.98, Jump=0.02)
                    y_t[reaction_idx:end_idx, 0] = 0.98  # P_run
                    y_t[reaction_idx:end_idx, 1] = 0.02  # P_jump
                    # Direction: Away from visual stim. If stim is at 0 (implicit), run to 180 (cos=-1)
                    # But Takeuchi says l/v affects direction. Simplified: Run backwards.
                    y_t[reaction_idx:end_idx, 2] = -1.0  # Cos_dir
                    y_t[reaction_idx:end_idx, 3] = 0.0  # Sin_dir

        # B. Audio/Wind Reaction (Lu et al. 2023)
        if has_audio_wind:
            # Reaction happens at Wind onset (wind_time_idx)
            reaction_idx = wind_time_idx
            if reaction_idx < self.seq_len:
                act_dur = int(0.1 / self.dt)
                end_idx = min(reaction_idx + act_dur, self.seq_len)

                # Check Memory Window
                # Did audio occur 1000-2000ms before wind?
                dt_sec = (wind_time_idx - audio_time_idx) * self.dt

                is_primed = (dt_sec >= self.audio_win_start) and (
                    dt_sec <= self.audio_win_end
                )

                if is_primed:
                    # Primed -> Jump
                    y_t[reaction_idx:end_idx, 0] = 0.1  # P_run
                    y_t[reaction_idx:end_idx, 1] = 0.9  # P_jump
                else:
                    # Unprimed Wind -> Run
                    y_t[reaction_idx:end_idx, 0] = 0.9  # P_run
                    y_t[reaction_idx:end_idx, 1] = 0.1  # P_jump

                # Wind Direction: Away from wind (Wind comes from 0 -> Run to 180)
                y_t[reaction_idx:end_idx, 2] = -1.0
                y_t[reaction_idx:end_idx, 3] = 0.0

        # Conflict Resolution (Simple overwrite logic for 'conflict' mode)
        # If both trigger around the same time, we need a rule.
        # For this generator, the later write overwrites the earlier one if they overlap.
        # In the actual Model (RNN), it needs to learn this arbitration.

        return x_t, y_t

    def generate_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a batch of data.
        Returns:
            X: (batch, seq_len, 6)
            Y: (batch, seq_len, 4)
        """
        X_list, Y_list = [], []
        modes = ["visual", "audio_wind", "mixed", "empty"]
        probs = [0.2, 0.4, 0.2, 0.2]

        for _ in range(batch_size):
            mode = np.random.choice(modes, p=probs)
            x, y = self.generate_trial(mode)
            X_list.append(x)
            Y_list.append(y)

        return np.stack(X_list), np.stack(Y_list)


# Quick Test Function
if __name__ == "__main__":
    # Mock Config for independent testing
    mock_config = {
        "simulation": {"dt": 0.002, "episode_length_ms": 3000},
        "model": {"input_dim": 6, "output_dim": 4},
        "visual": {"looming_threshold_deg": 41.0, "motor_delay_ms": 74.0},
        "audio": {"coincidence_window_start": 1000, "coincidence_window_end": 2000},
    }

    gen = CricketDataGenerator(mock_config)
    x, y = gen.generate_trial("mixed")

    print(f"Generate Shape: X={x.shape}, Y={y.shape}")
    print(f"Max Visual Theta: {np.max(x[:, 3]):.4f} rad")

    # Check if any action was triggered
    has_action = np.any(y[:, :2] > 0.5)
    print(f"Action Triggered: {has_action}")

    if has_action:
        # Find index of max action probability
        idx = np.argmax(np.sum(y[:, :2], axis=1))
        print(f"Action Peak at Step {idx} ({idx * 0.002:.2f}s)")
        print(f"Probabilities: Run={y[idx,0]:.2f}, Jump={y[idx,1]:.2f}")
