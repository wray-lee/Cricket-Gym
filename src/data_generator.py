import numpy as np
import math
from typing import Dict, Tuple

class CricketDataGenerator:
    """
    Biologically Constrained Data Generator.
    References:
    - Takeuchi (2025): Visual looming -> Mostly RUN (Jump < 2%), Threshold ~41 deg, Delay ~74ms.
    - Lu et al. (2023): Audio priming (1000ms delay) -> Increases JUMP prob to ~40% (Control ~20%).
    """

    def __init__(self, config: Dict):
        self.cfg = config
        self.dt = config["simulation"]["dt"]
        self.seq_len = int(config["simulation"]["episode_length_ms"] / 1000 / self.dt)

        self.input_dim = config["model"]["input_dim"]
        self.output_dim = config["model"]["output_dim"]

        self.vis_threshold = np.deg2rad(config["visual"]["looming_threshold_deg"])
        self.vis_delay_steps = int(config["visual"]["motor_delay_ms"] / 1000 / self.dt)
        self.audio_win_start = config["audio"]["coincidence_window_start"] / 1000
        self.audio_win_end = config["audio"]["coincidence_window_end"] / 1000

    def _compute_looming_trajectory(self, l_v_ratio: float, collision_time: float) -> Tuple[np.ndarray, np.ndarray]:
            time_axis = np.linspace(0, self.seq_len * self.dt, self.seq_len)
            epsilon = 0.05
            t_remain = np.clip(collision_time - time_axis, epsilon, None)
            theta = 2 * np.arctan(l_v_ratio / t_remain)
            d_theta = (2 * l_v_ratio) / (t_remain**2 + l_v_ratio**2)

            # Biological saturation constraint
            # Clip angular velocity to prevent gradient explosion while maintaining feature discrimination
            # Mimics retinal neuron saturation characteristics
            d_theta = np.clip(d_theta, 0, 4.0)

            mask = time_axis > collision_time
            theta[mask] = 0
            d_theta[mask] = 0
            return theta, d_theta

    def _sample_direction(self, mean_deg: float, kappa: float = 4.0) -> Tuple[float, float]:
        mean_rad = np.deg2rad(mean_deg)
        angle_rad = np.random.vonmises(mean_rad, kappa)
        return np.cos(angle_rad), np.sin(angle_rad)

    def generate_trial(self, trial_type: str = "mixed") -> Tuple[np.ndarray, np.ndarray]:
        x_t = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        y_t = np.zeros((self.seq_len, self.output_dim), dtype=np.float32)

        t_collision = np.random.uniform(1.5, 2.5)
        l_v = np.random.uniform(0.1, 0.4)

        # Wind timing
        wind_time_idx = np.random.randint(int(0.5 * self.seq_len), int(0.9 * self.seq_len))

        # Audio timing (Random offset to simulate different patterns)
        audio_offset_sec = np.random.uniform(0.0, 2.0) # 0 to 2s delay
        raw_audio_idx = wind_time_idx - int(audio_offset_sec / self.dt)
        audio_time_idx = max(0, raw_audio_idx)

        if trial_type == "empty": return x_t, y_t

        has_visual = trial_type in ["visual", "mixed", "conflict"]
        has_audio_wind = trial_type in ["audio_wind", "mixed", "conflict"]

        # --- 1. Inputs ---
        if has_audio_wind:
            # Wind (200ms duration)
            wind_dur = int(0.2 / self.dt)
            end = min(wind_time_idx + wind_dur, self.seq_len)
            x_t[wind_time_idx:end, 0] = 1.0 # Wind_Cos
            x_t[wind_time_idx:end, 1] = 0.0 # Wind_Sin

            # Audio (200ms duration, per Lu 2023)
            if audio_time_idx < self.seq_len:
                audio_dur = int(0.2 / self.dt) # [cite: 1220] 200ms tone
                end = min(audio_time_idx + audio_dur, self.seq_len)
                x_t[audio_time_idx:end, 2] = 1.0

        if has_visual:
            theta, d_theta = self._compute_looming_trajectory(l_v, t_collision)
            x_t[:, 3] = theta
            x_t[:, 4] = d_theta
            x_t[:, 5] = 1.0  # Visual stimulus present flag

        # --- 2. Ground Truth ---
        target_cos, target_sin = -1.0, 0.0 # Default backward

        if has_audio_wind:
            # Audio modulation of direction
            dt_sec = (wind_time_idx - audio_time_idx) * self.dt
            # Lu 2023: Audio 200-1000ms prior biases direction backward
            is_dir_modulated = (dt_sec >= 0.2) and (dt_sec <= 1.0)

            # Lu 2023: Jump prob increases only at ~1000ms
            is_jump_primed = (dt_sec >= self.audio_win_start) and (dt_sec <= self.audio_win_end)

            mean_angle = 110.0 if is_dir_modulated else 170.0
            target_cos, target_sin = self._sample_direction(mean_angle, kappa=10.0)

        elif has_visual:
            mean_angle = 180.0 - (l_v * 130.0)
            mean_angle = np.clip(mean_angle, 90, 180)
            target_cos, target_sin = self._sample_direction(mean_angle, kappa=15.0)

        # --- 3. Fill Outputs ---

        # A. Visual Response [Takeuchi 2025]
        # Mostly RUN (0.98), rarely JUMP (0.02)
        if has_visual:
            trigs = np.where(x_t[:, 3] > self.vis_threshold)[0]
            if len(trigs) > 0:
                noise_steps = np.random.randint(-2, 3)
                actual_delay = self.vis_delay_steps + noise_steps
                start = trigs[0] + actual_delay
                if start < self.seq_len:
                    # Ensure positive duration
                    dur = max(int(abs(np.random.normal(0.1, 0.02)) / self.dt), 1)
                    end = min(start + dur, self.seq_len)

                    # High base probability with small per-timestep noise for biological realism
                    window_size = end - start
                    y_t[start:end, 0] = np.clip(0.97 + np.random.uniform(-0.02, 0.02, window_size), 0.0, 1.0)
                    y_t[start:end, 1] = np.clip(0.02 + np.random.uniform(-0.01, 0.01, window_size), 0.0, 1.0)
                    y_t[start:end, 2] = target_cos
                    y_t[start:end, 3] = target_sin

        # B. Audio-Wind Response [Lu 2023]
        if has_audio_wind:
            start = wind_time_idx
            if start < self.seq_len:
                dur = int(0.1 / self.dt)
                end = min(start + dur, self.seq_len)

                # Jump Probability Logic
                if is_jump_primed:
                    # Primed condition: Higher jump probability (~40%)
                    # Clip probabilities to valid range [0, 1]
                    y_t[start:end, 0] = np.clip(0.55 + np.random.normal(0, 0.05), 0.0, 1.0) # Run
                    y_t[start:end, 1] = np.clip(0.45 + np.random.normal(0, 0.05), 0.0, 1.0) # Jump
                else:
                    # Control/Other: Jump ~ 21% -> 0.2
                    y_t[start:end, 0] = np.clip(0.85 + np.random.normal(0, 0.05), 0.0, 1.0) # Run
                    y_t[start:end, 1] = np.clip(0.15 + np.random.normal(0, 0.05), 0.0, 1.0) # Jump

                y_t[start:end, 2] = target_cos
                y_t[start:end, 3] = target_sin

        return x_t, y_t

    def generate_batch(self, batch_size=32):
        X, Y = [], []
        modes = ["visual", "audio_wind", "mixed", "empty"]
        probs = [0.4, 0.3, 0.2, 0.1]
        for _ in range(batch_size):
            m = np.random.choice(modes, p=probs)
            x, y = self.generate_trial(m)
            X.append(x)
            Y.append(y)
        return np.stack(X), np.stack(Y)