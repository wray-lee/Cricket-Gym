import numpy as np
import math
from typing import Dict, Tuple


class CricketDataGenerator:
    """
    Synthetic Data Generator for Cricket Multi-Sensory Integration (Phase 2).
    Updated with Von Mises directional sampling based on:
    - Lu et al. (2023): Audio-dependent angular bias.
    - Takeuchi (2025): Visual l/v-dependent angular control.
    """

    def __init__(self, config: Dict):
        self.cfg = config
        self.dt = config["simulation"]["dt"]  # seconds
        self.seq_len = int(config["simulation"]["episode_length_ms"] / 1000 / self.dt)

        # Inputs/Outputs Config
        self.input_dim = config["model"]["input_dim"]  # 6
        self.output_dim = config["model"]["output_dim"]  # 4 [P_run, P_jump, Cos, Sin]

        # Biological Constraints
        self.vis_threshold = np.deg2rad(config["visual"]["looming_threshold_deg"])
        self.vis_delay_steps = int(config["visual"]["motor_delay_ms"] / 1000 / self.dt)
        self.wind_delay_steps = int(config["visual"]["wind_delay_ms"] / 1000 / self.dt)
        self.audio_win_start = config["audio"]["coincidence_window_start"] / 1000
        self.audio_win_end = config["audio"]["coincidence_window_end"] / 1000

    def _compute_looming_trajectory(
        self, l_v_ratio: float, collision_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes theta and d_theta for a looming object."""
        time_axis = np.linspace(0, self.seq_len * self.dt, self.seq_len)
        epsilon = 0.05
        t_remain = np.clip(collision_time - time_axis, epsilon, None)
        theta = 2 * np.arctan(l_v_ratio / t_remain)
        d_theta = (2 * l_v_ratio) / (t_remain**2 + l_v_ratio**2)
        mask = time_axis > collision_time
        theta[mask] = 0
        d_theta[mask] = 0
        return theta, d_theta

    def _sample_direction(
        self, mean_deg: float, kappa: float = 4.0
    ) -> Tuple[float, float]:
        """
        Sample a direction from Von Mises distribution.
        Args:
            mean_deg: Mean angle in degrees (0 is front, 180 is back).
            kappa: Concentration parameter (Higher = less variance).
        Returns:
            cos_phi, sin_phi
        """
        # Convert to radians. Note: In our coordinate system, Wind comes from 0 (Front).
        # Escape is typically backwards (around 180 or +/- 100-130).
        mean_rad = np.deg2rad(mean_deg)

        # Sample angle
        angle_rad = np.random.vonmises(mean_rad, kappa)

        # Randomly flip sign for left/right symmetry (crickets escape to left or right rear)
        # We assume mean_deg is the absolute deviation from front.
        if np.random.rand() > 0.5:
            angle_rad = -angle_rad

        return np.cos(angle_rad), np.sin(angle_rad)

    def generate_trial(
        self, trial_type: str = "mixed"
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_t = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        y_t = np.zeros((self.seq_len, self.output_dim), dtype=np.float32)

        # Params
        t_collision = np.random.uniform(1.5, 2.5)
        l_v = np.random.uniform(0.1, 0.4)  # l/v ratio (Size/Velocity)

        wind_time_idx = np.random.randint(
            int(0.5 * self.seq_len), int(0.9 * self.seq_len)
        )
        audio_offset_sec = np.random.uniform(0.5, 2.5)
        audio_time_idx = wind_time_idx - int(audio_offset_sec / self.dt)

        if trial_type == "empty":
            return x_t, y_t

        has_visual = trial_type in ["visual", "mixed", "conflict"]
        has_audio_wind = trial_type in ["audio_wind", "mixed", "conflict"]

        # --- Inputs ---
        if has_audio_wind:
            wind_dur_steps = int(0.2 / self.dt)
            x_t[wind_time_idx : wind_time_idx + wind_dur_steps, 0] = 1.0  # Wind Cos
            x_t[wind_time_idx : wind_time_idx + wind_dur_steps, 1] = (
                0.0  # Wind Sin (From Front)
            )

        if has_audio_wind and audio_time_idx > 0:
            audio_dur_steps = int(0.05 / self.dt)
            x_t[audio_time_idx : audio_time_idx + audio_dur_steps, 2] = 1.0  # Audio

        if has_visual:
            theta, d_theta = self._compute_looming_trajectory(l_v, t_collision)
            x_t[:, 3] = theta
            x_t[:, 4] = d_theta
            x_t[:, 5] = 1.0

        # --- Outputs (Ground Truth) ---

        # 1. Determine Target Direction (Bio-Constraints)
        target_cos, target_sin = -1.0, 0.0  # Default directly back (fallback)

        if has_audio_wind:
            # Check Priming
            dt_sec = (wind_time_idx - audio_time_idx) * self.dt
            is_primed = (dt_sec >= self.audio_win_start) and (
                dt_sec <= self.audio_win_end
            )

            # Lu et al. (2023):
            # Control (Unprimed) -> ~100 deg (Side-backward)
            # Primed (Audio) -> ~125 deg (More backward)
            mean_angle = 125.0 if is_primed else 100.0
            target_cos, target_sin = self._sample_direction(mean_angle, kappa=5.0)

        elif has_visual:
            # Takeuchi (2025):
            # Fast approach (small l/v) -> 180 deg
            # Slow approach (large l/v) -> larger deviation
            # Mapping: l/v=0.1 -> 170 deg, l/v=0.4 -> 130 deg
            mean_angle = 180.0 - (l_v * 130.0)  # Simple linear approximation
            mean_angle = np.clip(mean_angle, 90, 180)
            target_cos, target_sin = self._sample_direction(
                mean_angle, kappa=8.0
            )  # Visual is more precise

        # 2. Fill Output Arrays
        # A. Visual Reaction
        if has_visual:
            trigger_indices = np.where(x_t[:, 3] > self.vis_threshold)[0]
            if len(trigger_indices) > 0:
                reaction_idx = trigger_indices[0] + self.vis_delay_steps
                if reaction_idx < self.seq_len:
                    act_dur = int(0.1 / self.dt)
                    end_idx = min(reaction_idx + act_dur, self.seq_len)

                    y_t[reaction_idx:end_idx, 0] = 0.98  # P_run
                    y_t[reaction_idx:end_idx, 1] = 0.02  # P_jump
                    y_t[reaction_idx:end_idx, 2] = target_cos
                    y_t[reaction_idx:end_idx, 3] = target_sin

        # B. Audio/Wind Reaction
        if has_audio_wind:
            reaction_idx = wind_time_idx + self.wind_delay_steps
            if reaction_idx < self.seq_len:
                act_dur = int(0.1 / self.dt)
                end_idx = min(reaction_idx + act_dur, self.seq_len)

                # Check Priming again for logic clarity
                dt_sec = (wind_time_idx - audio_time_idx) * self.dt
                is_primed = (dt_sec >= self.audio_win_start) and (
                    dt_sec <= self.audio_win_end
                )

                if is_primed:
                    y_t[reaction_idx:end_idx, 0] = 0.1  # P_run
                    y_t[reaction_idx:end_idx, 1] = 0.9  # P_jump
                else:
                    y_t[reaction_idx:end_idx, 0] = 0.9  # P_run
                    y_t[reaction_idx:end_idx, 1] = 0.1  # P_jump

                # Apply Direction
                y_t[reaction_idx:end_idx, 2] = target_cos
                y_t[reaction_idx:end_idx, 3] = target_sin

        return x_t, y_t

    def generate_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        X_list, Y_list = [], []
        # Increase 'mixed' and 'audio_wind' prob to learn direction better
        modes = ["visual", "audio_wind", "mixed", "empty"]
        probs = [0.2, 0.4, 0.3, 0.1]

        for _ in range(batch_size):
            mode = np.random.choice(modes, p=probs)
            x, y = self.generate_trial(mode)
            X_list.append(x)
            Y_list.append(y)

        return np.stack(X_list), np.stack(Y_list)
