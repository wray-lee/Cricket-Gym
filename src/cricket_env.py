import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple

from src.data_generator import CricketDataGenerator


class CricketGym(gym.Env):
    """
    Gymnasium Environment for Cricket Run/Jump Decision Making.

    Observation: (6,) [Wind_cos, Wind_sin, Audio, Vis_theta, Vis_dtheta, Vis_On]
    Action (Pred): (4,) [P_run, P_jump, Cos_dir, Sin_dir]

    This env wraps the synthetic data generator.
    In each step, it feeds the next time-step's sensory data.
    The 'reward' is calculated as the negative distance to the Ground Truth (Supervised Learning Signal).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

    def __init__(self, config: Dict):
        super(CricketGym, self).__init__()
        self.cfg = config

        # Dimensions
        self.input_dim = config["model"]["input_dim"]
        self.output_dim = config["model"]["output_dim"]

        # Define Spaces
        # Observation: Sensory Inputs (Normalized roughly -1 to 1 or 0 to 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.input_dim,), dtype=np.float32
        )

        # Action: Model Outputs (Probabilities + Direction Vectors)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.output_dim,), dtype=np.float32
        )

        # Internal State
        self.generator = CricketDataGenerator(config)
        self.current_step = 0
        self.max_steps = 0
        self.episode_data_x = None  # (Seq, 6)
        self.episode_data_y = None  # (Seq, 4)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment. Generates a new biological trial.
        """
        super().reset(seed=seed)

        # 1. Generate a new trial using the Phase 1 Generator
        # We can pass options to select specific trial types (e.g., 'visual', 'conflict')
        trial_type = "mixed"
        if options and "trial_type" in options:
            trial_type = options["trial_type"]

        self.episode_data_x, self.episode_data_y = self.generator.generate_trial(
            trial_type
        )

        # 2. Reset Counters
        self.current_step = 0
        self.max_steps = self.episode_data_x.shape[0]

        # 3. Get first observation
        observation = self.episode_data_x[self.current_step]
        info = {"trial_type": trial_type}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Args:
            action: The model's prediction for the current timestamp.
        Returns:
            observation: The sensory input for the NEXT timestamp.
        """
        # 1. Calculate Reward (Negative MSE against Ground Truth)
        # Ground Truth for THIS step
        gt = self.episode_data_y[self.current_step]

        # Simple MSE Loss as negative reward
        # We separate Probabilities (0-1) and Direction (-1 to 1)
        mse = np.mean((action - gt) ** 2)
        reward = -mse
        # Optional: Add bonus for low error?
        # reward = 1.0 - mse if mse < 1.0 else 0.0

        # 2. Advance Time
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 3. Get Next Observation
        if terminated:
            # If done, return zero observation or last frame (Gym convention requires valid shape)
            observation = np.zeros(self.input_dim, dtype=np.float32)
        else:
            observation = self.episode_data_x[self.current_step]

        info = {"step": self.current_step, "ground_truth": gt, "mse": mse}

        return observation, reward, terminated, truncated, info

    def render(self):
        # Optional visualization
        pass
