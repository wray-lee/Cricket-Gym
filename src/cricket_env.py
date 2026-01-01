import numpy as np
import matplotlib.pyplot as plt

# Predator initial distance range (cm)
dist_min = 30.0
dist_max = 40.0

# Predator speed range (cm/s) - tuned for 60-70% survival rate
speed_min = 5.0
speed_max = 7.0
class CricketEscapeEnv:
    def __init__(self, config):
        self.cfg = config
        self.dt = config["simulation"]["dt"]
        self.width = 100.0  # cm
        self.height = 100.0 # cm

        # State
        self.cricket_pos = np.array([50.0, 50.0])
        self.cricket_heading = np.random.uniform(0, 2*np.pi)

        # Stimulus
        self.predator_pos = np.array([50.0, 80.0]) # Start top center
        # [Modification 1] Reduce speed to 5.0 cm/s to allow reaction time > 74ms
        self.predator_vel = np.array([0.0, -5.0])
        self.is_collided = False

        # Physics limits
        self.run_speed = 35.0   # Cricket running speed (cm/s)
        self.jump_speed = 100.0 # cm/s (Burst)
        self.friction = 0.9     # speed decay
        self.current_speed = 0.0

        # History for plotting
        self.history = {'cricket': [], 'predator': [], 'action': []}

    def reset(self):
        self.cricket_pos = np.array([50.0, 20.0])
        self.cricket_heading = np.pi / 2 # Facing up (towards danger initially)

        # Initialize predator parameters
        initial_distance = np.random.uniform(dist_min, dist_max)
        self.predator_pos = np.array([50.0, 20.0 + initial_distance])

        speed = np.random.uniform(speed_min, speed_max)
        angle = np.random.uniform(-0.1, 0.1) - (np.pi/2) # Moving down roughly
        self.predator_vel = np.array([np.cos(angle)*speed, np.sin(angle)*speed])

        self.current_speed = 0.0
        self.is_collided = False
        self.history = {'cricket': [], 'predator': [], 'action': []}

        return self._get_obs()

    def step(self, action):
        # action: [P_run, P_jump, Cos, Sin]
        p_run, p_jump, d_cos, d_sin = action

        # 1. Decode Intention
        move_speed = 0.0
        action_type = "Stay"

        # Action probability thresholds
        RUN_TH = 0.2   # Run threshold
        JUMP_TH = 0.5  # Jump threshold

        if p_jump > JUMP_TH:
            move_speed = self.jump_speed
            action_type = "Jump"
        elif p_run > RUN_TH:
            move_speed = self.run_speed
            action_type = "Run"

        # 2. Decode Direction
        # Coordinate transformation: Model outputs direction in wind-relative frame
        # Wind frame: aligned with predator velocity direction
        # Need to transform from wind frame to global frame

        # Calculate wind direction in global frame (predator velocity direction)
        rel_pos = self.predator_pos - self.cricket_pos
        vel_norm = np.linalg.norm(self.predator_vel)
        if vel_norm > 0.1:
            wind_global_angle = np.arctan2(self.predator_vel[1], self.predator_vel[0])
        else:
            # Fallback: use position-based direction
            wind_global_angle = np.arctan2(rel_pos[1], rel_pos[0])

        # Model output angle in wind-relative frame
        # Example: (cos=-1, sin=0) -> 180° (away from wind)
        model_angle_in_wind_frame = np.arctan2(d_sin, d_cos)

        # Transform to global frame
        # Global escape direction = wind angle + 180° + model angle in wind frame
        # Add noise for trajectory variability (mixed distribution)
        if np.random.random() < 0.1:  # 10% chance of large deviation
            noise = np.random.uniform(-1.0, 1.0)  # ±57°
        else:
            noise = np.random.uniform(-0.3, 0.3)  # ±17°
        target_heading = wind_global_angle + np.pi + model_angle_in_wind_frame + noise

        # 3. Physics Update
        if move_speed > 0:
            # Smooth turn (simple)
            self.cricket_heading = target_heading
            # Acceleration
            self.current_speed = move_speed
        else:
            # Deceleration
            self.current_speed *= self.friction

        vel = np.array([np.cos(self.cricket_heading), np.sin(self.cricket_heading)]) * self.current_speed * self.dt
        self.cricket_pos += vel

        # Predator Update (Constant velocity)
        self.predator_pos += self.predator_vel * self.dt

        # 4. Check Collision
        dist = np.linalg.norm(self.cricket_pos - self.predator_pos)
        if dist < 2.0: # 2cm radius
            self.is_collided = True

        # 5. Record
        self.history['cricket'].append(self.cricket_pos.copy())
        self.history['predator'].append(self.predator_pos.copy())
        self.history['action'].append(action_type)

        return self._get_obs(), self.is_collided

    def _get_obs(self):
        # Calculate sensory inputs based on physical state

        # A. Visual (Looming)
        rel_pos = self.predator_pos - self.cricket_pos
        dist = np.linalg.norm(rel_pos)

        # Angle of predator relative to cricket heading
        global_angle = np.arctan2(rel_pos[1], rel_pos[0])
        ego_angle = global_angle - self.cricket_heading
        # Normalize to [-pi, pi]
        ego_angle = (ego_angle + np.pi) % (2 * np.pi) - np.pi

        # Check if predator is in Field of View (+- 120 deg)
        is_visible = np.abs(ego_angle) < np.deg2rad(120)

        theta = 0.0
        d_theta = 0.0
        vis_on = 0.0

        if is_visible and dist > 0.1:
            # Simple size model: size / dist
            # Predator size ~ 2cm
            half_angle = np.arctan(1.0 / dist)
            theta = 2 * half_angle # rad

            # Approximate d_theta
            rel_vel = self.predator_vel
            v_approach = -np.dot(rel_vel, rel_pos / dist)
            if v_approach > 0:
                d_theta = (2 * v_approach) / dist
                # Clip to match bio-constraints (Same as DataGenerator)
                d_theta = np.clip(d_theta, 0, 4.0)

            vis_on = 1.0

        # B. Wind stimulus
        # Wind represents airflow direction from predator to cricket
        # In real experiments: airflow blows from predator position toward cricket
        wind_cos = np.cos(global_angle)
        wind_sin = np.sin(global_angle)

        # C. Audio (None for simple chase)
        audio = 0.0

        # [Wind_x, Wind_y, Audio, Vis_Theta, Vis_dTheta, Vis_On]
        return np.array([wind_cos, wind_sin, audio, theta, d_theta, vis_on], dtype=np.float32)

    def render(self):
        # Static plot of the trajectory
        c_hist = np.array(self.history['cricket'])
        p_hist = np.array(self.history['predator'])
        acts = self.history['action']

        plt.figure(figsize=(6, 6))

        # [Modification 4] Mark Start point explicitly
        plt.scatter(c_hist[0,0], c_hist[0,1], c='blue', s=100, label='Start', marker='o')

        plt.plot(c_hist[:,0], c_hist[:,1], 'b-', linewidth=2, label='Cricket Path')
        plt.plot(p_hist[:,0], p_hist[:,1], 'r--', linewidth=2, label='Predator Path')

        # Mark Actions
        has_run = False
        for i, act in enumerate(acts):
            if act == "Jump":
                plt.scatter(c_hist[i,0], c_hist[i,1], c='orange', s=30, zorder=3)
            elif act == "Run" and not has_run:
                plt.scatter(c_hist[i,0], c_hist[i,1], c='cyan', s=30, zorder=3, marker='x')
                has_run = True

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("Closed-Loop Survival Test")
        plt.savefig("closed_loop_result.png")
        print("Saved trajectory to closed_loop_result.png")