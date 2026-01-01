import numpy as np
import matplotlib.pyplot as plt

dist_min = 30.0  # 进一步缩短距离以降低生存率
dist_max = 40.0

speed_min = 5.0  # 降低速度以提高生存率到60-70%
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
        self.run_speed = 35.0   # cm/s (提高到35以获得更长的逃避轨迹)
        self.jump_speed = 100.0 # cm/s (Burst)
        self.friction = 0.9     # speed decay
        self.current_speed = 0.0

        # History for plotting
        self.history = {'cricket': [], 'predator': [], 'action': []}

    def reset(self):
        self.cricket_pos = np.array([50.0, 20.0])
        self.cricket_heading = np.pi / 2 # Facing up (towards danger initially)

        # predator 参数设置
        initial_distance = np.random.uniform(dist_min, dist_max)  # 距离蟋蟀35-45cm
        self.predator_pos = np.array([50.0, 20.0 + initial_distance])

        speed = np.random.uniform(speed_min, speed_max)  # 提高速度以增加生存压力
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

        # Thresholds (平衡反应速度和移动能力)
        RUN_TH = 0.2  # 提高到0.4以进一步延迟反应
        JUMP_TH = 0.5

        if p_jump > JUMP_TH:
            move_speed = self.jump_speed
            action_type = "Jump"
        elif p_run > RUN_TH:
            move_speed = self.run_speed
            action_type = "Run"

        # 2. Decode Direction
        # [关键修复] 坐标系转换
        # 训练数据中: Wind表示气流流动方向（捕食者速度方向）
        #            target表示在Wind坐标系中的逃跑方向
        # 模型输出的(d_cos, d_sin)是在Wind坐标系中的逃跑方向向量
        # 需要将其从Wind坐标系旋转到全局坐标系

        # 计算Wind在全局坐标系中的角度（捕食者速度方向）
        rel_pos = self.predator_pos - self.cricket_pos
        vel_norm = np.linalg.norm(self.predator_vel)
        if vel_norm > 0.1:
            wind_global_angle = np.arctan2(self.predator_vel[1], self.predator_vel[0])
        else:
            # 后备：使用位置方向
            wind_global_angle = np.arctan2(rel_pos[1], rel_pos[0])

        # 模型输出在Wind坐标系中的角度
        # 例如: (cos=-1, sin=0) -> 180度（背离Wind）
        #      (cos=-0.342, sin=0.940) -> 110度（向左上逃离）
        model_angle_in_wind_frame = np.arctan2(d_sin, d_cos)

        # 转换到全局坐标系
        # 全局逃跑方向 = Wind全局角度 + 180度 + 模型在Wind坐标系中的角度
        # Wind指向蟋蟀，加180度反转后指向捕食者的反方向
        # 添加随机噪声以增加轨迹多样性
        # 使用混合分布：大部分时候小噪声，偶尔大幅偏离
        if np.random.random() < 0.1:  # 10%概率大幅偏离
            noise = np.random.uniform(-1.0, 1.0)  # ±1.0弧度 ≈ ±57度
        else:
            noise = np.random.uniform(-0.3, 0.3)  # ±0.3弧度 ≈ ±17度
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

        # B. Wind (气流刺激)
        # [最终修复] Wind表示气流吹来的方向
        # 在真实实验中：气流从捕食者位置吹向蟋蟀
        # Wind = 从捕食者指向蟋蟀的方向
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