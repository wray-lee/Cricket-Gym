import torch
import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.model import BioMoR_RNN
from src.cricket_env import CricketEscapeEnv

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_simulation():
    # 1. Config & Model
    cfg = load_config("configs/default.yaml")
    device = torch.device("cpu")

    model = BioMoR_RNN(cfg).to(device)
    model_path = "models/cricket_biomor.pth"

    if os.path.exists(model_path):
        print(f"Loading brain from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Error: Model weights not found. Train first!")
        return

    model.eval()

    # 2. Environment
    env = CricketEscapeEnv(cfg)
    obs = env.reset()

    h_router = None
    last_action = None

    max_steps = 300 # 6 seconds (足够长)

    print(f"Starting Simulation... (Predator at {env.predator_pos}, Cricket at {env.cricket_pos})")

    for t in range(max_steps):
        x_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_out, h_router, last_action = model(x_tensor, h_router, last_action)

        act_np = action_out.squeeze(0).numpy()
        p_run, p_jump = act_np[0], act_np[1]

        # [修改] 无条件打印关键帧
        vis_theta = obs[3]
        vis_deg = np.degrees(vis_theta)
        dist = np.linalg.norm(env.predator_pos - env.cricket_pos)

        # 每10帧打印一次，或者当 Run 概率高时打印
        if t % 10 == 0 or p_run > 0.1:
             print(f"Step {t:03d}: Dist={dist:.2f}cm | Theta={vis_deg:.1f}° | P_Run={p_run:.2f} P_Jump={p_jump:.2f}")

        obs, collided = env.step(act_np)

        if collided:
            print(f"Game Over at step {t}! (Eaten at Dist={dist:.2f}cm)")
            break

    if not collided:
        print("Success! Cricket Survived.")

    env.render()

if __name__ == "__main__":
    run_simulation()