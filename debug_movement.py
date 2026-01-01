"""
调试脚本：检查模型输出和蟋蟀移动
"""
import numpy as np
import torch
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.model import BioMoR_RNN
from src.cricket_env import CricketEscapeEnv

def main():
    # 加载模型
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model = BioMoR_RNN(cfg).to(device)
    model.load_state_dict(torch.load("models/cricket_biomor.pth", map_location=device))
    model.eval()

    env = CricketEscapeEnv(cfg)
    obs = env.reset()

    print("="*60)
    print("初始状态")
    print("="*60)
    print(f"蟋蟀位置: {env.cricket_pos}")
    print(f"捕食者位置: {env.predator_pos}")
    print(f"捕食者速度: {env.predator_vel}")
    print(f"距离: {np.linalg.norm(env.predator_pos - env.cricket_pos):.2f} cm")
    print(f"\n蟋蟀速度参数:")
    print(f"  run_speed: {env.run_speed} cm/s")
    print(f"  jump_speed: {env.jump_speed} cm/s")
    print(f"  dt: {env.dt} s")

    h_router = None
    last_action = None

    print(f"\n{'='*60}")
    print("前10步的模型输出")
    print("="*60)

    for t in range(10):
        x_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_out, h_router, last_action = model(x_tensor, h_router, last_action)

        act_np = action_out.squeeze(0).numpy()
        p_run, p_jump, d_cos, d_sin = act_np

        print(f"\n步骤 {t}:")
        print(f"  p_run: {p_run:.4f}, p_jump: {p_jump:.4f}")
        print(f"  方向: ({d_cos:.4f}, {d_sin:.4f})")
        print(f"  蟋蟀位置: {env.cricket_pos}")
        print(f"  蟋蟀速度: {env.current_speed:.4f} cm/s")

        obs, collided = env.step(act_np)

        if collided:
            print(f"\n在步骤 {t} 被抓!")
            break

    print(f"\n{'='*60}")
    print("诊断结果")
    print("="*60)
    if env.current_speed < 1.0:
        print("⚠️ 蟋蟀速度太低！模型可能没有输出足够的逃跑概率")
    if p_run < 0.5 and p_jump < 0.5:
        print("⚠️ 模型输出的逃跑概率太低！")

    print(f"\n实际移动距离: {np.linalg.norm(env.cricket_pos - np.array([50.0, 30.0])):.2f} cm")

if __name__ == "__main__":
    main()
