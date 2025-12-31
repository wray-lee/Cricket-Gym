import argparse
import yaml
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 确保导入路径正确
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_generator import CricketDataGenerator
from src.model import BioMoR_RNN, BaselineLSTM
from src.evaluator import NeuroEvaluator
from src.loss import BioMoRLoss


EPOCHS = 200
BATCH_SIZE = 64


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Cricket Multi-Sensory Integration Simulation"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--mode", type=str, choices=["data_gen", "train", "eval"], default="train"
    )
    parser.add_argument(
        "--arch", type=str, choices=["BioMoR", "LSTM"], default="BioMoR"
    )
    args = parser.parse_args()

    print(f"[Info] Loading configuration from {args.config}...")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Mode: Data Generation ---
    if args.mode == "data_gen":
        print("[Task] Starting Synthetic Data Generation...")
        gen = CricketDataGenerator(cfg)
        print("Generating a test batch (Batch Size = 32)...")
        X, Y = gen.generate_batch(32)
        print(f"Batch Shape: X={X.shape}, Y={Y.shape}")

        # === [新增] 可视化逻辑升级 (Phase 2) ===
        t = np.linspace(0, cfg["simulation"]["episode_length_ms"], X.shape[1])

        # 将 Cos, Sin 转换为角度 (Degrees) 用于直观显示
        # Y shape: [Batch, Seq, 4] -> [P_run, P_jump, Cos, Sin]
        gt_cos = Y[0, :, 2]
        gt_sin = Y[0, :, 3]
        # 使用 arctan2 计算角度，并转为度数
        gt_angles = np.degrees(np.arctan2(gt_sin, gt_cos))

        # 过滤掉非动作期间的角度 (设为 NaN 以免干扰绘图)
        action_mask = (Y[0, :, 0] + Y[0, :, 1]) > 0.01
        gt_angles[~action_mask] = np.nan

        plt.figure(figsize=(10, 10))  # 加高画布

        # 1. Visual
        plt.subplot(4, 1, 1)
        plt.plot(t, X[0, :, 3], label="Visual Theta", color="blue")
        plt.ylabel("Visual Input")
        plt.legend(loc="upper right")
        plt.title("Sample Trial Trace (Phase 2: Directional Control)")

        # 2. Audio/Wind
        plt.subplot(4, 1, 2)
        plt.plot(t, X[0, :, 2], label="Audio", color="magenta", alpha=0.8)
        plt.plot(t, X[0, :, 0], label="Wind (Cos)", color="cyan", alpha=0.6)
        plt.ylabel("Context Input")
        plt.legend(loc="upper right")

        # 3. Action Probabilities
        plt.subplot(4, 1, 3)
        plt.plot(t, Y[0, :, 0], label="P_run (GT)", color="green", linestyle="--")
        plt.plot(t, Y[0, :, 1], label="P_jump (GT)", color="orange")
        plt.ylabel("Action Prob")
        plt.legend(loc="upper right")

        # 4. [新增] Target Direction
        plt.subplot(4, 1, 4)
        plt.plot(t, gt_angles, label="Target Angle (deg)", color="purple", linewidth=2)
        plt.axhline(
            y=100, color="gray", linestyle=":", alpha=0.5, label="Control (~100°)"
        )
        plt.axhline(
            y=125, color="gray", linestyle=":", alpha=0.5, label="Primed (~125°)"
        )
        plt.ylabel("Direction (deg)")
        plt.ylim(0, 180)  # 限制在 0-180 度范围内显示
        plt.xlabel("Time (ms)")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig("data_gen_test.png")
        print("[Success] Validation plot saved to data_gen_test.png")

    # --- Mode: Training ---
    elif args.mode == "train":
        # LAMBDA_METABOLIC = 0.05 (Moved to Loss Class)
        loss_history = []

        print(f"[Task] Initializing Model Training: {args.arch}...")
        gen = CricketDataGenerator(cfg)

        if args.arch == "BioMoR":
            model = BioMoR_RNN(cfg).to(device)

            print("Applying 'Pulse-Sustain' Bio-Initialization...")

            # 确保 default.yaml 中必须有 bio_initialization 块
            if "bio_initialization" not in cfg:
                raise ValueError("Config missing 'bio_initialization' section!")

            bio_cfg = cfg["bio_initialization"]
            hidden_size = cfg["model"]["hidden_dim"]

            r_idx = slice(0, hidden_size)
            z_idx = slice(hidden_size, 2 * hidden_size)
            n_idx = slice(2 * hidden_size, 3 * hidden_size)

            for name, param in model.named_parameters():
                if "router_rnn" in name:
                    if "bias" in name:
                        param.data.fill_(0.0)
                        param.data[r_idx].fill_(bio_cfg["bias_reset_rest"])
                        param.data[z_idx].fill_(bio_cfg["bias_update_rest"])
                        param.data[n_idx].fill_(0.0)

            # 初始化 Bias Gate (连接 LAL -> Policy/Motor)
            # 初始化 Bias Gate (连接 LAL -> Policy/Motor)
            if "bias_gate.bias" in name:
                # 前2维是 Policy (Run/Jump)，后2维是 Motor (Cos/Sin)
                # 将 Policy Bias 设为负值 (e.g., -2.0)，防止 Audio 独自触发逃跑
                # -5 only wind can trigger escape
                param.data[0:2].fill_(-8.0)
                param.data[2:4].fill_(0.0)  # Motor bias 初始为 0

            # 3. [新增] Weight Initialization (切断 LAL 对 Trigger 的直接驱动)
            # 这里的 Weight 是 "连接强度"
            if "bias_gate.weight" in name:
                # param shape is [4, hidden_dim]
                # 前2行对应 Policy (Run/Jump)
                # 强行设为 0，意味着初始状态下，LAL 无论多兴奋，都无法直接推动 Policy
                param.data[0:2, :].fill_(0.0)

                # 后2行对应 Motor (Direction)，保持随机或设为较小值
                # 让 LAL 主要影响方向，而不是触发
                param.data[2:4, :].normal_(0, 0.01)
                
            # [CRITICAL] Boost Reflex Weights to overcome Deep Inhibition (-8.0)
            # Explicitly access policy_head instead of loop string matching to be safe
            if hasattr(model, "policy_head"):
                 model.policy_head.weight.data.normal_(0, 1.0)
                 print("[Init] Boosted policy_head weights for Reflex Pathway efficiency.")

            with torch.no_grad():
                w_ih = model.router_rnn.weight_ih
                w_ih.fill_(0.0)

                # Wind
                w_ih[z_idx, 0:2].fill_(bio_cfg["weight_wind_open"])
                w_ih[n_idx, 0:2].fill_(bio_cfg["weight_wind_drive"])

                # Audio
                w_ih[z_idx, 2].fill_(bio_cfg["weight_audio_open"])
                w_ih[n_idx, 2].fill_(bio_cfg["weight_audio_drive"])

                # Visual
                w_ih[z_idx, 3:].fill_(bio_cfg["weight_visual_open"])
                w_ih[n_idx, 3:].normal_(bio_cfg["weight_visual_drive"], 0.05)

            save_name = "cricket_biomor.pth"
        else:
            model = BaselineLSTM(cfg).to(device)
            save_name = "cricket_baseline.pth"

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # [Phase 2] Loss Function
        criterion = BioMoRLoss(lambda_dir=1.0, lambda_act=0.05).to(device)

        print("Starting training loop...")
        try:
            for epoch in range(EPOCHS):
                X_np, Y_np = gen.generate_batch(BATCH_SIZE)
                X = torch.tensor(X_np, dtype=torch.float32).to(device)
                Y = torch.tensor(Y_np, dtype=torch.float32).to(device)

                optimizer.zero_grad()

                if args.arch == "BioMoR":
                    Y_pred, router_state = model(X)
                    # print(
                    #     f"DEBUG: X shape: {X.shape}, Y shape: {Y.shape}, Y_pred shape: {Y_pred.shape}"
                    # )
                    total_loss, loss_dict = criterion(Y_pred, Y, router_state)

                else:
                    Y_pred, _ = model(X)
                    dummy_state = torch.zeros_like(Y_pred)
                    total_loss, loss_dict = criterion(Y_pred, Y, dummy_state)

                total_loss.backward()
                optimizer.step()

                loss_history.append(total_loss.item())
                if (epoch + 1) % 2 == 0:
                    print(
                        f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss.item():.6f} "
                        f"(Cls: {loss_dict['cls']:.4f}, Dir: {loss_dict['dir']:.4f}, Act: {loss_dict['act']:.4f})"
                    )

            torch.save(model.state_dict(), save_name)
            print(f"Saved model to {save_name}")

            # Plot Loss
            plt.figure()
            plt.plot(loss_history)
            plt.title(f"Training Loss ({args.arch})")
            plt.savefig(f"loss_{args.arch}.png")

        except KeyboardInterrupt:
            print("Interrupted. Saving model checkpoint...")
            torch.save(model.state_dict(), save_name)
            print(f"Saved model to {save_name} (Partial)")

    # --- Mode: Evaluation ---
    elif args.mode == "eval":
        print("[Task] Running Physiological Verification (Phase 2)...")
        model_path = "cricket_biomor.pth"

        if not os.path.exists(model_path):
            print(f"[Error] Model {model_path} not found!")
            return

        # 注意：Eval 阶段也需要更新，以支持方向分析
        evaluator = NeuroEvaluator(cfg, model_path, device)

        print("1. Analyzing Audio-Wind Priming (Jump Prob & Direction)...")
        evaluator.analyze_trial("audio_wind")

        print("2. Analyzing Visual Looming (Direction Control)...")
        evaluator.analyze_trial("visual")

        print("[Success] Evaluation Complete.")


if __name__ == "__main__":
    main()
