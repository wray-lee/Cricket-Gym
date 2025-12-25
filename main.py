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
from src.evaluator import NeuroEvaluator  # 确保在文件头导入


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

        # 简单绘图验证 (保持原样)
        t = np.linspace(0, cfg["simulation"]["episode_length_ms"], X.shape[1])
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, X[0, :, 3], label="Visual Theta")
        plt.legend()
        plt.ylabel("Visual")
        plt.subplot(3, 1, 2)
        plt.plot(t, X[0, :, 2], label="Audio", color="magenta")
        plt.legend()
        plt.ylabel("Audio")
        plt.subplot(3, 1, 3)
        plt.plot(t, Y[0, :, 1], label="P_jump (GT)", color="orange")
        plt.legend()
        plt.ylabel("Output")
        plt.savefig("data_gen_test.png")
        print("[Success] Plot saved.")

    # --- Mode: Training ---
    elif args.mode == "train":
        print(f"[Task] Initializing Model Training: {args.arch}...")
        gen = CricketDataGenerator(cfg)

        if args.arch == "BioMoR":
            model = BioMoR_RNN(cfg).to(device)

            print("Applying 'Pulse-Sustain' Bio-Initialization...")

            bio_cfg = cfg["bio_initialization"]  # Initialization Biology parameters
            hidden_size = cfg["model"]["hidden_dim"]

            # PyTorch GRU 参数切片索引
            r_idx = slice(0, hidden_size)  # Reset Gate
            z_idx = slice(hidden_size, 2 * hidden_size)  # Update Gate (Memory Control)
            n_idx = slice(2 * hidden_size, 3 * hidden_size)  # New Gate (Content)

            for name, param in model.named_parameters():
                if "router_rnn" in name:
                    if "bias" in name:
                        # 1. 初始化所有 Bias
                        param.data.fill_(0.0)

                        # Reset Gate (-3.0): 忽略旧杂讯
                        param.data[r_idx].fill_(bio_cfg["bias_reset_rest"])

                        # Update Gate (+6.0): 强记忆模式 (Crucial!)
                        # Sigmoid(6.0) ≈ 1.0 -> h_t = h_{t-1}. 只有强输入能打破这个状态。
                        # Strong Memory!
                        param.data[z_idx].fill_(bio_cfg["bias_update_rest"])

                        # New Gate (0.0): 静息电位
                        param.data[n_idx].fill_(0.0)

            with torch.no_grad():
                w_ih = model.router_rnn.weight_ih
                w_ih.fill_(0.0)

                # context indices: 0,1=Wind, 2=Audio, 3,4,5=Visual

                # === 1. Wind Input (Indices 0, 1) ===
                # 这是一个强刺激，需要瞬间打破 Memory 封锁
                # 强力开门 (Update Gate -> Negative)
                w_ih[z_idx, 0:2].fill_(bio_cfg["weight_wind_open"])
                # 强力驱动 (New Gate -> Positive)
                w_ih[n_idx, 0:2].fill_(bio_cfg["weight_wind_drive"])

                # === 2. Audio Input (Index 2) ===
                # 这是一个启动信号 (Priming)
                # 开门 (Update Gate)
                w_ih[z_idx, 2].fill_(bio_cfg["weight_audio_open"])
                # 驱动 (New Gate) - 稍微弱一点，维持 Base Level 即可
                w_ih[n_idx, 2].fill_(bio_cfg["weight_audio_drive"])

                # === 3. Visual Input (Index 3,4,5) ===
                w_ih[z_idx, 3:].fill_(bio_cfg["weight_visual_open"])
                w_ih[n_idx, 3:].normal_(bio_cfg["weight_visual_drive"], 0.05)

            save_name = "cricket_biomor.pth"
        else:
            model = BaselineLSTM(cfg).to(device)
            save_name = "cricket_baseline.pth"

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        EPOCHS = 200
        BATCH_SIZE = 64
        LAMBDA_METABOLIC = 0.05  # L1 regularization for sparsity
        loss_history = []

        print("Starting training loop...")
        try:
            for epoch in range(EPOCHS):
                X_np, Y_np = gen.generate_batch(BATCH_SIZE)
                X = torch.tensor(X_np, dtype=torch.float32).to(device)
                Y = torch.tensor(Y_np, dtype=torch.float32).to(device)

                optimizer.zero_grad()

                if args.arch == "BioMoR":
                    Y_pred, router_state = model(X)
                    mse_loss = criterion(Y_pred, Y)
                    # L1 Penalty: 鼓励稀疏发放，静息时不耗能
                    activity_loss = torch.mean(torch.abs(router_state))
                    total_loss = mse_loss + LAMBDA_METABOLIC * activity_loss
                else:
                    Y_pred, _ = model(X)
                    total_loss = criterion(Y_pred, Y)
                    mse_loss = total_loss  # Placeholder

                total_loss.backward()
                optimizer.step()

                loss_history.append(total_loss.item())
                if (epoch + 1) % 20 == 0:
                    act_val = activity_loss.item() if args.arch == "BioMoR" else 0.0
                    print(
                        f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss.item():.6f} (MSE: {mse_loss.item():.6f}, Act: {act_val:.6f})"
                    )

            torch.save(model.state_dict(), save_name)
            print(f"Saved model to {save_name}")

            # Plot Loss
            plt.figure()
            plt.plot(loss_history)
            plt.title(f"Training Loss ({args.arch})")
            plt.savefig(f"loss_{args.arch}.png")

        except KeyboardInterrupt:
            print("Interrupted.")

    # --- Mode: Evaluation ---
    elif args.mode == "eval":
        print("[Task] Running Physiological Verification...")
        model_path = "cricket_biomor.pth"

        if not os.path.exists(model_path):
            print(f"[Error] Model {model_path} not found!")
            return

        evaluator = NeuroEvaluator(cfg, model_path, device)
        print("Analyzing Audio-Wind Priming...")
        evaluator.analyze_trial("audio_wind")
        print("Analyzing Visual Looming...")
        evaluator.analyze_trial("visual")
        print("[Success] Evaluation Complete.")


if __name__ == "__main__":
    main()
