import argparse
import yaml
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure import paths are correct
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_generator import CricketDataGenerator
from src.model import BioMoR_RNN, BaselineLSTM
from src.evaluator import NeuroEvaluator
from src.loss import BioMoRLoss


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Cricket Multi-Sensory Integration Simulation (Phase 3)"
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
    print(f"[Info] Using device: {device}")

    # --- Mode: Data Generation (Validation) ---
    if args.mode == "data_gen":
        print("[Task] Starting Synthetic Data Generation Validation...")
        gen = CricketDataGenerator(cfg)

        print("Generating a PURE VISUAL trial for validation...")
        X_vis, Y_vis = gen.generate_trial("visual")

        X = X_vis[np.newaxis, :, :]
        Y = Y_vis[np.newaxis, :, :]

        t = np.linspace(0, cfg["simulation"]["episode_length_ms"], X.shape[1])
        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, X[0, :, 3], label="Visual Theta (rad)", color="blue")
        th_deg = cfg["visual"]["looming_threshold_deg"]
        plt.axhline(y=np.radians(th_deg), color='red', linestyle='--', label=f"Threshold {th_deg}°")
        plt.legend()
        plt.ylabel("Visual Input")
        plt.title("DataGen Check: Visual")

        plt.subplot(3, 1, 2)
        plt.plot(t, X[0, :, 0], label="Wind Cos", color="cyan")
        plt.plot(t, X[0, :, 2], label="Audio", color="magenta")
        plt.legend()
        plt.ylabel("Reflex Input")
        plt.title("DataGen Check: Wind/Audio (Should be 0)")

        plt.subplot(3, 1, 3)
        plt.plot(t, Y[0, :, 1], label="GT Jump Prob", color="green", linewidth=2)
        plt.legend()
        plt.ylabel("Output")
        plt.title("DataGen Check: GT Action")

        plt.tight_layout()
        plt.savefig("data_gen_test.png")
        print("[Success] Validation plot saved to data_gen_test.png")

    # --- Mode: Training ---
    elif args.mode == "train":

        EPOCHS = 700  # 增加轮数以获得更好收敛
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0005  # 降低学习率以提高稳定性
        GRAD_CLIP = 1.0

        print(f"[Task] Initializing Model Training: {args.arch}...")
        gen = CricketDataGenerator(cfg)

        if args.arch == "BioMoR":
            model = BioMoR_RNN(cfg).to(device)
            save_name = "models/cricket_biomor.pth"

            print("Applying 'Pulse-Sustain' Bio-Initialization (Fully Configurable)...")

            if "bio_initialization" not in cfg:
                raise ValueError("Config missing 'bio_initialization' section!")

            bio_cfg = cfg["bio_initialization"]
            hidden_size = cfg["model"]["hidden_dim"]

            # === 1. Router Internal Gates (Reset/Update) ===
            r_idx = slice(0, hidden_size)
            z_idx = slice(hidden_size, 2 * hidden_size)
            n_idx = slice(2 * hidden_size, 3 * hidden_size)

            for name, param in model.named_parameters():
                if "router_rnn" in name and "bias" in name:
                    param.data.fill_(0.0)
                    param.data[r_idx].fill_(bio_cfg.get("bias_reset_rest", -2.0))
                    param.data[z_idx].fill_(bio_cfg.get("bias_update_rest", 3.0))
                    param.data[n_idx].fill_(0.0)

            # === 2. Output Gating (Bias Gate) ===
            if hasattr(model, "bias_gate"):
                # 读取基础偏置
                gate_bias_val = bio_cfg.get("bias_gate", -1.5)

                # [关键] 读取分层权重 (不再硬编码)
                w_policy = bio_cfg.get("weight_gate_policy", 0.5) # Run/Jump
                w_motor = bio_cfg.get("weight_gate_motor", 1.0)   # Cos/Sin

                # Bias Init
                model.bias_gate.bias.data[0:2].fill_(gate_bias_val) # Policy inhibited
                model.bias_gate.bias.data[2:4].fill_(0.0)           # Motor neutral

                # Weight Init (Surgical)
                model.bias_gate.weight.data[0:2, :].normal_(0, w_policy) # Modest for Prob
                model.bias_gate.weight.data[2:4, :].normal_(0, w_motor)  # Strong for Direction

                print(f"[Init] Bias Gate: Bias={gate_bias_val}, W_Policy={w_policy}, W_Motor={w_motor}")

            # === 3. Policy Head ===
            if hasattr(model, "policy_head"):
                policy_bias_val = bio_cfg.get("bias_policy_head", -0.5)
                # [关键] 读取 Policy Head 权重 (不再硬编码)
                policy_weight_std = bio_cfg.get("weight_policy_head", 0.5)
                policy_bias_val = bio_cfg.get("bias_policy_head", -6.0)

                model.policy_head.bias.data.fill_(policy_bias_val)
                model.policy_head.weight.data.normal_(0, policy_weight_std)
                print(f"[Init] Policy Head: Bias={policy_bias_val}, W_Std={policy_weight_std}")
                print(f"[Init] Setting Policy Head Bias to: {policy_bias_val}")

            # === 4. Router Weights (Input Pathways & CD) ===
            with torch.no_grad():
                w_ih = model.router_rnn.weight_ih
                w_ih.fill_(0.0)

                # A. Corollary Discharge (Action -> Router Inhibition)
                # [关键] 读取 CD 抑制强度 (不再硬编码)
                cd_weight = bio_cfg.get("weight_corollary_discharge", -2.0)
                # Assumes Action (4 dims) are the LAST 4 columns of input
                w_ih[:, -4:].fill_(cd_weight)
                print(f"[Init] Corollary Discharge (Action->Router) set to: {cd_weight}")

                # B. Sensory Pathways
                # Wind (Idx 0,1)
                w_ih[z_idx, 0:2].fill_(bio_cfg["weight_wind_open"])
                w_ih[n_idx, 0:2].fill_(bio_cfg["weight_wind_drive"])

                # Audio (Idx 2)
                w_ih[z_idx, 2].fill_(bio_cfg["weight_audio_open"])
                w_ih[n_idx, 2].fill_(bio_cfg["weight_audio_drive"])

                # Visual (Idx 3+)
                w_ih[z_idx, 3:].fill_(bio_cfg["weight_visual_open"])
                w_ih[n_idx, 3:].normal_(bio_cfg["weight_visual_drive"], 0.05)

        else:
            model = BaselineLSTM(cfg).to(device)
            save_name = "models/cricket_baseline.pth"

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # === Loss Function Setup ===
        loss_dir_val = cfg["bio_initialization"].get("loss_dir", 2.0)
        loss_act_val = cfg["bio_initialization"].get("loss_act", 0.05) # Increased per new yaml

        criterion = BioMoRLoss(lambda_dir=loss_dir_val, lambda_act=loss_act_val).to(device)
        print(f"[Init] Loss Weights -> Dir: {loss_dir_val}, Act: {loss_act_val}")

        print("Starting training loop...")
        loss_history = []

        try:
            for epoch in range(EPOCHS):
                X_np, Y_np = gen.generate_batch(BATCH_SIZE)
                X = torch.tensor(X_np, dtype=torch.float32).to(device)
                Y = torch.tensor(Y_np, dtype=torch.float32).to(device)

                optimizer.zero_grad()

                if args.arch == "BioMoR":
                    Y_pred, router_state = model(X)
                    total_loss, loss_dict = criterion(Y_pred, Y, router_state)
                else:
                    Y_pred, _ = model(X)
                    dummy_state = torch.zeros_like(Y_pred)
                    total_loss, loss_dict = criterion(Y_pred, Y, dummy_state)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                loss_history.append(total_loss.item())
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss.item():.6f} "
                        f"(Cls: {loss_dict['cls']:.4f}, Dir: {loss_dict['dir']:.4f}, Act: {loss_dict['act']:.4f})"
                    )

            torch.save(model.state_dict(), save_name)
            print(f"Saved model to {save_name}")

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
        print("[Task] Running Physiological Evaluation...")
        model_path = "models/cricket_biomor.pth"

        evaluator = NeuroEvaluator(cfg, model_path, device)

        print("1. Analyzing Audio-Wind Trial...")
        evaluator.analyze_trial("audio_wind")

        print("2. Analyzing Visual Trial...")
        evaluator.analyze_trial("visual")

        print("[Success] Evaluation Complete.")


if __name__ == "__main__":
    main()