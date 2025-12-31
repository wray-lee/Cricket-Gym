import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BaselineLSTM(nn.Module):
    """
    Standard LSTM Baseline for comparison.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.input_dim = config["model"]["input_dim"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.output_dim = config["model"]["output_dim"]

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.head(lstm_out)
        probs = torch.sigmoid(logits[:, :, 0:2])
        dirs = torch.tanh(logits[:, :, 2:4])
        return torch.cat([probs, dirs], dim=-1), None


class BioMoR_RNN(nn.Module):
    """
    Biologically-Constrained Mixture-of-Recursions (MoR) - Phase 2
    Updated with explicit Policy Head (Classification) and Motor Head (Regression).
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        hidden_dim = config["model"]["hidden_dim"]
        # output_dim 依然是 4: [P_run, P_jump, Cos_phi, Sin_phi]

        # --- 1. Feature Engineering ---
        self.reflex_input_dim = 2  # Wind Only (Fast)
        self.context_input_dim = 6  # Wind + Audio + Visual (Slow/LAL)

        # --- 2. The "Router" (LAL - Lateral Accessory Lobe) ---
        self.router_rnn = nn.GRUCell(self.context_input_dim, hidden_dim)

        # Gates (Top-Down Modulation)
        # Gain: 调节 Reflex Backbone 的输入敏感度
        self.gain_gate = nn.Linear(hidden_dim, self.reflex_input_dim)

        # Bias: 包含 Policy Bias (Run/Jump倾向) 和 Motor Bias (方向偏差)
        # 这是一个关键的 Context 注入点，解释了 Lu et al. (2023) 中的 Biased Backward 现象
        self.bias_gate = nn.Linear(hidden_dim, 4)

        # --- 3. The "Reflex" (DNs - Descending Neurons) ---
        # A. Shared Backbone: 提取运动特征
        self.reflex_backbone = nn.Sequential(
            nn.Linear(self.reflex_input_dim, hidden_dim), nn.ReLU()
        )

        # B. Policy Head (Classification): Run vs Jump
        self.policy_head = nn.Linear(hidden_dim, 2)

        # C. Motor Head (Regression): Direction Control [Cos, Sin]
        self.motor_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        # Input Splitting
        x_reflex = x[:, :, 0:2]  # Wind
        x_audio = x[:, :, 2:3]
        x_visual = x[:, :, 3:6]
        # Context sees everything (Wind added per previous discussion)
        x_context = torch.cat([x[:, :, 0:2], x_audio, x_visual], dim=-1)

        # Initialize States
        h_router = torch.zeros(batch_size, self.cfg["model"]["hidden_dim"]).to(x.device)

        y_preds = []
        router_states = []

        for t in range(seq_len):
            # 1. Update Router (LAL)
            h_router = self.router_rnn(x_context[:, t, :], h_router)

            # 2. Compute Modulations
            gain = torch.sigmoid(self.gain_gate(h_router))  # Gain [0, 1]
            bias = self.bias_gate(h_router)  # Bias (unbounded)

            # Split Bias for different heads
            bias_policy = bias[:, 0:2]  # 影响 Run/Jump
            bias_motor = bias[:, 2:4]  # 影响方向 (e.g. shift to 130 deg)

            # 3. Reflex Pathway Execution
            # Apply Gain Gating
            modulated_input = x_reflex[:, t, :] * gain

            # Pass through Shared Backbone
            reflex_features = self.reflex_backbone(modulated_input)

            # --- Head 1: Policy (Classification) ---
            # Logits + Context Bias
            policy_logits = self.policy_head(reflex_features) + bias_policy

            # 移除sigmod,通过loss处理
            # probs = torch.sigmoid(policy_logits)
            probs = policy_logits

            # --- Head 2: Motor (Regression) ---
            # Logits + Directional Bias
            motor_logits = self.motor_head(reflex_features) + bias_motor
            # 使用 Tanh 强制输出在 [-1, 1] 之间，符合 Cos/Sin 定义
            dirs = torch.tanh(motor_logits)

            # 4. Concatenate Output
            y_t = torch.cat([probs, dirs], dim=1)

            y_preds.append(y_t.unsqueeze(1))
            router_states.append(h_router.unsqueeze(1))

        y_pred = torch.cat(y_preds, dim=1)
        router_activity = torch.cat(router_states, dim=1)

        return y_pred, router_activity
