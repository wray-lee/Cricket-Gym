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
    Biologically-Constrained Mixture-of-Recursions (MoR).
    RE-WIRED: LAL now receives Ascending Neuron (AN) inputs from Wind.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        hidden_dim = config["model"]["hidden_dim"]
        output_dim = config["model"]["output_dim"]

        # --- 1. Feature Engineering ---
        # Reflex (Fast): Wind_Cos(0), Wind_Sin(1) -> 2 dims
        self.reflex_input_dim = 2

        # Context (LAL Inputs):
        # NOW INCLUDES WIND (via Ascending Neurons)!
        # Wind(2) + Audio(1) + Visual(3) = 6 dims
        self.context_input_dim = 6

        # --- 2. The "Router" (LAL) ---
        self.router_rnn = nn.GRUCell(self.context_input_dim, hidden_dim)

        # Gates
        self.gain_gate = nn.Linear(hidden_dim, self.reflex_input_dim)
        self.bias_gate = nn.Linear(hidden_dim, output_dim)

        # --- 3. The "Reflex" (DNs) ---
        self.reflex_layer = nn.Sequential(
            nn.Linear(self.reflex_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        # Reflex Pathway: Only sees Wind
        x_reflex = x[:, :, 0:2]

        # Context Pathway: Sees EVERYTHING (Integrator)
        # Wind (0,1) + Audio (2) + Visual (3,4,5)
        x_context = x  # 直接使用全部输入

        # Initialize States
        h_router = torch.zeros(batch_size, self.cfg["model"]["hidden_dim"]).to(x.device)

        y_preds = []
        router_states = []

        for t in range(seq_len):
            # 1. Update Router (LAL) State
            h_router = self.router_rnn(x_context[:, t, :], h_router)

            # 2. Modulation
            gain = torch.sigmoid(self.gain_gate(h_router))
            bias = self.bias_gate(h_router)

            # 3. Reflex Execution
            modulated_input = x_reflex[:, t, :] * gain
            reflex_out = self.reflex_layer(modulated_input)
            final_logit = reflex_out + bias

            # 4. Decode
            probs = torch.sigmoid(final_logit[:, 0:2])
            dirs = torch.tanh(final_logit[:, 2:4])
            y_t = torch.cat([probs, dirs], dim=1)

            y_preds.append(y_t.unsqueeze(1))
            router_states.append(h_router.unsqueeze(1))

        y_pred = torch.cat(y_preds, dim=1)
        router_activity = torch.cat(router_states, dim=1)

        return y_pred, router_activity
