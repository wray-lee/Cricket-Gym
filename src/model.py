import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

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
    BioMoR Phase 3: With Corollary Discharge (Efference Copy).

    Feature:
    - Inputs: [Wind_x, Wind_y, Audio, Vis_Theta, Vis_dTheta, Vis_On]
    - Feedback: Last Action [P_run, P_jump, Cos, Sin] is fed back to Router.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.hidden_dim = config["model"]["hidden_dim"]

        # Dimensions
        self.reflex_input_dim = 2   # Wind (Direct)
        self.context_input_dim = 6  # All Sensory
        self.action_dim = 4         # [P_run, P_jump, Cos, Sin]

        # [关键] Router 输入 = 感官上下文 + 上一时刻动作 (Corollary Discharge)
        self.router_input_dim = self.context_input_dim + self.action_dim

        self.router_rnn = nn.GRUCell(self.router_input_dim, self.hidden_dim)

        # Gates
        self.gain_gate = nn.Linear(self.hidden_dim, self.reflex_input_dim)
        self.bias_gate = nn.Linear(self.hidden_dim, 4)

        # Reflex Backbone
        self.reflex_backbone = nn.Sequential(
            nn.Linear(self.reflex_input_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(self.hidden_dim, 2)
        self.motor_head = nn.Linear(self.hidden_dim, 2)

    def forward(self, x: torch.Tensor, h_router: Optional[torch.Tensor] = None, last_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor.
            h_router: Previous hidden state.
            last_action: Previous action (used in Step mode).

        Returns:
            output: Predictions
            hidden: New hidden state
            (In Step mode returns: pred, h_new, action_new)
        """
        batch_size = x.size(0)

        # --- Case 1: Step-by-Step Inference (Closed-Loop / Single Step) ---
        if x.dim() == 2:
            if h_router is None:
                h_router = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            if last_action is None:
                last_action = torch.zeros(batch_size, self.action_dim).to(x.device)

            x_context = x
            x_reflex = x[:, 0:2]

            # [CD Mechanism] Cat Context + Last Action
            router_input = torch.cat([x_context, last_action], dim=1)

            # 1. Update Router
            h_new = self.router_rnn(router_input, h_router)

            # 2. Reflex & Gates
            gain = torch.sigmoid(self.gain_gate(h_new))
            bias = self.bias_gate(h_new)

            modulated_input = x_reflex * gain
            reflex_feat = self.reflex_backbone(modulated_input)

            probs = torch.sigmoid(self.policy_head(reflex_feat) + bias[:, 0:2])
            dirs = torch.tanh(self.motor_head(reflex_feat) + bias[:, 2:4])

            current_action = torch.cat([probs, dirs], dim=1)

            # Return action for next step's feedback
            return current_action, h_new, current_action

        # --- Case 2: Sequence Training (Batch, Seq, Dim) ---
        else:
            seq_len = x.size(1)
            x_reflex = x[:, :, 0:2]
            x_context = x

            if h_router is None:
                h_router = torch.zeros(batch_size, self.hidden_dim).to(x.device)

            # Start with zero action
            current_last_action = torch.zeros(batch_size, self.action_dim).to(x.device)

            y_preds = []
            h_history = []

            for t in range(seq_len):
                # [CD Mechanism]
                router_input = torch.cat([x_context[:, t, :], current_last_action], dim=1)

                h_router = self.router_rnn(router_input, h_router)
                h_history.append(h_router.unsqueeze(1))

                gain = torch.sigmoid(self.gain_gate(h_router))
                bias = self.bias_gate(h_router)

                modulated_input = x_reflex[:, t, :] * gain
                reflex_feat = self.reflex_backbone(modulated_input)

                probs = torch.sigmoid(self.policy_head(reflex_feat) + bias[:, 0:2])
                dirs = torch.tanh(self.motor_head(reflex_feat) + bias[:, 2:4])

                y_t = torch.cat([probs, dirs], dim=1)
                y_preds.append(y_t.unsqueeze(1))


                # Update feedback for next step
                # [Critical Fix] Detach to prevent gradient backprop through time
                # This stabilizes training and reduces output oscillations
                current_last_action = y_t.detach()

            return torch.cat(y_preds, dim=1), torch.cat(h_history, dim=1)