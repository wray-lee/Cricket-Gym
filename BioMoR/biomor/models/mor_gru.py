import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

class MoRGRUCell(nn.Module):
    """
    Biological Mixture-of-Recursions GRU Cell.
    
    This module implements a dynamic recursive depth GRU (pondering GRU) 
    with biological constraints:
    1. Sensory Gating: Input signal attenuates/modulates based on internal state.
    2. Gumbel-Softmax Routing: Employs an annealed stochastic process to pick
       the discrete number of inner recursions (depth) for mechanistic interpretability.
    """
    def __init__(self, input_size: int, hidden_size: int, max_depth: int):
        super(MoRGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        
        # Core standard GRU cell (parameter shared across all internal depths)
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        
        # Router predicting depth log-probabilities (d = 1 to max_depth)
        self.router = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_depth)
        )
        
        # Sensory Gating Weights
        # Computes g^(d) = sigmoid(W_x * x_t + W_h * h_t^(d-1) + b_g)
        self.sensory_gate_W_x = nn.Linear(input_size, input_size, bias=False)
        self.sensory_gate_W_h = nn.Linear(hidden_size, input_size, bias=True)
        # Using a scalar or element-wise bias mapped through Linear logic

    def forward(self, 
                sensory_input_t: torch.Tensor, 
                hidden_state_t_prev: torch.Tensor, 
                tau: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes a single step with dynamic recursive pondering.
        
        Args:
            sensory_input_t: Tensor of shape (batch, input_size)
            hidden_state_t_prev: Tensor of shape (batch, hidden_size)
            tau: Gumbel-Softmax temperature. High = uniform exploration, Low (<0.1) = hard selection.
            
        Returns:
            final_hidden_state: The resulting state after MoR pondering.
            metrics_dict: Dictionary containing internal states for mechanistic analysis.
        """
        batch_size = sensory_input_t.size(0)
        
        # 1. Router Logic: Predict probabilities over Ponder Depth
        router_input = torch.cat([sensory_input_t, hidden_state_t_prev], dim=-1)
        router_logits = self.router(router_input)  # (batch, max_depth)
        
        # Obtain soft probabilities for the Metabolic Loss backpropagation
        soft_probs = F.softmax(router_logits, dim=-1)
        
        # Obtain the Gumbel-Softmax output for differentiable state summation
        # When tau -> 0, this approximates a one-hot vector (Hard MoR)
        # We explicitly set hard=True to enforce sparse computational choice mathematically
        gumbel_weights = F.gumbel_softmax(router_logits, tau=tau, hard=True, dim=-1)
        
        # Keep track of the index of the hard routing decision for logging
        hard_choice = torch.argmax(gumbel_weights, dim=-1) + 1  # Depth is 1-indexed (1 to max_depth)
        
        # 2. Iterative Pondering (Depth loops)
        current_hidden_state = hidden_state_t_prev
        
        # Mechanism tracing components
        depth_hidden_states = []
        sensory_gates = []
        
        # We process up to max_depth. The router weights will mask the final combination.
        for d in range(self.max_depth):
            # Calculate the explicit Sensory Gate
            # g^(d) = \sigma(W_x x_t + W_h h_t^(d-1) + b)
            gate_logits = self.sensory_gate_W_x(sensory_input_t) + self.sensory_gate_W_h(current_hidden_state)
            sensory_gate_d = torch.sigmoid(gate_logits)  # (batch, input_size)
            
            # Apply Sensory Gating to modulate incoming input
            gated_sensory_input = sensory_input_t * sensory_gate_d
            
            # Update Internal GRU state
            current_hidden_state = self.gru_cell(gated_sensory_input, current_hidden_state)
            
            # Save for interpretability and final summation
            depth_hidden_states.append(current_hidden_state)
            sensory_gates.append(sensory_gate_d)
            
        # Stack hidden states: shape (batch, max_depth, hidden_size)
        stacked_hidden_states = torch.stack(depth_hidden_states, dim=1)
        
        # 3. Apply Routing Weights (Weighted Context / Hard Masking)
        # We expand dimensions to match: gumbel_weights is (batch, max_depth, 1)
        gumbel_weights_expanded = gumbel_weights.unsqueeze(-1)
        
        # Because we used hard=True, this selects EXACTLY the hidden state at the chosen depth,
        # but gradient will flow naturally through the Gumbel approximation.
        final_hidden_state = torch.sum(stacked_hidden_states * gumbel_weights_expanded, dim=1)
        
        # 4. Expose White-Box Metrics
        metrics_dict = {
            "soft_probs": soft_probs,                  # Pure Softmax probabilities for BioMetabolicLoss
            "gumbel_weights": gumbel_weights,          # Action chosen by Router (One-hot if hard=True)
            "hard_choice": hard_choice,                # Integer discrete depth chosen
            "sensory_gates": sensory_gates,            # List[Tensor] tracking how stimulus attentuates
            "depth_hidden_states": depth_hidden_states # List[Tensor] internal trajectory
        }
        
        return final_hidden_state, metrics_dict
