import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List
from biomor.models.mor_gru import MoRGRUCell

class BioMoRNet(nn.Module):
    """
    Top-level wrapper for the Biological Mixture-of-Recursions Network.
    
    Includes:
    1. Sensory Encoder: Maps raw cerci/airflow arrays to network hidden dimension.
    2. MoR Core: The iterative ponder-based GRU processing temporal sequences.
    3. Motor Decoder: Maps final neural states to motor commands (e.g., angular velocity).
    """
    def __init__(self, raw_sensor_dim: int, hidden_size: int, action_dim: int, max_depth: int):
        super(BioMoRNet, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Sensory Encoder
        # Using a 1D-CNN to capture local temporal sensory features (e.g. wind acceleration)
        # kernel_size=3 acts as a sliding window over t-1, t, t+1
        self.sensory_encoder = nn.Sequential(
            nn.Conv1d(in_channels=raw_sensor_dim, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        
        # 2. Core MoR-GRU Loop
        self.mor_core = MoRGRUCell(
            input_size=hidden_size,  # Encoded dimension
            hidden_size=hidden_size,
            max_depth=max_depth
        )
        
        # 3. Motor Decoder
        # Maps the internal 'mushroom body/pre-motor' representation to specific kinematic outputs
        self.motor_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, action_dim)
        )

    def forward(self, 
                sensory_seq: torch.Tensor, 
                seq_lengths: torch.Tensor, 
                tau: float = 1.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            sensory_seq: Raw inputs of shape [batch, seq_len, raw_sensor_dim]
            seq_lengths: Valid lengths for each sequence in the batch [batch]
            tau: Annealing temperature for Gumbel-Softmax Router
            
        Returns:
            motor_commands: Tensor of shape [batch, seq_len, action_dim]
            aggregated_metrics: Struct full of internal states for downstream analysis
        """
        batch_size, seq_len, _ = sensory_seq.size()
        
        # Sensory Encoder requires [batch, channels, seq]
        x = sensory_seq.permute(0, 2, 1)
        encoded_x = self.sensory_encoder(x)
        # Permute back to [batch, seq, hidden]
        # Using contiguous to ensure optimal memory layout before view/RNN loop
        encoded_x = encoded_x.permute(0, 2, 1).contiguous()
        
        # Since we use LayerNorm on channel dim, in newer PyTorch returning from CNN might need
        # LayerNorm applied AFTER returning to [batch, seq, hidden]. Let's keep it robust explicitly:
        # Actually our LayerNorm was defined with `hidden_size`, which operates on the last dim natively!
        # Good thing we permuted it back BEFORE the layer norm? Wait, nn.Sequential executes in order.
        # So LayerNorm operated on the `seq` dimension in the Conv1d output, which is a bug!
        # Let's fix this dynamically in the forward pass to be mathematically correct:
        
        # Let's cleanly separate the encoder pieces to avoid the Conv1d LayerNorm silent tensor dim bug
        encoded_conv = self.sensory_encoder[0](sensory_seq.permute(0, 2, 1)) # -> [B, hidden, seq]
        encoded_act = self.sensory_encoder[1](encoded_conv)
        encoded_seq = encoded_act.permute(0, 2, 1).contiguous() # -> [B, seq, hidden]
        encoded_seq = self.sensory_encoder[2](encoded_seq) # Apply LayerNorm safely over hidden dim
        
        h_t = torch.zeros(batch_size, self.hidden_size, device=sensory_seq.device)
        
        outputs = []
        # Aggregator lists for white-box tracing
        all_soft_probs = []
        all_hard_choices = []
        all_sensory_gates = []
        all_depth_hidden_states = []
        
        # Temporal Loop
        for t in range(seq_len):
            x_t = encoded_seq[:, t, :]
            
            # Sub-Ponder Loop inside the cell
            h_t, step_metrics = self.mor_core(x_t, h_t, tau=tau)
            
            motor_t = self.motor_decoder(h_t)
            outputs.append(motor_t)
            
            all_soft_probs.append(step_metrics["soft_probs"])
            all_hard_choices.append(step_metrics["hard_choice"])
            all_sensory_gates.append(step_metrics["sensory_gates"])     # List of tensors
            all_depth_hidden_states.append(step_metrics["depth_hidden_states"])
            
        # Stack temporal outputs
        stacked_motor_commands = torch.stack(outputs, dim=1)           # [batch, seq, action_dim]
        stacked_soft_probs = torch.stack(all_soft_probs, dim=1)        # [batch, seq, max_depth]
        stacked_hard_choices = torch.stack(all_hard_choices, dim=1)    # [batch, seq]
        
        aggregated_metrics = {
            "soft_probs_seq": stacked_soft_probs,
            "hard_choice_seq": stacked_hard_choices,
            "sensory_gates_timeline": all_sensory_gates,
            "depth_hidden_states_timeline": all_depth_hidden_states
        }
        
        return stacked_motor_commands, aggregated_metrics
