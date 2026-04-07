import torch
import torch.optim as optim
import wandb
import numpy as np
import os
import math
from typing import Dict, Any

from biomor.config import BioMoRConfig
from biomor.models.network import BioMoRNet
from biomor.models.loss import BioMetabolicLoss
from biomor.data.dataset import get_dataloader_with_collate


# ---------------------------------------------------------
# Mechanism 1: Synaptic Pruning Annealing Scheduler
# ---------------------------------------------------------
class TauScheduler:
    """
    Handles the biological 'Synaptic Pruning' / 'Neural Maturation' schedule.
    Anneals the Gumbel-Softmax tau (temperature) parameter exponentially over training steps.
    """
    def __init__(self, initial_tau: float, min_tau: float, decay_steps: int):
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.decay_steps = decay_steps
        
        # Calculate strict exponential decay rate
        self.decay_rate = -math.log(self.min_tau / self.initial_tau) / self.decay_steps

    def get_tau(self, current_step: int) -> float:
        if current_step >= self.decay_steps:
            return self.min_tau
        return self.initial_tau * math.exp(-self.decay_rate * current_step)


# ---------------------------------------------------------
# Utility: Sequence Masking
# ---------------------------------------------------------
def create_boolean_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Creates a boolean mask avoiding padded zeros [batch, seq_len]."""
    # e.g., if lengths is [3, 1], returns [[True, True, True, False], [True, False, False, False]]
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# ---------------------------------------------------------
# Main Cognitive Console (Training Loop)
# ---------------------------------------------------------
def train_biomor():
    # Load configuration
    config = BioMoRConfig()
    
    # Initialize White-box Telemetry
    # You MUST login to wandb using `wandb login` in the terminal before running
    # Uncomment the wandb.init call below when ready to run on real server!
    wandb.init(
       project="BioMoR-NeuroAI",
       config=config.to_dict(),
       name="MoR_Biological_Annealing",
       mode="online" # change to online/offline as needed
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing BioMoR Observatory on: {device}")

    # Build Network and Loss
    # We specify arbitrary dimensions matching our DummyDataset configuration
    raw_sensor_dim = 4; action_dim = 2
    model = BioMoRNet(
        raw_sensor_dim=raw_sensor_dim, 
        hidden_size=config.hidden_size, 
        action_dim=action_dim, 
        max_depth=config.max_depth
    ).to(device)
    
    criterion = BioMetabolicLoss(lambda_energy=config.lambda_energy, task_loss_type=config.task_loss_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Build Dataset
    train_loader = get_dataloader_with_collate(
        config_or_batch_size=config.batch_size, 
        num_samples=500, 
        max_seq_len=60, 
        raw_sensor_dim=raw_sensor_dim, 
        action_dim=action_dim
    )
    
    # Initialize Annealing Scheduler
    tau_scheduler = TauScheduler(
        initial_tau=config.tau_initial, 
        min_tau=config.tau_min, 
        decay_steps=config.tau_decay_steps
    )
    
    global_step = 0
    num_epochs = 100
    
    print("[*] Observatory Online. Commencing biological simulation trace...")
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Fetch padded sequences and their actual valid biological durations
            inputs = batch_data['sensory_batch'].to(device)
            targets = batch_data['action_batch'].to(device)
            lengths = batch_data['lengths'].to(device)
            
            # Retrieve biological temperature (Neuroplasticity / Pruning factor)
            current_tau = tau_scheduler.get_tau(global_step)
            
            # --- Forward Pass ---
            predictions, aggregated_metrics = model(inputs, seq_lengths=lengths, tau=current_tau)
            
            # --- Mechanism 3: Padded Masking (Combat padding pollution) ---
            max_len = predictions.size(1)
            valid_mask = create_boolean_mask(lengths, max_len)
            
            # Mask predictions and targets to calculate loss strictly on valid physiological observations
            # We flatten the masked tensors: shape becomes [sum(lengths), dim]
            valid_predictions = predictions[valid_mask]
            valid_targets = targets[valid_mask]
            
            # Similarly mask the soft probabilities for energy calculation
            soft_probs_seq = aggregated_metrics['soft_probs_seq']
            valid_soft_probs = soft_probs_seq[valid_mask]
            
            # --- Compound Bio-Metabolic Loss Calculation ---
            loss, loss_dict = criterion(valid_predictions, valid_targets, valid_soft_probs)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients during internal GRU recursions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # --- Mechanism 2: White-box Telemetry (Scalars) ---
            wandb.log({
                "step": global_step,
                "epoch": epoch,
                "Gumbel_Tau": current_tau,
                "Loss/Total": loss_dict['total_loss'],
                "Loss/Kinematic_MSE": loss_dict['task_loss'],
                "Loss/ATP_Metabolic_Penalty": loss_dict['energy_penalty'],
                "Stats/Avg_Ponder_Depth": loss_dict['avg_expected_depth']
            })
            
            global_step += 1
            
        # --- White-box Telemetry (High-dimensional Modality Dumps) ---
        # Log complex tensors periodically to avoid massive data payloads
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Tau: {current_tau:.3f} | Total Loss: {loss.item():.4f} | Avg Depth: {loss_dict['avg_expected_depth']:.2f}")
            
            # 1. Routing Deep-dive: Heatmap of soft_probs for a single sequence
            # Extract the soft probabilities for the first element in the batch, up to its valid length
            sample_len = lengths[0].item()
            sample_probs = soft_probs_seq[0, :sample_len, :].detach().cpu().numpy()
            
            # Log as a Weights & Biases Image (Heatmap style)
            # Row = Time step, Col = Recursion Depth
            wandb.log({
                "Manifolds/Routing_Matrix_Heatmap": wandb.Image(
                    sample_probs, 
                    caption=f"Epoch {epoch}: Router Probabilities (Time x Depth)"
                )
            })
            
            # 2. Sensory Gating Trace: Histogram
            # Flatten all sensory gates produced across all time steps and depths in this batch
            sensory_gates_nested = aggregated_metrics['sensory_gates_timeline']
            gate_values = []
            for time_step_gates in sensory_gates_nested:      # list of length seq_len
                for depth_gate in time_step_gates:            # list of length max_depth
                    gate_values.append(depth_gate.detach().cpu().numpy().flatten())
            
            if len(gate_values) > 0:
                flat_gates = np.concatenate(gate_values)
                wandb.log({
                    "Manifolds/Sensory_Gate_Distribution": wandb.Histogram(flat_gates)
                })

    print("[*] Observatory Simulation Concluded Successfully.")
    wandb.finish()

if __name__ == "__main__":
    train_biomor()
