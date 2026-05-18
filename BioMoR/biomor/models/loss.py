import torch
import torch.nn as nn
from typing import Dict, Tuple

class BioMetabolicLoss(nn.Module):
    """
    Biological Metabolic Loss Function.
    
    Combines the functional task loss (e.g., predicting escape trajectory)
    with a biologically plausible metabolic energy penalty (ATP usage) 
    that scales with the internal recursion depth.
    """
    def __init__(self, lambda_energy: float = 0.01, task_loss_type: str = 'mse'):
        """
        Args:
            lambda_energy: The hyperparameter dictating the metabolic penalty strength.
            task_loss_type: The base loss for the behavioural modeling ('mse' or 'ce').
        """
        super(BioMetabolicLoss, self).__init__()
        self.lambda_energy = lambda_energy
        
        if task_loss_type.lower() == 'mse':
            self.task_criterion = nn.MSELoss()
        elif task_loss_type.lower() == 'ce':
            self.task_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported task_loss_type: {task_loss_type}")

    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor, 
                soft_probs_sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete biological loss.
        
        Args:
            predictions: Behavior prediction logits or regression values.
            targets: Experimental ground truth targets.
            soft_probs_sequence: Temporal tensor of shape (batch, sequence_length, max_depth)
                                 representing the router probabilities BEFORE Gumbel-Softmax.
                                 This is critical to maintain smooth gradients for the metabolic cost.
            
        Returns:
            total_loss: The final scalar loss to backpropagate.
            loss_components: Dict mapping metric names to floats for logging.
        """
        # 1. Behavioral Task Loss
        task_loss = self.task_criterion(predictions, targets)
        
        # 2. Metabolic Energy Cost (expected recursion depth)
        # Expected depth calculation: sum(p * depth)
        # Create a depth vector [1.0, 2.0, ..., max_depth]
        max_depth = soft_probs_sequence.size(-1)
        depth_tensor = torch.arange(1, max_depth + 1, dtype=torch.float32, device=soft_probs_sequence.device)
        
        # Multiply probabilities by the scalar depths: (batch, seq, max_depth) * (max_depth)
        expected_depths = torch.sum(soft_probs_sequence * depth_tensor, dim=-1)
        
        # Determine average expected depth across the sequence and batch to form the penalty
        average_energy_cost = torch.mean(expected_depths)
        
        energy_penalty = self.lambda_energy * average_energy_cost
        
        # 3. Total Compound Loss
        total_loss = task_loss + energy_penalty
        
        loss_components = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "energy_penalty": energy_penalty.item(),
            "avg_expected_depth": average_energy_cost.item()
        }
        
        return total_loss, loss_components
