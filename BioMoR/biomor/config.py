import json
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class BioMoRConfig:
    """
    Isolated Configuration for BioMoR.
    
    This dataclass encapsulates all hyperparameters required for the network architecture,
    biological constraints, and experimental settings. Keeping this isolated ensures
    that our model definitions are not hardcoded with magic numbers.
    """
    
    # -----------------------------------
    # 1. Architectural Parameters
    # -----------------------------------
    input_size: int = 4            # E.g., Cricket velocity, angle offset, looming size, looming velocity
    hidden_size: int = 64          # Neural population dimensionality
    max_depth: int = 5             # Maximum allowed internal recurrences (ponder steps) per dt
    
    # -----------------------------------
    # 2. Biological Constraints & Dynamics
    # -----------------------------------
    lambda_energy: float = 0.05    # Penalty coefficient for ATP/Ponder metabolic cost
    
    # Gumbel-Softmax Temperature Annealing Schedule
    # High tau = uniform random exploration (brain plasticity early on)
    # Low tau  = hard routing / sparse commitment (synaptic pruning / maturation)
    tau_initial: float = 2.0
    tau_min: float = 0.1
    tau_decay_steps: int = 5000    # Number of steps to anneal tau from initial to min
    
    # -----------------------------------
    # 3. Training Loop Parameters
    # -----------------------------------
    batch_size: int = 32
    learning_rate: float = 1e-3
    task_loss_type: str = 'mse'    # Objective for behaviour tracking
    
    # -----------------------------------
    # Utility Methods
    # -----------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return asdict(self)
    
    def save_yaml(self, path: str):
        """Save the config out to a YAML file for replicability."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "BioMoRConfig":
        """Instantiate tracking config strictly from a YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
        
    @classmethod
    def from_json(cls, path: str) -> "BioMoRConfig":
        """Fallback JSON parser."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
