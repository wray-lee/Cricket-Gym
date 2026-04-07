import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Tuple, Dict

class DummyCricketDataset(Dataset):
    """
    Adversarial Bio-DataLoader.
    
    Generates synthetic but biologically constrained data sequences simulating 
    cerci airflow inputs and corresponding escape behavioral outputs.
    Injects realistic Gaussian noise and random frame-dropping (masking).
    """
    def __init__(self, 
                 num_samples: int = 1000, 
                 max_seq_len: int = 100, 
                 raw_sensor_dim: int = 4, 
                 action_dim: int = 2,
                 frame_drop_prob: float = 0.1):
        """
        Args:
            num_samples: Number of simulated trajectories.
            max_seq_len: Maximum length of a trajectory.
            raw_sensor_dim: e.g. [cerci_left, cerci_right, looming_size, looming_vel].
            action_dim: e.g. [angular_velocity, forward_velocity].
            frame_drop_prob: Probability of a biological sensor 'misfiring' or camera losing frame.
        """
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.raw_sensor_dim = raw_sensor_dim
        self.action_dim = action_dim
        self.frame_drop_prob = frame_drop_prob
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Generate a random valid length representing variable observation duration/latency
        # Ensure minimum length of 10 for meaningful recurrent accumulation
        seq_len = int(np.random.randint(10, self.max_seq_len + 1))
        
        # 1. Base Signal Generation (Simulating an escalating wind flow / looming)
        # We use a linspace ramp modified by a sine wave to simulate an incoming threat
        time_vector = torch.linspace(0.1, 5.0, seq_len).unsqueeze(1)
        base_sensory = torch.sin(time_vector) * time_vector 
        sensory_signal = base_sensory.repeat(1, self.raw_sensor_dim)
        
        # Similarly, base action output is a delayed response curve
        action_signal = torch.cos(time_vector) * (time_vector ** 2)
        action_signal = action_signal.repeat(1, self.action_dim)
        
        # 2. Adversarial Injection: Biological / Tool Noise
        sensory_noise = torch.randn(seq_len, self.raw_sensor_dim) * 0.5
        action_noise = torch.randn(seq_len, self.action_dim) * 0.1
        
        sensory_signal = sensory_signal + sensory_noise
        action_signal = action_signal + action_noise
        
        # 3. Adversarial Injection: Frame Drops (Masking)
        # Simulating random missing sensor readings (Zeroed out)
        drop_mask = torch.rand(seq_len) > self.frame_drop_prob
        drop_mask = drop_mask.unsqueeze(1).float()
        
        # Apply drops to sensors (acting like sensor malfunction or lost tracked frame)
        sensory_signal = sensory_signal * drop_mask
        
        return sensory_signal, action_signal, seq_len

def bio_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable length sequences for our dynamic graph.
    
    Args:
        batch: List of tuples (sensory_seq, action_seq, length)
    """
    sensory_seqs, action_seqs, lengths = zip(*batch)
    
    # Pad sequences with 0.0 (Our network can handle this or use pack_padded_sequence later if strictly required)
    # batch_first=True -> [batch, seq_len, dim]
    padded_sensory = pad_sequence(sensory_seqs, batch_first=True, padding_value=0.0)
    padded_actions = pad_sequence(action_seqs, batch_first=True, padding_value=0.0)
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return {
        "sensory_batch": padded_sensory,
        "action_batch": padded_actions,
        "lengths": lengths_tensor
    }

def get_dataloader_with_collate(config_or_batch_size: int, **kwargs) -> DataLoader:
    """Helper to instantiate the dataset and loader correctly mapped to our collate."""
    dataset = DummyCricketDataset(**kwargs)
    loader = DataLoader(
        dataset, 
        batch_size=config_or_batch_size, 
        shuffle=True, 
        collate_fn=bio_collate_fn
    )
    return loader
