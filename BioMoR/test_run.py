import torch
import torch.optim as optim
import sys
import os

# Ensure the local biomor package can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from biomor.config import BioMoRConfig
from biomor.models.mor_gru import MoRGRUCell
from biomor.models.loss import BioMetabolicLoss

def test_biomor_forward_backward():
    print("--------------------------------------------------")
    print("Initializing BioMoR System Test...")
    
    # 1. Load Configuration
    config = BioMoRConfig(
        input_size=4,
        hidden_size=32,
        max_depth=5,
        lambda_energy=0.1
    )
    print(f"Config Loaded: {config}")

    # 2. Instantiate Components
    # Important: Set a manual seed to make outputs deterministic for this test
    torch.manual_seed(42)
    mor_cell = MoRGRUCell(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        max_depth=config.max_depth
    )
    
    criterion = BioMetabolicLoss(
        lambda_energy=config.lambda_energy,
        task_loss_type=config.task_loss_type
    )
    
    optimizer = optim.Adam(mor_cell.parameters(), lr=config.learning_rate)
    print("Model, Loss, and Optimizer initialized successfully.\n")

    # 3. Create Dummy Data
    batch_size = 2
    seq_length = 3
    
    # Let's say inputs are [velocity, angle, looming_size, looming_velocity]
    dummy_inputs = torch.randn(batch_size, seq_length, config.input_size)
    
    # Target is predicting the next behavior state (say, next state is just 2 dim for simplicity, 
    # but we will project the hidden state to match targets for the test)
    target_dim = 2
    dummy_targets = torch.randn(batch_size, seq_length, target_dim)
    
    # Readout layer (Mushroom Body Output -> Motor command)
    readout = torch.nn.Linear(config.hidden_size, target_dim)
    
    # 4. Forward Pass Simulation (Unrolling over time)
    # RNNs need an initial hidden state
    h_t = torch.zeros(batch_size, config.hidden_size)
    
    predictions = []
    soft_probs_seq = []
    
    tau = config.tau_initial  # Start exploration
    
    print("Starting Forward Time Sequence...")
    for t in range(seq_length):
        x_t = dummy_inputs[:, t, :]
        
        # Step through our MoR GRU Matrix
        h_t, metrics = mor_cell(x_t, h_t, tau=tau)
        
        # Project to target prediction
        pred_t = readout(h_t)
        
        predictions.append(pred_t)
        soft_probs_seq.append(metrics['soft_probs'])
        
        print(f"   Time Step {t}: Hard Depth Choice: {metrics['hard_choice'].tolist()}")
        print(f"   Time Step {t}: Soft Probabilities: {metrics['soft_probs'][0].detach().numpy().round(3)}")

    # Stack along temporal dimension
    stacked_predictions = torch.stack(predictions, dim=1)           # (batch, seq, target_dim)
    stacked_soft_probs = torch.stack(soft_probs_seq, dim=1)         # (batch, seq, max_depth)
    
    # 5. Loss Calculation & Backward Pass
    optimizer.zero_grad()
    
    print("\nCalculating BioMetabolic Loss...")
    loss, loss_dict = criterion(stacked_predictions, dummy_targets, stacked_soft_probs)
    
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")
        
    print("\nExecuting Gradient Backward Pass...")
    loss.backward()
    
    # Verify Gradients flowed to the Router (Critical check!)
    router_grad_norm = mor_cell.router[0].weight.grad.norm().item()
    gate_grad_norm = mor_cell.sensory_gate_W_x.weight.grad.norm().item()
    
    print(f"Router Gradient Norm: {router_grad_norm:.4f} (Must be > 0)")
    print(f"Sensory Gate Gradient Norm: {gate_grad_norm:.4f} (Must be > 0)")
    
    if router_grad_norm > 0 and gate_grad_norm > 0:
         print("\n✅ SUCCESS: Forward pass, metabolic constraints, and backpropagation graph strictly intact!")
    else:
         print("\n❌ FAILED: Gradient graph is broken somewhere in the MoR logic.")

if __name__ == "__main__":
    test_biomor_forward_backward()
