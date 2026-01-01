# Cricket Escape Simulation with BioMoR-RNN

A biologically-inspired neural network model for simulating cricket escape behavior in response to visual and audio-wind stimuli. This project implements a Mixture-of-Ramps (MoR) RNN architecture with bio-inspired initialization to model sensorimotor transformations in cricket escape responses.

## Overview

This project models how crickets make rapid escape decisions when threatened by predators. The model processes multi-modal sensory inputs (visual looming, audio tones, and wind stimuli) and generates escape actions (run/jump) with appropriate movement directions.

### Key Features

- **Bio-inspired Architecture**: MoR-RNN with corollary discharge and sensory gating mechanisms
- **Multi-modal Sensory Processing**: Visual (looming), audio, and wind stimuli
- **Realistic Escape Dynamics**: Biologically plausible action probabilities and movement trajectories
- **Publication-Quality Visualization**: Generates figures for scientific publications
- **Configurable Parameters**: Easy-to-modify YAML configuration

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Cricket-Gym.git
cd Cricket-Gym

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.10.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
pyyaml>=5.4.0
scipy>=1.7.0
```

## Quick Start

### 1. Train the Model

```bash
python main.py --mode train --arch BioMoR
```

This will:

- Generate training data with visual, audio-wind, and mixed stimuli
- Train the BioMoR-RNN model for 700 epochs
- Save the trained model to `models/cricket_biomor.pth`
- Generate training loss plot in `outputs/`

### 2. Evaluate the Model

```bash
python main.py --mode eval
```

This generates evaluation figures showing:

- Model responses to audio-wind stimuli
- Model responses to visual stimuli
- Saved to `outputs/eval/`

### 3. Run Escape Simulations

```bash
python visualize_simulator_results.py
```

This runs 50 escape trials and generates:

- Action probability curves (p_run, p_jump)
- Escape trajectories visualization
- Survival statistics
- Saved to `outputs/pub/publication_figure.png`

### 4. Generate Publication Figures

```bash
python generate_publication_figures.py
```

This generates publication-quality figures:

- `fig_audio_wind.png`: Audio-wind ISI analysis
- `fig_visual_lv.png`: Visual l/v analysis

## Project Structure

```
Cricket-Gym/
├── configs/
│   └── default.yaml          # Configuration file
├── src/
│   ├── model.py              # BioMoR-RNN architecture
│   ├── cricket_env.py        # Escape simulation environment
│   ├── data_generator.py     # Training data generation
│   ├── evaluator.py          # Model evaluation metrics
│   └── loss.py               # Multi-task loss function
├── main.py                   # Training/evaluation entry point
├── main_simulator.py         # Simulator entry point
├── visualize_simulator_results.py  # Escape visualization
├── generate_publication_figures.py # Publication figures
├── models/                   # Trained model checkpoints
└── outputs/                  # Generated figures and results
```

## Model Architecture

### BioMoR-RNN

The model uses a Mixture-of-Ramps (MoR) RNN architecture with:

- **Input**: 6-dimensional sensory vector

  - Audio tone (binary)
  - Wind direction (x, y components)
  - Visual looming (angular velocity)
  - Predator distance
  - Predator angle

- **Hidden Layer**: 64-dimensional MoR-RNN with:

  - Reset and update gates
  - Sensory gating mechanisms
  - Corollary discharge (action feedback)

- **Output**: 4-dimensional action vector
  - Run probability
  - Jump probability
  - Movement direction (cos, sin)

### Bio-inspired Mechanisms

1. **Sensory Gating**: Different sensory modalities have different gate opening strengths
2. **Corollary Discharge**: Inhibitory feedback from motor output to sensory processing
3. **Metabolic Cost**: Penalizes excessive neural activity
4. **Motor Delay**: 74ms delay from visual trigger to action (biologically realistic)

## Configuration

Edit `configs/default.yaml` to modify:

### Simulation Parameters

```yaml
simulation:
  dt: 0.02 # Time step (20ms)
  episode_length_ms: 3000 # Episode duration
```

### Visual Stimulus

```yaml
visual:
  looming_threshold_deg: 41.0 # Trigger threshold
  motor_delay_ms: 74.0 # Motor delay
```

### Model Architecture

```yaml
model:
  hidden_dim: 64 # Hidden layer size
  rnn_type: "MoR" # RNN type
```

### Bio-initialization Weights

```yaml
bio_initialization:
  weight_audio_drive: 5.0 # Audio input strength
  weight_wind_drive: 2.0 # Wind input strength
  weight_visual_drive: 0.5 # Visual input strength
  weight_corollary_discharge: -5.0 # Inhibitory feedback
```

## Training Data

The model is trained on a mixture of:

- **40%** Visual stimuli (looming with varying l/v ratios)
- **30%** Audio-wind stimuli (with varying ISI patterns)
- **20%** Mixed stimuli (visual + audio-wind)
- **10%** Empty trials (no stimulus)

Each trial generates:

- Sensory input sequences
- Target action probabilities
- Target movement directions

## Escape Simulation

The escape simulator models:

### Cricket Parameters

- Run speed: 35 cm/s
- Jump speed: 100 cm/s
- Response thresholds: p_run > 0.2, p_jump > 0.5

### Predator Parameters

- Initial distance: 30-40 cm
- Speed: 5-7 cm/s
- Approaches from random angle

### Trajectory Generation

- Movement direction with ±17° noise (90% of time)
- Occasional large deviations ±57° (10% of time)
- Realistic escape patterns

## Publication Figures

### Audio-Wind ISI Analysis (`fig_audio_wind.png`)

- **Panel A**: Movement direction vs ISI patterns
- **Panel B**: Jumping probability vs ISI patterns
- **Panel C**: ISI schematic diagram

### Visual l/v Analysis (`fig_visual_lv.png`)

- **Panel A**: Movement direction vs l/v ratios
- **Panel B**: Jumping probability vs l/v ratios
- **Panel C**: Looming stimulus curves

## Results

The trained model achieves:

- **Survival Rate**: 60-70% (biologically realistic)
- **Response Latency**: ~74ms for visual stimuli
- **Movement Direction**: Consistently away from predator (±30° deviation)
- **Action Selection**: Appropriate run/jump probabilities based on threat urgency

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cricket-escape-2024,
  title={Bio-inspired Neural Network Model for Cricket Escape Behavior},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## References

- Mixture-of-Ramps RNN architecture
- Cricket escape behavior literature
- Visual looming detection mechanisms
- Audio-wind multimodal integration

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

This project implements bio-inspired mechanisms based on cricket neuroscience research.
