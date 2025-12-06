# Reinforcement Learning for CarRacing-v3

This folder contains a complete PPO (Proximal Policy Optimization) implementation for training an agent to drive in the CarRacing-v3 environment.

## Features
- PPO algorithm implementation
- CNN-based policy network
- Frame stacking for temporal information
- Reward shaping for better learning
- Checkpoint saving/loading
- Evaluation and visualization

## Installation

```bash
pip install torch torchvision gym[box2d] numpy opencv-python
```

## Usage

### Training
```bash
python main.py train
```

### Evaluating
```bash
python main.py evaluate
```

### Continue Training from Checkpoint
```bash
python main.py train --checkpoint checkpoints/ppo_carracing_best.pt
```

## Files
- `main.py`: Entry point for training and evaluation
- `ppo_agent.py`: PPO algorithm implementation
- `network.py`: Neural network architectures
- `environment.py`: Environment wrapper with preprocessing
- `utils.py`: Utility functions for training
