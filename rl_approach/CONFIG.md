# Configuration Guide for PPO CarRacing

## Hyperparameter Tuning Guide

### Learning Rate (`--lr`)
- Default: `3e-4`
- Range: `1e-5` to `1e-3`
- Lower values make training more stable but slower
- Higher values speed up training but may cause instability

### Discount Factor (`--gamma`)
- Default: `0.99`
- Range: `0.95` to `0.999`
- Controls how much the agent values future rewards
- Higher values make the agent more forward-thinking

### GAE Lambda (`--gae-lambda`)
- Default: `0.95`
- Range: `0.90` to `0.99`
- Controls bias-variance tradeoff in advantage estimation
- Higher values reduce bias but increase variance

### Clip Epsilon (`--clip-epsilon`)
- Default: `0.2`
- Range: `0.1` to `0.3`
- Controls how much the policy can change per update
- Lower values make training more conservative

### Update Frequency (`--update-freq`)
- Default: `2048`
- Range: `1024` to `4096`
- Number of steps before policy update
- Higher values are more stable but slower to adapt

### Batch Size (`--batch-size`)
- Default: `64`
- Range: `32` to `256`
- Size of mini-batches during PPO update
- Larger batches are more stable but use more memory

### Frame Skip (`--frame-skip`)
- Default: `2`
- Range: `1` to `4`
- Number of frames to skip (action is repeated)
- Higher values speed up training but reduce control precision

### Entropy Coefficient (`--entropy-coef`)
- Default: `0.01`
- Range: `0.001` to `0.1`
- Encourages exploration
- Higher values increase randomness in early training

## Quick Start Commands

### Basic Training
```bash
python main.py train
```

### Training with Rendering (slower, for visualization)
```bash
python main.py train --render --render-freq 10
```

### Fast Training (less stable)
```bash
python main.py train --lr 5e-4 --update-freq 1024 --frame-skip 3
```

### Stable Training (slower but more reliable)
```bash
python main.py train --lr 1e-4 --update-freq 4096 --clip-epsilon 0.1
```

### Resume from Checkpoint
```bash
python main.py train --checkpoint checkpoints/ppo_carracing_ep500.pt
```

### Evaluate Best Model
```bash
python main.py evaluate
```

### Evaluate Specific Checkpoint
```bash
python main.py evaluate --checkpoint checkpoints/ppo_carracing_ep1000.pt --eval-episodes 10
```

## Training Tips

1. **Start with default parameters** - They work well for most cases
2. **Monitor the logs** - Check `logs/` folder for training progress
3. **Be patient** - Good performance usually appears after 500+ episodes
4. **Watch for overfitting** - If eval performance drops, reduce learning rate
5. **Use GPU if available** - Training is much faster with CUDA

## Expected Performance

- **Episodes 0-100**: Random behavior, exploring the environment
- **Episodes 100-300**: Car starts to stay on track for short distances
- **Episodes 300-700**: Car can complete some laps
- **Episodes 700+**: Consistent track completion and optimization

## Troubleshooting

### Car doesn't improve after many episodes
- Reduce learning rate: `--lr 1e-4`
- Increase update frequency: `--update-freq 4096`
- Reduce entropy: `--entropy-coef 0.005`

### Training is unstable (loss spikes)
- Reduce learning rate: `--lr 1e-4`
- Reduce clip epsilon: `--clip-epsilon 0.1`
- Increase batch size: `--batch-size 128`

### Training is too slow
- Increase frame skip: `--frame-skip 3`
- Reduce update frequency: `--update-freq 1024`
- Reduce PPO epochs: `--ppo-epochs 5`

### Out of memory
- Reduce batch size: `--batch-size 32`
- Reduce update frequency: `--update-freq 1024`
