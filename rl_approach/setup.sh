#!/bin/bash

# Quick start script for RL CarRacing training

echo "=========================================="
echo "RL CarRacing - Quick Start"
echo "=========================================="

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

echo ""
echo "Setup complete!"
echo ""
echo "To train the agent, run:"
echo "  python main.py train"
echo ""
echo "To evaluate a trained agent, run:"
echo "  python main.py evaluate"
echo ""
echo "For more options, run:"
echo "  python main.py --help"
echo ""
echo "Training tips:"
echo "  - Training will take several hours (recommend 1000+ episodes)"
echo "  - The car will be random at first, but will improve over time"
echo "  - Checkpoints are saved every 100 episodes"
echo "  - Best model is saved automatically"
echo "  - Use --render flag to watch training (slows down training)"
echo ""
echo "=========================================="
