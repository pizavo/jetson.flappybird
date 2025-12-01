#!/bin/bash
# Quick launch script for training on Linux/Jetson Nano
echo ""
echo "========================================"
echo "Flappy Bird AI Training"
echo "========================================"
echo ""
echo "Starting training with default settings..."
echo "This will train for 1000 episodes."
echo ""
echo "Press Ctrl+C to stop training at any time."
echo "Models are saved every 50 episodes."
echo ""
read -p "Press Enter to start training..."
python3 train_ai.py

