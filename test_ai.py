#!/usr/bin/env python3
"""
Test the trained AI model in the Flappy Bird environment
"""

import torch
import numpy as np
import sys
import os

from config import GAME_CONFIG, NETWORK_CONFIG, IO_CONFIG

SCREEN_HEIGHT = GAME_CONFIG['screen_height']
BIRD_SIZE = GAME_CONFIG['bird_size']
MODEL_DIR = IO_CONFIG['model_dir']
INPUT_SIZE = NETWORK_CONFIG['input_size']
HIDDEN_SIZE = NETWORK_CONFIG['hidden_size']
OUTPUT_SIZE = NETWORK_CONFIG['output_size']
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_ai import DQN, FlappyBirdEnv, device

def load_model(model_path):
    """Load a trained model"""
    model = DQN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()
    return model

def test_model(model_path, num_episodes=10, render=False):
    """Test the trained model"""
    model = load_model(model_path)
    env = FlappyBirdEnv()

    scores = []

    print(f"Testing model: {model_path}")
    print(f"Device: {device}")
    print("-" * 50)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # Select action using the trained model (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                # Print state information
                print(f"Step {steps}: Bird Y: {env.bird_y:.1f}, "
                      f"Velocity: {env.bird_velocity:.2f}, "
                      f"Score: {env.score}, Action: {'JUMP' if action == 1 else 'NOTHING'}")

            if done:
                break

        score = info.get('score', 0)
        scores.append(score)

        print(f"Episode {episode + 1}: Score = {score}, Steps = {steps}, "
              f"Total Reward = {total_reward:.2f}")

    print("-" * 50)
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Std Dev: {np.std(scores):.2f}")

def compare_models(model_dir='models', num_episodes=20):
    """Compare multiple models"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    if not model_files:
        print(f"No models found in {model_dir}")
        return

    print(f"Found {len(model_files)} models")
    print("=" * 60)

    results = {}

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path)
        env = FlappyBirdEnv()

        scores = []
        for _ in range(num_episodes):
            state = env.reset()
            while True:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, done, info = env.step(action)
                state = next_state

                if done:
                    scores.append(info.get('score', 0))
                    break

        avg_score = np.mean(scores)
        results[model_file] = avg_score

        print(f"{model_file:30s} | Avg Score: {avg_score:.2f} | Max: {max(scores)}")

    print("=" * 60)
    best_model = max(results, key=results.get)
    print(f"Best Model: {best_model} with average score: {results[best_model]:.2f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained Flappy Bird AI model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to test')
    parser.add_argument('--render', action='store_true',
                        help='Print detailed state information')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models in the models directory')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR,
                        help='Directory containing model files')

    args = parser.parse_args()

    if args.compare:
        compare_models(args.model_dir, args.episodes)
    else:
        test_model(args.model, args.episodes, args.render)
