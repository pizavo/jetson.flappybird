#!/usr/bin/env python3
"""
Comprehensive test of all saved models
"""

import torch
import numpy as np
import os
import glob

from train_ai import DQN, FlappyBirdEnv, device
from config import NETWORK_CONFIG, IO_CONFIG

INPUT_SIZE = NETWORK_CONFIG['input_size']
HIDDEN_SIZE = NETWORK_CONFIG['hidden_size']
OUTPUT_SIZE = NETWORK_CONFIG['output_size']
MODEL_DIR = IO_CONFIG['model_dir']

def load_model(model_path):
    """Load a trained model"""
    model = DQN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()
    return model

def test_single_model(model_path, num_episodes=50):
    """Test a single model extensively"""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(model_path)}")
    print(f"{'='*70}")

    model = load_model(model_path)
    env = FlappyBirdEnv()

    scores = []
    steps_list = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        score = info.get('score', 0)
        scores.append(score)
        steps_list.append(steps)

        # Print notable episodes
        if score > 1 or (episode < 5) or (episode % 10 == 0):
            print(f"  Episode {episode+1:3d}: Score={score:2d}, Steps={steps:4d}, Reward={total_reward:7.2f}")

    print(f"\n{'-'*70}")
    print(f"Results over {num_episodes} episodes:")
    print(f"  Average Score: {np.mean(scores):.2f}")
    print(f"  Max Score:     {max(scores)}")
    print(f"  Min Score:     {min(scores)}")
    print(f"  Std Dev:       {np.std(scores):.2f}")
    print(f"  Avg Steps:     {np.mean(steps_list):.1f}")
    print(f"  Max Steps:     {max(steps_list)}")

    # Score distribution
    from collections import Counter
    score_dist = Counter(scores)
    print(f"\n  Score Distribution:")
    for score in sorted(score_dist.keys()):
        count = score_dist[score]
        pct = 100 * count / num_episodes
        bar = '█' * int(pct / 2)
        print(f"    Score {score:2d}: {count:3d} times ({pct:5.1f}%) {bar}")

    return {
        'model': os.path.basename(model_path),
        'avg_score': np.mean(scores),
        'max_score': max(scores),
        'std_score': np.std(scores),
        'scores': scores
    }

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*70)
    print(f"Device: {device}")

    # Find all model files
    model_files = []
    for pattern in ['best_model.pth', 'final_model.pth', 'checkpoint_*.pth']:
        model_files.extend(glob.glob(os.path.join(MODEL_DIR, pattern)))

    if not model_files:
        print(f"\nNo models found in {MODEL_DIR}/")
        return

    print(f"\nFound {len(model_files)} model(s):")
    for mf in model_files:
        print(f"  - {os.path.basename(mf)}")

    # Test each model
    results = []
    for model_file in sorted(model_files):
        if os.path.exists(model_file):
            result = test_single_model(model_file, num_episodes=50)
            results.append(result)

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'Avg Score':>10} {'Max Score':>10} {'Std Dev':>10}")
        print(f"{'-'*70}")
        for r in sorted(results, key=lambda x: x['avg_score'], reverse=True):
            print(f"{r['model']:<25} {r['avg_score']:>10.2f} {r['max_score']:>10d} {r['std_score']:>10.2f}")

    print(f"\n{'='*70}")
    print("TESTING COMPLETE")
    print(f"{'='*70}")

    # Recommendations
    best_result = max(results, key=lambda x: x['avg_score'])
    print(f"\nRecommendation: Use {best_result['model']}")
    print(f"  Best avg score: {best_result['avg_score']:.2f}")
    print(f"  Best max score: {best_result['max_score']}")

    if best_result['max_score'] > 2:
        print(f"\n✓ Model CAN score {best_result['max_score']}! It learned something!")
    else:
        print(f"\n✗ Model stuck at score {best_result['max_score']} - didn't learn much")

if __name__ == '__main__':
    main()

