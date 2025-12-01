#!/usr/bin/env python3
"""
Visualize training progress and model performance
"""

import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Jetson Nano
import matplotlib.pyplot as plt
import numpy as np

from config import IO_CONFIG

DEFAULT_STATS_PATH = os.path.join(IO_CONFIG['model_dir'], 'training_stats.json')
DEFAULT_MODELS_DIR = IO_CONFIG['model_dir']
DEFAULT_PLOTS_DIR = IO_CONFIG['plot_dir']

def plot_training_stats(stats_file=DEFAULT_STATS_PATH, output_dir=DEFAULT_PLOTS_DIR):
    """Plot training statistics"""

    if not os.path.exists(stats_file):
        print(f"Stats file not found: {stats_file}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load stats
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    scores = stats['scores']
    episodes = list(range(1, len(scores) + 1))

    # Calculate moving average
    window_size = 50
    moving_avg = []
    for i in range(len(scores)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(scores[start_idx:i+1]))

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Flappy Bird AI Training Results', fontsize=16, fontweight='bold')

    # Plot 1: Score over episodes
    axes[0, 0].plot(episodes, scores, alpha=0.3, label='Episode Score')
    axes[0, 0].plot(episodes, moving_avg, linewidth=2, label=f'{window_size}-Episode Moving Average')
    axes[0, 0].axhline(y=stats['best_score'], color='r', linestyle='--', label=f'Best Score: {stats["best_score"]}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Score distribution histogram
    axes[0, 1].hist(scores, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
    axes[0, 1].axvline(x=np.median(scores), color='g', linestyle='--', label=f'Median: {np.median(scores):.2f}')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cumulative best score
    cumulative_best = []
    current_best = 0
    for score in scores:
        current_best = max(current_best, score)
        cumulative_best.append(current_best)

    axes[1, 0].plot(episodes, cumulative_best, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Best Score')
    axes[1, 0].set_title('Cumulative Best Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(episodes, cumulative_best, alpha=0.3, color='green')

    # Plot 4: Performance improvement rate
    window = 100
    improvement_rate = []
    for i in range(window, len(scores)):
        old_avg = np.mean(scores[i-window:i-window//2])
        new_avg = np.mean(scores[i-window//2:i])
        improvement = new_avg - old_avg
        improvement_rate.append(improvement)

    if improvement_rate:
        axes[1, 1].plot(range(window, len(scores)), improvement_rate, linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Score Improvement')
        axes[1, 1].set_title(f'Learning Rate (per {window//2} episodes)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(range(window, len(scores)), improvement_rate, alpha=0.3,
                                color='purple', where=[x >= 0 for x in improvement_rate])

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, 'training_results.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Training visualization saved to: {output_file}")

    # Create a summary text file
    summary_file = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FLAPPY BIRD AI TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Date: {stats.get('training_date', 'Unknown')}\n")
        f.write(f"Total Episodes: {len(scores)}\n\n")

        f.write("Performance Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Score:      {stats['best_score']}\n")
        f.write(f"Average Score:   {np.mean(scores):.2f}\n")
        f.write(f"Median Score:    {np.median(scores):.2f}\n")
        f.write(f"Std Deviation:   {np.std(scores):.2f}\n")
        f.write(f"Min Score:       {min(scores)}\n")
        f.write(f"Max Score:       {max(scores)}\n\n")

        # Performance by phase
        phase_size = len(scores) // 4
        f.write("Performance by Training Phase:\n")
        f.write("-" * 40 + "\n")
        for i, phase_name in enumerate(['Early', 'Mid-Early', 'Mid-Late', 'Late']):
            start = i * phase_size
            end = (i + 1) * phase_size if i < 3 else len(scores)
            phase_scores = scores[start:end]
            f.write(f"{phase_name:12s} (Ep {start+1:4d}-{end:4d}): "
                   f"Avg={np.mean(phase_scores):6.2f}, "
                   f"Max={max(phase_scores):3d}, "
                   f"Std={np.std(phase_scores):6.2f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Training summary saved to: {summary_file}")

    plt.close()

def plot_model_comparison(models_dir=DEFAULT_MODELS_DIR, test_episodes=20, output_dir=DEFAULT_PLOTS_DIR):
    """Compare different model checkpoints"""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from test_ai import load_model
    from train_ai import FlappyBirdEnv

    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])

    if not model_files:
        print(f"No models found in {models_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    model_names = []
    avg_scores = []
    max_scores = []
    min_scores = []

    print("Evaluating models...")
    for model_file in model_files:
        print(f"  Testing {model_file}...")
        model_path = os.path.join(models_dir, model_file)

        try:
            model = load_model(model_path)
            env = FlappyBirdEnv()

            scores = []
            for _ in range(test_episodes):
                state = env.reset()
                while True:
                    import torch
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = model(state_tensor)
                        action = q_values.argmax().item()

                    next_state, reward, done, info = env.step(action)
                    state = next_state

                    if done:
                        scores.append(info.get('score', 0))
                        break

            model_names.append(model_file.replace('.pth', ''))
            avg_scores.append(np.mean(scores))
            max_scores.append(max(scores))
            min_scores.append(min(scores))
        except Exception as e:
            print(f"    Error testing {model_file}: {e}")

    if not model_names:
        print("No models could be evaluated")
        return

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Checkpoint Comparison', fontsize=16, fontweight='bold')

    x = np.arange(len(model_names))
    width = 0.35

    # Bar chart
    ax1.bar(x, avg_scores, width, label='Average Score', alpha=0.8)
    ax1.errorbar(x, avg_scores,
                 yerr=[np.array(avg_scores) - np.array(min_scores),
                       np.array(max_scores) - np.array(avg_scores)],
                 fmt='none', ecolor='red', capsize=5, label='Min/Max Range')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Average Performance per Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Line plot
    ax2.plot(x, avg_scores, 'o-', linewidth=2, markersize=8, label='Average')
    ax2.fill_between(x, min_scores, max_scores, alpha=0.3, label='Min-Max Range')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Trend')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nModel comparison saved to: {output_file}")

    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Flappy Bird AI training')
    parser.add_argument('--stats', type=str, default=DEFAULT_STATS_PATH,
                        help='Path to training stats JSON file')
    parser.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR,
                        help='Directory containing model checkpoints')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PLOTS_DIR,
                        help='Output directory for plots')
    parser.add_argument('--compare', action='store_true',
                        help='Compare model checkpoints')
    parser.add_argument('--test-episodes', type=int, default=20,
                        help='Number of test episodes for model comparison')

    args = parser.parse_args()

    print("Generating training visualizations...")
    plot_training_stats(args.stats, args.output_dir)

    if args.compare:
        print("\nComparing model checkpoints...")
        plot_model_comparison(args.models_dir, args.test_episodes, args.output_dir)

    print("\nVisualization complete!")
