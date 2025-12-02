#!/usr/bin/env python3
"""
Watch the trained model play with detailed output
"""

import torch
import time
from train_ai import DQN, FlappyBirdEnv, device
from config import NETWORK_CONFIG, IO_CONFIG
import os

INPUT_SIZE = NETWORK_CONFIG['input_size']
HIDDEN_SIZE = NETWORK_CONFIG['hidden_size']
OUTPUT_SIZE = NETWORK_CONFIG['output_size']
MODEL_DIR = IO_CONFIG['model_dir']

def load_model(model_path):
    model = DQN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()
    return model

def watch_model_play(model_path, num_episodes=3, delay=0.05):
    """Watch the model play with detailed frame-by-frame output"""
    print("="*80)
    print(f"WATCHING MODEL: {os.path.basename(model_path)}")
    print("="*80)

    model = load_model(model_path)
    env = FlappyBirdEnv()

    for episode in range(num_episodes):
        print(f"\n{'-'*80}")
        print(f"EPISODE {episode + 1}")
        print(f"{'-'*80}")

        state = env.reset()
        total_reward = 0
        steps = 0
        last_score = 0

        print(f"\nInitial state:")
        print(f"  Bird: x={env.bird_x:.1f}, y={env.bird_y:.1f}, velocity={env.bird_velocity:.2f}")
        print(f"  Pipes: {len(env.pipes)}")
        for i, pipe in enumerate(env.pipes):
            print(f"    Pipe {i}: x={pipe['x']:.1f}, gap={pipe['gap_y']:.1f}-{pipe['gap_y']+pipe['gap_height']:.1f}")

        print(f"\n{'Step':<5} {'Action':<8} {'Bird Y':<8} {'Velocity':<9} {'Pipes':<6} {'Score':<6} {'Reward':<8} {'Note'}")
        print("-"*80)

        while True:
            # Get action from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
                q0, q1 = q_values[0].cpu().numpy()

            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            # Check for score change
            current_score = info.get('score', 0)
            note = ""
            if current_score > last_score:
                note = f"🎉 SCORED! ({last_score} → {current_score})"
                last_score = current_score

            # Print every step or when something interesting happens
            if steps <= 10 or steps % 10 == 0 or note or done:
                action_name = "JUMP" if action == 1 else "nothing"
                pipe_info = f"{len(env.pipes)}"
                if env.pipes:
                    closest_pipe = min(env.pipes, key=lambda p: abs(p['x'] - env.bird_x))
                    pipe_info += f" (x={closest_pipe['x']:.0f})"

                print(f"{steps:<5} {action_name:<8} {env.bird_y:<8.1f} {env.bird_velocity:<9.2f} "
                      f"{pipe_info:<6} {current_score:<6} {reward:<8.2f} {note}")

            if done:
                print(f"\n{'='*80}")
                if env.bird_y <= 0:
                    death_reason = "HIT CEILING"
                elif env.bird_y >= 568:
                    death_reason = "HIT FLOOR"
                else:
                    death_reason = "HIT PIPE"

                print(f"DIED: {death_reason}")
                print(f"Final Score: {current_score}")
                print(f"Total Steps: {steps}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"{'='*80}")
                break

            if delay > 0:
                time.sleep(delay)

        input(f"\nPress Enter to continue to episode {episode + 2}...")

def main():
    import sys

    model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"\nAvailable models:")
        import glob
        for mf in glob.glob(os.path.join(MODEL_DIR, '*.pth')):
            print(f"  - {mf}")
        return

    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    print(f"\nThis will show frame-by-frame what the AI is doing.")
    print(f"Watch for:")
    print(f"  - When it jumps vs stays still")
    print(f"  - When pipes appear and where")
    print(f"  - When it scores")
    print(f"  - How it dies\n")

    input("Press Enter to start...")

    watch_model_play(model_path, num_episodes=5, delay=0.0)

if __name__ == '__main__':
    main()

