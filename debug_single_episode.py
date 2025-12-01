#!/usr/bin/env python3
"""
Debug script to see what's happening in a single episode
"""

import sys
import importlib

# Force reload
import config
importlib.reload(config)

from train_ai import FlappyBirdEnv, DQNAgent
from config import GAME_CONFIG

def debug_episode():
    """Run a single episode with detailed logging"""
    print("=" * 80)
    print("DEBUGGING SINGLE EPISODE")
    print("=" * 80)

    print("\nGame Config:")
    print(f"  Screen: {GAME_CONFIG['screen_width']}x{GAME_CONFIG['screen_height']}")
    print(f"  Bird start: x={GAME_CONFIG['bird_start_x']}, y={GAME_CONFIG['bird_start_y']}")
    print(f"  Gravity: {GAME_CONFIG['gravity']}")
    print(f"  Jump strength: {GAME_CONFIG['jump_strength']}")
    print(f"  Pipe gap: {GAME_CONFIG['pipe_gap']}")
    print(f"  Pipe speed: {GAME_CONFIG['pipe_speed']}")

    env = FlappyBirdEnv()
    agent = DQNAgent()

    state = env.reset()

    print(f"\nInitial state:")
    print(f"  Bird: y={env.bird_y:.1f}, velocity={env.bird_velocity:.1f}")
    print(f"  Pipes: {len(env.pipes)}")
    if env.pipes:
        print(f"    Pipe 0: x={env.pipes[0]['x']:.1f}, gap_y={env.pipes[0]['gap_y']:.1f}-{env.pipes[0]['gap_y'] + env.pipes[0]['gap_height']:.1f}")

    print(f"\n{'Step':<6} {'Action':<8} {'Bird Y':<10} {'Velocity':<10} {'Pipe X':<10} {'Reward':<10} {'Alive':<6}")
    print("-" * 80)

    for step in range(100):
        # Let agent choose action
        action = agent.select_action(state)
        action_name = "JUMP" if action == 1 else "nothing"

        next_state, reward, done, info = env.step(action)

        pipe_x = env.pipes[0]['x'] if env.pipes else -1

        print(f"{step:<6} {action_name:<8} {env.bird_y:<10.1f} {env.bird_velocity:<10.2f} {pipe_x:<10.1f} {reward:<10.3f} {env.alive!s:<6}")

        if done:
            print("\n" + "=" * 80)
            print(f"EPISODE ENDED")
            print(f"  Final score: {info.get('score', 0)}")
            print(f"  Total steps: {step + 1}")
            print(f"  Death reason: ", end="")
            if env.bird_y <= 0:
                print("Hit ceiling")
            elif env.bird_y >= GAME_CONFIG['screen_height'] - GAME_CONFIG['bird_size']:
                print("Hit floor")
            else:
                print("Hit pipe")
            print("=" * 80)
            break

        state = next_state
    else:
        print(f"\nSurvived 100 steps! Score: {env.score}")

if __name__ == '__main__':
    debug_episode()

