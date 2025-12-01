#!/usr/bin/env python3
"""Quick test to see if bird can now pass pipes"""

import sys
sys.path.insert(0, '.')

from train_ai import FlappyBirdEnv
from config import GAME_CONFIG

print("=" * 70)
print("TESTING ULTRA-EASY SETTINGS")
print("=" * 70)

print(f"\nConfig:")
print(f"  Gravity: {GAME_CONFIG['gravity']}")
print(f"  Jump strength: {GAME_CONFIG['jump_strength']}")
print(f"  Pipe gap: {GAME_CONFIG['pipe_gap']}")
print(f"  Pipe speed: {GAME_CONFIG['pipe_speed']}")

env = FlappyBirdEnv()

print(f"\nTesting different jump patterns:")
print("-" * 70)

patterns = [
    ("Jump every 10 frames", lambda s: 1 if s % 10 == 0 else 0),
    ("Jump every 12 frames", lambda s: 1 if s % 12 == 0 else 0),
    ("Jump every 15 frames", lambda s: 1 if s % 15 == 0 else 0),
    ("Jump every 18 frames", lambda s: 1 if s % 18 == 0 else 0),
    ("Jump every 20 frames", lambda s: 1 if s % 20 == 0 else 0),
]

for name, pattern in patterns:
    total_score = 0
    total_steps = 0

    for _ in range(3):
        env.reset()
        for step in range(300):
            action = pattern(step)
            _, _, done, info = env.step(action)
            if done:
                total_score += info.get('score', 0)
                total_steps += step + 1
                break

    avg_score = total_score / 3
    avg_steps = total_steps / 3

    status = "✓" if avg_score > 0 else "✗"
    print(f"{status} {name:25s}: Avg Score={avg_score:.1f}, Avg Steps={avg_steps:.0f}")

print("\n" + "=" * 70)

# Detailed run of best pattern
print("\nDetailed run (jump every 15 frames):")
env.reset()
print(f"First pipe: x={env.pipes[0]['x']:.0f}, gap={env.pipes[0]['gap_y']:.0f}-{env.pipes[0]['gap_y']+env.pipes[0]['gap_height']:.0f}")

for step in range(300):
    action = 1 if step % 15 == 0 else 0
    obs, reward, done, info = env.step(action)

    if step < 10 or step % 20 == 0 or info.get('score', 0) > 0 or done:
        pipe_x = env.pipes[0]['x'] if env.pipes else -999
        print(f"Step {step:3d}: y={env.bird_y:6.1f}, v={env.bird_velocity:6.2f}, pipe_x={pipe_x:6.1f}, score={info.get('score', 0)}, alive={env.alive}")

    if done:
        print(f"\n{'='*70}")
        print(f"Final: Score={info.get('score', 0)}, Steps={step+1}")
        if info.get('score', 0) > 0:
            print("✓✓✓ SUCCESS! Bird passed a pipe! ✓✓✓")
        else:
            print("✗ Failed - bird died before passing pipe")
        print(f"{'='*70}")
        break
else:
    print(f"\nSurvived 300 steps! Score: {env.score}")
    if env.score > 0:
        print("✓✓✓ SUCCESS! ✓✓✓")

