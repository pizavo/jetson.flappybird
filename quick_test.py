#!/usr/bin/env python3
"""Simple test to verify pipes spawn close enough"""

import sys
sys.path.insert(0, '.')

from train_ai import FlappyBirdEnv

env = FlappyBirdEnv()
env.reset()

print("First pipe:", env.pipes[0]['x'])

for i in range(100):
    action = 1 if i % 18 == 0 else 0
    _, _, done, info = env.step(action)
    if done:
        print(f"Died at step {i}, score: {info['score']}")
        break
else:
    print(f"Survived 100 steps, score: {env.score}")

print("Test complete!")

