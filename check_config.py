#!/usr/bin/env python3
"""Quick script to check config values"""

from config import REWARD_CONFIG, TRAINING_CONFIG

print("=" * 60)
print("CURRENT CONFIG VALUES")
print("=" * 60)
print("\nREWARD_CONFIG:")
for key, value in REWARD_CONFIG.items():
    print(f"  {key:25s}: {value}")

print("\nTRAINING_CONFIG:")
for key, value in TRAINING_CONFIG.items():
    print(f"  {key:25s}: {value}")
print("=" * 60)

# Also check what train_ai.py loaded
print("\nVALUES LOADED IN train_ai.py:")
from train_ai import ALIVE_REWARD, JUMP_PENALTY, DEATH_PENALTY, PIPE_PASS_BONUS
from train_ai import USE_DISTANCE_REWARD, DISTANCE_REWARD_SCALE
print(f"  ALIVE_REWARD             : {ALIVE_REWARD}")
print(f"  JUMP_PENALTY             : {JUMP_PENALTY}")
print(f"  DEATH_PENALTY            : {DEATH_PENALTY}")
print(f"  PIPE_PASS_BONUS          : {PIPE_PASS_BONUS}")
print(f"  USE_DISTANCE_REWARD      : {USE_DISTANCE_REWARD}")
print(f"  DISTANCE_REWARD_SCALE    : {DISTANCE_REWARD_SCALE}")
print("=" * 60)

