#!/usr/bin/env python3
"""
Force clear all Python cache and verify config is correct
"""

import os
import sys
import subprocess

print("=" * 70)
print("FORCE CLEARING ALL PYTHON CACHE")
print("=" * 70)

# Find and delete all __pycache__ directories
for root, dirs, files in os.walk('.'):
    for dir_name in dirs:
        if dir_name == '__pycache__':
            cache_path = os.path.join(root, dir_name)
            print(f"Removing: {cache_path}")
            try:
                import shutil
                shutil.rmtree(cache_path)
            except Exception as e:
                print(f"  Error: {e}")

# Find and delete all .pyc files
for root, dirs, files in os.walk('.'):
    for file_name in files:
        if file_name.endswith('.pyc'):
            pyc_path = os.path.join(root, file_name)
            print(f"Removing: {pyc_path}")
            try:
                os.remove(pyc_path)
            except Exception as e:
                print(f"  Error: {e}")

print("\n" + "=" * 70)
print("VERIFYING CONFIG.PY FILE CONTENT")
print("=" * 70)

# Read and display key values from config.py file
with open('config.py', 'r') as f:
    content = f.read()

    # Extract key values
    import re

    pipe_gap = re.search(r"'pipe_gap':\s*(\d+)", content)
    gravity = re.search(r"'gravity':\s*([\d.]+)", content)
    jump_strength = re.search(r"'jump_strength':\s*(-?[\d.]+)", content)
    pipe_speed = re.search(r"'pipe_speed':\s*([\d.]+)", content)
    pipe_pass_bonus = re.search(r"'pipe_pass_bonus':\s*(\d+)", content)
    death_penalty = re.search(r"'death_penalty':\s*(-?\d+)", content)
    alive_reward = re.search(r"'alive_reward':\s*([\d.]+)", content)

    print("\nValues in config.py FILE:")
    print(f"  pipe_gap: {pipe_gap.group(1) if pipe_gap else 'NOT FOUND'} (should be 350)")
    print(f"  gravity: {gravity.group(1) if gravity else 'NOT FOUND'} (should be 0.4)")
    print(f"  jump_strength: {jump_strength.group(1) if jump_strength else 'NOT FOUND'} (should be -8.0)")
    print(f"  pipe_speed: {pipe_speed.group(1) if pipe_speed else 'NOT FOUND'} (should be 1.5)")
    print(f"  pipe_pass_bonus: {pipe_pass_bonus.group(1) if pipe_pass_bonus else 'NOT FOUND'} (should be 20)")
    print(f"  death_penalty: {death_penalty.group(1) if death_penalty else 'NOT FOUND'} (should be -5)")
    print(f"  alive_reward: {alive_reward.group(1) if alive_reward else 'NOT FOUND'} (should be 0.5)")

print("\n" + "=" * 70)
print("ALL CACHE CLEARED - NOW RUN:")
print("  python3 -B train_ai.py")
print("=" * 70)

