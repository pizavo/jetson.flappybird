#!/usr/bin/env python3
"""Test that pipes spawn correctly and bird can reach them"""

from train_ai import FlappyBirdEnv
from config import GAME_CONFIG

print("=" * 70)
print("TESTING PIPE SPAWNING BUG FIX")
print("=" * 70)

env = FlappyBirdEnv()
env.reset()

print(f"\nConfig:")
print(f"  Pipe speed: {GAME_CONFIG['pipe_speed']}")
print(f"  Pipe spawn interval: {GAME_CONFIG['pipe_spawn_interval']}")
print(f"  Bird at x={env.bird_x}")

print(f"\nInitial state:")
print(f"  First pipe at x={env.pipes[0]['x']} (should be 10.0 - behind bird)")

print(f"\nSimulating 200 steps with periodic jumping:")
print(f"{'Step':<6} {'Pipes':<7} {'Score':<7} {'Pipe Positions'}")
print("-" * 70)

prev_pipe_count = len(env.pipes)
prev_score = env.score
for step in range(200):
    action = 1 if step % 18 == 0 else 0
    obs, reward, done, info = env.step(action)

    if step % 10 == 0 or info.get('score', 0) > prev_score or len(env.pipes) != prev_pipe_count:
        pipe_positions = ", ".join([f"x={p['x']:.0f}" for p in env.pipes])
        print(f"{step:<6} {len(env.pipes):<7} {info['score']:<7} {pipe_positions}")

    prev_pipe_count = len(env.pipes)
    prev_score = info.get('score', prev_score)

    if done:
        print(f"\nDied at step {step}")
        break
else:
    print(f"\nSurvived 200 steps!")

print(f"\nFinal score: {env.score}")
print(f"Total pipes encountered: {env.score}")

expected_pipes = (200 // GAME_CONFIG['pipe_spawn_interval']) + 1
print(f"Expected pipes to spawn: ~{expected_pipes}")

if env.score > 1:
    print(f"\n✓✓✓ FIX WORKED! Bird scored {env.score} (more than 1!)")
else:
    print(f"\n✗ Still broken - bird only scored {env.score}")

print("=" * 70)

# Calculate how long it takes for second pipe to reach bird
pipe_spawn_x = 200.0
bird_x = 100.0
distance = pipe_spawn_x - bird_x
speed = GAME_CONFIG['pipe_speed']
frames_to_reach = distance / speed

print(f"\nPipe spawn analysis:")
print(f"  Second pipe spawns at x={pipe_spawn_x}")
print(f"  Distance to bird: {distance} pixels")
print(f"  Speed: {speed} pixels/frame")
print(f"  Time to reach bird: {frames_to_reach:.0f} frames")
print(f"  Bird survival in test: {step} frames")

if frames_to_reach < 200:
    print(f"  ✓ Bird should reach second pipe!")
else:
    print(f"  ✗ Pipe takes too long to reach bird")

