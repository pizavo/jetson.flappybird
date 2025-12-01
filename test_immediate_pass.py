#!/usr/bin/env python3
"""Test that bird can now pass the first pipe immediately"""

from train_ai import FlappyBirdEnv

print("=" * 70)
print("TESTING IMMEDIATE PIPE PASSING")
print("=" * 70)

env = FlappyBirdEnv()
env.reset()

print(f"\nInitial setup:")
print(f"  Bird position: x={env.bird_x}, y={env.bird_y}")
print(f"  First pipe: x={env.pipes[0]['x']}, right_edge={env.pipes[0]['x'] + env.pipes[0]['width']}")
print(f"  Pipe gap: {env.pipes[0]['gap_y']}-{env.pipes[0]['gap_y'] + env.pipes[0]['gap_height']}")
print(f"  Bird is at y={env.bird_y}, gap is {env.pipes[0]['gap_y']}-{env.pipes[0]['gap_y'] + env.pipes[0]['gap_height']}")
print(f"  Bird in gap? {env.pipes[0]['gap_y'] < env.bird_y < env.pipes[0]['gap_y'] + env.pipes[0]['gap_height']}")

# Test with no jumping (just let bird fall)
print(f"\nTest 1: No jumping (bird falls)")
for step in range(200):
    obs, reward, done, info = env.step(0)  # No jump

    if step < 5 or info.get('score', 0) > 0 or done:
        pipe_info = ""
        if env.pipes:
            p = env.pipes[0]
            pipe_info = f"pipe_x={p['x']:.1f}, right={p['x']+p['width']:.1f}, passed={p['passed']}"
        print(f"  Step {step:3d}: y={env.bird_y:6.1f}, score={info.get('score', 0)}, {pipe_info}, alive={env.alive}")

    if info.get('score', 0) > 0:
        print(f"\n{'='*70}")
        print(f"✓✓✓ SUCCESS! Scored on step {step}!")
        print(f"{'='*70}")
        break

    if done:
        print(f"\n{'='*70}")
        print(f"✗ Died at step {step} without scoring")
        print(f"  Final score: {info.get('score', 0)}")
        print(f"{'='*70}")
        break
else:
    print(f"\n✓ Survived 200 steps! Score: {env.score}")

# Test 2: With periodic jumping
print(f"\nTest 2: Jumping every 15 frames")
env.reset()
for step in range(200):
    action = 1 if step % 15 == 0 else 0
    obs, reward, done, info = env.step(action)

    if step < 5 or info.get('score', 0) > 0 or done:
        pipe_info = ""
        if env.pipes:
            p = env.pipes[0]
            pipe_info = f"pipe_x={p['x']:.1f}, right={p['x']+p['width']:.1f}, passed={p['passed']}"
        print(f"  Step {step:3d}: y={env.bird_y:6.1f}, score={info.get('score', 0)}, {pipe_info}, alive={env.alive}")

    if info.get('score', 0) > 0:
        print(f"\n{'='*70}")
        print(f"✓✓✓ SUCCESS! Scored on step {step}!")
        print(f"{'='*70}")
        break

    if done:
        print(f"\n{'='*70}")
        print(f"✗ Died at step {step}")
        print(f"  Final score: {info.get('score', 0)}")
        print(f"{'='*70}")
        break
else:
    print(f"\n✓ Survived 200 steps! Score: {env.score}")

print("\n" + "=" * 70)
print("If either test shows 'SUCCESS! Scored', the fix worked!")
print("=" * 70)

