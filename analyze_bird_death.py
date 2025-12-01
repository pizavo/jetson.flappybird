#!/usr/bin/env python3
"""Analyze why bird dies so quickly"""

from train_ai import FlappyBirdEnv

print("=" * 70)
print("Why does bird die at step 35-40?")
print("=" * 70)

env = FlappyBirdEnv()
env.reset()

print(f"\nBird starts at: y={env.bird_y}, velocity={env.bird_velocity}")
print(f"Screen height: {600}, bird size: {32}")
print(f"Dies if: y <= 0 or y >= {600-32}")

print(f"\nSimulating bird with NO jumps (just falling):")
for step in range(100):
    obs, reward, done, info = env.step(0)  # No jump

    if step < 10 or step % 10 == 0 or done:
        print(f"Step {step:3d}: y={env.bird_y:7.2f}, v={env.bird_velocity:7.2f}, alive={env.alive}")

    if done:
        if env.bird_y >= 568:
            print(f"\n→ Hit FLOOR at y={env.bird_y:.2f}")
        else:
            print(f"\n→ Hit CEILING at y={env.bird_y:.2f}")
        break

print(f"\n" + "=" * 70)
print(f"Simulating bird jumping every 15 frames:")
env.reset()
for step in range(100):
    action = 1 if step % 15 == 0 else 0
    obs, reward, done, info = env.step(action)

    if step < 10 or step % 10 == 0 or done:
        print(f"Step {step:3d}: y={env.bird_y:7.2f}, v={env.bird_velocity:7.2f}, alive={env.alive}")

    if done:
        if env.bird_y >= 568:
            print(f"\n→ Hit FLOOR at y={env.bird_y:.2f}")
        elif env.bird_y <= 0:
            print(f"\n→ Hit CEILING at y={env.bird_y:.2f}")
        else:
            print(f"\n→ Hit pipe")
        break
else:
    print(f"\n→ Survived 100 steps!")

print(f"\n" + "=" * 70)
print(f"Testing various jump frequencies:")
for freq in [10, 12, 15, 17, 20, 25]:
    env.reset()
    for step in range(150):
        action = 1 if step % freq == 0 else 0
        obs, reward, done, info = env.step(action)
        if done:
            break

    result = "✓ Good" if step > 50 else "✗ Dies too soon"
    print(f"Jump every {freq:2d} frames: Survived {step:3d} steps - {result}")

