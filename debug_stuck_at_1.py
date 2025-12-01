#!/usr/bin/env python3
"""Debug why AI isn't learning beyond the free first pipe"""

from train_ai import FlappyBirdEnv, DQNAgent
import torch

print("=" * 70)
print("DEBUGGING: Why is AI stuck at score 1?")
print("=" * 70)

# Test 1: What happens after the free pipe?
print("\n1. What happens after free first pipe?")
env = FlappyBirdEnv()
env.reset()

print(f"   First pipe at x={env.pipes[0]['x']}, passed={env.pipes[0]['passed']}")

for step in range(200):
    action = 0  # No jumping
    obs, reward, done, info = env.step(action)

    if step < 3 or step == 10 or step == 50 or step == 100 or done or len(env.pipes) > 1:
        print(f"   Step {step:3d}: score={info['score']}, pipes={len(env.pipes)}, bird_y={env.bird_y:.1f}, alive={env.alive}")
        if len(env.pipes) > 0:
            print(f"           Pipe 0: x={env.pipes[0]['x']:.1f}, passed={env.pipes[0]['passed']}")
        if len(env.pipes) > 1:
            print(f"           Pipe 1: x={env.pipes[1]['x']:.1f}, gap={env.pipes[1]['gap_y']:.1f}-{env.pipes[1]['gap_y']+env.pipes[1]['gap_height']:.1f}")

    if done:
        print(f"\n   Died at step {step}: score={info['score']}, y={env.bird_y:.1f}")
        break

# Test 2: Check when second pipe spawns
print(f"\n2. When does second pipe spawn?")
from config import GAME_CONFIG
print(f"   pipe_spawn_interval = {GAME_CONFIG['pipe_spawn_interval']}")
print(f"   First pipe passes on step 0")
print(f"   Second pipe should spawn around step {GAME_CONFIG['pipe_spawn_interval']}")

# Test 3: Can bird survive to second pipe?
print(f"\n3. Can bird survive to second pipe with simple strategy?")
env.reset()
for step in range(300):
    action = 1 if step % 18 == 0 else 0  # Jump every 18 frames
    obs, reward, done, info = env.step(action)

    if done:
        print(f"   Died at step {step}, score={info['score']}")
        break

    if info['score'] > 1:
        print(f"   SUCCESS! Reached score {info['score']} at step {step}")
        break
else:
    print(f"   Survived 300 steps, score={env.score}")

# Test 4: Check reward structure
print(f"\n4. Check reward structure:")
from config import REWARD_CONFIG
print(f"   alive_reward = {REWARD_CONFIG['alive_reward']}")
print(f"   pipe_pass_bonus = {REWARD_CONFIG['pipe_pass_bonus']}")
print(f"   death_penalty = {REWARD_CONFIG['death_penalty']}")

# Test 5: Check if AI is actually training
print(f"\n5. Check if network is training:")
agent = DQNAgent()
print(f"   Initial policy_net weights sample: {list(agent.policy_net.parameters())[0][0][0:3]}")

# Add some experiences
env.reset()
for _ in range(100):
    state = env.reset()
    for step in range(50):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

print(f"   Memory size: {len(agent.memory)}")

if len(agent.memory) >= agent.batch_size:
    loss_before = agent.train()
    print(f"   Training loss: {loss_before:.6f}")

    # Train a few more times
    for _ in range(10):
        agent.train()

    print(f"   Policy_net weights after training: {list(agent.policy_net.parameters())[0][0][0:3]}")
    print(f"   (Weights should be different if training is working)")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

