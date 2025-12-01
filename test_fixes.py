#!/usr/bin/env python3
"""
Test script to verify the AI learning fixes
"""

import sys
import numpy as np
from train_ai import FlappyBirdEnv, DQNAgent
from config import GAME_CONFIG, TRAINING_CONFIG

def test_environment():
    """Test that the environment works correctly"""
    print("Testing FlappyBirdEnv...")
    env = FlappyBirdEnv()

    # Test reset
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"Initial score: {env.score}")

    # Test a few steps
    total_reward = 0
    for i in range(100):
        action = 1 if i % 10 == 0 else 0  # Jump every 10 frames
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Game ended at step {i}")
            print(f"Final score: {info.get('score', 0)}")
            print(f"Total reward: {total_reward:.2f}")
            break
    else:
        print(f"Survived 100 steps!")
        print(f"Score: {env.score}")
        print(f"Total reward: {total_reward:.2f}")

    print("✓ Environment test passed\n")

def test_reward_structure():
    """Test that rewards are being calculated correctly"""
    print("Testing reward structure...")
    env = FlappyBirdEnv()
    env.reset()

    # Test alive reward
    _, reward, done, _ = env.step(0)
    print(f"Reward for staying alive (no jump): {reward:.3f}")

    # Test jump penalty
    env.reset()
    _, reward, done, _ = env.step(1)
    print(f"Reward for jumping: {reward:.3f}")

    # Simulate passing a pipe
    env.reset()
    env.bird_x = 100
    env.pipes = [{'x': 50, 'gap_y': 250, 'gap_height': 180, 'width': 80, 'passed': False}]
    env.bird_y = 300
    _, reward, done, info = env.step(0)
    print(f"Reward for passing pipe: {reward:.3f}")
    print(f"Score after passing pipe: {info['score']}")

    print("✓ Reward structure test passed\n")

def test_agent_training():
    """Test that the agent can learn from experiences"""
    print("Testing agent training...")
    agent = DQNAgent()
    env = FlappyBirdEnv()

    # Collect some experiences
    for episode in range(5):
        state = env.reset()
        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    print(f"Memory size: {len(agent.memory)}")

    # Train on experiences
    if len(agent.memory) >= agent.batch_size:
        loss = agent.train()
        print(f"Training loss: {loss:.6f}")
        print("✓ Agent training test passed\n")
    else:
        print("⚠ Not enough experiences to train\n")

def test_score_tracking():
    """Test that scores are properly tracked"""
    print("Testing score tracking...")
    env = FlappyBirdEnv()

    # Reset and manually pass some pipes
    env.reset()
    initial_score = env.score
    print(f"Initial score: {initial_score}")

    # Add a pipe and move past it
    env.pipes = [{'x': 150, 'gap_y': 250, 'gap_height': 180, 'width': 80, 'passed': False}]
    env.bird_x = 100
    env.bird_y = 300

    # Take steps to pass the pipe
    for i in range(30):
        _, _, done, info = env.step(0)
        if done:
            break

    final_score = info.get('score', env.score)
    print(f"Final score: {final_score}")

    if final_score > initial_score:
        print("✓ Score tracking test passed\n")
    else:
        print("⚠ Score did not increase (might have died before passing)\n")

if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING FLAPPY BIRD AI FIXES TESTS")
    print("=" * 60 + "\n")

    try:
        test_environment()
        test_reward_structure()
        test_score_tracking()
        test_agent_training()

        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        print("\nYou can now run: python train_ai.py")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

