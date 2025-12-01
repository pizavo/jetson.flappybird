#!/bin/bash
# Complete setup verification and training script

echo "========================================================================"
echo "FLAPPY BIRD AI - FINAL SETUP VERIFICATION"
echo "========================================================================"
echo ""

# Step 1: Clear ALL Python cache
echo "STEP 1: Clearing all Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
echo "✓ Cache cleared"
echo ""

# Step 2: Verify config.py values
echo "STEP 2: Verifying config.py has correct values..."
echo "----------------------------------------------------------------------"
grep "'gravity':" config.py
grep "'jump_strength':" config.py
grep "'pipe_gap':" config.py
grep "'pipe_speed':" config.py
grep "'pipe_spawn_interval':" config.py
echo ""
grep "'alive_reward':" config.py
grep "'death_penalty':" config.py
grep "'pipe_pass_bonus':" config.py
grep "'use_distance_reward':" config.py
echo ""
echo "EXPECTED VALUES:"
echo "  gravity: 0.4"
echo "  jump_strength: -8.0"
echo "  pipe_gap: 350"
echo "  pipe_speed: 1.5"
echo "  pipe_spawn_interval: 100"
echo "  alive_reward: 0.5"
echo "  death_penalty: -5"
echo "  pipe_pass_bonus: 20"
echo "  use_distance_reward: False"
echo ""

# Step 3: Test that Python loads correct values
echo "STEP 3: Testing that Python loads correct values..."
python3 -B -c "
from config import GAME_CONFIG, REWARD_CONFIG
print('Values loaded by Python:')
print(f\"  gravity: {GAME_CONFIG['gravity']} (should be 0.4)\")
print(f\"  jump_strength: {GAME_CONFIG['jump_strength']} (should be -8.0)\")
print(f\"  pipe_gap: {GAME_CONFIG['pipe_gap']} (should be 350)\")
print(f\"  pipe_speed: {GAME_CONFIG['pipe_speed']} (should be 1.5)\")
print(f\"  pipe_pass_bonus: {REWARD_CONFIG['pipe_pass_bonus']} (should be 20)\")
print(f\"  death_penalty: {REWARD_CONFIG['death_penalty']} (should be -5)\")
print(f\"  alive_reward: {REWARD_CONFIG['alive_reward']} (should be 0.5)\")
print(f\"  use_distance_reward: {REWARD_CONFIG['use_distance_reward']} (should be False)\")
print()

# Check first pipe spawn position
from train_ai import FlappyBirdEnv
env = FlappyBirdEnv()
env.reset()
print(f\"First pipe spawns at x={env.pipes[0]['x']} (should be 250.0)\")
print(f\"Pipe gap height: {env.pipes[0]['gap_height']} (should be 350.0)\")
print()

# Quick survival test
score = 0
for _ in range(5):
    env.reset()
    steps = 0
    for s in range(200):
        action = 1 if s % 15 == 0 else 0
        _, _, done, info = env.step(action)
        steps += 1
        if done:
            score += info.get('score', 0)
            break
    print(f\"  Test run: {steps} steps, score: {info.get('score', 0)}\")

print(f\"Average score with periodic jumping: {score/5:.1f}\")
if score > 0:
    print('✓ AI CAN score with easier settings!')
else:
    print('⚠ Still difficult - may need more training')
"

echo ""
echo "========================================================================"
echo "VERIFICATION COMPLETE!"
echo ""
echo "If all values match 'should be' values above, run:"
echo "  python3 -B train_ai.py"
echo ""
echo "Expected results:"
echo "  - Episodes 1-50: Some scores > 0"
echo "  - Episodes 50-200: Regular scores 1-10"
echo "  - Episodes 200-500: Scores 10-50+"
echo "========================================================================"

