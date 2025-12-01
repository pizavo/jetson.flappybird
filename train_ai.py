#!/usr/bin/env python3
"""
Flappy Bird AI Training using Deep Q-Learning with PyTorch and CUDA
Optimized for Jetson Nano
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json
import os
from datetime import datetime

from config import (
    GAME_CONFIG,
    NETWORK_CONFIG,
    TRAINING_CONFIG,
    REWARD_CONFIG,
    OPTIMIZER_CONFIG,
    LOSS_CONFIG,
    IO_CONFIG,
    CUDA_CONFIG,
)

# Check CUDA availability respecting config toggle
if CUDA_CONFIG.get('use_cuda', True) and torch.cuda.is_available():
    device = torch.device(f"cuda:{CUDA_CONFIG.get('cuda_device', 0)}")
    torch.backends.cudnn.benchmark = CUDA_CONFIG.get('cudnn_benchmark', True)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA Device: {torch.cuda.get_device_name(device.index or 0)}")
    print(f"CUDA Version: {torch.version.cuda}")

SCREEN_HEIGHT = GAME_CONFIG['screen_height']
BIRD_SIZE = GAME_CONFIG['bird_size']
PIPE_SPEED = GAME_CONFIG['pipe_speed']
PIPE_GAP = GAME_CONFIG['pipe_gap']
PIPE_WIDTH = GAME_CONFIG['pipe_width']
GRAVITY = GAME_CONFIG['gravity']
JUMP_STRENGTH = GAME_CONFIG['jump_strength']
PIPE_SPAWN_INTERVAL = GAME_CONFIG['pipe_spawn_interval']
BIRD_START_Y = GAME_CONFIG['bird_start_y']
BIRD_START_X = GAME_CONFIG['bird_start_x']

INPUT_SIZE = NETWORK_CONFIG['input_size']
HIDDEN_SIZE = NETWORK_CONFIG['hidden_size']
OUTPUT_SIZE = NETWORK_CONFIG['output_size']

NUM_EPISODES = TRAINING_CONFIG['num_episodes']
LEARNING_RATE = TRAINING_CONFIG['learning_rate']
GAMMA = TRAINING_CONFIG['gamma']
EPSILON_START = TRAINING_CONFIG['epsilon_start']
EPSILON_MIN = TRAINING_CONFIG['epsilon_min']
EPSILON_DECAY = TRAINING_CONFIG['epsilon_decay']
BATCH_SIZE = TRAINING_CONFIG['batch_size']
BUFFER_CAPACITY = TRAINING_CONFIG['buffer_capacity']
TARGET_UPDATE_FREQUENCY = TRAINING_CONFIG['target_update_frequency']
CHECKPOINT_FREQUENCY = TRAINING_CONFIG['checkpoint_frequency']
MAX_STEPS_PER_EPISODE = TRAINING_CONFIG['max_steps_per_episode']

ALIVE_REWARD = REWARD_CONFIG['alive_reward']
JUMP_PENALTY = REWARD_CONFIG['jump_penalty']
DEATH_PENALTY = REWARD_CONFIG['death_penalty']
PIPE_PASS_BONUS = REWARD_CONFIG['pipe_pass_bonus']
USE_DISTANCE_REWARD = REWARD_CONFIG['use_distance_reward']
DISTANCE_REWARD_SCALE = REWARD_CONFIG['distance_reward_scale']

OPTIMIZER_NAME = OPTIMIZER_CONFIG['optimizer']
ADAM_BETAS = OPTIMIZER_CONFIG['adam_betas']
ADAM_EPS = OPTIMIZER_CONFIG['adam_eps']
WEIGHT_DECAY = OPTIMIZER_CONFIG['weight_decay']
CLIP_GRAD_NORM = OPTIMIZER_CONFIG['clip_grad_norm']

LOSS_FUNCTION = LOSS_CONFIG['loss_function']

MODEL_DIR = IO_CONFIG['model_dir']
SAVE_BEST = IO_CONFIG['save_best_model']
SAVE_CHECKPOINTS = IO_CONFIG['save_checkpoints']
SAVE_FINAL = IO_CONFIG['save_final_model']
SAVE_STATS = IO_CONFIG['save_training_stats']


class DQN(nn.Module):
    """Deep Q-Network for Flappy Bird"""

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class FlappyBirdEnv:
    """Simplified Flappy Bird environment for training"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the environment"""
        self.bird_y = float(BIRD_START_Y)
        self.bird_velocity = 0.0
        self.bird_x = float(BIRD_START_X)
        self.pipes = []
        self.score = 0
        self.alive = True
        self.frame_count = 0
        self.last_pipe_spawn = 0
        self.pipe_spawn_interval = PIPE_SPAWN_INTERVAL  # frames

        # Spawn initial pipe MUCH closer for faster learning
        self._spawn_pipe(initial=True)

        return self._get_observation()

    def _spawn_pipe(self, initial=False):
        """Spawn a new pipe"""
        # Center gap EXACTLY around bird's starting position
        # Bird starts at y=300, gap is 500 tall
        # Gap from 50 to 550 means bird at 300 is perfectly centered
        gap_y = 50.0  # Fixed position, perfectly centered

        if initial:
            # Spawn first pipe so bird is ALREADY passing through it!
            # Bird at x=100
            # Spawn pipe at x=10, width=80, right edge at 90
            # Bird at 100 > 90, so on FIRST STEP pipe moves to x=9.5, right edge 89.5
            # Bird checks: 89.5 < 100? YES! Pipe passed immediately!
            spawn_x = 10.0
        else:
            spawn_x = float(GAME_CONFIG['screen_width'])

        self.pipes.append({
            'x': spawn_x,
            'gap_y': gap_y,
            'gap_height': float(PIPE_GAP),
            'width': float(PIPE_WIDTH),
            'passed': False
        })
        self.last_pipe_spawn = self.frame_count

    def _get_observation(self):
        """Get the current observation state"""
        # Find the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + pipe['width'] > self.bird_x:
                next_pipe = pipe
                break

        if next_pipe:
            obs = [
                self.bird_y / 600.0,  # Normalize bird y position
                self.bird_velocity / 20.0,  # Normalize velocity
                (next_pipe['x'] - self.bird_x) / 800.0,  # Normalize distance to pipe
                next_pipe['gap_y'] / 600.0,  # Normalize gap top
                (next_pipe['gap_y'] + next_pipe['gap_height']) / 600.0  # Normalize gap bottom
            ]
        else:
            obs = [
                self.bird_y / 600.0,
                self.bird_velocity / 20.0,
                1.0,
                0.5,
                0.833
            ]

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment"""
        if not self.alive:
            return self._get_observation(), 0, True, {}

        # Apply action (0 = do nothing, 1 = jump)
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH

        # Apply gravity
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Check boundaries
        if self.bird_y <= 0.0 or self.bird_y >= SCREEN_HEIGHT - BIRD_SIZE:
            self.alive = False
            return self._get_observation(), DEATH_PENALTY, True, {'score': self.score}

        # Update pipes
        reward_bonus = 0.0  # Initialize reward bonus outside the loop
        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED

            # Check if bird passed the pipe
            if not pipe['passed'] and pipe['x'] + pipe['width'] < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                reward_bonus += PIPE_PASS_BONUS  # Accumulate bonuses

            # Check collision
            if self._check_collision(pipe):
                self.alive = False
                return self._get_observation(), DEATH_PENALTY, True, {'score': self.score}

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p['x'] + p['width'] > -10.0]

        # Spawn new pipes
        if self.frame_count - self.last_pipe_spawn >= self.pipe_spawn_interval:
            self._spawn_pipe()

        self.frame_count += 1

        # Calculate reward
        reward = ALIVE_REWARD  # Small reward for staying alive
        if action == 1:
            reward += JUMP_PENALTY  # Penalty for jumping (encourages efficiency)

        if USE_DISTANCE_REWARD and self.pipes:
            # Find the next pipe
            next_pipe = None
            for pipe in self.pipes:
                if pipe['x'] + pipe['width'] > self.bird_x:
                    next_pipe = pipe
                    break

            if next_pipe:
                # Reward for being vertically centered in the gap
                gap_center = next_pipe['gap_y'] + next_pipe['gap_height'] / 2.0
                vertical_distance = abs(self.bird_y - gap_center)
                vertical_reward = -DISTANCE_REWARD_SCALE * (vertical_distance / SCREEN_HEIGHT)
                reward += vertical_reward

        reward += reward_bonus

        return self._get_observation(), reward, False, {'score': self.score}

    def _check_collision(self, pipe):
        """Check if bird collides with pipe"""
        bird_left = self.bird_x
        bird_right = self.bird_x + 32.0
        bird_top = self.bird_y
        bird_bottom = self.bird_y + 32.0

        pipe_left = pipe['x']
        pipe_right = pipe['x'] + pipe['width']

        # Check horizontal overlap
        if bird_right > pipe_left and bird_left < pipe_right:
            # Check vertical collision (top pipe or bottom pipe)
            if bird_top < pipe['gap_y'] or bird_bottom > pipe['gap_y'] + pipe['gap_height']:
                return True

        return False


class DQNAgent:
    """Deep Q-Learning Agent"""

    def __init__(
        self,
        state_size=INPUT_SIZE,
        action_size=OUTPUT_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Create Q-networks
        self.policy_net = DQN(state_size, HIDDEN_SIZE, action_size).to(device)
        self.target_net = DQN(state_size, HIDDEN_SIZE, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer selection
        if OPTIMIZER_NAME.lower() == 'sgd':
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        elif OPTIMIZER_NAME.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        else:
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=learning_rate,
                betas=ADAM_BETAS,
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
            )

        # Loss selection
        if LOSS_FUNCTION == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(BUFFER_CAPACITY)

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), CLIP_GRAD_NORM)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_agent(episodes=NUM_EPISODES, save_dir=MODEL_DIR):
    """Train the DQN agent"""
    os.makedirs(save_dir, exist_ok=True)

    env = FlappyBirdEnv()
    agent = DQNAgent()

    scores = []
    best_score = 0

    print("Starting training...")
    print(f"Device: {device}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        score = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.train()

            state = next_state
            total_reward += reward
            steps += 1
            score = info.get('score', score)  # Update score each step

            if done:
                break

        scores.append(score)

        # Update target network every 10 episodes
        if TARGET_UPDATE_FREQUENCY > 0 and episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Save best model
        if SAVE_BEST and score > best_score:
            best_score = score
            agent.save(os.path.join(save_dir, 'best_model.pth'))

        # Save periodic checkpoint
        if SAVE_CHECKPOINTS and CHECKPOINT_FREQUENCY > 0 and episode % CHECKPOINT_FREQUENCY == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_{episode}.pth'))

        # Print progress
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        print(f"Episode {episode + 1}/{episodes} | Score: {score} | Avg Score: {avg_score:.2f} | "
              f"Best: {best_score} | Steps: {steps} | Epsilon: {agent.epsilon:.3f}")

    # Save final model
    if SAVE_FINAL:
        agent.save(os.path.join(save_dir, 'final_model.pth'))

    if SAVE_STATS:
        stats = {
            'scores': scores,
            'best_score': best_score,
            'avg_score': float(np.mean(scores)),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(os.path.join(save_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best score: {best_score}")
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Models saved to: {save_dir}")


if __name__ == '__main__':
    train_agent(episodes=1000)
