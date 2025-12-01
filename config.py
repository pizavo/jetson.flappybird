"""
Configuration file for Flappy Bird AI Training
Modify these parameters to customize the training process
"""

# ==================== GAME SETTINGS ====================
GAME_CONFIG = {
    'screen_width': 800,
    'screen_height': 600,
    'bird_size': 32,
    'bird_start_x': 100,
    'bird_start_y': 300,
    'gravity': 0.4,  # REDUCED: Slower falling for easier learning
    'jump_strength': -8.0,  # REDUCED: Gentler jumps, more control
    'pipe_width': 80,
    'pipe_gap': 350,  # INCREASED: Much larger gap, easier to pass
    'pipe_speed': 1.5,  # REDUCED: Slower pipes, more time to react
    'pipe_spawn_interval': 100,  # INCREASED: More time between pipes
}

# ==================== NEURAL NETWORK SETTINGS ====================
NETWORK_CONFIG = {
    'input_size': 5,
    'hidden_size': 128,
    'output_size': 2,
    'activation': 'relu',
}

# ==================== TRAINING SETTINGS ====================
TRAINING_CONFIG = {
    # Number of training episodes
    'num_episodes': 1000,

    # Learning rate for the optimizer
    'learning_rate': 0.0005,

    # Discount factor for future rewards
    'gamma': 0.99,

    # Exploration settings
    'epsilon_start': 1.0,
    'epsilon_min': 0.1,  # INCREASED: Keep more exploration
    'epsilon_decay': 0.9995,  # MUCH SLOWER: Explore for longer

    # Batch size for training
    'batch_size': 32,

    # Replay buffer capacity
    'buffer_capacity': 10000,

    # Target network update frequency (in episodes)
    'target_update_frequency': 10,

    # Save checkpoint frequency (in episodes)
    'checkpoint_frequency': 50,

    # Maximum steps per episode (0 = unlimited)
    'max_steps_per_episode': 0,
}

# ==================== REWARD SETTINGS ====================
REWARD_CONFIG = {
    # Reward for staying alive per frame
    'alive_reward': 0.5,  # INCREASED: Stronger survival reward

    # Penalty for jumping (DISABLED - jumping is necessary!)
    'jump_penalty': 0.0,

    # Penalty for collision/death
    'death_penalty': -5,  # REDUCED: Lighter penalty to not overwhelm positives

    # Bonus for passing a pipe
    'pipe_pass_bonus': 20,  # INCREASED: Much bigger reward for success!

    # Reward shaping (distance-based rewards)
    'use_distance_reward': False,  # DISABLED: Simplify learning, remove confusing signals
    'distance_reward_scale': 0.0,
}

# ==================== OPTIMIZER SETTINGS ====================
OPTIMIZER_CONFIG = {
    'optimizer': 'adam',  # 'adam', 'sgd', 'rmsprop'
    'adam_betas': (0.9, 0.999),
    'adam_eps': 1e-08,
    'weight_decay': 0,

    # Gradient clipping
    'clip_grad_norm': 1.0,
}

# ==================== LOSS FUNCTION SETTINGS ====================
LOSS_CONFIG = {
    'loss_function': 'smooth_l1',  # 'smooth_l1', 'mse'
}

# ==================== SAVE/LOAD SETTINGS ====================
IO_CONFIG = {
    'model_dir': 'models',
    'plot_dir': 'plots',
    'save_best_model': True,
    'save_checkpoints': True,
    'save_final_model': True,
    'save_training_stats': True,
}

# ==================== CUDA SETTINGS ====================
CUDA_CONFIG = {
    # Use CUDA if available
    'use_cuda': True,

    # CUDA device ID (for multi-GPU systems)
    'cuda_device': 0,

    # Enable cudnn benchmark for performance
    'cudnn_benchmark': True,
}

# ==================== LOGGING SETTINGS ====================
LOGGING_CONFIG = {
    # Print frequency (in episodes)
    'print_frequency': 1,

    # Verbose output
    'verbose': True,

    # Log file path (None = no file logging)
    'log_file': None,
}

# ==================== ADVANCED SETTINGS ====================
ADVANCED_CONFIG = {
    # Double DQN
    'use_double_dqn': False,

    # Dueling DQN
    'use_dueling_dqn': False,

    # Prioritized Experience Replay
    'use_prioritized_replay': False,
    'priority_alpha': 0.6,
    'priority_beta_start': 0.4,
    'priority_beta_frames': 100000,

    # Noisy Networks
    'use_noisy_networks': False,

    # Multi-step learning
    'n_step': 1,

    # Categorical DQN
    'use_categorical_dqn': False,
    'num_atoms': 51,
    'v_min': -10,
    'v_max': 10,
}


def get_config():
    """Get the complete configuration dictionary"""
    return {
        'game': GAME_CONFIG,
        'network': NETWORK_CONFIG,
        'training': TRAINING_CONFIG,
        'reward': REWARD_CONFIG,
        'optimizer': OPTIMIZER_CONFIG,
        'loss': LOSS_CONFIG,
        'io': IO_CONFIG,
        'cuda': CUDA_CONFIG,
        'logging': LOGGING_CONFIG,
        'advanced': ADVANCED_CONFIG,
    }


def print_config():
    """Print the current configuration"""
    config = get_config()
    print("=" * 60)
    print("FLAPPY BIRD AI TRAINING CONFIGURATION")
    print("=" * 60)

    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        print("-" * 40)
        for key, value in settings.items():
            print(f"  {key:30s}: {value}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_config()

