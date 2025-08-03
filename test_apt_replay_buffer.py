import torch
import numpy as np
from algorithms.APT_PPO import APT_PPO, ReplayBuffer
import gym
from env_wrapper import CustomObservationWrapper
from mini_behavior.envs.multitoy import MultiToyEnv

def test_replay_buffer():
    """Test the replay buffer functionality."""
    print("Testing Replay Buffer...")
    
    device = torch.device("cpu")
    capacity = 1000
    obs_dim = 50
    num_envs = 4
    
    # Create replay buffer
    buffer = ReplayBuffer(capacity, obs_dim, num_envs, device)
    
    # Test adding observations
    for i in range(100):
        for env_idx in range(num_envs):
            obs = torch.randn(obs_dim, device=device)
            buffer.add(obs, env_idx)
    
    # Test sampling
    for env_idx in range(num_envs):
        batch = buffer.sample(32, env_idx)
        assert batch.shape == (32, obs_dim), f"Expected shape (32, {obs_dim}), got {batch.shape}"
        print(f"✓ Env {env_idx}: Successfully sampled batch of shape {batch.shape}")
    
    # Test capacity limit
    for i in range(2000):  # Add more than capacity
        buffer.add(torch.randn(obs_dim, device=device), 0)
    
    assert buffer.buffers[0]['size'] == capacity, f"Buffer size should be capped at {capacity}"
    print(f"✓ Buffer correctly capped at capacity: {capacity}")
    
    print("\nReplay Buffer tests passed!")

def test_apt_with_replay():
    """Test APT algorithm with replay buffer integration."""
    print("\nTesting APT with Replay Buffer...")
    
    # Create environment
    env_kwargs = {
        'room_size': 12
    }
    
    # Create vectorized environment
    from stable_baselines3.common.vec_env import DummyVecEnv
    def make_env():
        def _init():
            env = MultiToyEnv(**env_kwargs)
            env = CustomObservationWrapper(env)
            return env
        return _init
    
    num_envs = 2
    env = DummyVecEnv([make_env() for i in range(num_envs)])
    
    # Create APT agent with new replay buffer parameters
    apt = APT_PPO(
        env=env,
        env_id="MultiToyEnv",
        env_kwargs=env_kwargs,
        save_dir="test_apt_replay",
        num_envs=num_envs,
        num_steps=32,  # Small for testing
        total_timesteps=1024,  # Must be divisible by batch_size (32*2=64)
        k=12,  # URLB parameter
        replay_buffer_size=10000,  # Smaller for testing
        batch_size=64  # Smaller for testing
    )
    
    print(f"APT Configuration:")
    print(f"  - k-NN k: {apt.k}")
    print(f"  - Replay buffer size: {apt.replay_buffer_size}")
    print(f"  - APT batch size: {apt.apt_batch_size}")
    print(f"  - Rollout steps: {apt.num_steps}")
    print(f"  - Parallel envs: {apt.num_envs}")
    
    # Run a short training to test functionality
    try:
        print("\nRunning short training test...")
        apt.train()
        print("✓ Training completed successfully with replay buffer!")
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        raise
    
    env.close()

if __name__ == "__main__":
    print("=== APT Replay Buffer Implementation Test ===\n")
    
    # Test replay buffer
    test_replay_buffer()
    
    # Test APT integration
    test_apt_with_replay()
    
    print("\n=== All tests completed! ===")