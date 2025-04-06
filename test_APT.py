import os
import gym
from algorithms.APT_PPO import APT_PPO
from env_wrapper import CustomObservationWrapper

# ===== Parameters =====
TOTAL_TIMESTEPS = int(2e6)
NUM_ENVS = 8
NUM_EPS = 5
SAVE_FREQUENCY = 100
TEST_STEPS = 500

ENV_NAME = 'MiniGrid-MultiToy-8x8-N2-v0'
ENV_KWARGS = {"room_size": 8, "max_steps": 10000}
SEED = 1

def make_env(env_id: str, seed: int, idx: int, env_kwargs: dict):
    """
    Create a callable that returns an environment instance.
    """
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk

def init_env(env_name: str, num_envs: int, seed: int, env_kwargs: dict):
    """
    Initialize a vectorized (synchronous) environment.
    Note: Use the `single_observation_space` attribute if available.
    """
    return gym.vector.SyncVectorEnv(
        [make_env(env_name, seed, i, env_kwargs) for i in range(num_envs)]
    )

if __name__ == "__main__":
    # Instantiate a vectorized training environment.
    env = init_env(ENV_NAME, NUM_ENVS, SEED, ENV_KWARGS)
    
    save_dir = "APT_PPO"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Begin training")
    print("\n=== Environment Observation Space ===")
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    print("Shape:", obs_space.shape)
    print("Type:", obs_space.dtype)

    # Instantiate the APT_PPO agent.
    model = APT_PPO(
        env=env,
        env_id=ENV_NAME,
        env_kwargs=ENV_KWARGS,
        save_dir=save_dir,
        num_envs=NUM_ENVS,
        total_timesteps=TOTAL_TIMESTEPS,
        save_freq=SAVE_FREQUENCY,
        test_steps=TEST_STEPS
    )
    
    # Train the agent.
    model.train()
    
    # Test the agent after training.
    model.test_agent(num_episodes=NUM_EPS, max_steps_per_episode=TEST_STEPS, save_episode=True)
