import os
import gym
from algorithms.APT_PPO import APT_PPO
from env_wrapper import CustomObservationWrapper

# ===== Parameters =====
TOTAL_TIMESTEPS = int(1e4)
NUM_ENVS = 8
SAVE_FREQUENCY = 100
TEST_STEPS = 500

env_name = 'MiniGrid-MultiToy-8x8-N2-v0'
env_kwargs = {"room_size": 8, "max_steps": 1000}

# ===== Helper Functions =====
def make_env(env_id, seed, idx, env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk

def init_env(num_envs: int, seed: int):
    return gym.vector.SyncVectorEnv(
        [make_env(env_name, seed, i, env_kwargs) for i in range(num_envs)]
    )

# ===== Main Block =====
if __name__ == "__main__":
    # Instantiate a vectorized training environment.
    env = init_env(NUM_ENVS, seed=1)
    
    save_dir = f"models/APT_PPO"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Begin training")
    # Use the same env for training and testing by passing the same id and kwargs.
    model = APT_PPO(
        env=env,
        env_id=env_name,
        env_kwargs=env_kwargs,
        save_dir=save_dir,
        num_envs=NUM_ENVS,
        total_timesteps=TOTAL_TIMESTEPS,
        save_freq=SAVE_FREQUENCY,
        test_steps=TEST_STEPS
    )
    
    print("\n=== Observation Space ===")
    print("Shape:", env.observation_space.shape)
    print("Type:", env.observation_space.dtype)
    
    model.train()
    model.test_agent(save_episode="final", num_episodes=1, max_steps_per_episode=TEST_STEPS)
