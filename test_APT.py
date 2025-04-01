import os
import gym
from algorithms.APT_PPO import APT_PPO
from mini_behavior.register import register
from env_wrapper import CustomObservationWrapper

# ===== Parameters =====
TASK = 'MultiToy'
ROOM_SIZE = 16
MAX_STEPS = 1000
TOTAL_TIMESTEPS = 1e6
DENSE_REWARD = False
POLICY_TYPE = 'CnnPolicy'
NUM_ENVS = 8
NUM_STEPS = 125
SAVE_FREQUENCY = 10
TEST_STEPS = 500

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

env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v2"
env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
test_env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v3"
test_env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "test_env": True}
# ===== Main Block =====
if __name__ == "__main__":
    register(
         id=env_name,
         entry_point=f'mini_behavior.envs:{TASK}Env',
         kwargs=env_kwargs
     )
    register(
         id=test_env_name, 
         entry_point=f'mini_behavior.envs:{TASK}Env',
         kwargs=test_env_kwargs
     )
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
