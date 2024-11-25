import gym
import os
from mini_behavior.register import register
from algorithms.APT_PPO import APT_PPO
from env_wrapper import CustomObservationWrapper


TASK = 'MultiToy'
PARTIAL_OBS = True
ROOM_SIZE = 16
MAX_STEPS = 1000
TOTAL_TIMESTEPS = 1e6
DENSE_REWARD = False
POLICY_TYPE = 'CnnPolicy'
NUM_ENVS = 8
NUM_STEPS = 125
SAVE_FREQUENCY = 100

env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-v0"
env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}


def make_env(env_id, seed, idx, env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk


def init_env(num_envs: int, seed):

    return gym.vector.SyncVectorEnv(
                [make_env(env_name, seed, i, env_kwargs) for i in range(num_envs)]
                )
    

        
if __name__ == "__main__":
    register(
        id=env_name,
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=env_kwargs
    )
    env = init_env(NUM_ENVS, seed = 1)
        
    print('begin training')
    # Policy training
    model = APT_PPO(env, env_id = env_name, num_envs=NUM_ENVS, total_timesteps = TOTAL_TIMESTEPS, num_steps=NUM_STEPS, save_freq = SAVE_FREQUENCY)

    model.train()

    # Define the directory path
    save_dir = "models/ATP_PPO_MultiToy"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(f"{save_dir}/{env_name}", env_kwargs = env_kwargs)