import gym
import os
from mini_behavior.register import register
from algorithms.APT_PPO import APT_PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import wandb



TASK = 'PlayAlligator'
PARTIAL_OBS = True
ROOM_SIZE = 10
MAX_STEPS = 1000
TOTAL_TIMESTEPS = 1e6
DENSE_REWARD = False
POLICY_TYPE = 'CnnPolicy'
NUM_ENVS = 8
NUM_STEPS = 125
env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-v0"
env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "exploration_type": "ATP"}

def get_single_env() -> gym.Env:
    '''
    policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128)
)
    '''

    # Env wrapping
    env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-v0"

    kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "exploration_type": "ATP"}

    if DENSE_REWARD:
        assert TASK in ["PuttingAwayDishesAfterCleaning", "WashingPotsAndPans"]
        kwargs["dense_reward"] = True

    register(
        id=env_name,
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=kwargs
    )

    '''
    config = {
        "policy_type": POLICY_TYPE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "env_name": env_name,
    }
    '''
    
    env = gym.make(env_name)

    return env


def init_env(num_envs: int):

    env_fns = [lambda: get_single_env() for _ in range(num_envs)]
    if num_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)
    
    
if __name__ == "__main__":
    env = init_env(NUM_ENVS)
        
    print('begin training')
    # Policy training
    model = APT_PPO(env, env_id = env_name, num_envs=NUM_ENVS, total_timesteps = TOTAL_TIMESTEPS, num_steps=NUM_STEPS, save_freq = 100)

    model.train()

    # Define the directory path
    save_dir = "models/ATP_PPO_PlayAlligator"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(f"{save_dir}/{env_name}", env_kwargs = env_kwargs)