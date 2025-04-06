#train_rnd.py
import gym
import os
import numpy as np
from mini_behavior.register import register
from RND_PPO import RND_PPO
import torch
import wandb
wandb.login()

TASK = 'MultiToy'
ROOM_SIZE = 8
MAX_STEPS = 1000
TOTAL_TIMESTEPS = 3e6

# Training settings
NUM_ENVS = 8
NUM_STEPS = 125
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FREQUENCY = 100000 

env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v0"
env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
test_env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v1"
test_env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "test_env": True}

save_dir = f"models/RND_PPO_{TASK}_Run4_32x32"

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
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("\nStarting Training")
    print(f"Task: {TASK}, Room Size: {ROOM_SIZE}x{ROOM_SIZE}")
    print(f"Total Steps: {TOTAL_TIMESTEPS:,}, Num Envs: {NUM_ENVS}")

    wandb.init(
        project="rnd-ppo-minigrid",
        name=f"RND_PPO_{TASK}_{ROOM_SIZE}x{ROOM_SIZE}",
        config={
            "env_name": env_name,
            "task": TASK,
            "room_size": ROOM_SIZE,
            "max_steps": MAX_STEPS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "num_steps": NUM_STEPS,
            "device": DEVICE,
            "rnd_reward_scale": 0.1
        }
    )
    
    model = RND_PPO(
        env_id=env_name,
        device=DEVICE,
        total_timesteps=int(TOTAL_TIMESTEPS),
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        rnd_reward_scale=0.1,
        seed=1
    )
    
    model.train(save_freq=SAVE_FREQUENCY, save_path=save_dir)
    
    model.save(f"{save_dir}/final_model.pt")
    print(f"Model saved to {save_dir}/final_model.pt")
    
    wandb.finish()