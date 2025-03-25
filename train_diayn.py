#train_diayn.py
import gym
import os
import numpy as np
from mini_behavior.register import register
from DIAYN_PPO import DIAYN
import torch
import wandb
wandb.login()

TASK = 'MultiToy'
ROOM_SIZE = 8
MAX_STEPS = 1000
TOTAL_TIMESTEPS = 2e6
N_SKILLS = 64 #test this out, 64?

NUM_ENVS = 8
NUM_STEPS = 125
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FREQUENCY = 100000 

env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v2"
env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
test_env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v3"
test_env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "test_env": True}

save_dir = f"models/DIAYN_{TASK}_Run3_64skills"

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
        project="diayn-minigrid",
        name=f"DIAYN_{TASK}_{ROOM_SIZE}x{ROOM_SIZE}",
        config={
            "env_name": env_name,
            "task": TASK,
            "room_size": ROOM_SIZE,
            "max_steps": MAX_STEPS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "num_steps": NUM_STEPS,
            "device": DEVICE,
            "n_skills": N_SKILLS,
            "disc_coef": 0.1
        }
    )
    
    model = DIAYN(
        env_id=env_name,
        device=DEVICE,
        total_timesteps=int(TOTAL_TIMESTEPS),
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        n_skills=N_SKILLS,
        disc_coef=0.1,
        seed=1
    )
    
    model.train(save_freq=SAVE_FREQUENCY, save_path=save_dir)
    
    model.save(f"{save_dir}/final_model.pt")
    print(f"Model saved to {save_dir}/final_model.pt")
    
    wandb.finish()