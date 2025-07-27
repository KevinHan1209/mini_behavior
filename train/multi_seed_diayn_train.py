# multi_seed_diayn_train.py

import os
import time
import random
import numpy as np
import torch
import gym
import wandb
import tyro
from dataclasses import dataclass
from mini_behavior.register import register
from algorithms.DIAYN_PPO_3 import DIAYN
from env_wrapper import CustomObservationWrapper

@dataclass
class Args:
    exp_name: str = "diayn_multi_seed"
    seed_start: int = 1
    seed_count: int = 5

    task: str = "MultiToy"
    room_size: int = 8
    max_steps: int = 1000

    total_timesteps: int = int(3e6)
    num_envs: int = 8
    num_steps: int = 125
    n_skills: int = 16

    cuda: bool = True

    track: bool = True
    wandb_project_name: str = "diayn-minigrid"
    wandb_entity: str = None

    save_freq: int = 100000
    save_dir: str = "models/DIAYN_multi_seed"

def make_env(env_id, seed, idx, env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk

def init_env(env_id, num_envs, seed, env_kwargs):
    return gym.vector.SyncVectorEnv([make_env(env_id, seed, i, env_kwargs) for i in range(num_envs)])

def setup_env_registration(args):
    env_name = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-LP-v2"
    env_kwargs = {"room_size": args.room_size, "max_steps": args.max_steps}

    test_env_name = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-LP-v3"
    test_env_kwargs = {"room_size": args.room_size, "max_steps": args.max_steps, "test_env": True}

    from mini_behavior.register import env_list

    if env_name not in env_list:
        register(id=env_name, entry_point=f'mini_behavior.envs:{args.task}Env', kwargs=env_kwargs)

    if test_env_name not in env_list:
        register(id=test_env_name, entry_point=f'mini_behavior.envs:{args.task}Env', kwargs=test_env_kwargs)

    return env_name, env_kwargs

def setup_experiment_directory(args, seed):
    run_id = int(time.time())
    seed_save_dir = f"{args.save_dir}/{args.task}_{args.room_size}x{args.room_size}_seed{seed}_{run_id}"
    os.makedirs(seed_save_dir, exist_ok=True)
    return seed_save_dir

def run_seed(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_name, env_kwargs = setup_env_registration(args)
    env = init_env(env_name, args.num_envs, seed, env_kwargs)
    seed_save_dir = setup_experiment_directory(args, seed)

    if args.track:
        run_name = f"{args.exp_name}_{args.task}_{args.room_size}x{args.room_size}_seed{seed}"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "env_name": env_name,
                "task": args.task,
                "room_size": args.room_size,
                "max_steps": args.max_steps,
                "total_timesteps": args.total_timesteps,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "device": str(device),
                "n_skills": args.n_skills,
                "seed": seed
            },
            reinit=True
        )

    print(f"\n{'='*50}")
    print(f"Training DIAYN with seed {seed}")
    print(f"{'='*50}\n")

    model = DIAYN(
        env_id=env_name,
        device=device,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        n_skills=args.n_skills,
        seed=seed
    )

    model.train(save_freq=args.save_freq, save_path=seed_save_dir)
    model.save(f"{seed_save_dir}/final_model.pt")

    if args.track:
        wandb.finish()

    return f"{seed_save_dir}/final_model.pt"

if __name__ == "__main__":
    args = tyro.cli(Args)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}
    for seed in seeds:
        model_path = run_seed(args, seed)
        results[seed] = model_path

    print("\nTraining complete for all seeds!\n")
    for seed, path in results.items():
        print(f"Seed {seed}: {path}")
