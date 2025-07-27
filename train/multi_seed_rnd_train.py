# multi_seed_rnd_train.py
import os
import time
import random
import numpy as np
import torch
import gym
import wandb
from dataclasses import dataclass
import tyro
from mini_behavior.register import register
from algorithms.RND_PPO import RND_PPO
from env_wrapper import CustomObservationWrapper

@dataclass
class Args:
    # Experiment settings
    exp_name: str = "rnd_ppo_multi_seed"
    """experiment name for logging"""
    seed_start: int = 3
    """starting seed number"""
    seed_count: int = 4
    """number of seeds to run sequentially"""
    
    # Environment settings
    task: str = "MultiToy"
    """the task name"""
    room_size: int = 8
    """room size for the environment"""
    max_steps: int = 1000
    """maximum steps per episode"""
    
    # Training settings
    total_timesteps: int = int(25e5)
    """total timesteps for each training run"""
    num_envs: int = 8
    """number of parallel environments"""
    num_steps: int = 200
    """number of steps per environment per update"""
    rnd_reward_scale: float = 0.1
    """scaling factor for intrinsic rewards"""
    
    # Hardware settings
    cuda: bool = False
    """whether to use CUDA if available"""
    
    # Logging settings
    track: bool = True
    """whether to track with wandb"""
    wandb_project_name: str = "rnd-ppo-minigrid-multi-seed"
    """wandb project name"""
    wandb_entity: str = None
    """wandb entity name"""
    
    # Saving settings
    save_freq: int = 100000
    """how often to save model (in steps)"""
    save_dir: str = "models/RND_PPO_multi_seed"
    """directory to save models"""

def make_env(env_id, seed, idx, env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk

def init_env(env_id, num_envs, seed, env_kwargs):
    return gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, env_kwargs) for i in range(num_envs)]
    )

def setup_env_registration(args):
    """Register the training and test environments"""
    env_name = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-LP-v4"
    env_kwargs = {"room_size": args.room_size, "max_steps": args.max_steps}
    
    test_env_name = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-LP-v5"
    test_env_kwargs = {"room_size": args.room_size, "max_steps": args.max_steps, "test_env": True}
    
    # Import the env_list to check if environments are already registered
    from mini_behavior.register import env_list
    
    # Only register environments if they're not already registered
    if env_name not in env_list:
        register(
            id=env_name,
            entry_point=f'mini_behavior.envs:{args.task}Env',
            kwargs=env_kwargs
        )
    
    if test_env_name not in env_list:
        register(
            id=test_env_name, 
            entry_point=f'mini_behavior.envs:{args.task}Env',
            kwargs=test_env_kwargs
        )
    
    return env_name, env_kwargs, test_env_name, test_env_kwargs

def setup_experiment_directory(args, seed):
    """Create directories for saving models and logs"""
    run_id = int(time.time())
    seed_save_dir = f"{args.save_dir}/{args.task}_{args.room_size}x{args.room_size}_seed{seed}_{run_id}"
    
    if not os.path.exists(seed_save_dir):
        os.makedirs(seed_save_dir, exist_ok=True)
        
    return seed_save_dir

def run_seed(args, seed):
    """Run a complete training with the specified random seed"""
    # Set the seed for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True
    
    # Setup environment
    env_name, env_kwargs, _, _ = setup_env_registration(args)
    env = init_env(env_name, args.num_envs, seed, env_kwargs)
    
    # Setup directories
    seed_save_dir = setup_experiment_directory(args, seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Initialize WandB for this seed
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
                "rnd_reward_scale": args.rnd_reward_scale,
                "seed": seed
            },
            reinit=True  # Allow multiple runs in one script
        )
    
    print(f"\n{'='*50}")
    print(f"Starting training with seed {seed}")
    print(f"Task: {args.task}, Room Size: {args.room_size}x{args.room_size}")
    print(f"Total Steps: {args.total_timesteps:,}, Num Envs: {args.num_envs}")
    print(f"Save directory: {seed_save_dir}")
    print(f"{'='*50}\n")
    
    # Initialize your RND_PPO model
    model = RND_PPO(
        env=env,
        env_id=env_name,
        device=device,
        total_timesteps=int(args.total_timesteps),
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        rnd_reward_scale=args.rnd_reward_scale,
        seed=seed
    )
    
    # Train the model
    model.train(save_freq=args.save_freq, save_path=seed_save_dir)
    
    # Save final model
    final_model_path = f"{seed_save_dir}/final_model.pt"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    if args.track:
        wandb.finish()
    
    return final_model_path

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Generate seeds based on seed_start and seed_count
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    
    # Ensure the base save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Run all specified seeds
    results = {}
    for seed in seeds:
        model_path = run_seed(args, seed)
        results[seed] = model_path
    
    # Print summary
    print("\nTraining complete for all seeds!")
    print("\nSummary of trained models:")
    for seed, path in results.items():
        print(f"Seed {seed}: {path}")