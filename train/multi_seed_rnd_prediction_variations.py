# multi_seed_rnd_prediction_variations.py
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
    exp_name: str = "rnd_ppo_variations"
    """experiment name for logging"""
    seed_start: int = 3
    """starting seed number"""
    seed_count: int = 3
    """number of seeds to run sequentially"""
    
    # Environment settings
    task: str = "MultiToy"
    """the task name"""
    room_size: int = 8
    """room size for the environment"""
    max_steps: int = 1000
    """maximum steps per episode"""
    
    # Training settings
    total_timesteps: int = int(2e6)
    """total timesteps for each training run"""
    num_envs: int = 8
    """number of parallel environments"""
    num_steps: int = 200
    """number of steps per environment per update"""
    rnd_reward_scale: float = 0.1
    """scaling factor for intrinsic rewards"""
    ent_coef: float = 0.01
    """entropy coefficient (fixed based on previous experiments)"""
    
    # RND Prediction Network Variations
    rnd_update_freq_values: list = None
    """list of RND predictor update frequencies to test (1 = every step, 2 = every 2 steps, etc.)"""
    rnd_weight_decay_values: list = None
    """list of weight decay values for RND predictor"""
    
    # Hardware settings
    cuda: bool = False
    """whether to use CUDA if available"""
    
    # Logging settings
    track: bool = True
    """whether to track with wandb"""
    wandb_project_name: str = "rnd_ppo_variations"
    """wandb project name"""
    wandb_entity: str = None
    """wandb entity name"""
    
    # Saving settings
    save_freq: int = 100000
    """how often to save model (in steps)"""
    save_dir: str = "models/RND_PPO_variations"
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
    env_kwargs = {"room_size": args.room_size, "max_steps": args.max_steps,
                   "extrinsic_rewards": {"noise": 0.1, "interaction": 0.1, "location_change": 0.1}
        }
    
    test_env_name = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-LP-v5"
    test_env_kwargs = {"room_size": args.room_size, "max_steps": args.max_steps,
                   "extrinsic_rewards": {"noise": 0.1, "interaction": 0.1, "location_change": 0.1}, "test_env": True}
    
    from mini_behavior.register import env_list
    
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

def setup_experiment_directory(args, rnd_update_freq, rnd_weight_decay, seed):
    """Create directories for saving models and logs"""
    run_id = int(time.time())
    freq_str = f"freq{rnd_update_freq}"
    decay_str = f"decay{str(rnd_weight_decay).replace('.', '_').replace('-', 'neg')}"
    seed_save_dir = f"{args.save_dir}/{args.task}_{args.room_size}x{args.room_size}_{freq_str}_{decay_str}_seed{seed}_{run_id}"
    
    if not os.path.exists(seed_save_dir):
        os.makedirs(seed_save_dir, exist_ok=True)
        
    return seed_save_dir

def run_experiment(args, rnd_update_freq, rnd_weight_decay, seed):
    """Run a complete training with the specified RND prediction network settings"""
    # Set the seed for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True
    
    env_name, env_kwargs, _, _ = setup_env_registration(args)
    env = init_env(env_name, args.num_envs, seed, env_kwargs)
    
    seed_save_dir = setup_experiment_directory(args, rnd_update_freq, rnd_weight_decay, seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    if args.track:
        freq_str = f"freq{rnd_update_freq}"
        decay_str = f"decay{str(rnd_weight_decay).replace('.', '_').replace('-', 'neg')}"
        run_name = f"{args.exp_name}_{args.task}_{args.room_size}x{args.room_size}_{freq_str}_{decay_str}_seed{seed}"
        
        project_name = f"{args.wandb_project_name}-{freq_str}-{decay_str}"
        
        wandb.init(
            project=project_name,
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
                "ent_coef": args.ent_coef,
                "rnd_update_freq": rnd_update_freq,
                "rnd_weight_decay": rnd_weight_decay,
                "seed": seed
            },
            reinit=True  # Allow multiple runs in one script
        )
    
    print(f"\n{'='*60}")
    print(f"Starting training with RND update_freq={rnd_update_freq}, weight_decay={rnd_weight_decay}, seed={seed}")
    print(f"Task: {args.task}, Room Size: {args.room_size}x{args.room_size}")
    print(f"Total Steps: {args.total_timesteps:,}, Num Envs: {args.num_envs}")
    print(f"Entropy Coef: {args.ent_coef} (fixed)")
    print(f"Save directory: {seed_save_dir}")
    print(f"{'='*60}\n")
    
    model = RND_PPO(
        env=env,
        env_id=env_name,
        device=device,
        total_timesteps=int(args.total_timesteps),
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        rnd_reward_scale=args.rnd_reward_scale,
        ent_coef=args.ent_coef,
        rnd_update_freq=rnd_update_freq,  # New parameter
        rnd_weight_decay=rnd_weight_decay,  # New parameter
        seed=seed
    )
    
    model.train(save_freq=args.save_freq, save_path=seed_save_dir)
    
    final_model_path = f"{seed_save_dir}/final_model.pt"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    if args.track:
        wandb.finish()
    
    return final_model_path

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Default values for RND prediction network variations
    if args.rnd_update_freq_values is None:
        args.rnd_update_freq_values = [2, 4]  
    
    if args.rnd_weight_decay_values is None:
        args.rnd_weight_decay_values = [0.0] #[0.0, 1e-4, 1e-3, 1e-2]  # Range of weight decay values
    
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    results = {}
    total_experiments = len(args.rnd_update_freq_values) * len(args.rnd_weight_decay_values) * len(seeds)
    experiment_count = 0
    
    print(f"\nStarting RND prediction network variations experiment")
    print(f"Total experiments to run: {total_experiments}")
    print(f"RND Update Frequencies: {args.rnd_update_freq_values}")
    print(f"RND Weight Decay values: {args.rnd_weight_decay_values}")
    print(f"Seeds: {seeds}")
    print(f"Fixed Entropy Coefficient: {args.ent_coef}")
    
    for seed in seeds:
        for rnd_update_freq in args.rnd_update_freq_values:
            for rnd_weight_decay in args.rnd_weight_decay_values:
                key = (rnd_update_freq, rnd_weight_decay)
                if key not in results:
                    results[key] = {}
                
                experiment_count += 1
                print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                print(f"RND Update Freq: {rnd_update_freq}, Weight Decay: {rnd_weight_decay}, Seed: {seed}")
                
                model_path = run_experiment(args, rnd_update_freq, rnd_weight_decay, seed)
                results[key][seed] = model_path
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total experiments completed: {total_experiments}")
    print(f"RND Update Frequencies tested: {args.rnd_update_freq_values}")
    print(f"RND Weight Decay values tested: {args.rnd_weight_decay_values}")
    print(f"Seeds used: {seeds}")
    print("\nSummary of all trained models:")
    print("-" * 80)
    
    for (rnd_update_freq, rnd_weight_decay) in results.keys():
        print(f"\nRND Update Freq: {rnd_update_freq}, Weight Decay: {rnd_weight_decay}")
        for seed in seeds:
            print(f"  Seed {seed}: {results[(rnd_update_freq, rnd_weight_decay)][seed]}")