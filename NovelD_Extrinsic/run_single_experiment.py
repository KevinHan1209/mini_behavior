#!/usr/bin/env python3
"""
Run a single NovelD experiment with extrinsic rewards
"""
import sys
import os
import json
import argparse
import shutil
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.NovelD_PPO import NovelD_PPO
import torch
import gym
from env_wrapper import CustomObservationWrapper
from mini_behavior.utils.states_base import RelativeObjectState
from mini_behavior.register import register

def build_obs_header(env):
    """Construct labeled header matching CustomObservationWrapper observation layout.
    Layout:
      - agent_x, agent_y, agent_dir
      - For each object: {obj_type}_{idx}_x, {obj_type}_{idx}_y, then non-default binary flags
    """
    default_states = [
        'atsamelocation',
        'infovofrobot',
        'inleftreachofrobot',
        'inrightreachofrobot',
        'inside',
        'nextto',
        'inlefthandofrobot',
        'inrighthandofrobot',
    ]
    header = ['agent_x', 'agent_y', 'agent_dir']
    # Iterate objects in the same order as CustomObservationWrapper
    for obj_type, obj_list in env.objs.items():
        for idx, obj in enumerate(obj_list):
            # Position
            header.append(f"{obj_type}_{idx}_x")
            header.append(f"{obj_type}_{idx}_y")
            # Binary non-relative states except defaults
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    if state_name not in default_states:
                        header.append(f"{obj_type}_{idx}_{state_name}")
    return header


def run_experiment(config_path, output_dir='results', log_obs=False, obs_csv_path=None, max_obs_steps=500):
    """Run a single experiment based on config file"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    exp_name = config['name']
    hyperparams = config['hyperparameters']
    extrinsic_rewards = config['extrinsic_rewards']
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_name}")
    print(f"Description: {config['description']}")
    print(f"Extrinsic Rewards: {extrinsic_rewards}")
    print(f"{'='*60}\n")
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Save config to output directory
    config_copy_path = os.path.join(exp_output_dir, 'experiment_config.json')
    shutil.copy2(config_path, config_copy_path)
    
    # Register environment with extrinsic rewards
    env_id = 'MiniGrid-MultiToy-8x8-N2-Extrinsic-v0'
    TASK = 'MultiToy'
    ROOM_SIZE = 8
    MAX_STEPS = 1000
    env_kwargs = {
        "room_size": ROOM_SIZE, 
        "max_steps": MAX_STEPS,
        "extrinsic_rewards": extrinsic_rewards
    }
    register(
        id=env_id,
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=env_kwargs
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create temporary checkpoint directory for this experiment
    original_checkpoint_dir = "checkpoints"
    exp_checkpoint_dir = os.path.join(exp_output_dir, "checkpoints")
    
    # Temporarily rename the original checkpoints directory if it exists
    checkpoint_backup = None
    if os.path.exists(original_checkpoint_dir):
        checkpoint_backup = f"{original_checkpoint_dir}_backup"
        # Remove any existing backup first
        if os.path.exists(checkpoint_backup):
            if os.path.islink(checkpoint_backup):
                os.unlink(checkpoint_backup)
            else:
                shutil.rmtree(checkpoint_backup)
        os.rename(original_checkpoint_dir, checkpoint_backup)
    
    try:
        # Create symlink to experiment checkpoint directory
        os.makedirs(exp_checkpoint_dir, exist_ok=True)
        os.symlink(os.path.abspath(exp_checkpoint_dir), original_checkpoint_dir)
        
        # Determine wandb project based on experiment category (v2 for new runs)
        wandb_project = "NovelD_Extrinsic_v2"  # Default
        if 'location' in exp_name.lower():
            wandb_project = "NovelD_Location_v2"
        elif 'interaction' in exp_name.lower():
            wandb_project = "NovelD_Interaction_v2"
        elif 'noise' in exp_name.lower():
            wandb_project = "NovelD_Noise_v2"
        elif 'pure_intrinsic' in exp_name.lower():
            wandb_project = "NovelD_Intrinsic_v2"
        
        # Create agent with experiment hyperparameters
        agent = NovelD_PPO(
            env_id,
            device,
            total_timesteps=hyperparams['total_timesteps'],
            learning_rate=hyperparams['learning_rate'],
            num_envs=hyperparams['num_envs'],
            num_steps=hyperparams['num_steps'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            num_minibatches=hyperparams['num_minibatches'],
            update_epochs=hyperparams['update_epochs'],
            clip_coef=hyperparams['clip_coef'],
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            int_coef=hyperparams['int_coef'],
            ext_coef=hyperparams['ext_coef'],
            int_gamma=hyperparams['int_gamma'],
            alpha=hyperparams['alpha'],
            update_proportion=hyperparams['update_proportion'],
            wandb_project=wandb_project,
            wandb_run_name=exp_name
        )
        
        # Train agent
        print(f"\nStarting training with hyperparameters:")
        print(f"  alpha: {hyperparams['alpha']}")
        print(f"  update_proportion: {hyperparams['update_proportion']}")
        print(f"  int_gamma: {hyperparams['int_gamma']}")
        print(f"  ent_coef: {hyperparams['ent_coef']}")
        print(f"  int_coef: {hyperparams['int_coef']}")
        print(f"  ext_coef: {hyperparams['ext_coef']}")
        print(f"  total_timesteps: {hyperparams['total_timesteps']:,}")
        print(f"\nExtrinsic Rewards:")
        for reward_type, value in extrinsic_rewards.items():
            print(f"  {reward_type}: {value}")
        
        agent.train()
        
        print(f"\nTraining completed for {exp_name}")
        
        # Optional: log one episode of observations with labeled header
        if log_obs:
            # Build a single test env matching training obs (CustomObservationWrapper)
            test_env = CustomObservationWrapper(gym.make(env_id))
            # Header
            env_unwrapped = getattr(test_env, 'env', test_env)
            header = build_obs_header(env_unwrapped)
            # Actions may be multi-discrete; we will log action vector columns upfront
            action_dims = getattr(agent, 'action_dims', None)
            act_cols = [f"act_{i}" for i in range(len(action_dims))] if action_dims is not None else ["act"]
            # Prepare CSV path
            obs_dir = os.path.join(exp_output_dir, 'obs_logs')
            os.makedirs(obs_dir, exist_ok=True)
            if obs_csv_path:
                csv_path = obs_csv_path
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(obs_dir, f"obs_{exp_name}_{ts}.csv")
            # Write CSV
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Combined header: step, actions..., reward, done, then labeled obs fields
                writer.writerow(['step'] + act_cols + ['reward', 'done'] + header)
                # Run episode
                obs = test_env.reset()
                done = False
                step = 0
                while not done and step < max_obs_steps:
                    obs_vec = obs if isinstance(obs, (list, tuple,)) else os
                    obs_arr = None
                    try:
                        import numpy as _np
                        obs_arr = _np.asarray(obs, dtype=_np.float32).reshape(-1)
                    except Exception:
                        # Fallback: if dict-like, skip
                        pass
                    # Policy action
                    import numpy as np
                    import torch as th
                    obs_tensor = th.tensor(obs_arr if obs_arr is not None else obs, dtype=th.float32, device=agent.device).unsqueeze(0)
                    with th.no_grad():
                        action, _, _, _, _ = agent.agent.get_action_and_value(obs_tensor)
                    act_np = action.detach().cpu().numpy()
                    # Flatten action to list
                    if act_np.ndim > 1:
                        act_list = act_np[0].tolist()
                    else:
                        act_list = [int(act_np.item())]
                    # Step
                    obs, reward, done, _ = test_env.step(action.cpu().numpy()[0] if act_np.ndim > 1 else int(act_np.item()))
                    # Pad / truncate action columns to match act_cols
                    if len(act_list) < len(act_cols):
                        act_list = act_list + [""] * (len(act_cols) - len(act_list))
                    elif len(act_list) > len(act_cols):
                        act_list = act_list[:len(act_cols)]
                    # Ensure obs array
                    if obs_arr is None:
                        try:
                            obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                        except Exception:
                            obs_arr = np.array([], dtype=np.float32)
                    # Row
                    writer.writerow([step] + act_list + [float(reward), 1 if done else 0] + obs_arr.tolist())
                    step += 1
            print(f"Saved observation log to {csv_path}")
        
        # Create success marker
        with open(os.path.join(exp_output_dir, 'SUCCESS'), 'w') as f:
            f.write(f"Experiment {exp_name} completed successfully\n")
        
    finally:
        # Clean up symlink
        if os.path.islink(original_checkpoint_dir):
            os.unlink(original_checkpoint_dir)
        
        # Restore original checkpoints directory
        if checkpoint_backup and os.path.exists(checkpoint_backup):
            os.rename(checkpoint_backup, original_checkpoint_dir)
    
    return exp_output_dir

def main():
    parser = argparse.ArgumentParser(description='Run a single NovelD experiment with extrinsic rewards')
    parser.add_argument('config', help='Path to experiment configuration JSON file')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--log-obs', action='store_true',
                       help='After training, run one episode and log labeled observations to CSV')
    parser.add_argument('--obs-csv', default=None,
                       help='Optional explicit path for observation CSV; defaults to results/<exp_name>/obs_logs/obs_<exp>_<ts>.csv')
    parser.add_argument('--max-obs-steps', type=int, default=500,
                       help='Max steps to log in the post-training observation episode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    try:
        output_dir = run_experiment(args.config, args.output_dir, log_obs=args.log_obs, obs_csv_path=args.obs_csv, max_obs_steps=args.max_obs_steps)
        print(f"\nResults saved to: {output_dir}")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()