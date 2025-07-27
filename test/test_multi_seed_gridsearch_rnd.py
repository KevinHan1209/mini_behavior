# test_multi_seed_rnd.py
import gym
from algorithms.RND_PPO import RND_PPO
import os
import numpy as np
import torch
import time
import wandb
import argparse
import pandas as pd
from array2gif import write_gif
from env_wrapper import CustomObservationWrapper
from mini_behavior.utils.states_base import RelativeObjectState
from mini_behavior.register import register


def count_binary_flags(env):
    """
    Count the total number of non-relative (binary) state flags in the environment.
    For each object, we skip the two position values and then count each state 
    that is not an instance of RelativeObjectState.
    """
    num_flags = 0
    for obj_list in env.objs.values():
        for obj in obj_list:
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    num_flags += 1
    return num_flags


def extract_binary_flags(obs, env):
    index = 4  # or 3, etc.
    flags = []
    
    for obj_list in env.objs.values():
        for obj in obj_list:
            # Instead of blindly skipping 2, let's do the "stop-when-you-run-out" approach:
            possible_pos = 0
            if index + 2 <= len(obs):
                possible_pos = 2  # try skipping up to 2 for position
            index += possible_pos

            # Now read each non-relative state in the object
            for state_name, state in obj.states.items():
                # Make sure we haven't run out of obs
                if index >= len(obs):
                    break
                # If it's not a RelativeObjectState, read a binary flag
                if not isinstance(state, RelativeObjectState):
                    flags.append(obs[index])
                    index += 1

            # If we've already used up obs, better break out of the loop
            if index >= len(obs):
                break

    return np.array(flags)


def generate_flag_mapping(env):
    """
    Generate a mapping (list of dictionaries) that tells you which binary flag 
    (by its order in the observation) corresponds to which object's non-relative state.
    
    Each mapping entry contains:
      - object_type: the key from env.objs (e.g., "toy", "key", etc.)
      - object_index: the index of the object in the list for that type
      - state_name: the name of the state (e.g., "is_active", "has_been_used", etc.)
    """
    mapping = []
    for obj_type_name, obj_list in env.objs.items():
        for obj_index, obj in enumerate(obj_list):
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    mapping.append({
                        "object_type": obj_type_name,
                        "object_index": obj_index,
                        "state_name": state_name
                    })
    return mapping

def extract_entropy_coef_from_dirname(dirname):

    if "ent0_" not in dirname:
        return 0.01  # Default value if not found
    
    # Extract the part after "ent0_"
    ent_part = dirname.split("ent0_")[1].split("_")[0]
    
    # Convert to float
    if ent_part == "001":
        return 0.001
    elif ent_part == "01":
        return 0.01
    elif ent_part == "1":
        return 0.1
    elif ent_part == "2":
        return 0.2
    elif ent_part == "05":
        return 0.05
    else:
        # Try to parse as float with decimal point
        try:
            return float(f"0.{ent_part}")
        except ValueError:
            print(f"Warning: Could not parse entropy coefficient from '{ent_part}', using default 0.01")
            return 0.01

def make_single_env(env_id, seed, env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    env = CustomObservationWrapper(env)
    env.seed(seed)
    return env


def setup_env_registration(task, room_size, max_steps):
    """Register the training and test environments if not already registered"""
    env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-LP-v4"
    test_env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-LP-v5"
    env_kwargs = {"room_size": room_size, "max_steps": max_steps}
    test_env_kwargs = {"room_size": room_size, "max_steps": max_steps, "test_env": True}
    
    # Import the env_list to check if environments are already registered
    from mini_behavior.register import env_list
    
    # Only register environments if they're not already registered
    if env_name not in env_list:
        register(
            id=env_name,
            entry_point=f'mini_behavior.envs:{task}Env',
            kwargs=env_kwargs
        )
    
    if test_env_name not in env_list:
        register(
            id=test_env_name, 
            entry_point=f'mini_behavior.envs:{task}Env',
            kwargs=test_env_kwargs
        )
    
    return env_name, test_env_name, env_kwargs, test_env_kwargs


def find_model_checkpoint(model_dir, step=None):
    """Find the model checkpoint to test, either at specific step or latest"""
    if step:
        model_path = f"{model_dir}/model_step_{step}.pt"
        if os.path.exists(model_path):
            return model_path
            
    # If step not specified or not found, use the latest checkpoint
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_step_")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
        model_path = os.path.join(model_dir, checkpoints[0])
        print(f"Using checkpoint: {model_path}")
        return model_path
    
    # If no checkpoints, try using final model
    final_model_path = f"{model_dir}/final_model.pt"
    if os.path.exists(final_model_path):
        print(f"Using final model: {final_model_path}")
        return final_model_path
        
    return None


def test_agent(env_id, model, device, task, room_size, step, model_seed, ent_coef, num_episodes=5, max_steps_per_episode=200):
    print(f"\n=== Testing Agent (Seed {model_seed}): {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="rnd-ppo-test-multiseed-gridsearch",
               name=f"RND_PPO_{task}_{room_size}x{room_size}_seed{model_seed}_ent{ent_coef}_step{step}",
               config={"env_id": env_id,
                       "mode": "testing",
                       "model_seed": model_seed,
                       "entropy_coef": ent_coef,
                       "step": step,
                       "num_episodes": num_episodes,
                       "max_steps": max_steps_per_episode})

    test_env_kwargs = {"room_size": room_size, "max_steps": max_steps_per_episode, "test_env": True}
    test_env = make_single_env(env_id, seed=42, env_kwargs=test_env_kwargs)  # Use consistent test seed

    print("Test Env Observation Space Dim:", test_env.observation_space.shape)
    
    # Get the underlying (unwrapped) environment for accessing object info
    env_unwrapped = getattr(test_env, 'env', test_env)

    # Compute the total number of binary flags and generate the mapping
    num_binary_flags = count_binary_flags(env_unwrapped)
    flag_mapping = generate_flag_mapping(env_unwrapped)

    mapping_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name"])
    for idx, mapping in enumerate(flag_mapping):
        mapping_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"])
    wandb.log({"flag_mapping": mapping_table})
    
    # Output directory for GIFs
    #gif_dir = f"img/rnd_{room_size}x{room_size}_seed{model_seed}_step{step}"
    #os.makedirs(gif_dir, exist_ok=True)
    output_dir = f"results/rnd_{room_size}x{room_size}_seed{model_seed}_ent{ent_coef}_step{step}"
    gif_dir = f"{output_dir}/gifs"
    csv_dir = f"{output_dir}/csvs"
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    episode_novelties = []
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        novelty_values = []
        frames = []
        activity = [0] * num_binary_flags
        prev_flags = None
        
        while not done and steps < max_steps_per_episode:
            # Render the environment for visualization
            frame = test_env.render()
            if frame is not None:
                frames.append(np.moveaxis(frame, 2, 0))
            
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
                action, logp, entropy, value_ext, value_int = model.agent.get_action_and_value(obs_tensor)
                action_np = action[0].cpu().numpy().tolist()  # e.g. [4, 0, 1]
            
            # Take action in environment
            obs, reward, done, _ = test_env.step(action_np)  
            total_reward += reward
            steps += 1

            # Calculate novelty
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                novelty = model.calculate_novelty(obs_tensor)
                novelty_values.append(novelty.item())
            
            # Track binary flag changes
            current_flags = extract_binary_flags(obs, env_unwrapped)
            if prev_flags is not None:
                differences = (current_flags != prev_flags).astype(int)
                activity = [a + d for a, d in zip(activity, differences)]
            prev_flags = current_flags

            # Log step information to wandb
            wandb.log({
                "step_reward": reward,
                "step_novelty": novelty.item(),
                "step_value": value_int.item(),
                "episode": episode,
                "step": steps
            })
            
            # Print step info
            try:
                action_name = test_env.actions(action.item()).name
            except Exception:
                action_name = str(action_np)
            print(f"\nStep {steps}/{max_steps_per_episode}")
            print(f"Action Taken: {action_name} | Reward: {reward:.2f}")
            print(f"Novelty Score: {novelty.item():.4f} | Value: {value_int.item():.4f}")
        
        # Track episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_novelties.append(np.mean(novelty_values))
        
        # Log episode information to wandb
        wandb.log({
            "episode_total_reward": total_reward,
            "episode_length": steps,
            "episode_mean_novelty": np.mean(novelty_values),
            "activity": activity,
            "episode": episode
        })

        # Save episode as GIF if frames were captured
        if frames and episode==4:
            gif_path = f"{gif_dir}/episode_{episode + 1}.gif"
            write_gif(np.array(frames), gif_path, fps=1)
            wandb.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})
        
        # Log activity per binary flag
        activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
        activity_data = []
        for idx, count in enumerate(activity):
            mapping = flag_mapping[idx]
            activity_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"], count)
            activity_data.append({
                "flag_id": idx,
                "object_type": mapping["object_type"],
                "object_index": mapping["object_index"],
                "state_name": mapping["state_name"],
                "activity_count": count
            })

        activity_df = pd.DataFrame(activity_data)
        csv_path = f"{csv_dir}/episode_{episode + 1}_activity.csv"
        activity_df.to_csv(csv_path, index=False)
        wandb.log({f"episode_{episode + 1}_activity": activity_table})
        
        # Print episode summary
        print(f"\n=== Episode {episode + 1} Summary ===")
        print(f"Total Reward: {total_reward:.2f} | Steps: {steps} | Mean Novelty: {np.mean(novelty_values):.4f}")
        print("\nActivity per binary flag (changes detected):")
        for idx, count in enumerate(activity):
            mapping = flag_mapping[idx]
            print(f"- Flag {idx} ({mapping['object_type']} #{mapping['object_index']} - {mapping['state_name']}): {count}")

    # Log overall testing results
    wandb.log({
        "mean_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_episode_novelty": np.mean(episode_novelties)
    })
    
    print("\n=== Overall Testing Results ===")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"Mean Novelty: {np.mean(episode_novelties):.4f}")

    test_env.close()
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Test RND_PPO models from multi-seed training")
    parser.add_argument("--task", type=str, default="MultiToy", help="Task name")
    parser.add_argument("--room_size", type=int, default=8, help="Room size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step", type=int, default=None, help="Specific step to test (uses latest if not specified)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to test (tests all seeds if not specified)")
    parser.add_argument("--base_dir", type=str, default="models/RND_PPO_grid_search", help="Base directory for saved models")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=200, help="Max steps per test episode")
    parser.add_argument("--ent_coef", type=float, default=None, help="Specific entropy coefficient to test (tests all if not specified)")
    args = parser.parse_args()

    # Register environments
    env_name, test_env_name, env_kwargs, test_env_kwargs = setup_env_registration(
        args.task, args.room_size, args.max_steps
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all seed directories if specific seed not provided
    if args.seed is not None:
        seed_dirs = [d for d in os.listdir(args.base_dir) 
                    if os.path.isdir(os.path.join(args.base_dir, d)) and 
                    f"seed{args.seed}_" in d]
    else:
        seed_dirs = [d for d in os.listdir(args.base_dir) 
                    if os.path.isdir(os.path.join(args.base_dir, d)) and 
                    "seed" in d]
    
    if not seed_dirs:
        print(f"No seed directories found in {args.base_dir}")
        return
    
    print(f"Found {len(seed_dirs)} seed directories to test")
    
    for seed_dir in seed_dirs:
        seed = int(seed_dir.split("seed")[1].split("_")[0])
        ent_coef = extract_entropy_coef_from_dirname(seed_dir)
        model_dir = os.path.join(args.base_dir, seed_dir)
        
        model_path = find_model_checkpoint(model_dir, args.step)
        if not model_path:
            print(f"No model checkpoint found in {model_dir}")
            continue
        
        test_env = make_single_env(
            env_id=test_env_name,
            seed=1,
            env_kwargs=test_env_kwargs
        )
        
        model = RND_PPO(
            env=test_env,
            env_id=test_env_name,
            device=device,
            seed=seed,
            ent_coef=ent_coef
        )
        
        #model.load(model_path)
        print(f"Loaded model from {model_path}")
        
        if "step_" in model_path:
            step = int(model_path.split("step_")[1].split(".")[0])
        else:
            step = "final"
        
        test_agent(
            env_id=test_env_name,
            model=model,
            device=device,
            task=args.task,
            room_size=args.room_size,
            step=step,
            model_seed=seed,
            ent_coef=ent_coef,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps_per_episode
        )


if __name__ == "__main__":
    main()