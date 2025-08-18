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
import pickle
from scipy.stats import entropy
import re
from collections import deque


# Object name mapping (from the analysis script)
obj_name_mapping = {
    'a': 'alligator busy box',
    'b': 'broom',
    'bs': 'broom set', 
    'bu': 'bucket',
    'ca': 'cart',
    'cu': 'cube',
    'f': 'farm toy',
    'g': 'gear',
    'm': 'music toy',
    'p': 'piggie bank',
    'pb': 'pink beach ball',
    'r': 'rattle',
    'rb': 'red spiky ball',
    's': 'stroller',
    'sh': 'shape sorter',
    't': 'tree busy box',
    'c': 'winnie cabinet',
    'y': 'yellow donut ring'
}


def transform_object_name(obj_name):
    """Transform object name by replacing underscores with spaces and removing trailing numbers"""
    # Special case for beach ball
    if obj_name.startswith('beach_ball'):
        return 'pink beach ball'
    
    # Replace underscores with spaces
    obj_name = obj_name.replace('_', ' ')
    # Remove trailing numbers and any remaining underscores
    obj_name = re.sub(r'_\d+$', '', obj_name)
    return obj_name


def load_agent_distributions(pkl_path="post_processing/averaged_state_distributions.pkl"):
    """Load the averaged distributions from the pickle file and convert to flat distribution"""
    with open(pkl_path, 'rb') as f:
        agent_dists = pickle.load(f)
    
    # Convert to flat distribution
    flat_dist = {}
    for obj, states in agent_dists.items():
        # Transform object name using mapping
        if obj in obj_name_mapping:
            obj = obj_name_mapping[obj]
        for state, percentages in states.items():
            # Only keep the false percentage
            if isinstance(percentages, dict):
                false_percentage = percentages.get('false_percentage', 100)
            else:
                false_percentage = percentages
            flat_dist[(obj, state)] = false_percentage
    
    return flat_dist


def calculate_js_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions"""
    # Convert to numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Calculate the average distribution
    m = 0.5 * (p + q)
    
    # Calculate JS divergence
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    
    return js_div


def calculate_kl_divergence(p, q):
    """Calculate KL divergence from p to q"""
    # Convert to numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Calculate KL divergence
    kl_div = entropy(p, q)
    
    return kl_div


def calculate_current_distribution(activity_counts, total_opportunities, env_unwrapped):
    """Calculate current distribution from activity counts"""
    flat_dist = {}
    
    # Define object pairs to average
    object_pairs = {
        'coin': 'piggie bank',
        'gear_toy': 'gear',
        'shape_toy': 'shape sorter'
    }
    
    # Get flag mapping
    flag_mapping = generate_flag_mapping(env_unwrapped)
    
    # Group activity counts by object type and state
    object_activities = {}
    for idx, count in enumerate(activity_counts):
        if idx >= len(flag_mapping):
            continue
        mapping = flag_mapping[idx]
        obj_type = mapping['object_type']
        obj_idx = mapping['object_index']
        state_name = mapping['state_name']
        
        # Skip robot states
        if 'robot' in state_name.lower():
            continue
        
        key = (obj_type, obj_idx, state_name)
        object_activities[key] = count
    
    # Process each object type
    processed_objects = set()
    for (obj_type, obj_idx, state_name), count in object_activities.items():
        if obj_type in object_pairs:
            # For objects that need to be averaged
            final_obj_name = object_pairs[obj_type]
            state_key = (final_obj_name, state_name)
            
            if state_key not in processed_objects:
                # Sum all counts for this object type and state
                total_count = sum(c for (ot, oi, sn), c in object_activities.items() 
                                if ot == obj_type and sn == state_name)
                if total_count > 0:
                    false_percentage = 100 - (total_count / total_opportunities * 100)
                else:
                    false_percentage = 100
                flat_dist[state_key] = false_percentage
                processed_objects.add(state_key)
        else:
            # For individual objects
            obj_name = transform_object_name(f"{obj_type}_{obj_idx}")
            state_key = (obj_name, state_name)
            
            if total_opportunities > 0:
                false_percentage = 100 - (count / total_opportunities * 100)
            else:
                false_percentage = 100
            flat_dist[state_key] = false_percentage
    
    return flat_dist


def check_convergence(js_history, kl_history, window_size=10, threshold=0.001):
    """Check if JS and KL divergences have converged"""
    if len(js_history) < window_size * 2:
        return False
    
    # Calculate standard deviation of recent window
    recent_js = js_history[-window_size:]
    recent_kl = kl_history[-window_size:]
    
    js_std = np.std(recent_js)
    kl_std = np.std(recent_kl)
    
    # Check if both have low variance (converged)
    return js_std < threshold and kl_std < threshold


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


def find_all_model_checkpoints(model_dir):
    """Find all model checkpoints in the directory, sorted by step number"""
    checkpoints = []
    
    # Find all model_step_*.pt files
    for filename in os.listdir(model_dir):
        if filename.startswith("model_step_") and filename.endswith(".pt"):
            try:
                step = int(filename.split("_")[2].split(".")[0])
                model_path = os.path.join(model_dir, filename)
                checkpoints.append((step, model_path))
            except (ValueError, IndexError):
                continue
    
    # Add final model if it exists
    final_model_path = os.path.join(model_dir, "final_model.pt")
    if os.path.exists(final_model_path):
        checkpoints.append(("final", final_model_path))
    
    # Sort by step number (put "final" at the end)
    checkpoints.sort(key=lambda x: x[0] if isinstance(x[0], int) else float('inf'))
    
    return checkpoints


def test_agent(env_id, model, device, task, room_size, step, model_seed, ent_coef, 
               agent_dist, num_episodes=5, max_steps_per_episode=1000, 
               convergence_window=10, convergence_threshold=0.001):
    print(f"\n=== Testing Agent (Seed {model_seed}): {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="rnd-ppo-test-multiseed-convergence",
               name=f"RND_PPO_{task}_{room_size}x{room_size}_seed{model_seed}_ent{ent_coef}_step{step}",
               config={"env_id": env_id,
                       "mode": "testing",
                       "model_seed": model_seed,
                       "entropy_coef": ent_coef,
                       "step": step,
                       "num_episodes": num_episodes,
                       "max_steps": max_steps_per_episode,
                       "convergence_window": convergence_window,
                       "convergence_threshold": convergence_threshold})

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
    output_dir = f"results_5/rnd_{room_size}x{room_size}_seed{model_seed}_ent{ent_coef}_step{step}"
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
        
        # Initialize convergence tracking
        js_history = deque(maxlen=convergence_window * 3)  # Keep more history for stability
        kl_history = deque(maxlen=convergence_window * 3)
        
        while not done and steps < max_steps_per_episode:
            # Render the environment for visualization (only for last episode to save memory)
            #if episode == num_episodes - 1:
            #    frame = test_env.render()
            #    if frame is not None:
            #        frames.append(np.moveaxis(frame, 2, 0))
            
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

            # Calculate current distribution and divergences every 10 steps
            if steps % 10 == 0 and steps > 0:
                total_opportunities = steps
                current_dist = calculate_current_distribution(activity, total_opportunities, env_unwrapped)
                
                # Get common keys between agent and current distributions
                common_keys = set(agent_dist.keys()) & set(current_dist.keys())
                
                if len(common_keys) > 0:
                    # Calculate JS and KL divergences for each state
                    js_divergences = []
                    kl_divergences = []
                    
                    for key in common_keys:
                        # Create binary distributions for this state
                        agent_value = np.array([agent_dist[key], 100 - agent_dist[key]])
                        current_value = np.array([current_dist[key], 100 - current_dist[key]])
                        
                        # Normalize to probabilities
                        agent_value = agent_value / 100.0
                        current_value = current_value / 100.0
                        
                        # Calculate divergences
                        js_div = calculate_js_divergence(agent_value, current_value)
                        kl_div = calculate_kl_divergence(current_value, agent_value)
                        
                        js_divergences.append(js_div)
                        kl_divergences.append(kl_div)
                    
                    # Average divergences
                    avg_js = np.mean(js_divergences)
                    avg_kl = np.mean(kl_divergences)
                    
                    js_history.append(avg_js)
                    kl_history.append(avg_kl)
                    
                    # Log divergences to wandb
                    wandb.log({
                        "step_reward": reward,
                        "step_novelty": novelty.item(),
                        "step_value": value_int.item(),
                        "js_divergence": avg_js,
                        "kl_divergence": avg_kl,
                        "episode": episode,
                        "step": steps,
                        "num_common_states": len(common_keys)
                    })
                    
                    # Check for convergence
                    if check_convergence(list(js_history), list(kl_history), 
                                       convergence_window, convergence_threshold):
                        print(f"Convergence detected at step {steps}")
                        print(f"Final JS divergence: {avg_js:.6f}")
                        print(f"Final KL divergence: {avg_kl:.6f}")
                        wandb.log({
                            "convergence_step": steps,
                            "converged": True,
                            "final_js_divergence": avg_js,
                            "final_kl_divergence": avg_kl
                        })
                        break
                else:
                    # Log step information without divergences
                    wandb.log({
                        "step_reward": reward,
                        "step_novelty": novelty.item(),
                        "step_value": value_int.item(),
                        "episode": episode,
                        "step": steps
                    })
            else:
                # Log step information without divergences
                wandb.log({
                    "step_reward": reward,
                    "step_novelty": novelty.item(),
                    "step_value": value_int.item(),
                    "episode": episode,
                    "step": steps
                })
            
            # Print step info every 50 steps
            if steps % 50 == 0:
                try:
                    action_name = test_env.actions(action.item()).name
                except Exception:
                    action_name = str(action_np)
                print(f"Step {steps}/{max_steps_per_episode} | Action: {action_name} | Reward: {reward:.2f}")
                if len(js_history) > 0:
                    print(f"Current JS divergence: {js_history[-1]:.6f} | KL divergence: {kl_history[-1]:.6f}")
        
        # Track episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_novelties.append(np.mean(novelty_values))
        
        # Calculate final divergences for this episode
        total_opportunities = steps
        final_dist = calculate_current_distribution(activity, total_opportunities, env_unwrapped)
        common_keys = set(agent_dist.keys()) & set(final_dist.keys())
        
        final_js_divergences = []
        final_kl_divergences = []
        for key in common_keys:
            agent_value = np.array([agent_dist[key], 100 - agent_dist[key]]) / 100.0
            final_value = np.array([final_dist[key], 100 - final_dist[key]]) / 100.0
            
            js_div = calculate_js_divergence(agent_value, final_value)
            kl_div = calculate_kl_divergence(final_value, agent_value)
            
            final_js_divergences.append(js_div)
            final_kl_divergences.append(kl_div)
        
        final_avg_js = np.mean(final_js_divergences) if final_js_divergences else np.nan
        final_avg_kl = np.mean(final_kl_divergences) if final_kl_divergences else np.nan
        
        # Log episode information to wandb
        wandb.log({
            "episode_total_reward": total_reward,
            "episode_length": steps,
            "episode_mean_novelty": np.mean(novelty_values),
            "episode_final_js_divergence": final_avg_js,
            "episode_final_kl_divergence": final_avg_kl,
            "activity": activity,
            "episode": episode
        })

        # Save episode as GIF if frames were captured
        #if frames and episode == num_episodes - 1:
        #    gif_path = f"{gif_dir}/episode_{episode + 1}.gif"
        #    write_gif(np.array(frames), gif_path, fps=1)
        #    wandb.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})
        
        # Log activity per binary flag
        activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
        activity_data = []
        for idx, count in enumerate(activity):
            if idx < len(flag_mapping):
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
        print(f"Final JS Divergence: {final_avg_js:.6f} | Final KL Divergence: {final_avg_kl:.6f}")

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
    parser = argparse.ArgumentParser(description="Test RND_PPO models from multi-seed training with convergence")
    parser.add_argument("--task", type=str, default="MultiToy", help="Task name")
    parser.add_argument("--room_size", type=int, default=8, help="Room size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step", type=int, default=None, help="Specific step to test (uses latest if not specified)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to test (tests all seeds if not specified)")
    parser.add_argument("--base_dir", type=str, default="models/RND_PPO_grid_search_5", help="Base directory for saved models")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per test episode")
    parser.add_argument("--ent_coef", type=float, default=None, help="Specific entropy coefficient to test (tests all if not specified)")
    parser.add_argument("--pkl_path", type=str, default="post_processing/averaged_state_distributions.pkl", 
                       help="Path to agent distribution pickle file")
    parser.add_argument("--convergence_window", type=int, default=10, 
                       help="Window size for convergence detection")
    parser.add_argument("--convergence_threshold", type=float, default=0.001, 
                       help="Threshold for convergence detection")
    args = parser.parse_args()

    # Load agent distributions
    if not os.path.exists(args.pkl_path):
        print(f"Error: Agent distribution file '{args.pkl_path}' not found")
        return
    
    agent_dist = load_agent_distributions(args.pkl_path)
    print(f"Loaded agent distributions with {len(agent_dist)} object-state pairs")

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
        
        # Get all checkpoints for this seed
        checkpoints = find_all_model_checkpoints(model_dir)
        if not checkpoints:
            print(f"No model checkpoints found in {model_dir}")
            continue
        
        print(f"\nFound {len(checkpoints)} checkpoints for seed {seed}")
        
        # Test each checkpoint
        for step, model_path in checkpoints:
            print(f"\nTesting checkpoint: {model_path}")
            
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
            
            model.load(model_path)
        
            test_agent(
                    env_id=test_env_name,
                    model=model,
                    device=device,
                    task=args.task,
                    room_size=args.room_size,
                    step=step,
                    model_seed=seed,
                    ent_coef=ent_coef,
                    agent_dist=agent_dist,
                    num_episodes=args.num_episodes,
                    max_steps_per_episode=args.max_steps_per_episode,
                    convergence_window=args.convergence_window,
                    convergence_threshold=args.convergence_threshold
            )


if __name__ == "__main__":
    main()