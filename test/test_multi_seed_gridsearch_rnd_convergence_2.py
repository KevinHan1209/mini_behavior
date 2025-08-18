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

def save_action_probabilities(action_probs_data, checkpoint_step, model_seed, base_output_dir):
    """Save action probability data for later box plot analysis"""
    import pickle
    import os
    
    # Create directory for action probability data  
    action_prob_dir = os.path.join(base_output_dir, f"action_probabilities_seed{model_seed}")
    os.makedirs(action_prob_dir, exist_ok=True)
    
    # Save the data
    filename = f"action_probs_step_{checkpoint_step}.pkl"
    filepath = os.path.join(action_prob_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(action_probs_data, f)
    
    print(f"Saved action probabilities to {filepath}")
    print(f"  Total episodes: {len(action_probs_data.get('episodes_data', []))}")
    
    # Debug: Check if episodes have action_probs data
    episodes_data = action_probs_data.get('episodes_data', [])
    for i, episode in enumerate(episodes_data[:2]):  # Check first 2 episodes
        action_probs = episode.get('action_probs', [])
        print(f"  Episode {i}: {len(action_probs)} action probability entries")
    
    return filepath


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
        
def extract_rnd_params_from_dirname(dirname):
    """Extract RND update frequency and weight decay from directory name
    
    Expected format: MultiToy_8x8_freq1_decay0_0001_seed5_1754942590
    """
    rnd_update_freq = 1  # default
    rnd_weight_decay = 0.0  # default
    
    # Extract update frequency
    freq_match = re.search(r'freq(\d+)', dirname)
    if freq_match:
        rnd_update_freq = int(freq_match.group(1))
    
    # Extract weight decay - handle the decimal format with underscores
    decay_match = re.search(r'decay(\d+_\d+)', dirname)
    if decay_match:
        decay_str = decay_match.group(1)
        
        if decay_str == "0_0":
            rnd_weight_decay = 0.0
        else:
            try:
                # Convert "0_0001" to "0.0001"
                clean_str = decay_str.replace("_", ".", 1)  # Only replace first underscore
                rnd_weight_decay = float(clean_str)
            except ValueError:
                print(f"Warning: Could not parse weight decay from '{decay_str}', using default 0.0")
                rnd_weight_decay = 0.0
    
    return rnd_update_freq, rnd_weight_decay

def make_single_env(env_id, seed, env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    env = CustomObservationWrapper(env)
    env.seed(seed)
    return env


def setup_env_registration(task, room_size, max_steps):
    """Register the training and test environments if not already registered"""
    env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-LP-v4"
    test_env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-LP-v5"
    env_kwargs = {"room_size": room_size, "max_steps": max_steps,
                   "extrinsic_rewards": {"noise": 0.1, "interaction": 0.1, "location_change": 0.1}}
    test_env_kwargs = {"room_size": room_size, "max_steps": max_steps, "test_env": True,
                   "extrinsic_rewards": {"noise": 0.1, "interaction": 0.1, "location_change": 0.1}}
    
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


def test_agent(env_id, model, device, task, room_size, step, model_seed, rnd_update_freq, rnd_weight_decay,
               agent_dist, num_episodes=5, max_steps_per_episode=1000, 
               convergence_window=10, convergence_threshold=0.001):
    print(f"\n=== Testing Agent (Seed {model_seed}): {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="rnd-ppo_w_external",
           name=f"RND_PPO_{task}_{room_size}x{room_size}_freq{rnd_update_freq}_decay{str(rnd_weight_decay).replace('.', '_')}_seed{model_seed}_step{step}",
               config={"env_id": env_id,
                       "mode": "testing",
                       "model_seed": model_seed,
                       "rnd_update_freq": rnd_update_freq,
                       "rnd_weight_decay": rnd_weight_decay,
                       "step": step,
                       "num_episodes": num_episodes,
                       "max_steps": max_steps_per_episode,
                       "convergence_window": convergence_window,
                       "convergence_threshold": convergence_threshold})

    test_env_kwargs = {"room_size": room_size, "max_steps": max_steps_per_episode, "test_env": True,
                   "extrinsic_rewards": {"noise": 0.1, "interaction": 0.1, "location_change": 0.1}}
    test_env = make_single_env(env_id, seed=42, env_kwargs=test_env_kwargs)

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
    
    # Output directory for results
    freq_str = f"freq{rnd_update_freq}"
    decay_str = f"decay{str(rnd_weight_decay).replace('.', '_')}"
    output_dir = f"results_variations/rnd_{room_size}x{room_size}_{freq_str}_{decay_str}_seed{model_seed}_step{step}"
    gif_dir = f"{output_dir}/gifs"
    csv_dir = f"{output_dir}/csvs"
    action_prob_dir = f"{output_dir}/action_probabilities"  # NEW: Add action prob directory
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(action_prob_dir, exist_ok=True)  # NEW: Create action prob directory
    
    episode_rewards = []
    episode_lengths = []
    episode_novelties = []
    
    # NEW: Track action probabilities across all episodes
    all_action_probs = {
        'checkpoint_step': step,
        'model_seed': model_seed,
        'action_dimensions': len(model.agent.action_dims),
        'action_dims_sizes': model.agent.action_dims,
        'episodes_data': []
    }
    
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
        
        # NEW: Track action probabilities for this episode
        episode_action_probs = []
        
        # Initialize convergence tracking
        js_history = deque(maxlen=convergence_window * 3)
        kl_history = deque(maxlen=convergence_window * 3)
        
        while not done and steps < max_steps_per_episode:
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
                action, logp, entropy, value_ext, value_int = model.agent.get_action_and_value(obs_tensor)
                action_np = action[0].cpu().numpy().tolist()
                
                # NEW: Extract action probabilities - FIXED VERSION
                hidden = model.agent.network(obs_tensor)
                actor_h = model.agent.actor_hidden(hidden)
                logits = model.agent.actor_logits(actor_h)
                
                # Split logits and get probabilities for each action dimension
                logits_split = torch.split(logits, tuple(model.agent.action_dims), dim=-1)
                step_action_probs = []
                for dim_idx, logit in enumerate(logits_split):
                    probs = torch.softmax(logit, dim=-1)
                    step_action_probs.append(probs[0].cpu().numpy().tolist())
                
                # Store the action probabilities for this step
                episode_action_probs.append({
                    'step': steps,
                    'action_taken': action_np,
                    'action_probs': step_action_probs  # List of probability arrays for each action dimension
                })
            
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
        
        # Store episode action probabilities - FIXED: Store in the right place
        all_action_probs['episodes_data'].append({
            'episode': episode,
            'episode_length': steps,
            'total_reward': total_reward,
            'action_probs': episode_action_probs  # This is the key fix - store the actual data
        })
        
        # DEBUG: Print action prob collection status
        print(f"Episode {episode}: collected {len(episode_action_probs)} action probability entries")
        if episode_action_probs:
            first_entry = episode_action_probs[0]
            print(f"  Sample entry: {len(first_entry['action_probs'])} dimensions, "
                  f"sizes: {[len(dim) for dim in first_entry['action_probs']]}")
        
        # Track episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_novelties.append(np.mean(novelty_values))
        
        # ... rest of existing convergence and logging code ...
        
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

    # NEW: Save action probabilities for this checkpoint - FIXED: Save in checkpoint directory
    action_prob_file = os.path.join(action_prob_dir, f"action_probs_step_{step}.pkl")
    
    # DEBUG: Final check before saving
    print(f"\nFinal action prob data check:")
    print(f"Total episodes: {len(all_action_probs['episodes_data'])}")
    for i, episode in enumerate(all_action_probs['episodes_data']):
        print(f"Episode {i}: {len(episode['action_probs'])} action prob entries")
    
    with open(action_prob_file, 'wb') as f:
        pickle.dump(all_action_probs, f)
    print(f"Saved action probabilities to: {action_prob_file}")
    
    # Log summary action probability statistics to wandb
    total_steps_all_episodes = sum(len(ep['action_probs']) for ep in all_action_probs['episodes_data'])
    wandb.log({
        "total_action_probability_samples": total_steps_all_episodes,
        "action_prob_file": action_prob_file
    })

def load_all_action_probabilities(base_dir, seed=None):
    """Load all action probability files for analysis"""
    import glob
    import pickle
    import os
    
    if seed is not None:
        pattern = os.path.join(base_dir, f"action_probabilities_seed{seed}", "action_probs_step_*.pkl")
    else:
        pattern = os.path.join(base_dir, "action_probabilities_seed*", "action_probs_step_*.pkl")
    
    files = glob.glob(pattern)
    
    all_data = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)
    
    # Sort by checkpoint step
    all_data.sort(key=lambda x: x['checkpoint_step'] if x['checkpoint_step'] != 'final' else float('inf'))
    
    return all_data


def create_action_probability_boxplots(action_prob_data, save_path=None):
    """Create box plots showing action probability evolution across checkpoints"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Prepare data for plotting
    plot_data = []
    
    for checkpoint_data in action_prob_data:
        checkpoint_step = checkpoint_data['checkpoint_step']
        
        for episode_data in checkpoint_data['episodes_data']:
            for step_data in episode_data['action_probs']:
                action_probs = step_data['action_probs']
                
                # For each action dimension
                for dim_idx, dim_probs in enumerate(action_probs):
                    # For each action in this dimension
                    for action_idx, prob in enumerate(dim_probs):
                        plot_data.append({
                            'checkpoint_step': checkpoint_step,
                            'action_dimension': dim_idx,
                            'action_index': action_idx,
                            'probability': prob,
                            'action_label': f"Dim{dim_idx}_Action{action_idx}"
                        })
    
    df = pd.DataFrame(plot_data)
    
    # Create separate plots for each action dimension
    action_dims = df['action_dimension'].unique()
    
    fig, axes = plt.subplots(len(action_dims), 1, figsize=(15, 5*len(action_dims)))
    if len(action_dims) == 1:
        axes = [axes]
    
    for dim_idx, ax in enumerate(axes):
        dim_data = df[df['action_dimension'] == dim_idx]
        
        # Create box plot
        sns.boxplot(data=dim_data, x='checkpoint_step', y='probability', 
                   hue='action_label', ax=ax)
        
        ax.set_title(f'Action Dimension {dim_idx} - Probability Distribution Evolution')
        ax.set_xlabel('Checkpoint Step')
        ax.set_ylabel('Action Probability')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels if there are many checkpoints
        if len(dim_data['checkpoint_step'].unique()) > 8:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved box plot to {save_path}")
    
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Test RND_PPO models from multi-seed training with convergence")
    parser.add_argument("--task", type=str, default="MultiToy", help="Task name")
    parser.add_argument("--room_size", type=int, default=8, help="Room size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--step", type=int, default=None, help="Specific step to test (uses latest if not specified)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to test (tests all seeds if not specified)")
    parser.add_argument("--base_dir", type=str, default="models/RND_PPO_variations", help="Base directory for saved models")
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
        #ent_coef = extract_entropy_coef_from_dirname(seed_dir)
        rnd_update_freq, rnd_weight_decay = extract_rnd_params_from_dirname(seed_dir)
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
                rnd_update_freq=rnd_update_freq,  # Add this
                rnd_weight_decay=rnd_weight_decay,
                ent_coef=0.01
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
                    #ent_coef=ent_coef,
                    rnd_update_freq=rnd_update_freq,  # Add this
                    rnd_weight_decay=rnd_weight_decay,
                    agent_dist=agent_dist,
                    num_episodes=args.num_episodes,
                    max_steps_per_episode=args.max_steps_per_episode,
                    convergence_window=args.convergence_window,
                    convergence_threshold=args.convergence_threshold
            )


if __name__ == "__main__":
    main()