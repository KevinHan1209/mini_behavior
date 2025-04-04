import os
import numpy as np
import torch
import wandb
from array2gif import write_gif
import gym
import random

from DIAYN_PPO import DIAYN, DIAYNObservationWrapper
from mini_behavior.utils.states_base import RelativeObjectState
from mini_behavior.register import register

def process_obs(obs_dict):
    """
    Process the observation exactly as during training.
    """
    if not isinstance(obs_dict, dict):
        # If the environment doesn't return a dict, just flatten
        return obs_dict.reshape(obs_dict.shape[0], -1)

    flattened_obs = []
    for key in sorted(obs_dict.keys()):
        if key == 'skill':
            continue  # skip skill
        val = obs_dict[key]
        if isinstance(val, np.ndarray):
            # Flatten the environment dimension
            flattened_obs.append(val.reshape(val.shape[0], -1))

    return np.concatenate(flattened_obs, axis=1)

def count_binary_flags(env):
    """
    Count the total number of non-relative (binary) state flags in the environment.
    """
    num_flags = 0
    for obj_list in env.objs.values():
        for obj in obj_list:
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    num_flags += 1
    return num_flags

def generate_flag_mapping(env):
    """
    Generate a mapping (list of dictionaries) that tells you which binary flag 
    corresponds to which object's non-relative state.
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

def get_current_flags_from_env(env):
    """
    Extract binary flags directly from the environment state,
    bypassing the observation.
    Properly passes the environment to the get_value method.
    """
    flags = []
    for obj_list in env.objs.values():
        for obj in obj_list:
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    # Pass the environment to get_value
                    flags.append(state.get_value(env))
    return np.array(flags)

class FixedLayoutWrapper(gym.Wrapper):
    """
    A wrapper that ensures the environment uses the same layout for all resets.
    """
    def __init__(self, env, fixed_seed=None):
        super().__init__(env)
        self.fixed_seed = fixed_seed if fixed_seed is not None else random.randint(0, 10000)
        self.initial_state = None
        
    def reset(self):
        # Set environment's random seed to ensure consistent layout
        self.env.seed(self.fixed_seed)
        obs = self.env.reset()
        
        # If this is the first reset, store the initial state 
        # (not used now but might be useful for future implementations)
        if self.initial_state is None:
            # Try to capture the environment's initial state
            try:
                # This is a simplified attempt - your environment might need 
                # specific code to properly capture and restore state
                self.initial_state = {
                    'agent_pos': self.env.agent_pos.copy() if hasattr(self.env, 'agent_pos') else None,
                    'agent_dir': getattr(self.env, 'agent_dir', None),
                }
                
                # Log info about the fixed layout
                print(f"Fixed environment layout with seed {self.fixed_seed}")
                if hasattr(self.env, 'objs'):
                    # Log positions of objects
                    for obj_type, obj_list in self.env.objs.items():
                        for i, obj in enumerate(obj_list):
                            if hasattr(obj, 'cur_pos'):
                                print(f"  {obj_type}[{i}] at position {obj.cur_pos}")
            except:
                print("Could not fully capture initial state, but seed is fixed.")
                
        return obs

def test_all_skills(
    env_id,
    model,
    device,
    fixed_seed=None,
    n_skills=16,
    target_dim=792,
    num_episodes=3,
    max_steps_per_episode=200,
    project_name="diayn-ppo-test"
):
    """
    Tests all skills using the same fixed environment layout.
    """
    # Create a fixed layout environment once
    fixed_layout_seed = fixed_seed if fixed_seed is not None else random.randint(0, 10000)
    print(f"Using fixed layout with seed: {fixed_layout_seed}")
    
    # Start a single wandb run for all skills
    wandb.init(
        project=project_name,
        name=f"DIAYN_Run7_seed_{fixed_layout_seed}",
        config={
            "env_id": env_id,
            "layout_seed": fixed_layout_seed,
            "n_skills": n_skills,
            "num_episodes": num_episodes,
            "max_steps": max_steps_per_episode
        }
    )
    
    # Create the base environment with fixed layout
    base_env = gym.make(env_id)
    base_env = FixedLayoutWrapper(base_env, fixed_seed=fixed_layout_seed)
    
    # Get access to the unwrapped environment
    env_unwrapped = base_env
    while hasattr(env_unwrapped, 'env'):
        env_unwrapped = env_unwrapped.env
        if hasattr(env_unwrapped, 'objs'):
            break
    
    # Count and map binary flags
    num_binary_flags = count_binary_flags(env_unwrapped)
    flag_mapping = generate_flag_mapping(env_unwrapped)
    
    print(f"Found {num_binary_flags} binary flags in environment")
    
    # Log the flag mapping
    mapping_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name"])
    for idx, fm in enumerate(flag_mapping):
        mapping_table.add_data(idx, fm["object_type"], fm["object_index"], fm["state_name"])
    wandb.log({"flag_mapping": mapping_table})
    
    # Set model to evaluation mode
    model.agent.eval()
    
    # Create a summary table for all skills
    skill_summary = []
    
    # Test each skill
    for skill_id in range(n_skills):
        print(f"\n===== Testing Skill {skill_id} =====")
        
        # Create a new DIAYN wrapper with this skill
        env = base_env
        env = DIAYNObservationWrapper(env, skill_z=skill_id)
        
        # Run episodes for this skill
        skill_activity = np.zeros(num_binary_flags)
        skill_total_reward = 0
        skill_total_steps = 0
        
        for episode in range(num_episodes):
            obs_dict = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0
            frames = []
            
            # For counting changes in binary flags
            episode_activity = np.zeros(num_binary_flags)
            prev_flags = None
            
            print(f"Episode {episode+1} started. Skill={skill_id}")
            
            while not done and step_count < max_steps_per_episode:
                # Render for GIF
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(np.moveaxis(frame, 2, 0))
    
                # Process observation
                processed = process_obs(obs_dict)
                if step_count == 0 and episode == 0:
                    print(f"Processed observation shape: {processed.shape}")
                    
                obs_tensor = torch.FloatTensor(processed).to(device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                # Adapt tensor dimensions if needed
                if obs_tensor.shape[1] < target_dim:
                    padding = torch.zeros(obs_tensor.shape[0], target_dim - obs_tensor.shape[1], device=device)
                    obs_tensor = torch.cat([obs_tensor, padding], dim=1)
    
                # Get action
                with torch.no_grad():
                    try:
                        action, logprob, entropy, value = model.agent.get_action_and_value(obs_tensor)
                    except Exception as e:
                        print(f"Error during forward pass: {e}")
                        print(f"Input tensor shape: {obs_tensor.shape}")
                        break
    
                action_item = action.cpu().numpy()[0] if action.ndim > 0 else action.item()
                obs_dict, reward, done, info = env.step(action_item)
    
                ep_reward += reward
                step_count += 1
                
                # Track binary flags using direct environment access
                try:
                    current_flags = get_current_flags_from_env(env_unwrapped)
                    if prev_flags is not None:
                        diff = (current_flags != prev_flags).astype(int)
                        episode_activity += diff
                    prev_flags = current_flags
                except Exception as e:
                    print(f"Error tracking binary flags: {e}")
    
                # Log step info
                wandb.log({
                    "skill_id": skill_id,
                    "episode": episode,
                    "step": step_count,
                    "step_reward": reward,
                    "step_value": value.mean().item()
                })
    
            # Save GIF
            gif_path = f"img/diayn_test/1.5M/diayn_skill_{skill_id}_ep_{episode+1}.gif"
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            if frames:
                write_gif(np.array(frames), gif_path, fps=5)
                wandb.log({f"skill_{skill_id}_ep_{episode+1}_replay": wandb.Video(gif_path, fps=5, format="gif")})
    
            wandb.log({
                "skill_id": skill_id,
                "episode": episode,
                "episode_reward": ep_reward,
                "episode_length": step_count
            })
            
            skill_activity += episode_activity
            skill_total_reward += ep_reward
            skill_total_steps += step_count
            
            activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
            for idx, count in enumerate(episode_activity):
                if idx < len(flag_mapping):
                    fm = flag_mapping[idx]
                    activity_table.add_data(idx, fm["object_type"], fm["object_index"], fm["state_name"], int(count))
            wandb.log({f"skill_{skill_id}_ep_{episode+1}_activity": activity_table})
    
            print(f"[Skill={skill_id}] Episode {episode+1}/{num_episodes} ended. Reward={ep_reward:.2f}, Steps={step_count}")
            
            print("\nEpisode toy interaction summary:")
            for idx, count in enumerate(episode_activity):
                if count > 0 and idx < len(flag_mapping):
                    fm = flag_mapping[idx]
                    print(f"  {fm['object_type']}[{fm['object_index']}].{fm['state_name']}: {int(count)} interactions")
        
        skill_avg_reward = skill_total_reward / num_episodes
        skill_avg_steps = skill_total_steps / num_episodes
        
        skill_summary.append({
            "skill_id": skill_id,
            "avg_reward": skill_avg_reward,
            "avg_steps": skill_avg_steps,
            "activity": skill_activity
        })
        
        skill_activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
        for idx, count in enumerate(skill_activity):
            if idx < len(flag_mapping):
                fm = flag_mapping[idx]
                skill_activity_table.add_data(idx, fm["object_type"], fm["object_index"], fm["state_name"], int(count))
        wandb.log({f"skill_{skill_id}_total_activity": skill_activity_table})
        
        # Print skill summary
        print(f"\nSkill {skill_id} Summary:")
        print(f"  Average Reward: {skill_avg_reward:.2f}")
        print(f"  Average Steps: {skill_avg_steps:.1f}")
        print("  Toy interactions:")
        for idx, count in enumerate(skill_activity):
            if count > 0 and idx < len(flag_mapping):
                fm = flag_mapping[idx]
                print(f"    {fm['object_type']}[{fm['object_index']}].{fm['state_name']}: {int(count)} interactions")
    
    comparison_table = wandb.Table(columns=["skill_id", "avg_reward", "avg_steps", "most_active_flags"])
    for summary in skill_summary:
        top_indices = np.argsort(summary["activity"])[-3:][::-1]
        top_flags = []
        for idx in top_indices:
            if idx < len(flag_mapping) and summary["activity"][idx] > 0:
                fm = flag_mapping[idx]
                flag_str = f"{fm['object_type']}[{fm['object_index']}].{fm['state_name']}: {int(summary['activity'][idx])}"
                top_flags.append(flag_str)
        
        comparison_table.add_data(
            summary["skill_id"],
            summary["avg_reward"],
            summary["avg_steps"],
            ", ".join(top_flags)
        )
    
    wandb.log({"skill_comparison": comparison_table})
    wandb.finish()

def main():
    TASK = 'MultiToy'
    ROOM_SIZE = 8
    N_SKILLS = 8
    CHECKPOINT_STEP = 1500000
    EPISODES_PER_SKILL = 5
    MAX_STEPS_PER_EPISODE = 500
    TARGET_DIM = 792  # set to expected input dimension 792 with 8 skills
    FIXED_SEED = 1

    env_id = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v2"

    register(
        id=env_id,
        entry_point='mini_behavior.envs:MultiToyEnv',
        kwargs={"room_size": ROOM_SIZE, "max_steps": 1000}
    )

    model_dir = f"models/DIAYN_{TASK}_Run9_lr_match"
    model_path = os.path.join(model_dir, f"model_step_{CHECKPOINT_STEP}.pt")
    if not os.path.exists(model_path):
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_step_")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
            model_path = os.path.join(model_dir, checkpoints[0])
            print(f"Using latest checkpoint: {model_path}")
        else:
            print("No model found. Please train a model first.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DIAYN(
        env_id=env_id,
        device=device,
        total_timesteps=0,
        num_envs=1,
        num_steps=1,
        n_skills=N_SKILLS,
        disc_coef=0.1,
        seed=1
    )
    model.load(model_path)
    print(f"Loaded DIAYN model from {model_path}")

    # Test all skills with the same fixed layout
    test_all_skills(
        env_id=env_id,
        model=model,
        device=device,
        fixed_seed=FIXED_SEED,
        n_skills=N_SKILLS,
        target_dim=TARGET_DIM,
        num_episodes=EPISODES_PER_SKILL,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        project_name="diayn-ppo-test"
    )

if __name__ == "__main__":
    main()