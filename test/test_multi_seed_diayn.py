import os
import numpy as np
import torch
import wandb
from array2gif import write_gif
import gym
import random
import argparse

from algorithms.DIAYN_PPO_3 import DIAYN, DIAYNObservationWrapper
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
    STEP,
    fixed_seed=None,
    n_skills=8,
    target_dim=792,
    num_episodes=5,
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
        name=f"DIAYN_Run15_new_env_{STEP}",
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
                        #action, logprob, entropy, value = model.agent.get_action_and_value(obs_tensor)
                        action, logprob, entropy, value_ext, value_int = (model.agent.get_action_and_value(obs_tensor))
                        value = value_int 
                    except Exception as e:
                        print(f"Error during forward pass: {e}")
                        print(f"Input tensor shape: {obs_tensor.shape}")
                        break

                action_np = action[0].cpu().numpy().tolist()
                obs_dict, reward, done, info = env.step(action_np)

                ep_reward += reward
                step_count += 1
                
                # Track binary flags using a most defensive approach
                try:
                    if hasattr(env_unwrapped, 'objs') and hasattr(env_unwrapped.objs, 'values'):
                        current_flags = []
                        for obj_type_name, obj_list in env_unwrapped.objs.items():
                            for obj_index, obj in enumerate(obj_list):
                                for state_name, state in obj.states.items():
                                    if not isinstance(state, RelativeObjectState):
                                        try:
                                            # Skip problematic states that we know cause errors
                                            if state_name == "infovofrobot" or state_name.startswith("info"):
                                                # Skip this state since it's causing errors
                                                current_flags.append(0)
                                                continue
                                                
                                            # Try to get value very carefully
                                            if hasattr(state, 'value'):
                                                # Direct value attribute
                                                current_flags.append(state.value)
                                            elif hasattr(state, 'cur_value'):
                                                # Maybe it's called cur_value
                                                current_flags.append(state.cur_value)
                                            elif hasattr(state, 'get_value'):
                                                # Only try get_value if other methods don't work
                                                try:
                                                    # Try without arguments first
                                                    val = state.get_value()
                                                except Exception:
                                                    try:
                                                        # Then with dummy list
                                                        val = state.get_value([])
                                                    except Exception:
                                                        # Default to 0
                                                        val = 0
                                                current_flags.append(val)
                                            else:
                                                # If nothing works, just use 0
                                                current_flags.append(0)
                                        except Exception as sub_e:
                                            print(f"Skipping state {state_name} for {obj_type_name}[{obj_index}]: {sub_e}")
                                            current_flags.append(0)
                                            
                        current_flags = np.array(current_flags)
                        if prev_flags is not None and len(prev_flags) == len(current_flags):
                            diff = (current_flags != prev_flags).astype(int)
                            episode_activity += diff
                        else:
                            # Initialize prev_flags if length doesn't match
                            prev_flags = current_flags
                    else:
                        # If environment doesn't have the expected object structure
                        print("Environment structure doesn't match expectations; using placeholder for tracking")
                        if prev_flags is None:
                            prev_flags = np.zeros(num_binary_flags)
                except Exception as e:
                    print(f"Skipping binary flag tracking for this step: {e}")
                    # Make sure prev_flags exists even if we have errors
                    if prev_flags is None:
                        prev_flags = np.zeros(num_binary_flags)
    
                # Log step info
                wandb.log({
                    "skill_id": skill_id,
                    "episode": episode,
                    "step": step_count,
                    "step_reward": reward,
                    "step_value": value.mean().item()
                })
    
            # Save GIF
            gif_path = f"img/diayn_8x8_new_env/{STEP}/diayn_skill_{skill_id}_ep_{episode+1}.gif"
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            if frames and episode==4:
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

def test_multi_seed_diayn(
    base_dir,
    env_id,
    task,
    room_size,
    fixed_seed,
    num_episodes,
    max_steps_per_episode,
    target_dim,
    default_n_skills=16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and "seed" in d
    ]

    if not seed_dirs:
        print(f"No seed directories found in {base_dir}")
        return

    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.split("seed")[1].split("_")[0])
        except:
            print(f"Could not extract seed from {seed_dir}, skipping.")
            continue

        model_dir = os.path.join(base_dir, seed_dir)
        model_path = os.path.join(model_dir, "final_model.pt")

        if not os.path.exists(model_path):
            checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_step_")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
                model_path = os.path.join(model_dir, checkpoints[0])
                print(f"[Seed {seed}] Using latest checkpoint: {model_path}")
            else:
                print(f"[Seed {seed}] No valid model found.")
                continue
        else:
            print(f"[Seed {seed}] Using final model: {model_path}")

        # Infer n_skills from directory or default
        n_skills = default_n_skills

        model = DIAYN(
            env_id=env_id,
            device=device,
            n_skills=n_skills,
            seed=seed
        )
        model.load(model_path)

        # Use step from filename if available
        if "step_" in model_path:
            STEP = int(model_path.split("step_")[1].split(".")[0])
        else:
            STEP = "final"

        print(f"\n=== Testing model from seed {seed} ===")
        test_all_skills(
            env_id=env_id,
            model=model,
            device=device,
            STEP=STEP,
            fixed_seed=fixed_seed,
            n_skills=n_skills,
            target_dim=target_dim,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            project_name="diayn-ppo-test"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MultiToy")
    parser.add_argument("--room_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--base_dir", type=str, default="models/DIAYN_multi_seed")
    parser.add_argument("--target_dim", type=int, default=2564)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps_per_episode", type=int, default=200)
    parser.add_argument("--n_skills", type=int, default=16, help="Number of skills")
    args = parser.parse_args()

    env_id = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-LP-v2"

    register(
        id=env_id,
        entry_point=f'mini_behavior.envs:{args.task}Env',
        kwargs={"room_size": args.room_size, "max_steps": args.max_steps}
    )

    test_multi_seed_diayn(
        base_dir=args.base_dir,
        env_id=env_id,
        task=args.task,
        room_size=args.room_size,
        fixed_seed=args.fixed_seed,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        target_dim=args.target_dim,
        default_n_skills=args.n_skills
    )


def find_model_checkpoint(model_dir, step=None):
    if step is not None:
        model_path = os.path.join(model_dir, f"model_step_{step}.pt")
        if os.path.exists(model_path):
            return model_path

    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_step_")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
        return os.path.join(model_dir, checkpoints[0])

    final_model_path = os.path.join(model_dir, "final_model.pt")
    return final_model_path if os.path.exists(final_model_path) else None


def setup_env_registration(task, room_size, max_steps):
    env_id = f"MiniGrid-{task}-{room_size}x{room_size}-N2-LP-v2"
    register(
        id=env_id,
        entry_point=f"mini_behavior.envs:{task}Env",
        kwargs={"room_size": room_size, "max_steps": max_steps}
    )
    return env_id


def main():
    parser = argparse.ArgumentParser(description="Multi-seed DIAYN evaluation")
    parser.add_argument("--task", type=str, default="MultiToy")
    parser.add_argument("--room_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to test")
    parser.add_argument("--base_dir", type=str, default="models/DIAYN_multi_seed")
    parser.add_argument("--n_skills", type=int, default=16)
    parser.add_argument("--target_dim", type=int, default=2564)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps_per_episode", type=int, default=200)
    args = parser.parse_args()

    env_id = setup_env_registration(args.task, args.room_size, args.max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dirs = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d)) and "seed" in d]
    if not seed_dirs:
        print(f"No seed directories found in {args.base_dir}")
        return

    for seed_dir in seed_dirs:
        seed_path = os.path.join(args.base_dir, seed_dir)
        seed = int(seed_dir.split("seed")[1].split("_")[0])
        model_path = find_model_checkpoint(seed_path, args.step)
        if not model_path:
            print(f"No checkpoint found in {seed_path}")
            continue

        if "step_" in model_path:
            step = int(model_path.split("step_")[1].split(".")[0])
        else:
            step = "final"

        model = DIAYN(env_id=env_id, device=device, n_skills=args.n_skills, seed=seed)
        model.load(model_path)
        print(f"Loaded model from {model_path}")

        test_all_skills(
            env_id=env_id,
            model=model,
            device=device,
            STEP=step,
            fixed_seed=1,  # fixed layout
            n_skills=args.n_skills,
            target_dim=args.target_dim,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            project_name="diayn-ppo-test"
        )


if __name__ == "__main__":
    main()