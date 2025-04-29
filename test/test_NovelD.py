import gym
from algorithms.NovelD_PPO import NovelD_PPO
import numpy as np
import torch
import wandb
from array2gif import write_gif
from env_wrapper import CustomObservationWrapper
from mini_behavior.utils.states_base import RelativeObjectState
from mini_behavior.register import register
import os


def count_binary_flags(env):
    """
    Count the total number of non-relative (binary) state flags in the environment.
    For each object, we skip the two position values and then count each state 
    that is not an instance of RelativeObjectState and not in default_states.
    """
    default_states = [
        'atsamelocation',
        'infovofrobot',
        'inleftreachofrobot',
        'inrightreachofrobot',
        'inside',
        'nextto',
    ]
    
    num_flags = 0
    for obj_list in env.objs.values():
        for obj in obj_list:
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState) and state_name not in default_states:
                    num_flags += 1
    return num_flags


def extract_binary_flags(obs, env):
    """
    Given an observation vector and the environment, extract only the binary flags.
    
    The observation is built as follows:
      - First three values: agent_x, agent_y, agent_dir.
      - For each object:
          - Two position values: pos_x, pos_y.
          - Then each non-relative state is appended as a binary flag.
    
    This function skips the agent state and object positions, returning only the flags.
    """
    # Same default states list used in CustomObservationWrapper.gen_obs
    default_states = [
        'atsamelocation',
        'infovofrobot',
        'inleftreachofrobot',
        'inrightreachofrobot',
        'inside',
        'nextto',
    ]
    
    flags = []
    index = 3  # skip agent state (x, y, direction)
    
    for obj_list in env.objs.values():
        for obj in obj_list:
            index += 2  # skip the object's position (2 values)
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    # Only process states that aren't in the default_states list
                    # This matches the behavior in gen_obs
                    if state_name not in default_states:
                        flags.append(obs[index])
                        index += 1
                    # Don't increment index for skipped states
    
    return np.array(flags)


def generate_flag_mapping(env):
    """
    Generate a mapping (list of dictionaries) that tells you which binary flag 
    corresponds to which object's non-relative state.
    """
    default_states = [
        'atsamelocation',
        'infovofrobot',
        'inleftreachofrobot',
        'inrightreachofrobot',
        'inside',
        'nextto',
    ]
    
    mapping = []
    for obj_type_name, obj_list in env.objs.items():
        for obj_index, obj in enumerate(obj_list):
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState) and state_name not in default_states:
                    mapping.append({
                        "object_type": obj_type_name,
                        "object_index": obj_index,
                        "state_name": state_name
                    })
    return mapping


def train_agent(env_id, device):
    print("\n=== Starting Agent Training ===")
    try:
        noveld_ppo = NovelD_PPO(env_id, device)
        noveld_ppo.train()
        return noveld_ppo
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


def test_agent(env_id, noveld_ppo, device, num_episodes=5, max_steps_per_episode=200, use_wandb=True):
    print(f"🧪 Testing Agent | {env_id} | Episodes: {num_episodes}")
    
    # Initialize wandb for testing if enabled
    if use_wandb:
        wandb.init(
            project="noveld-ppo-test",
            config={
                "env_id": env_id,
                "mode": "testing",
                "num_episodes": num_episodes,
                "max_steps": max_steps_per_episode
            }
        )

    test_env = gym.make(env_id)
    test_env = CustomObservationWrapper(test_env)
    
    # Get the underlying environment for accessing object info
    env_unwrapped = getattr(test_env, 'env', test_env)
    num_binary_flags = count_binary_flags(env_unwrapped)
    flag_mapping = generate_flag_mapping(env_unwrapped)

    # Log flag mapping once if using wandb
    if use_wandb:
        mapping_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name"])
        for idx, mapping in enumerate(flag_mapping):
            mapping_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"])
        wandb.log({"flag_mapping": mapping_table}, commit=False)

    noveld_ppo.agent.network.to(device)
    
    for episode in range(num_episodes):
        print(f"\n🔄 Episode {episode + 1}/{num_episodes}")
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        novelty_values = []
        frames = []
        # Track state changes
        activity = [0] * num_binary_flags
        prev_flags = None
        
        while not done and steps < max_steps_per_episode:
            # Collect frames if rendering is available
            frame = test_env.render()
            if frame is not None:
                frames.append(np.moveaxis(frame, 2, 0))
            
            # Get action from agent
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, ext_value, int_value = noveld_ppo.agent.get_action_and_value(obs_tensor)
                action_np = action[0].cpu().numpy().tolist()
        
            # Take action in environment
            obs, reward, done, _ = test_env.step(action_np)
            total_reward += reward
            steps += 1
            
            # Calculate novelty and track values
            novelty = noveld_ppo.calculate_novelty(torch.FloatTensor(obs).unsqueeze(0).to(device))
            novelty_values.append(novelty)
            
            # Convert tensor values to Python scalars
            novelty_val = novelty.item() if torch.is_tensor(novelty) else novelty
            ext_val = ext_value.item() if torch.is_tensor(ext_value) else ext_value
            int_val = int_value.item() if torch.is_tensor(int_value) else int_value

            # Track state changes
            current_flags = extract_binary_flags(obs, env_unwrapped)
            if prev_flags is not None:
                differences = (current_flags != prev_flags).astype(int)
                activity = [a + d for a, d in zip(activity, differences)]
            prev_flags = current_flags

            # Log step data to wandb if enabled
            if use_wandb:
                wandb.log({
                    "test/step": steps,
                    "test/step_reward": reward,
                    "test/step_novelty": novelty_val,
                    "test/step_ext_value": ext_val,
                    "test/step_int_value": int_val,
                    "test/episode": episode + 1
                }, commit=False)
            
            # Print concise step info (only every 10 steps to avoid clutter)
            if steps % 10 == 0 or steps == 1:
                try:
                    action_name = test_env.actions(action_np[0]).name
                except:
                    action_name = str(action_np)
                print(f"  Step {steps:3d} | Action: {action_name:10s} | Reward: {reward:.2f} | Novelty: {novelty_val:.2f}")
        
        # Save GIF of episode
        if frames:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            gif_path = os.path.join(checkpoint_dir, f"episode_{episode + 1}.gif")
            write_gif(np.array(frames), gif_path, fps=1)
            
            if use_wandb:
                wandb.log({"test/episode_replay": wandb.Video(gif_path, fps=10, format="gif")})

        # Log episode summary
        if use_wandb:
            # Create activity table with all states (including those with zero changes)
            full_activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
            for idx, count in enumerate(activity):
                m = flag_mapping[idx]
                full_activity_table.add_data(idx, m["object_type"], m["object_index"], m["state_name"], count)
            
            # Keep the filtered table for states that changed (for backward compatibility)
            filtered_activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
            for idx, count in enumerate(activity):
                if count > 0:  # Only log states that changed
                    m = flag_mapping[idx]
                    filtered_activity_table.add_data(idx, m["object_type"], m["object_index"], m["state_name"], count)
            
            # Log episode data with explicit commit=True to ensure it's saved
            wandb.log({
                "test/episode_total_reward": total_reward,
                "test/episode_length": steps,
                "test/activity_table": filtered_activity_table,
                "test/full_activity_table": full_activity_table,
                f"episode_{episode + 1}_activity": full_activity_table,  # Match RND naming convention
                "test/episode_complete": episode + 1
            }, commit=True)
            
            # Add explicit summary table logging with unique step to prevent overwriting
            wandb.log({
                f"activity_summary_{episode + 1}": filtered_activity_table,
            }, step=10000000 + episode)  # Use large offset to avoid collision with regular steps
        
        # Print episode summary
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Steps: {steps}")
        print(f"   Active States: {sum(1 for c in activity if c > 0)}/{len(activity)}")


def main():
    env_id = 'MiniGrid-MultiToy-16x16-N2-v0'
    TASK = 'MultiToy'
    ROOM_SIZE = 16
    MAX_STEPS = 10000
    env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    register(
        id=env_id,
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=env_kwargs
    )
    # Train model
    noveld_ppo = train_agent(env_id, device)
    
    # Test model
    test_agent(env_id, noveld_ppo, device)

    # # Test checkpoints
    # noveld_ppo = NovelD_PPO(env_id, device)
    # noveld_ppo.load_checkpoint('checkpoints/checkpoint_400000.pt')
    # test_agent(env_id, noveld_ppo, device)


if __name__ == "__main__":
    main()
