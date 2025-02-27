# test.py
import gym
from NovelD_PPO import NovelD_PPO
import numpy as np
import torch
import time
import wandb
from array2gif import write_gif
from env_wrapper import CustomObservationWrapper
from mini_behavior.utils.states_base import RelativeObjectState


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
    """
    Given an observation vector and the environment, extract only the binary flags.
    
    The observation is built as follows:
      - First three values: agent_x, agent_y, agent_dir.
      - For each object:
          - Two position values: pos_x, pos_y.
          - Then each non-relative state is appended as a binary flag.
    
    This function skips the agent state and object positions, returning only the flags.
    """
    flags = []
    index = 3  # skip agent state (x, y, direction)
    for obj_list in env.objs.values():
        for obj in obj_list:
            index += 2  # skip the object's position (2 values)
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    flags.append(obs[index])
                    index += 1
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


def train_agent(env_id, device):
    print("\n=== Starting Agent Training ===")
    try:
        noveld_ppo = NovelD_PPO(env_id, device)
        noveld_ppo.train()
        return noveld_ppo
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


def test_agent(env_id, noveld_ppo, device, num_episodes=5, max_steps_per_episode=100):
    print(f"\n=== Testing Agent: {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="noveld-ppo-test",
               config={"env_id": env_id,
                       "mode": "testing",
                       "num_episodes": num_episodes,
                       "max_steps": max_steps_per_episode})

    test_env = gym.make(env_id)
    test_env = CustomObservationWrapper(test_env)
    
    # Get the underlying (unwrapped) environment for accessing object info.
    env_unwrapped = getattr(test_env, 'env', test_env)

    # Compute the total number of binary flags and generate the mapping.
    num_binary_flags = count_binary_flags(env_unwrapped)
    flag_mapping = generate_flag_mapping(env_unwrapped)

    # Log flag mapping to wandb as a table.
    mapping_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name"])
    for idx, mapping in enumerate(flag_mapping):
        mapping_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"])
    wandb.log({"flag_mapping": mapping_table})

    noveld_ppo.agent.network.to(device)
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        novelty_values = []
        frames = []
        # Initialize the activity counter (one per binary flag).
        activity = [0] * num_binary_flags
        prev_flags = None
        
        while not done and steps < max_steps_per_episode:
            # Render and store the current frame (if available)
            frame = test_env.render()
            if frame is not None:
                frames.append(np.moveaxis(frame, 2, 0))
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, ext_value, int_value = noveld_ppo.agent.get_action_and_value(obs_tensor)
            
            # Take an action in the environment.
            obs, reward, done, _ = test_env.step(action.cpu().numpy()[0])
            total_reward += reward
            steps += 1
            novelty = noveld_ppo.calculate_novelty(torch.FloatTensor(obs).unsqueeze(0).to(device))
            novelty_values.append(novelty)
            
            # Convert tensor values to Python scalars.
            novelty_val = novelty.item() if torch.is_tensor(novelty) else novelty
            ext_val = ext_value.item() if torch.is_tensor(ext_value) else ext_value
            int_val = int_value.item() if torch.is_tensor(int_value) else int_value

            # Extract the binary flags from the current observation.
            current_flags = extract_binary_flags(obs, env_unwrapped)
            if prev_flags is not None:
                # Compare flags: if a flag has changed (0->1 or 1->0), count it as an activity.
                differences = (current_flags != prev_flags).astype(int)
                activity = [a + d for a, d in zip(activity, differences)]
            prev_flags = current_flags

            # Log step metrics to wandb.
            wandb.log({
                "step_reward": reward,
                "step_novelty": novelty_val,
                "step_ext_value": ext_val,
                "step_int_value": int_val,
                "episode": episode,
                "step": steps
            })
            
            # Print step information.
            try:
                action_name = test_env.actions(action.item()).name
            except Exception as e:
                action_name = str(action.item())
            print(f"\nStep {steps}/{max_steps_per_episode}")
            print(f"Action Taken: {action_name} | Reward: {reward:.2f}")
            print(f"Novelty Score: {novelty_val:.4f} | External Value: {ext_val:.4f} | Internal Value: {int_val:.4f}")
        
        # Log episode-level metrics including the activity (cumulative flag changes).
        wandb.log({
            "episode_total_reward": total_reward,
            "episode_length": steps,
            # "episode_mean_novelty": np.mean(novelty_values),
            "activity": activity,
            "episode": episode
        })

        # Create and log a gif replay of the episode.
        gif_path = f"episode_{episode + 1}.gif"
        if frames:
            write_gif(np.array(frames), gif_path, fps=1)
            wandb.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})
        
        # Instead of printing the activity details, log them as a table in wandb.
        activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
        for idx, count in enumerate(activity):
            mapping = flag_mapping[idx]
            activity_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"], count)
        wandb.log({f"episode_{episode + 1}_activity": activity_table})
        
        print(f"\n=== Episode {episode + 1} Summary ===")
        print(f"Total Reward: {total_reward:.2f} | Steps: {steps}")
        print("\nActivity per binary flag (changes detected):")
        for idx, count in enumerate(activity):
            mapping = flag_mapping[idx]
            print(f"- Flag {idx} ({mapping['object_type']} #{mapping['object_index']} - {mapping['state_name']}): {count}")

    
    test_env.close()
    wandb.finish()


def main():
    env_id = 'MiniGrid-MultiToy-8x8-N2-v0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model.
    noveld_ppo = train_agent(env_id, device)
    
    # Test model.
    test_agent(env_id, noveld_ppo, device)


if __name__ == "__main__":
    main()
