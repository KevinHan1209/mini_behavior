# test_rnd.py
import gym
from RND_PPO import RND_PPO
import os
import numpy as np
import torch
import time
import wandb
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


def test_agent(env_id, model, device, TASK, ROOM_SIZE, STEP, num_episodes=5, max_steps_per_episode=200):
    print(f"\n=== Testing Agent: {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="rnd-ppo-test",
               name=f"RND_PPO_{TASK}_{ROOM_SIZE}x{ROOM_SIZE}_STEP{STEP}",
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

    #model.agent.network.to(device)
    
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
                obs_tensor = torch.FloatTensor(obs).to(device)
                action, _, _, value = model.agent.get_action_and_value(obs_tensor)
            
            # Take an action in the environment.
            #print(f"Action tensor: {action}, shape: {action.shape}")
            #obs, reward, done, _ = test_env.step(action.cpu().numpy()[0])
            obs, reward, done, _ = test_env.step(action.cpu().item())
            total_reward += reward
            steps += 1

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                if obs_tensor.dim() == 1:  # Ensure obs is 2D for batch processing
                    obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

                novelty = model.calculate_novelty(obs_tensor)
                #novelty = model.calculate_novelty(torch.FloatTensor(obs).to(device))
                novelty_values.append(novelty.item())
            

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
                "step_novelty": novelty.item(),
                "step_value": value.item(),
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
            print(f"Novelty Score: {novelty.item():.4f} | Value: {value.item():.4f}")
        
        # Log episode-level metrics including the activity (cumulative flag changes).
        wandb.log({
            "episode_total_reward": total_reward,
            "episode_length": steps,
            "episode_mean_novelty": np.mean(novelty_values),
            "activity": activity,
            "episode": episode
        })

        # Create and log a gif replay of the episode.
        gif_path = f"img/rnd32x32/0/episode_{episode + 1}.gif"
        #print(activity)
        #if frames:
        #    write_gif(np.array(frames), gif_path, fps=1)
        #    wandb.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})
        
        # Instead of printing the activity details, log them as a table in wandb.
        activity_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"])
        for idx, count in enumerate(activity):
            mapping = flag_mapping[idx]
            activity_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"], count)
        wandb.log({f"episode_{episode + 1}_activity": activity_table})
        
        print(f"\n=== Episode {episode + 1} Summary ===")
        print(f"Total Reward: {total_reward:.2f} | Steps: {steps} | Mean Novelty: {np.mean(novelty_values):.4f}")
        print("\nActivity per binary flag (changes detected):")
        for idx, count in enumerate(activity):
            mapping = flag_mapping[idx]
            print(f"- Flag {idx} ({mapping['object_type']} #{mapping['object_index']} - {mapping['state_name']}): {count}")

    
    test_env.close()
    wandb.finish()


def main():

    TASK = 'MultiToy'
    ROOM_SIZE = 32
    MAX_STEPS = 1000
    STEP = 0
    
    env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v0"
    env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
    test_env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v1" 
    test_env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "test_env": True}
    
    register(
        id=env_name,
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=env_kwargs
    )
    register(
        id=test_env_name, 
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=test_env_kwargs
    )

    #Pull in trained model
    model_dir = "models/RND_PPO_MultiToy_Run4_32x32"
    model_path = f"{model_dir}/model_step_{STEP}.pt"

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
    
    model = RND_PPO(
        env_id=test_env_name,
        device=device,
        seed=1
    )
    #model.load(model_path)
    
    # Test model.
    test_agent(test_env_name, model, device, TASK, ROOM_SIZE, STEP, num_episodes=5, max_steps_per_episode=500)


if __name__ == "__main__":
    main()
