# test_rnd.py
import gym
from algorithms.RND_PPO import RND_PPO
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
    num_flags = 0
    for obj_list in env.objs.values():
        for obj in obj_list:
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    num_flags += 1
    return num_flags


def extract_binary_flags(obs, env):
    index = 4
    flags = []
    for obj_list in env.objs.values():
        for obj in obj_list:
            if index + 2 <= len(obs):
                index += 2
            for state_name, state in obj.states.items():
                if index >= len(obs):
                    break
                if not isinstance(state, RelativeObjectState):
                    flags.append(obs[index])
                    index += 1
            if index >= len(obs):
                break
    return np.array(flags)


def generate_flag_mapping(env):
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


def make_single_env(env_id, seed, env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    env = CustomObservationWrapper(env)
    env.seed(seed)
    return env


def compare_models(env_id, model_trained, model_untrained, device, TASK, ROOM_SIZE, STEP, num_episodes=5, max_steps_per_episode=200):
    print("\n=== Running Comparison: Trained vs Untrained ===")
    results = {}
    for label, model in zip(["trained", "untrained"], [model_trained, model_untrained]):
        run_name = f"{label.upper()}_RND_{TASK}_{ROOM_SIZE}x{ROOM_SIZE}_STEP{STEP}"
        wandb.init(project="rnd-ppo-compare", name=run_name, config={"model": label})
        stats = []
        for episode in range(num_episodes):
            test_env = make_single_env(env_id, seed=episode, env_kwargs={"room_size": ROOM_SIZE, "max_steps": max_steps_per_episode, "test_env": True})
            obs = test_env.reset()
            done = False
            logp_values = []
            novelty_values = []
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
                    action, logp, _, _, _ = model.agent.get_action_and_value(obs_tensor)
                    obs, reward, done, _ = test_env.step(action[0].cpu().numpy().tolist())
                    logp_values.append(logp.item())
                    novelty = model.calculate_novelty(torch.FloatTensor(obs).unsqueeze(0).to(device))
                    novelty_values.append(novelty.item())
            wandb.log({
                "logp_mean": np.mean(logp_values),
                "logp_std": np.std(logp_values),
                "novelty_mean": np.mean(novelty_values),
                "episode": episode
            })
            stats.append((np.mean(logp_values), np.std(logp_values), np.mean(novelty_values)))
            test_env.close()
        wandb.finish()
        results[label] = stats
    return results


def main():
    TASK = 'MultiToy'
    ROOM_SIZE = 8
    MAX_STEPS = 1000
    STEP = 1000000

    env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v4"
    test_env_name = f"MiniGrid-{TASK}-{ROOM_SIZE}x{ROOM_SIZE}-N2-LP-v5"
    env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
    test_env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS, "test_env": True}

    register(id=env_name, entry_point=f'mini_behavior.envs:{TASK}Env', kwargs=env_kwargs)
    register(id=test_env_name, entry_point=f'mini_behavior.envs:{TASK}Env', kwargs=test_env_kwargs)

    model_dir = "models/RND_PPO_MultiToy_Run9_8x8_new_env"
    model_path = f"{model_dir}/model_step_{STEP}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_env = make_single_env(env_id=test_env_name, seed=1, env_kwargs=test_env_kwargs)

    model_trained = RND_PPO(env=test_env, env_id=test_env_name, device=device, seed=1)
    if os.path.exists(model_path):
        model_trained.load(model_path)
    else:
        print("No trained model found. Exiting.")
        return

    model_untrained = RND_PPO(env=test_env, env_id=test_env_name, device=device, seed=1)

    compare_models(test_env_name, model_trained, model_untrained, device, TASK, ROOM_SIZE, STEP)


if __name__ == "__main__":
    main()
