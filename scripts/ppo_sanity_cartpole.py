#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch

# Ensure project root is on sys.path to allow `import algorithms.NovelD_PPO`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Prefer gymnasium if available; fall back to gym
try:
    import gymnasium as gym
except Exception:
    import gym

from algorithms.NovelD_PPO import NovelD_PPO


def reset_env(env, seed=None):
    try:
        if seed is not None:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()
        return obs, info
    except Exception:
        # Older gym API
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        obs = env.reset()
        info = {}
        return obs, info


def step_env(env, action):
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    except ValueError:
        # Older gym API
        obs, reward, done, info = env.step(action)
        return obs, float(reward), bool(done), info


def evaluate_agent(ppo: NovelD_PPO, env_id: str, episodes: int = 5, render: bool = False) -> float:
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = reset_env(env)
        ep_ret = 0.0
        done = False
        while not done:
            if render:
                env.render()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
            action_np = action.detach().cpu().numpy()
            # For single Discrete action spaces, squeeze and cast to int
            if len(ppo.action_dims) == 1:
                action_env = int(action_np.squeeze())
            else:
                # MultiDiscrete (not used for CartPole)
                action_env = action_np[0]
            obs, reward, done, _ = step_env(env, action_env)
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="PPO optimizer sanity check on CartPole")
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=50, help="number of PPO updates (total_timesteps = num_envs*num_steps*updates)")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pre_episodes", type=int, default=5)
    parser.add_argument("--post_episodes", type=int, default=5)
    parser.add_argument("--target_return", type=float, default=150.0, help="success threshold for average return after training")
    parser.add_argument("--render_eval", action="store_true")
    args = parser.parse_args()

    total_timesteps = args.num_envs * args.num_steps * args.updates

    # Instantiate NovelD_PPO with wrapper disabled and extrinsic-only optimization
    ppo = NovelD_PPO(
        env_id=args.env_id,
        device=args.device,
        total_timesteps=total_timesteps,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
        int_coef=0.0,
        ext_coef=1.0,
        use_wandb=False,
        wrap_observations=False,
        norm_adv=True,
    )

    print("Evaluating before training...")
    pre_return = evaluate_agent(ppo, args.env_id, episodes=args.pre_episodes, render=args.render_eval)
    print(f"Average return before training: {pre_return:.2f}")

    print("\nStarting training...")
    ppo.train()

    print("\nEvaluating after training...")
    post_return = evaluate_agent(ppo, args.env_id, episodes=args.post_episodes, render=args.render_eval)
    print(f"Average return after training: {post_return:.2f}")

    improved = post_return > pre_return + 20.0  # require at least 20 point improvement
    meets_target = post_return >= args.target_return

    print("\nSanity check result:")
    print(f"Improved: {improved} (Δ={post_return - pre_return:.1f}) | Meets target ({args.target_return}): {meets_target}")

    # Exit code semantics: 0 if passing the sanity check, 1 otherwise
    if meets_target or improved:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()

