#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import numpy as np
import torch

# Ensure project root is on sys.path to allow `import algorithms.NovelD_PPO`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Prefer gymnasium if available; fall back to gym
try:
    import gymnasium as gym
    GYMNASIUM = True
except Exception:
    import gym
    GYMNASIUM = False

# Optional deps
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    plt = None
    MPL_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None
    IMAGEIO_AVAILABLE = False

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
                try:
                    env.render()
                except Exception:
                    pass
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


def get_frame(env):
    try:
        frame = env.render()
        if frame is not None:
            return frame
    except Exception:
        pass
    # Fallback for older gym
    try:
        return env.render(mode="rgb_array")
    except Exception:
        return None


def record_policy_video(ppo: NovelD_PPO, env_id: str, max_steps: int = 500, fps: int = 20, out_dir: str = None):
    if not IMAGEIO_AVAILABLE:
        return None, None
    # Try to create env with rgb_array rendering
    try:
        env = gym.make(env_id, render_mode="rgb_array") if GYMNASIUM else gym.make(env_id)
    except Exception:
        env = gym.make(env_id)
    obs, _ = reset_env(env)
    frames = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
        action_np = action.detach().cpu().numpy()
        action_env = int(action_np.squeeze()) if len(ppo.action_dims) == 1 else action_np[0]
        obs, reward, done, _ = step_env(env, action_env)
        frame = get_frame(env)
        if frame is not None:
            frames.append(frame)
        steps += 1
    env.close()
    if not frames:
        return None, None

    # Save gif
    out_dir = out_dir or os.path.join(REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "ppo_cartpole_episode.gif")
    try:
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path, frames
    except Exception:
        return None, frames


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
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="PPO-Sanity")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_video", action="store_true", help="record and log a video after training")
    args = parser.parse_args()

    if args.wandb and not WANDB_AVAILABLE:
        print("wandb not installed; proceeding without wandb logging.")
        args.wandb = False

    total_timesteps = args.num_envs * args.num_steps * args.updates

    # Initialize W&B (optional)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "env_id": args.env_id,
                "device": args.device,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "updates": args.updates,
                "total_timesteps": total_timesteps,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "int_coef": 0.0,
                "ext_coef": 1.0,
            },
        )

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
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wrap_observations=False,
        norm_adv=True,
    )

    print("Evaluating before training...")
    pre_return = evaluate_agent(ppo, args.env_id, episodes=args.pre_episodes, render=args.render_eval)
    print(f"Average return before training: {pre_return:.2f}")
    if args.wandb:
        wandb.log({"eval/pre_return": pre_return})

    print("\nStarting training...")
    ppo.train()

    print("\nEvaluating after training...")
    post_return = evaluate_agent(ppo, args.env_id, episodes=args.post_episodes, render=args.render_eval)
    print(f"Average return after training: {post_return:.2f}")

    improved = post_return > pre_return + 20.0  # require at least 20 point improvement
    meets_target = post_return >= args.target_return

    if args.wandb:
        wandb.log({
            "eval/post_return": post_return,
            "eval/improvement": post_return - pre_return,
            "sanity/improved": improved,
            "sanity/meets_target": meets_target,
        })

    # Optional: record and log a video
    video_path = None
    if args.log_video:
        gif_path, frames = record_policy_video(ppo, args.env_id, max_steps=500, fps=20)
        if gif_path:
            print(f"Saved evaluation video to {gif_path}")
            video_path = gif_path
            if args.wandb:
                wandb.log({"eval/video": wandb.Video(gif_path, fps=20, format="gif")})
        else:
            print("Could not record video (imageio not available or env rendering unsupported).")

    # Optional: simple visualization of pre vs post return
    if MPL_AVAILABLE:
        fig = plt.figure(figsize=(4, 3))
        plt.bar(["pre", "post"], [pre_return, post_return], color=["#9999ff", "#66cc66"]) 
        plt.ylabel("Average Return")
        plt.title("PPO Sanity Check: CartPole")
        plt.tight_layout()
        # Save locally
        out_dir = os.path.join(REPO_ROOT, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, "ppo_sanity_returns.png")
        plt.savefig(fig_path)
        print(f"Saved return plot to {fig_path}")
        if args.wandb:
            wandb.log({"eval/return_plot": wandb.Image(fig)})
        plt.close(fig)

    print("\nSanity check result:")
    print(f"Improved: {improved} (Δ={post_return - pre_return:.1f}) | Meets target ({args.target_return}): {meets_target}")

    # Exit code semantics: 0 if passing the sanity check, 1 otherwise
    exit_code = 0 if (meets_target or improved) else 1

    if args.wandb:
        # Ensure the wandb run is closed if we opened it
        try:
            wandb.finish()
        except Exception:
            pass

    exit(exit_code)


if __name__ == "__main__":
    main()

