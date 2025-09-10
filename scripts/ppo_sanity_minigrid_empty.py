#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch

# Ensure project root on path
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

# Try minigrid imports from both namespaces
MINIGRID_AVAILABLE = True
try:
    # Newer package name
    import minigrid
    from minigrid.wrappers import FullyObsWrapper
except Exception:
    try:
        # Older package name
        import gym_minigrid as minigrid  # type: ignore
        from gym_minigrid.wrappers import FullyObsWrapper  # type: ignore
    except Exception:
        MINIGRID_AVAILABLE = False

# Optional deps
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

# Optional video deps
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None
    IMAGEIO_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

from algorithms.NovelD_PPO import NovelD_PPO


class FlatObsWrapper(gym.ObservationWrapper):
    """Flattens MiniGrid observations to a 1D float32 vector.
    - If observation is a dict with 'image', uses that and flattens (normalized to [0,1])
    - If observation is already a numpy array, flattens it
    """
    def __init__(self, env):
        super().__init__(env)
        # Probe one reset to derive shape
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        flat_len = self._flat_len(obs)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(flat_len,), dtype=np.float32)

    def _flat_len(self, obs):
        if isinstance(obs, dict):
            img = obs.get("image", None)
            if img is None:
                # Fallback to first ndarray entry
                for v in obs.values():
                    if isinstance(v, np.ndarray):
                        img = v
                        break
            if img is None:
                raise RuntimeError("FlatObsWrapper: could not find an image-like ndarray in observation dict")
            return int(np.prod(img.shape))
        elif isinstance(obs, np.ndarray):
            return int(np.prod(obs.shape))
        else:
            raise RuntimeError(f"Unsupported obs type for flattening: {type(obs)}")

    def observation(self, obs):
        if isinstance(obs, dict):
            img = obs.get("image", None)
            if img is None:
                # Fallback to any ndarray value
                for v in obs.values():
                    if isinstance(v, np.ndarray):
                        img = v
                        break
            arr = np.asarray(img, dtype=np.float32)
            # Normalize if looks like pixels
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr.reshape(-1)
        elif isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32, copy=False)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr.reshape(-1)
        else:
            raise RuntimeError(f"Unsupported obs type for flattening: {type(obs)}")


def minigrid_env_factory(env_id: str, seed: int, idx: int):
    env = gym.make(env_id)
    # Disable partial observability by wrapping with FullyObsWrapper
    env = FullyObsWrapper(env)
    # Flatten to 1D vector to be similar to current repo style
    env = FlatObsWrapper(env)
    # Seed
    try:
        env.reset(seed=seed + idx)
    except TypeError:
        if hasattr(env, 'seed'):
            env.seed(seed + idx)
    return env


def evaluate_agent(ppo: NovelD_PPO, env_id: str, episodes: int = 5) -> float:
    env = minigrid_env_factory(env_id, seed=0, idx=999)
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        ep_ret = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
            act = int(action.detach().cpu().numpy().squeeze())
            step_out = env.step(act)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out
            ep_ret += float(reward)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def _get_frame_minigrid(env):
    try:
        return env.render(mode="rgb_array")
    except Exception:
        try:
            return env.render()
        except Exception:
            return None


def record_minigrid_video(ppo: NovelD_PPO, env_id: str, max_steps: int = 500, fps: int = 20, fmt: str = "mp4"):
    if not IMAGEIO_AVAILABLE:
        print("[viz] imageio not available. Install with `pip install imageio imageio-ffmpeg`.")
        return None
    # Build env consistent with training obs (FullyObs + Flat) so the agent input matches
    env = minigrid_env_factory(env_id, seed=0, idx=123)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    frames = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        # Frame BEFORE step to include initial state
        frame = _get_frame_minigrid(env)
        if frame is not None:
            frames.append(frame)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
        act = int(action.detach().cpu().numpy().squeeze())
        step_out = env.step(act)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out
        steps += 1
    env.close()

    if not frames:
        print("[viz] No frames captured from MiniGrid.")
        return None
    out_dir = os.path.join(REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "ppo_minigrid_episode")
    try:
        if fmt == "gif":
            path = base + ".gif"
            imageio.mimsave(path, frames, fps=fps)
            print(f"[viz] Saved MiniGrid GIF with {len(frames)} frames to {path}")
            return path
        elif fmt == "mp4":
            path = base + ".mp4"
            writer = imageio.get_writer(path, fps=fps)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"[viz] Saved MiniGrid MP4 with {len(frames)} frames to {path}")
            return path
        else:
            print(f"[viz] Unsupported video format: {fmt}")
            return None
    except Exception as e:
        print(f"[viz] Could not write video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="PPO sanity check on MiniGrid Empty with full observability and flat obs")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="PPO-Sanity-Minigrid")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_video", action="store_true", help="record and log a MiniGrid video after training")
    parser.add_argument("--video_format", choices=["gif", "mp4"], default="mp4")
    args = parser.parse_args()

    if not MINIGRID_AVAILABLE:
        raise RuntimeError("MiniGrid is not installed. Install with `pip install gym-minigrid==1.0.3` or `pip install minigrid`.")

    total_timesteps = args.num_envs * args.num_steps * args.updates

    if args.wandb and not WANDB_AVAILABLE:
        print("wandb not installed; proceeding without wandb logging.")
        args.wandb = False

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
                "obs": "FullyObs + Flat",
            },
        )
        try:
            wandb.define_metric("global_step")
            wandb.define_metric("eval/return", step_metric="global_step")
        except Exception:
            pass

    # PPO-only
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
        use_intrinsic=False,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wrap_observations=False,  # we provide our own wrappers
        env_make_fn=minigrid_env_factory,
        norm_adv=True,
    )

    # Per-update evaluation
    def on_update(update, global_step, agent_ref):
        if args.eval_every > 0 and (update % args.eval_every == 0):
            avg_ret = evaluate_agent(agent_ref, args.env_id, episodes=args.eval_episodes)
            print(f"[Eval] Update {update}: avg_return={avg_ret:.2f}")
            if args.wandb:
                wandb.log({
                    "global_step": global_step,
                    "update": update,
                    "eval/return": avg_ret,
                }, step=global_step)
    ppo.update_callback = on_update

    print("Evaluating before training...")
    pre = evaluate_agent(ppo, args.env_id, episodes=args.eval_episodes)
    print(f"Average return before training: {pre:.2f}")
    if args.wandb:
        wandb.log({"global_step": 0, "eval/pre": pre}, step=0)

    print("\nStarting training...")
    ppo.train()

    print("\nEvaluating after training...")
    post = evaluate_agent(ppo, args.env_id, episodes=args.eval_episodes)
    print(f"Average return after training: {post:.2f}")
    if args.wandb:
        wandb.log({
            "global_step": total_timesteps,
            "eval/post": post,
            "eval/improvement": post - pre
        }, step=total_timesteps)

    # Optional video logging
    if args.log_video:
        vid_path = record_minigrid_video(ppo, args.env_id, max_steps=500, fps=20, fmt=args.video_format)
        if vid_path and args.wandb:
            wandb.log({"eval/video_env": wandb.Video(vid_path, fps=20, format=args.video_format)}, step=total_timesteps)

    if args.wandb:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

