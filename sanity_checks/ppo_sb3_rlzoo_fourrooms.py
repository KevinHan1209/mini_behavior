#!/usr/bin/env python3
"""
Train Stable-Baselines3 PPO on MiniGrid FourRooms and optionally run RL Zoo commands.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, Type

import numpy as np
import torch

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Prefer gymnasium when available
try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

MINIGRID_AVAILABLE = True
flat_wrapper_candidates: dict[str, Type[gym.ObservationWrapper]] = {}

try:
    import minigrid  # type: ignore

    try:
        from minigrid.wrappers import FlatObsWrapper as _MinigridFlatObsWrapper  # type: ignore
        flat_wrapper_candidates["minigrid.wrappers.FlatObsWrapper"] = _MinigridFlatObsWrapper
    except Exception:
        pass
    try:
        import minigrid.envs  # type: ignore
    except Exception:
        pass
except Exception:
    MINIGRID_AVAILABLE = False

try:
    import gym_minigrid  # type: ignore

    MINIGRID_AVAILABLE = True
    try:
        from gym_minigrid.wrappers import FlatObsWrapper as _GymMinigridFlatObsWrapper  # type: ignore
        flat_wrapper_candidates["gym_minigrid.wrappers.FlatObsWrapper"] = _GymMinigridFlatObsWrapper
    except Exception:
        pass
    try:
        import gym_minigrid.envs  # type: ignore
    except Exception:
        pass
except Exception:
    pass

if not MINIGRID_AVAILABLE:
    raise RuntimeError("MiniGrid is required. Install with `pip install minigrid` or `pip install gym-minigrid`.")

try:
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except Exception:
    SB3PPO = None  # type: ignore
    Monitor = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    STABLE_BASELINES_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None  # type: ignore
    IMAGEIO_AVAILABLE = False


class LocalFlatObsWrapper(gym.ObservationWrapper):
    """Fallback MiniGrid observation flattener."""

    def __init__(self, env):
        super().__init__(env)
        sample_obs = env.reset()
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        flat_len = self._flat_len(sample_obs)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(flat_len,), dtype=np.float32)

    def _flat_len(self, obs):
        if isinstance(obs, dict):
            img = obs.get("image")
            if img is None:
                for value in obs.values():
                    if isinstance(value, np.ndarray):
                        img = value
                        break
            if img is None:
                raise RuntimeError("LocalFlatObsWrapper: no ndarray found in observation dict")
            return int(np.prod(img.shape))
        if isinstance(obs, np.ndarray):
            return int(np.prod(obs.shape))
        raise RuntimeError(f"Unsupported observation type: {type(obs)}")

    def observation(self, obs):
        if isinstance(obs, dict):
            img = obs.get("image")
            if img is None:
                for value in obs.values():
                    if isinstance(value, np.ndarray):
                        img = value
                        break
            arr = np.asarray(img, dtype=np.float32)
        elif isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"Unsupported observation type: {type(obs)}")
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr = arr / 255.0
        return arr.reshape(-1)


class Float32ObsWrapper(gym.ObservationWrapper):
    """Ensure observations are float32 for vectorized envs."""

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Float32ObsWrapper expects Box observation space")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32)

    def observation(self, observation):
        arr = np.asarray(observation, dtype=np.float32)
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr = arr / 255.0
        return arr


class GymV26ToV21Wrapper(gym.Wrapper):
    """Bridge Gymnasium API (terminated/truncated) to SB3's Gym API."""

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, _ = result
            return obs
        return result

    def step(self, action):
        outcome = self.env.step(action)
        if len(outcome) == 5:
            obs, reward, terminated, truncated, info = outcome
            done = bool(terminated or truncated)
            if truncated:
                info = dict(info)
                info["TimeLimit.truncated"] = True
            return obs, reward, done, info
        return outcome


def resolve_flat_wrapper(requested: Optional[str]) -> Type[gym.ObservationWrapper]:
    if requested:
        if requested in flat_wrapper_candidates:
            return flat_wrapper_candidates[requested]
        module, _, attr = requested.rpartition(".")
        if not module:
            raise ValueError(f"Invalid wrapper path: {requested}")
        module_obj = __import__(module, fromlist=[attr])
        wrapper_cls = getattr(module_obj, attr)
        if not issubclass(wrapper_cls, gym.ObservationWrapper):
            raise TypeError(f"{requested} is not an ObservationWrapper")
        return wrapper_cls
    if "gym_minigrid.wrappers.FlatObsWrapper" in flat_wrapper_candidates:
        return flat_wrapper_candidates["gym_minigrid.wrappers.FlatObsWrapper"]
    if "minigrid.wrappers.FlatObsWrapper" in flat_wrapper_candidates:
        return flat_wrapper_candidates["minigrid.wrappers.FlatObsWrapper"]
    return LocalFlatObsWrapper


def make_single_env(env_id: str, seed: int, idx: int, flat_wrapper_cls: Type[gym.ObservationWrapper]):
    env = gym.make(env_id)
    env = GymV26ToV21Wrapper(env)
    env = flat_wrapper_cls(env)
    env = Float32ObsWrapper(env)
    try:
        env.reset(seed=seed + idx)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed + idx)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed + idx)
    return env


def make_render_env(env_id: str, seed: int, idx: int, flat_wrapper_cls: Type[gym.ObservationWrapper]):
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)
    env = GymV26ToV21Wrapper(env)
    env = flat_wrapper_cls(env)
    env = Float32ObsWrapper(env)
    try:
        env.reset(seed=seed + idx)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed + idx)
    return env


def evaluate_sb3_policy(model, env_id: str, seed: int, episodes: int, flat_wrapper_cls: Type[gym.ObservationWrapper], eval_idx: int = 10_000) -> float:
    env = make_single_env(env_id, seed, eval_idx, flat_wrapper_cls)
    returns = []
    try:
        for ep in range(episodes):
            try:
                obs = env.reset(seed=seed + eval_idx + ep)
            except TypeError:
                obs = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    if action.shape == ():
                        action = int(action)
                    else:
                        action = action.item()
                obs, reward, done, _ = env.step(action)
                ep_return += float(reward)
            returns.append(ep_return)
    finally:
        env.close()
    return float(np.mean(returns)) if returns else 0.0


def record_policy_video(policy_fn, env_id: str, seed: int, idx: int, flat_wrapper_cls: Type[gym.ObservationWrapper], max_steps: int, out_path: str, video_format: str, fps: int) -> bool:
    if not IMAGEIO_AVAILABLE:
        print("[video] imageio not available; skipping recording")
        return False
    env = make_render_env(env_id, seed, idx, flat_wrapper_cls)
    frames = []
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        steps = 0
        while not done and steps < max_steps:
            frame = env.render("rgb_array") if hasattr(env, "render") else None
            if frame is not None:
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)
            action = policy_fn(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out
            if isinstance(obs, tuple):
                obs = obs[0]
            steps += 1
        final_frame = env.render("rgb_array") if hasattr(env, "render") else None
        if final_frame is not None:
            final_frame = np.asarray(final_frame)
            if final_frame.dtype != np.uint8:
                final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            frames.append(final_frame)
    finally:
        env.close()
    if not frames:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        if video_format == "gif":
            imageio.mimsave(out_path, frames, fps=fps)
        elif video_format == "mp4":
            writer = imageio.get_writer(out_path, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        else:
            raise ValueError(f"Unsupported video format: {video_format}")
    except Exception as exc:
        print(f"[video] Failed to save video: {exc}")
        return False
    print(f"[video] Saved evaluation rollout to {out_path}")
    return True


def run_sb3_training(env_id: str, seed: int, total_timesteps: int, n_envs: int, log_dir: str, flat_wrapper_cls: Type[gym.ObservationWrapper], eval_episodes: int, record_eval: bool, pre_video_path: Optional[str], post_video_path: Optional[str], final_eval_dir: str, final_eval_seeds: list[int], video_format: str, video_fps: int, video_max_steps: int, video_seed_idx: int):
    if not STABLE_BASELINES_AVAILABLE:
        raise RuntimeError("stable-baselines3 is required. Install with `pip install stable-baselines3`.")

    os.makedirs(log_dir, exist_ok=True)

    def env_fn(rank: int):
        def _init():
            env = make_single_env(env_id, seed, rank, flat_wrapper_cls)
            return Monitor(env, filename=None)

        return _init

    vec_env = DummyVecEnv([env_fn(i) for i in range(n_envs)])
    try:
        model = SB3PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            n_steps=512,
            batch_size=64,
            learning_rate=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            n_epochs=10,
        )

        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                if action.shape == ():
                    return int(action)
                return action.item()
            return int(action)

        if eval_episodes > 0:
            pre_return = evaluate_sb3_policy(model, env_id, seed, eval_episodes, flat_wrapper_cls)
            print(f"[sb3] Average return before training: {pre_return:.2f}")
        if record_eval and pre_video_path:
            record_policy_video(policy_fn, env_id, seed, video_seed_idx, flat_wrapper_cls, video_max_steps, pre_video_path, video_format, video_fps)

        model.learn(total_timesteps=total_timesteps)

        if eval_episodes > 0:
            post_return = evaluate_sb3_policy(model, env_id, seed, eval_episodes, flat_wrapper_cls, eval_idx=20_000)
            print(f"[sb3] Average return after training: {post_return:.2f}")
        if record_eval and post_video_path:
            record_policy_video(policy_fn, env_id, seed, video_seed_idx + 1, flat_wrapper_cls, video_max_steps, post_video_path, video_format, video_fps)

        if IMAGEIO_AVAILABLE:
            print("[video] Recording final evaluation rollouts (10 episodes)...")
            for rollout_idx, rollout_seed in enumerate(final_eval_seeds, start=1):
                rollout_path = os.path.join(final_eval_dir, f"rollout_{rollout_idx:02d}.gif")
                record_policy_video(
                    policy_fn,
                    env_id,
                    seed,
                    rollout_seed,
                    flat_wrapper_cls,
                    video_max_steps,
                    rollout_path,
                    "gif",
                    video_fps,
                )
        else:
            print("[video] imageio not available; skipping final evaluation rollouts.")

        save_path = os.path.join(log_dir, "sb3_ppo_minigrid_fourrooms")
        model.save(save_path)
        print(f"[sb3] Saved model to {save_path}.zip")
    finally:
        vec_env.close()


def run_rl_zoo_training(env_id: str, log_root: str, dry_run: bool = False, push_to_hub: bool = False):
    os.makedirs(log_root, exist_ok=True)
    train_cmd = [
        sys.executable,
        "-m",
        "rl_zoo3.train",
        "--algo",
        "ppo",
        "--env",
        env_id,
        "-f",
        log_root,
    ]
    push_cmd = [
        sys.executable,
        "-m",
        "rl_zoo3.push_to_hub",
        "--algo",
        "ppo",
        "--env",
        env_id,
        "-f",
        log_root,
        "-orga",
        "sb3",
    ]

    if dry_run:
        print("[rlzoo] Dry run: ", " ".join(train_cmd))
        if push_to_hub:
            print("[rlzoo] Dry run: ", " ".join(push_cmd))
        return

    print("[rlzoo] Running:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    if push_to_hub:
        print("[rlzoo] Running:", " ".join(push_cmd))
        subprocess.run(push_cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SB3 PPO on MiniGrid FourRooms and optionally invoke RL Zoo.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-FourRooms-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sb3-total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--sb3-n-envs", type=int, default=8)
    parser.add_argument("--sb3-log-dir", type=str, default=os.path.join(REPO_ROOT, "logs", "sb3_fourrooms"))
    parser.add_argument("--skip-sb3", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--record-eval", action="store_true")
    parser.add_argument("--video-dir", type=str, default=os.path.join(REPO_ROOT, "outputs", "videos"))
    parser.add_argument("--video-format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--video-max-steps", type=int, default=500)
    parser.add_argument("--video-seed-idx", type=int, default=4242)
    parser.add_argument(
        "--eval-video-offset",
        type=int,
        default=100,
        help="Base offset added to the seed for post-training evaluation rollouts.",
    )
    parser.add_argument("--rlzoo-log-root", type=str, default=os.path.join(REPO_ROOT, "logs"))
    parser.add_argument("--run-rlzoo", action="store_true")
    parser.add_argument("--rlzoo-dry-run", action="store_true")
    parser.add_argument("--rlzoo-push", action="store_true")
    parser.add_argument("--env-wrapper", type=str, default="gym_minigrid.wrappers.FlatObsWrapper")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    flat_wrapper_cls = resolve_flat_wrapper(args.env_wrapper)

    safe_env = args.env_id.replace("/", "_")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_root = os.path.abspath(args.video_dir)
    final_eval_dir = os.path.join(video_root, safe_env, f"eval_rollouts_{run_timestamp}")
    os.makedirs(final_eval_dir, exist_ok=True)
    seeds_manifest = os.path.join(final_eval_dir, "seeds.txt")
    if os.path.exists(seeds_manifest):
        final_eval_seeds: list[int] = []
        with open(seeds_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                try:
                    _, raw_seed = line.split(":", 1)
                    final_eval_seeds.append(int(raw_seed.strip()))
                except ValueError:
                    continue
        if len(final_eval_seeds) != 10:
            final_eval_seeds = [args.video_seed_idx + args.eval_video_offset + i for i in range(10)]
    else:
        final_eval_seeds = [args.video_seed_idx + args.eval_video_offset + i for i in range(10)]
        with open(seeds_manifest, "w", encoding="utf-8") as f:
            for idx, seed_val in enumerate(final_eval_seeds, start=1):
                f.write(f"rollout_{idx:02d}: {seed_val}\n")

    pre_video_path = None
    post_video_path = None
    if args.record_eval:
        if not IMAGEIO_AVAILABLE:
            print("[video] imageio not available; disabling eval recording")
            args.record_eval = False
        else:
            os.makedirs(video_root, exist_ok=True)
            pre_video_path = os.path.join(video_root, f"{safe_env}_sb3_pre_{run_timestamp}.{args.video_format}")
            post_video_path = os.path.join(video_root, f"{safe_env}_sb3_post_{run_timestamp}.{args.video_format}")

    if not args.skip_sb3:
        print(f"[sb3] Training PPO on {args.env_id} with {args.sb3_n_envs} envs for {args.sb3_total_timesteps} steps (seed {args.seed}).")
        device = torch.device(args.device)
        torch.set_default_tensor_type(torch.FloatTensor if device.type == "cpu" else torch.cuda.FloatTensor)
        run_sb3_training(
            env_id=args.env_id,
            seed=args.seed,
            total_timesteps=args.sb3_total_timesteps,
            n_envs=args.sb3_n_envs,
            log_dir=args.sb3_log_dir,
            flat_wrapper_cls=flat_wrapper_cls,
            eval_episodes=args.eval_episodes,
            record_eval=args.record_eval,
            pre_video_path=pre_video_path,
            post_video_path=post_video_path,
            final_eval_dir=final_eval_dir,
            final_eval_seeds=final_eval_seeds,
            video_format=args.video_format,
            video_fps=args.video_fps,
            video_max_steps=args.video_max_steps,
            video_seed_idx=args.video_seed_idx,
        )
    else:
        print("[sb3] Skipping SB3 training (--skip-sb3)")

    if args.run_rlzoo:
        run_rl_zoo_training(
            env_id=args.env_id,
            log_root=args.rlzoo_log_root,
            dry_run=args.rlzoo_dry_run,
            push_to_hub=args.rlzoo_push,
        )
    else:
        print("[rlzoo] Not executing rl_zoo3 commands. Use --run-rlzoo to enable.")


if __name__ == "__main__":
    main()
