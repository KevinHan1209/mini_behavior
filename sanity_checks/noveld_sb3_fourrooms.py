#!/usr/bin/env python3
"""
Quick sanity check runner for the SB3 NovelD_PPO implementation on MiniGrid FourRooms.
"""
import argparse
import os
import sys
import warnings
from datetime import datetime
from enum import IntEnum
from typing import Callable, Optional, Type

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import gymnasium as gym  # type: ignore
except Exception:
    import gym  # type: ignore

try:
    from gym_minigrid.wrappers import FlatObsWrapper as DefaultFlatObsWrapper  # type: ignore
except Exception:
    try:
        from minigrid.wrappers import FlatObsWrapper as DefaultFlatObsWrapper  # type: ignore
    except Exception:
        DefaultFlatObsWrapper = None  # type: ignore

from stable_baselines3 import NovelD_PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None  # type: ignore
    IMAGEIO_AVAILABLE = False


class LocalFlatObsWrapper(gym.ObservationWrapper):
    """
    Minimal fallback wrapper to flatten MiniGrid observations when the standard wrapper is unavailable.
    """

    def __init__(self, env):
        super().__init__(env)
        sample_obs = env.reset()
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        flat_len = self._infer_flat_size(sample_obs)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(flat_len,), dtype=np.float32)

    def _infer_flat_size(self, obs):
        if isinstance(obs, dict):
            for value in obs.values():
                if isinstance(value, np.ndarray):
                    return int(np.prod(value.shape))
        elif isinstance(obs, np.ndarray):
            return int(np.prod(obs.shape))
        raise RuntimeError(f"Unsupported observation type: {type(obs)}")

    def observation(self, observation):
        if isinstance(observation, dict):
            for value in observation.values():
                if isinstance(value, np.ndarray):
                    arr = value
                    break
            else:
                raise RuntimeError("LocalFlatObsWrapper: no ndarray found in observation dict")
        else:
            arr = observation
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr = arr / 255.0
        return arr.reshape(-1)


class Float32ObsWrapper(gym.ObservationWrapper):
    """
    Ensure observations are float32 so that SB3 policies receive consistent tensors.
    """

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Float32ObsWrapper expects a Box observation space")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32)

    def observation(self, observation):
        arr = np.asarray(observation, dtype=np.float32)
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr = arr / 255.0
        return arr


class GymV26ToV21Wrapper(gym.Wrapper):
    """
    Bridge Gymnasium's (obs, info) API to Gym v0.21-style tuples expected by SB3.
    """

    def reset(self, **kwargs):
        outcome = self.env.reset(**kwargs)
        if isinstance(outcome, tuple):
            obs, _ = outcome
            return obs
        return outcome

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


class LegacyActionWrapper(gym.ActionWrapper):
    """
    Restore the classic discrete MiniGrid action space for environments that expose newer multi-discrete actions.
    """

    def __init__(self, env):
        super().__init__(env)
        base_env = env.unwrapped
        if not hasattr(base_env, "actions"):
            class Actions(IntEnum):
                left = 0
                right = 1
                forward = 2
                pickup = 3
                drop = 4
                toggle = 5
                done = 6

            base_env.actions = Actions  # type: ignore[attr-defined]
        self._actions_enum = base_env.actions  # type: ignore[attr-defined]
        self.action_space = gym.spaces.Discrete(len(self._actions_enum))

    def action(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)
        # Convert ints back to the IntEnum expected by classic MiniGrid step logic
        return self._actions_enum(int(action))


def build_env_fn(env_id: str, seed: int, wrapper_cls: Type[gym.ObservationWrapper]) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_id)
        env = GymV26ToV21Wrapper(env)
        env = wrapper_cls(env)
        env = Float32ObsWrapper(env)
        env = LegacyActionWrapper(env)
        env = Monitor(env)
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        return env

    return thunk


def make_single_env(env_id: str, seed: int, idx: int, wrapper_cls: Type[gym.ObservationWrapper]) -> gym.Env:
    env = gym.make(env_id)
    env = GymV26ToV21Wrapper(env)
    env = wrapper_cls(env)
    env = Float32ObsWrapper(env)
    env = LegacyActionWrapper(env)
    try:
        env.reset(seed=seed + idx)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed + idx)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed + idx)
    return env


def make_render_env(env_id: str, seed: int, idx: int, wrapper_cls: Type[gym.ObservationWrapper]) -> gym.Env:
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)
    env = GymV26ToV21Wrapper(env)
    env = wrapper_cls(env)
    env = Float32ObsWrapper(env)
    env = LegacyActionWrapper(env)
    try:
        env.reset(seed=seed + idx)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed + idx)
    return env


def evaluate_noveld_policy(
    model: NovelD_PPO,
    env_id: str,
    seed: int,
    episodes: int,
    wrapper_cls: Type[gym.ObservationWrapper],
    eval_idx: int = 10_000,
) -> float:
    env = make_single_env(env_id, seed, eval_idx, wrapper_cls)
    returns: list[float] = []
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
                obs, reward, done, _ = env.step(int(action))
                ep_return += float(reward)
            returns.append(ep_return)
    finally:
        env.close()
    return float(np.mean(returns)) if returns else 0.0


def record_policy_video(
    policy_fn: Callable[[np.ndarray], int],
    env_id: str,
    seed: int,
    idx: int,
    wrapper_cls: Type[gym.ObservationWrapper],
    max_steps: int,
    out_path: str,
    video_format: str,
    fps: int,
) -> bool:
    if not IMAGEIO_AVAILABLE:
        print("[video] imageio not available; skipping recording")
        return False
    env = make_render_env(env_id, seed, idx, wrapper_cls)
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
            action = policy_fn(np.asarray(obs))
            step_out = env.step(int(action))
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


def resolve_flat_wrapper() -> Type[gym.ObservationWrapper]:
    if DefaultFlatObsWrapper is not None:
        return DefaultFlatObsWrapper
    return LocalFlatObsWrapper


def ensure_minigrid_registered() -> None:
    """
    Import MiniGrid modules so that Gym registers the environments.
    """
    for module_name in ("minigrid", "gym_minigrid"):
        try:
            __import__(module_name)
            return
        except Exception:
            continue


def resolve_env_id(requested_id: str) -> str:
    """
    Try to use the requested env id, but gracefully fall back to v0 variants when only they exist.
    """
    ensure_minigrid_registered()
    try:
        gym.spec(requested_id)
        return requested_id
    except Exception as exc:  # gymnasium/gym raise different subclasses
        if requested_id.endswith("-v1"):
            base, _, _ = requested_id.rpartition("-v")
            fallback = f"{base}-v0"
            try:
                ensure_minigrid_registered()
                gym.spec(fallback)
                warnings.warn(f"{requested_id} unavailable; falling back to {fallback}.", RuntimeWarning)
                return fallback
            except Exception:
                pass
        raise exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NovelD_PPO on MiniGrid FourRooms for a quick smoke test.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-FourRooms-v1", help="MiniGrid environment id.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Training timesteps.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--n-envs", type=int, default=8, help="Vectorised environments.")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout length per environment.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Optimizer learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy regularisation coefficient.")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO optimisation epochs.")
    parser.add_argument(
        "--int-coef",
        type=float,
        default=0.0,
        help="Intrinsic reward scale (set >0.0 to re-enable NovelD intrinsic rewards).",
    )
    parser.add_argument("--ext-coef", type=float, default=1.0, help="Extrinsic reward scale.")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional path to save the trained model.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device, e.g. 'cpu', 'cuda', 'cuda:0'.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for mean return eval.")
    parser.add_argument("--record-eval", action="store_true", help="Record pre/post training evaluation rollouts.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs", "videos"),
        help="Directory for saving evaluation videos.",
    )
    parser.add_argument("--video-format", choices=["gif", "mp4"], default="gif", help="Evaluation video format.")
    parser.add_argument("--video-fps", type=int, default=15, help="Frames per second for evaluation videos.")
    parser.add_argument("--video-max-steps", type=int, default=500, help="Max steps per evaluation rollout video.")
    parser.add_argument("--video-seed-idx", type=int, default=4242, help="Seed offset used for evaluation videos.")
    parser.add_argument(
        "--eval-video-offset",
        type=int,
        default=100,
        help="Base offset added to the seed for post-training evaluation rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.int_coef <= 0.0:
        print("[noveld] Intrinsic rewards disabled; running NovelD_PPO in pure PPO mode.")

    flat_wrapper_cls = resolve_flat_wrapper()
    env_id = resolve_env_id(args.env_id)
    if env_id != args.env_id:
        print(f"[env] Requested {args.env_id} unavailable; using {env_id} instead.")

    safe_env = env_id.replace("/", "_")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_root = os.path.abspath(args.video_dir)
    final_eval_dir = os.path.join(video_root, safe_env, f"eval_rollouts_{run_timestamp}")
    os.makedirs(final_eval_dir, exist_ok=True)
    seeds_manifest = os.path.join(final_eval_dir, "seeds.txt")
    if os.path.exists(seeds_manifest):
        final_eval_seeds: list[int] = []
        with open(seeds_manifest, "r", encoding="utf-8") as manifest:
            for line in manifest:
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
        with open(seeds_manifest, "w", encoding="utf-8") as manifest:
            for idx, seed_val in enumerate(final_eval_seeds, start=1):
                manifest.write(f"rollout_{idx:02d}: {seed_val}\n")

    pre_video_path: Optional[str] = None
    post_video_path: Optional[str] = None
    if args.record_eval:
        if not IMAGEIO_AVAILABLE:
            print("[video] imageio not available; disabling eval recording")
            args.record_eval = False
        else:
            os.makedirs(video_root, exist_ok=True)
            pre_video_path = os.path.join(video_root, f"{safe_env}_noveld_pre_{run_timestamp}.{args.video_format}")
            post_video_path = os.path.join(video_root, f"{safe_env}_noveld_post_{run_timestamp}.{args.video_format}")

    device_arg = args.device
    if device_arg != "auto" and device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] CUDA requested but not available; falling back to cpu.")
        device_arg = "cpu"
    env_fns = [
        build_env_fn(env_id, seed=args.seed + idx, wrapper_cls=flat_wrapper_cls) for idx in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    if device_arg != "auto":
        print(f"[device] Using {device_arg} device")
        if device_arg.startswith("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    model = NovelD_PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        int_coef=args.int_coef,
        ext_coef=args.ext_coef,
        device=device_arg,
    )

    def policy_fn(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            if action.shape == ():
                return int(action)
            return int(action.item())
        return int(action)

    try:
        if args.eval_episodes > 0:
            pre_return = evaluate_noveld_policy(
                model,
                env_id=env_id,
                seed=args.seed,
                episodes=args.eval_episodes,
                wrapper_cls=flat_wrapper_cls,
            )
            print(f"[noveld] Average return before training: {pre_return:.2f}")
        if args.record_eval and pre_video_path:
            record_policy_video(
                policy_fn,
                env_id,
                args.seed,
                args.video_seed_idx,
                flat_wrapper_cls,
                args.video_max_steps,
                pre_video_path,
                args.video_format,
                args.video_fps,
            )

        model.learn(total_timesteps=args.total_timesteps, log_interval=10)

        if args.eval_episodes > 0:
            post_return = evaluate_noveld_policy(
                model,
                env_id=env_id,
                seed=args.seed,
                episodes=args.eval_episodes,
                wrapper_cls=flat_wrapper_cls,
                eval_idx=20_000,
            )
            print(f"[noveld] Average return after training: {post_return:.2f}")
        if args.record_eval and post_video_path:
            record_policy_video(
                policy_fn,
                env_id,
                args.seed,
                args.video_seed_idx + 1,
                flat_wrapper_cls,
                args.video_max_steps,
                post_video_path,
                args.video_format,
                args.video_fps,
            )

        if IMAGEIO_AVAILABLE:
            print("[video] Recording final evaluation rollouts (10 episodes)...")
            for rollout_idx, rollout_seed in enumerate(final_eval_seeds, start=1):
                rollout_path = os.path.join(final_eval_dir, f"rollout_{rollout_idx:02d}.gif")
                record_policy_video(
                    policy_fn,
                    env_id,
                    args.seed,
                    rollout_seed,
                    flat_wrapper_cls,
                    args.video_max_steps,
                    rollout_path,
                    "gif",
                    args.video_fps,
                )
        else:
            print("[video] imageio not available; skipping final evaluation rollouts.")

        if args.model_path:
            save_dir = os.path.dirname(args.model_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            model.save(args.model_path)
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
