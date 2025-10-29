#!/usr/bin/env python3
"""
Run NovelD's PPO implementation on the MiniGrid FourRooms environment with
hyperparameters chosen to mirror the Stable-Baselines3 defaults provided by the user.
"""
import argparse
import importlib
import os
import sys
import types
from typing import Optional, Type

import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Ensure project root is on the Python path so we can import NovelD_PPO.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Prefer gymnasium if available; fall back to gym.
try:
    import gymnasium as gym
except Exception:  # gymnasium not available
    import gym  # type: ignore

# Try MiniGrid imports from both namespaces so the script works with either package.
MINIGRID_AVAILABLE = True
FullyObsWrapper: Optional[Type[gym.ObservationWrapper]] = None
flat_wrapper_candidates: dict[str, Type[gym.ObservationWrapper]] = {}

try:
    import minigrid  # type: ignore

    try:
        from minigrid.wrappers import FullyObsWrapper as _MinigridFullyObsWrapper  # type: ignore

        FullyObsWrapper = _MinigridFullyObsWrapper
    except Exception:
        pass
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
        from gym_minigrid.wrappers import FullyObsWrapper as _GymMinigridFullyObsWrapper  # type: ignore

        if FullyObsWrapper is None:
            FullyObsWrapper = _GymMinigridFullyObsWrapper
    except Exception:
        pass
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
    if FullyObsWrapper is None:
        MINIGRID_AVAILABLE = False

if not MINIGRID_AVAILABLE or FullyObsWrapper is None:
    raise RuntimeError(
        "MiniGrid is not installed or FullyObsWrapper is unavailable. "
        "Install with `pip install minigrid` or `pip install gym-minigrid`."
    )

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None  # type: ignore
    IMAGEIO_AVAILABLE = False

from algorithms.NovelD_PPO import NovelD_PPO
from networks.actor_critic import MultiCategorical


class LocalFlatObsWrapper(gym.ObservationWrapper):
    """
    Fall-back wrapper that flattens MiniGrid observations to a 1D float32 vector.
    This mirrors the behaviour of gym_minigrid.wrappers.FlatObsWrapper while staying
    compatible across gym and gymnasium return conventions.
    """

    def __init__(self, env):
        super().__init__(env)
        sample_obs = env.reset()
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        flat_len = self._flat_len(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(flat_len,), dtype=np.float32
        )

    def _flat_len(self, obs):
        if isinstance(obs, dict):
            img = obs.get("image")
            if img is None:
                for value in obs.values():
                    if isinstance(value, np.ndarray):
                        img = value
                        break
            if img is None:
                raise RuntimeError(
                    "LocalFlatObsWrapper: could not locate an ndarray inside the observation dict."
                )
            return int(np.prod(img.shape))
        if isinstance(obs, np.ndarray):
            return int(np.prod(obs.shape))
        raise RuntimeError(f"Unsupported observation type for flattening: {type(obs)}")

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
            raise RuntimeError(f"Unsupported observation type for flattening: {type(obs)}")
        if arr.max(initial=0) > 1.0:
            arr = arr / 255.0
        return arr.reshape(-1)


def resolve_flat_wrapper(path: Optional[str]) -> Type[gym.ObservationWrapper]:
    """
    Resolve the requested flattening wrapper path. Falls back to LocalFlatObsWrapper
    if the import fails or no path is provided.
    """
    if path:
        if path in flat_wrapper_candidates:
            return flat_wrapper_candidates[path]
        try:
            module_name, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            wrapper_cls = getattr(module, class_name)
            if not issubclass(wrapper_cls, gym.ObservationWrapper):
                raise TypeError(f"{path} is not an ObservationWrapper.")
            return wrapper_cls
        except Exception as exc:
            print(f"[warn] Could not import {path}: {exc}. Falling back to LocalFlatObsWrapper.")
    # Prefer the gym_minigrid variant if it was imported successfully.
    if "gym_minigrid.wrappers.FlatObsWrapper" in flat_wrapper_candidates:
        return flat_wrapper_candidates["gym_minigrid.wrappers.FlatObsWrapper"]
    if "minigrid.wrappers.FlatObsWrapper" in flat_wrapper_candidates:
        return flat_wrapper_candidates["minigrid.wrappers.FlatObsWrapper"]
    return LocalFlatObsWrapper


class Float32ObsWrapper(gym.ObservationWrapper):
    """Casts flattened observations to float32; scales to [0,1] if values look like bytes."""
    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Float32ObsWrapper requires Box observation space")
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        # If it looks like bytes or categorical encoded up to 255, scale down.
        if arr.size > 0 and np.nanmax(arr) > 1.0:
            arr = arr / 255.0
        return arr


class GymV26ToV21Wrapper(gym.Wrapper):
    """Bridge Gymnasium API (terminated/truncated) to the classic 4-tuple API NovelD expects."""

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
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


def _sb3_layer_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class SB3LikeAgent(nn.Module):
    """Minimal SB3-style (64,64, tanh) policy/value network compatible with NovelD_PPO."""

    def __init__(self, obs_dim: int, action_dims: list[int]):
        super().__init__()
        self.action_dims = action_dims
        tanh_gain = nn.init.calculate_gain("tanh")
        hidden_layers = [
            _sb3_layer_init(nn.Linear(obs_dim, 64), tanh_gain),
            nn.Tanh(),
            _sb3_layer_init(nn.Linear(64, 64), tanh_gain),
            nn.Tanh(),
        ]
        self.backbone = nn.Sequential(*hidden_layers)
        self.policy = _sb3_layer_init(nn.Linear(64, sum(action_dims)), 0.01)
        # Keep a separate intrinsic head so NovelD_PPO can call into both.
        self.value_ext = _sb3_layer_init(nn.Linear(64, 1), 1.0)
        self.value_int = _sb3_layer_init(nn.Linear(64, 1), 1.0)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        hidden = self.backbone(x)
        logits = self.policy(hidden)
        dist = MultiCategorical(logits, self.action_dims)
        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, self.value_ext(hidden), self.value_int(hidden)

    def get_value(self, x: torch.Tensor):
        hidden = self.backbone(x)
        return self.value_ext(hidden), self.value_int(hidden)


def _safe_render_frame(env):
    for fn in (
        lambda: env.render(),
        lambda: env.render("rgb_array"),
        lambda: env.render(mode="rgb_array"),
        lambda: getattr(env, "get_frame", lambda: None)(),
        lambda: env.unwrapped.render("rgb_array") if hasattr(env, "unwrapped") else None,
    ):
        try:
            frame = fn()
        except Exception:
            continue
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    return None


def make_minigrid_factory(
    flat_wrapper_cls: Type[gym.ObservationWrapper],
    fully_observable: bool,
):
    """
    Build a factory compatible with NovelD_PPO.env_make_fn signature.
    """

    def _factory(env_id: str, seed: int, idx: int):
        env = gym.make(env_id)
        env = GymV26ToV21Wrapper(env)
        if fully_observable:
            env = FullyObsWrapper(env)
        env = flat_wrapper_cls(env)
        # Ensure dtype consistency for vectorized stacking and agent input
        env = Float32ObsWrapper(env)
        try:
            env.reset(seed=seed + idx)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed + idx)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + idx)
        return env

    return _factory


def make_render_env(
    env_id: str,
    seed: int,
    idx: int,
    flat_wrapper_cls: Type[gym.ObservationWrapper],
    fully_observable: bool,
):
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)
    env = GymV26ToV21Wrapper(env)
    if fully_observable:
        env = FullyObsWrapper(env)
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


def _save_video(frames: list[np.ndarray], path: str, fmt: str, fps: int) -> bool:
    if not frames:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        return False
    try:
        if fmt == "gif":
            imageio.mimsave(path, frames, fps=fps)
        elif fmt == "mp4":
            writer = imageio.get_writer(path, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        else:
            raise ValueError(f"Unsupported video format: {fmt}")
    except Exception as exc:
        print(f"[video] Failed to write {path}: {exc}")
        return False
    return True


def record_policy_video(
    policy_fn,
    env_id: str,
    seed: int,
    idx: int,
    flat_wrapper_cls: Type[gym.ObservationWrapper],
    fully_observable: bool,
    max_steps: int,
    out_path: str,
    video_format: str,
    fps: int,
) -> bool:
    if not IMAGEIO_AVAILABLE:
        print("[video] imageio not available; skipping video export.")
        return False
    env = make_render_env(env_id, seed, idx, flat_wrapper_cls, fully_observable)
    frames: list[np.ndarray] = []
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        steps = 0
        while not done and steps < max_steps:
            frame = _safe_render_frame(env)
            if frame is not None:
                frames.append(frame)
            action = policy_fn(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out
            steps += 1
        final_frame = _safe_render_frame(env)
        if final_frame is not None:
            frames.append(final_frame)
    finally:
        env.close()
    if not _save_video(frames, out_path, video_format, fps):
        return False
    print(f"[video] Saved evaluation roll-out to {out_path}")
    return True


def evaluate_agent(
    ppo: NovelD_PPO,
    env_id: str,
    episodes: int,
    flat_wrapper_cls: Type[gym.ObservationWrapper],
    fully_observable: bool,
    seed: int,
) -> float:
    env = make_minigrid_factory(flat_wrapper_cls, fully_observable)(env_id, seed=seed, idx=10)
    returns: list[float] = []
    for ep in range(episodes):
        try:
            obs = env.reset(seed=seed + 10 + ep)
        except TypeError:
            obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor, deterministic=True)
            act = int(action.detach().cpu().numpy().squeeze())
            step_out = env.step(act)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = step_out
            total_reward += float(reward)
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NovelD_PPO on MiniGrid FourRooms with SB3-style hyperparameters."
    )
    parser.add_argument("--env-id", type=str, default="MiniGrid-FourRooms-v0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=float, default=5_000_000.0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64, help="Desired minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluate every N updates (0 disables).")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--env-wrapper",
        type=str,
        default="gym_minigrid.wrappers.FlatObsWrapper",
        help="Observation wrapper to flatten MiniGrid observations.",
    )
    parser.add_argument("--fully-observable", action="store_true", default=False)
    parser.add_argument("--no-fully-observable", action="store_false", dest="fully_observable")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--norm-obs", action="store_true", default=False)
    parser.add_argument("--no-norm-obs", action="store_false", dest="norm_obs")
    parser.add_argument("--norm-reward", action="store_true", default=False)
    parser.add_argument("--no-norm-reward", action="store_false", dest="norm_reward")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="NovelD-PPO-FourRooms")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--record-eval", action="store_true", default=True)
    parser.add_argument("--no-record-eval", action="store_false", dest="record_eval")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs", "videos"),
    )
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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.wandb and not WANDB_AVAILABLE:
        print("[warn] wandb not available; continuing without logging.")
        args.wandb = False

    if not args.normalize:
        args.norm_obs = False
        args.norm_reward = False

    # Fix random seeds across libraries for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    flat_wrapper_cls = resolve_flat_wrapper(args.env_wrapper)

    total_timesteps = int(args.total_timesteps)

    pre_video_path = None
    post_video_path = None
    safe_env = args.env_id.replace("/", "_")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_root = os.path.abspath(args.video_dir)
    final_eval_dir = os.path.join(video_root, safe_env, f"eval_rollouts_{run_timestamp}")
    os.makedirs(final_eval_dir, exist_ok=True)

    if args.record_eval:
        if not IMAGEIO_AVAILABLE:
            print("[video] imageio not available; disabling eval recording.")
            args.record_eval = False
        else:
            os.makedirs(video_root, exist_ok=True)
            pre_video_path = os.path.join(
                video_root, f"{safe_env}_noveld_pre_{run_timestamp}.{args.video_format}"
            )
            post_video_path = os.path.join(
                video_root, f"{safe_env}_noveld_post_{run_timestamp}.{args.video_format}"
            )

    wandb_run = None
    if args.wandb:
        config_dict = {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "n_epochs": args.n_epochs,
            "total_timesteps": total_timesteps,
            "fully_observable": args.fully_observable,
            "env_wrapper": args.env_wrapper,
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=config_dict,
        )

    rollout_batch = args.n_envs * args.n_steps
    if rollout_batch % args.batch_size != 0:
        raise ValueError(
            f"rollout batch ({rollout_batch}) must be divisible by batch_size ({args.batch_size})."
        )
    num_minibatches = rollout_batch // args.batch_size

    env_factory = make_minigrid_factory(flat_wrapper_cls, fully_observable=args.fully_observable)

    # Instantiate NovelD_PPO configured as plain PPO (disable intrinsic rewards).
    ppo = NovelD_PPO(
        env_id=args.env_id,
        device=args.device,
        total_timesteps=total_timesteps,
        learning_rate=args.learning_rate,
        num_envs=args.n_envs,
        num_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=num_minibatches,
        update_epochs=args.n_epochs,
        clip_coef=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        int_coef=0.0,
        ext_coef=1.0,
        use_intrinsic=False,
        norm_adv=True,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wrap_observations=False,
        env_make_fn=env_factory,
    )
    ppo.agent = SB3LikeAgent(obs_dim=ppo.obs_dim, action_dims=ppo.action_dims).to(ppo.device)
    ppo.anneal_lr = False

    def ppo_policy(obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor, deterministic=True)
        return int(action.detach().cpu().numpy().squeeze())

    if not args.norm_reward:
        def _identity_rewards(self, rewards):
            return rewards

        def _identity_reward(self, reward):
            return reward

        ppo.normalize_rewards = types.MethodType(_identity_rewards, ppo)
        ppo.normalize_reward = types.MethodType(_identity_reward, ppo)

    if not args.norm_obs:
        def _identity_obs(self, obs):
            return obs

        ppo.normalize_obs = types.MethodType(_identity_obs, ppo)

    def on_update(update_idx: int, global_step: int, agent_ref: NovelD_PPO):
        if args.eval_every <= 0 or update_idx % args.eval_every != 0:
            return
        avg_return = evaluate_agent(
            agent_ref,
            args.env_id,
            episodes=args.eval_episodes,
            flat_wrapper_cls=flat_wrapper_cls,
            fully_observable=args.fully_observable,
            seed=args.seed,
        )
        print(f"[Eval] Update {update_idx:04d} | global_step={global_step} | avg_return={avg_return:.2f}")
        if args.wandb:
            wandb.log(
                {
                    "global_step": global_step,
                    "update": update_idx,
                    "eval/return": avg_return,
                },
                step=global_step,
            )

    ppo.update_callback = on_update

    print("Evaluating before training...")
    pre_return = evaluate_agent(
        ppo,
        args.env_id,
        episodes=args.eval_episodes,
        flat_wrapper_cls=flat_wrapper_cls,
        fully_observable=args.fully_observable,
        seed=args.seed,
    )
    print(f"Average return before training: {pre_return:.2f}")
    if args.record_eval and pre_video_path:
        record_policy_video(
            ppo_policy,
            env_id=args.env_id,
            seed=args.seed,
            idx=args.video_seed_idx,
            flat_wrapper_cls=flat_wrapper_cls,
            fully_observable=args.fully_observable,
            max_steps=args.video_max_steps,
            out_path=pre_video_path,
            video_format=args.video_format,
            fps=args.video_fps,
        )
    if args.wandb:
        wandb.log({"global_step": 0, "eval/pre": pre_return}, step=0)

    print("\nStarting training...")
    ppo.train()

    print("\nEvaluating after training...")
    post_return = evaluate_agent(
        ppo,
        args.env_id,
        episodes=args.eval_episodes,
        flat_wrapper_cls=flat_wrapper_cls,
        fully_observable=args.fully_observable,
        seed=args.seed,
    )
    print(f"Average return after training: {post_return:.2f}")
    if args.record_eval and post_video_path:
        record_policy_video(
            ppo_policy,
            env_id=args.env_id,
            seed=args.seed,
            idx=args.video_seed_idx + 1,
            flat_wrapper_cls=flat_wrapper_cls,
            fully_observable=args.fully_observable,
            max_steps=args.video_max_steps,
            out_path=post_video_path,
            video_format=args.video_format,
            fps=args.video_fps,
        )

    if IMAGEIO_AVAILABLE:
        seeds_manifest = os.path.join(final_eval_dir, "seeds.txt")
        if os.path.exists(seeds_manifest):
            eval_rollout_seeds: list[int] = []
            with open(seeds_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    try:
                        _, raw_seed = line.split(":", 1)
                        eval_rollout_seeds.append(int(raw_seed.strip()))
                    except ValueError:
                        continue
            if len(eval_rollout_seeds) != 10:
                eval_rollout_seeds = [args.video_seed_idx + args.eval_video_offset + i for i in range(10)]
        else:
            eval_rollout_seeds = [args.video_seed_idx + args.eval_video_offset + i for i in range(10)]
            with open(seeds_manifest, "w", encoding="utf-8") as f:
                for idx, seed_val in enumerate(eval_rollout_seeds, start=1):
                    f.write(f"rollout_{idx:02d}: {seed_val}\n")
        print("[video] Recording final evaluation rollouts (10 episodes)...")
        for rollout_idx, rollout_seed in enumerate(eval_rollout_seeds, start=1):
            rollout_path = os.path.join(final_eval_dir, f"rollout_{rollout_idx:02d}.gif")
            record_policy_video(
                ppo_policy,
                env_id=args.env_id,
                seed=args.seed,
                idx=rollout_seed,
                flat_wrapper_cls=flat_wrapper_cls,
                fully_observable=args.fully_observable,
                max_steps=args.video_max_steps,
                out_path=rollout_path,
                video_format="gif",
                fps=args.video_fps,
            )
    else:
        print("[video] imageio not available; skipping final evaluation rollouts.")
    if args.wandb:
        wandb.log(
            {
                "global_step": total_timesteps,
                "eval/post": post_return,
                "eval/improvement": post_return - pre_return,
            },
            step=total_timesteps,
        )
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
