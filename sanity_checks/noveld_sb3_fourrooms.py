#!/usr/bin/env python3
"""
Quick sanity check runner for the SB3 NovelD_PPO implementation on MiniGrid FourRooms.
"""
import argparse
import os
import sys
import warnings
from enum import IntEnum
from typing import Callable

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


def build_env_fn(env_id: str, seed: int, wrapper_factory: Callable[[gym.Env], gym.Env]) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_id)
        env = GymV26ToV21Wrapper(env)
        env = wrapper_factory(env)
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


def resolve_flat_wrapper() -> Callable[[gym.Env], gym.Env]:
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
    parser.add_argument("--int-coef", type=float, default=1.0, help="Intrinsic reward scale.")
    parser.add_argument("--ext-coef", type=float, default=1.0, help="Extrinsic reward scale.")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional path to save the trained model.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device, e.g. 'cpu', 'cuda', 'cuda:0'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    flat_wrapper_cls = resolve_flat_wrapper()
    env_id = resolve_env_id(args.env_id)
    if env_id != args.env_id:
        print(f"[env] Requested {args.env_id} unavailable; using {env_id} instead.")
    device_arg = args.device
    if device_arg != "auto" and device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] CUDA requested but not available; falling back to cpu.")
        device_arg = "cpu"
    env_fns = [
        build_env_fn(env_id, seed=args.seed + idx, wrapper_factory=flat_wrapper_cls) for idx in range(args.n_envs)
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

    model.learn(total_timesteps=args.total_timesteps, log_interval=10)

    if args.model_path:
        save_dir = os.path.dirname(args.model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.save(args.model_path)

    vec_env.close()


if __name__ == "__main__":
    main()
