#!/usr/bin/env python3
"""Evaluate maskable SB3 checkpoints with consistent action spaces."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3.common import utils
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_maskable_noveld import MaskableNovelD_PPO

from mini_behavior.register import env_list, register
from SB3_Extrinsic_Experiments.run_experiment import make_vec_env, validate_action_space

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)_steps\.zip$")


def load_experiment_config(result_path: Path) -> dict:
    config_path = result_path / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.json at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    checkpoints = []
    for path in checkpoint_dir.glob("checkpoint_*_steps.zip"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return [p for _, p in sorted(checkpoints, key=lambda item: item[0])]


def _to_nvec(nvec_raw) -> np.ndarray:
    if nvec_raw is None:
        raise ValueError("Checkpoint is missing action_space.nvec")
    if isinstance(nvec_raw, str):
        vals = [int(x) for x in nvec_raw.strip("[]").split()]
        return np.asarray(vals, dtype=np.int64)
    return np.asarray(nvec_raw, dtype=np.int64)


def checkpoint_metadata(checkpoint_path: Path) -> Tuple[np.ndarray, str]:
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
    action_space = data.get("action_space") or {}
    nvec = _to_nvec(action_space.get("nvec"))
    policy_field = data.get("policy_class") or ""
    policy_text = json.dumps(policy_field)
    return nvec, policy_text


def validate_checkpoint(checkpoint_path: Path, expected_nvec: np.ndarray) -> None:
    ckpt_nvec, policy_text = checkpoint_metadata(checkpoint_path)
    if "MaskableActorCriticPolicy" not in policy_text:
        raise ValueError(f"{checkpoint_path.name} is not a maskable checkpoint. Retrain/save with MaskableActorCriticPolicy.")
    if ckpt_nvec.shape != expected_nvec.shape or not np.array_equal(ckpt_nvec, expected_nvec):
        raise ValueError(
            f"{checkpoint_path.name} uses nvec {ckpt_nvec.tolist()}, expected env nvec {expected_nvec.tolist()}."
        )


def evaluate_mean_return(
    model: MaskableNovelD_PPO,
    vec_env,
    n_episodes: int,
    deterministic: bool,
) -> Tuple[float, float]:
    obs = vec_env.reset()
    running_returns = np.zeros(vec_env.num_envs, dtype=np.float64)
    completed: List[float] = []
    while len(completed) < n_episodes:
        action_masks = get_action_masks(vec_env)
        actions, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        obs, rewards, dones, _infos = vec_env.step(actions)
        running_returns += np.asarray(rewards, dtype=np.float64)
        for idx, done in enumerate(dones):
            if done:
                completed.append(float(running_returns[idx]))
                running_returns[idx] = 0.0
                if len(completed) >= n_episodes:
                    break
    mean = float(np.mean(completed)) if completed else float("nan")
    std = float(np.std(completed)) if completed else float("nan")
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate maskable checkpoints and print masked ep_rew_mean.")
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path to a results directory (e.g., SB3_Extrinsic_Experiments/results/sb3_noise_only).",
    )
    parser.add_argument("--n-episodes", type=int, default=8, help="Number of eval episodes (default: 8).")
    parser.add_argument("--seed", type=int, default=1, help="Base seed for evaluation envs (default: 1).")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_path)
    checkpoint_dir = result_path / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory at {checkpoint_dir}")

    config = load_experiment_config(result_path)
    env_config = config["env_config"]
    extrinsic_rewards = config["extrinsic_rewards"]

    env_id = env_config["env_id"]
    env_kwargs = {
        "room_size": env_config["room_size"],
        "max_steps": env_config["max_steps"],
        "extrinsic_rewards": extrinsic_rewards,
    }
    if env_id not in env_list:
        register(id=env_id, entry_point=f'mini_behavior.envs:{env_config["task"]}Env', kwargs=env_kwargs)

    if args.n_episodes <= 0:
        raise ValueError("--n-episodes must be positive")

    utils.set_random_seed(args.seed)
    vec_env = make_vec_env(env_id, env_kwargs, seed=args.seed, num_envs=args.n_episodes)
    validate_action_space(vec_env)
    env_nvec = np.asarray(vec_env.action_space.nvec, dtype=np.int64)

    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    print(f"Evaluating {len(checkpoints)} checkpoints from {checkpoint_dir}")
    for ckpt in checkpoints:
        validate_checkpoint(ckpt, expected_nvec=env_nvec)
        model = MaskableNovelD_PPO.load(
            str(ckpt),
            env=vec_env,
            device="cpu",
            custom_objects={"policy_class": MaskableActorCriticPolicy},
        )
        model.use_masking = True
        mean_reward, std_reward = evaluate_mean_return(
            model,
            vec_env,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
        )
        print(f"{ckpt.name}: ep_rew_mean_masked={mean_reward:.3f} (std={std_reward:.3f})")

    vec_env.close()


if __name__ == "__main__":
    main()
