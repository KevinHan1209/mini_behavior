#!/usr/bin/env python3
"""Roll out a maskable checkpoint without updating the policy."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Make repository imports available
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sb3_maskable_noveld import MaskableNovelD_PPO
from SB3_Extrinsic_Experiments.eval_checkpoints import checkpoint_metadata
from SB3_Extrinsic_Experiments.run_experiment import (
    EvalEpRewMeanCallback,
    EpisodeReturnLogger,
    WandbOutputFormat,
    mask_fn,
    make_vec_env,
    validate_action_space,
)
from mini_behavior.register import env_list, register

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("checkpoint_*_steps.zip"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
    return candidates[-1]


def load_experiment_config(result_path: Path) -> dict:
    config_path = result_path / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.json at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_checkpoint(checkpoint_path: Path, env_nvec: np.ndarray) -> None:
    ckpt_nvec, policy_text = checkpoint_metadata(checkpoint_path)
    if "MaskableActorCriticPolicy" not in policy_text:
        raise ValueError(f"{checkpoint_path.name} is not maskable. Retrain/save with MaskableActorCriticPolicy.")
    if ckpt_nvec.shape != env_nvec.shape or not np.array_equal(ckpt_nvec, env_nvec):
        raise ValueError(
            f"{checkpoint_path.name} uses nvec {ckpt_nvec.tolist()}, expected env nvec {env_nvec.tolist()}."
        )


def patch_no_train(model: MaskableNovelD_PPO):
    """Replace model.train with a no-op so learn() only collects rollouts."""
    original_train = model.train

    def _no_train(self):
        return None

    model.train = _no_train.__get__(model, MaskableNovelD_PPO)
    return original_train


def main():
    parser = argparse.ArgumentParser(description="Roll out a checkpoint without updating the policy.")
    parser.add_argument("--result-path", required=True, help="Path to experiment result directory.")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint .zip to load (defaults to latest).")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Number of rollout timesteps (no updates).")
    parser.add_argument("--seed", type=int, default=1, help="Base seed for envs (matches training seeds when unchanged).")
    parser.add_argument("--num-envs", type=int, default=None, help="Override num_envs (defaults to value from config).")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions during rollout.")
    parser.add_argument("--wandb", action="store_true", help="Log rollout metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default=None, help="W&B project (defaults to config or 'sb3-noveld-extrinsic').")
    parser.add_argument("--wandb-group", default=None, help="W&B group (defaults to config or 'no_update_rollout').")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (defaults to checkpoint stem).")
    args = parser.parse_args()

    result_path = Path(args.result_path).resolve()
    checkpoint_dir = result_path / "checkpoints"
    config = load_experiment_config(result_path)
    env_config = config["env_config"]
    env_id = env_config["env_id"]
    env_kwargs = {
        "room_size": env_config["room_size"],
        "max_steps": env_config["max_steps"],
        "extrinsic_rewards": config["extrinsic_rewards"],
    }
    if env_id not in env_list:
        register(id=env_id, entry_point=f'mini_behavior.envs:{env_config.get("task", "MultiToy")}Env', kwargs=env_kwargs)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    num_envs = args.num_envs or int(config["hyperparameters"]["num_envs"])
    utils.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vec_env = make_vec_env(env_id, env_kwargs, seed=args.seed, num_envs=num_envs)
    validate_action_space(vec_env)
    env_nvec = np.asarray(vec_env.action_space.nvec, dtype=np.int64)
    for e in vec_env.envs:
        mask_fn(e)
    validate_checkpoint(checkpoint_path, env_nvec)

    eval_env = make_vec_env(env_id, env_kwargs, seed=args.seed, num_envs=num_envs)
    validate_action_space(eval_env)
    for e in eval_env.envs:
        mask_fn(e)

    model: MaskableNovelD_PPO = MaskableNovelD_PPO.load(
        str(checkpoint_path),
        env=vec_env,
        device=device,
        custom_objects={"policy_class": MaskableActorCriticPolicy},
    )
    model.use_masking = True

    logger = utils.configure_logger(
        model.verbose,
        model.tensorboard_log,
        tb_log_name="no_update_rollout",
        reset_num_timesteps=False,
    )
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
    if args.wandb and WANDB_AVAILABLE:
        wandb_cfg = config.get("wandb_config", {})
        project = args.wandb_project or wandb_cfg.get("project", "sb3-noveld-extrinsic")
        group = args.wandb_group or wandb_cfg.get("group", "no_update_rollout")
        name = args.wandb_name or checkpoint_path.stem
        wandb_run = wandb.init(
            project=project,
            group=group,
            name=name,
            tags=["no_update", "rollout"],
            config={
                "checkpoint": str(checkpoint_path),
                "total_timesteps": args.total_timesteps,
                "seed": args.seed,
                "num_envs": num_envs,
                "deterministic": args.deterministic,
                "use_masks": True,
            },
        )
        if not any(isinstance(fmt, WandbOutputFormat) for fmt in logger.output_formats):
            logger.output_formats.append(WandbOutputFormat())
    else:
        # Remove WandbOutputFormat if present; keep this run local
        logger.output_formats = [fmt for fmt in logger.output_formats if not isinstance(fmt, WandbOutputFormat)]
    model.set_logger(logger)

    callbacks = [EpisodeReturnLogger(num_envs=num_envs)]
    callbacks.append(
        EvalEpRewMeanCallback(
            eval_env=eval_env,
            n_eval_episodes=num_envs,
            eval_freq=num_envs * model.n_steps,
            use_masks=True,
            deterministic=args.deterministic,
            verbose=1,
        )
    )
    callback = CallbackList(callbacks)

    original_train = patch_no_train(model)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="no_update_rollout",
            progress_bar=False,
            log_interval=1,
        )
    finally:
        model.train = original_train
        vec_env.close()
        eval_env.close()
        if wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    print("Finished rollout with frozen policy.")


if __name__ == "__main__":
    main()
