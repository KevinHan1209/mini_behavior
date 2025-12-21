#!/usr/bin/env python3
"""
Quick sanity check for GIF rendering with the maskable SB3 pipeline.

Creates a tiny model, runs the GifLogger rollout once, and writes the GIF to a temp file.
Use to debug rendering without running a full experiment.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Make repo imports available
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio
from sb3_maskable_noveld import MaskableNovelD_PPO

from SB3_Extrinsic_Experiments.run_experiment import GifLoggerCallback, make_vec_env, validate_action_space
from mini_behavior.register import env_list, register


def main():
    env_id = "MiniGrid-MultiToy-8x8-N2-v0"
    env_kwargs = {
        "room_size": 8,
        "max_steps": 200,
        "extrinsic_rewards": {"interaction": 0.1},
    }
    seed = 42

    if env_id not in env_list:
        register(id=env_id, entry_point="mini_behavior.envs:MultiToyEnv", kwargs=env_kwargs)

    vec_env = make_vec_env(env_id, env_kwargs, seed=seed, num_envs=1)
    validate_action_space(vec_env)

    model = MaskableNovelD_PPO(
        policy=MaskableActorCriticPolicy,
        env=vec_env,
        learning_rate=1e-3,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        int_coef=0.0,
        ext_coef=1.0,
        int_gamma=0.99,
        update_proportion=0.25,
        use_masking=True,
        verbose=0,
    )

    gif_callback = GifLoggerCallback(
        env_id=env_id,
        env_kwargs=env_kwargs,
        log_interval=1,
        gif_steps=50,
        seed=seed,
        deterministic=False,
        verbose=1,
    )
    gif_callback.model = model

    rollout = gif_callback._rollout_frames()
    if rollout is None:
        raise RuntimeError("GifLogger rollout returned None")
    frames, total_reward, stats = rollout
    if not frames:
        raise RuntimeError("GifLogger produced no frames")

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        imageio.mimsave(tmp.name, frames, fps=10)
        out_path = tmp.name

    vec_env.close()
    print(f"Saved GIF to {out_path} (total_reward={total_reward:.3f}, frames={len(frames)})")
    print("Stats summary:")
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"  {key}: {{...}}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
