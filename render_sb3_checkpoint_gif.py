#!/usr/bin/env python3
"""
Render a short rollout from an SB3 checkpoint to a GIF.

Defaults target the interaction-only checkpoint from SB3_Extrinsic_Experiments,
but paths and reward weights can be overridden with CLI flags.
"""
import argparse
from pathlib import Path

import gym
import imageio
import numpy as np
from typing import Optional, Dict

from stable_baselines3 import NovelD_PPO
from env_wrapper import CustomObservationWrapper
from mini_behavior.register import env_list, register


def make_env(room_size: int, max_steps: int, extrinsic_rewards: dict):
    env_id = "MiniGrid-MultiToy-8x8-N2-v0"
    task = "MultiToy"
    env_kwargs = {
        "room_size": room_size,
        "max_steps": max_steps,
        "extrinsic_rewards": extrinsic_rewards,
    }
    if env_id not in env_list:
        register(
            id=env_id,
            entry_point=f"mini_behavior.envs:{task}Env",
            kwargs=env_kwargs,
        )
    env = gym.make(env_id, **env_kwargs)
    return CustomObservationWrapper(env)


def rollout_to_gif(
    checkpoint: Path,
    gif_path: Path,
    steps: int = 200,
    room_size: int = 8,
    max_steps: int = 1000,
    extrinsic_rewards: Optional[Dict[str, float]] = None,
    deterministic: bool = True,
):
    extrinsic_rewards = extrinsic_rewards or {"interaction": 0.1}
    env = make_env(room_size, max_steps, extrinsic_rewards)
    base_env = env.env if hasattr(env, "env") else env

    model = NovelD_PPO.load(str(checkpoint))

    frames = []
    obs = env.reset()
    done = False
    total_reward = 0.0
    for _ in range(steps):
        frame = base_env.render(mode="rgb_array")
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, _info = env.step(action)
        total_reward += float(reward)
        if done:
            break

    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Saved GIF to {gif_path} | Episode reward: {total_reward:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Render SB3 checkpoint to GIF")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "SB3_Extrinsic_Experiments/results/sb3_interaction_only/checkpoints/checkpoint_500000_steps.zip"
        ),
        help="Path to SB3 checkpoint (.zip)",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=Path("interaction_only_500k.gif"),
        help="Output GIF path",
    )
    parser.add_argument("--steps", type=int, default=200, help="Max steps to render")
    parser.add_argument("--room-size", type=int, default=8, help="Env room size")
    parser.add_argument("--max-steps", type=int, default=1000, help="Env max steps")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions",
    )
    parser.add_argument(
        "--reward-type",
        choices=["interaction", "location_change", "noise"],
        default="interaction",
        help="Which extrinsic reward weight to enable (0.1)",
    )

    args = parser.parse_args()
    extrinsic_rewards = {args.reward_type: 0.1}

    rollout_to_gif(
        checkpoint=args.checkpoint,
        gif_path=args.gif,
        steps=args.steps,
        room_size=args.room_size,
        max_steps=args.max_steps,
        extrinsic_rewards=extrinsic_rewards,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
