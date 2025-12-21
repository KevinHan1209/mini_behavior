import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
import mini_behavior.envs  # Preload env registrations before runner imports
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from SB3_Extrinsic_Experiments.eval_checkpoints import checkpoint_metadata
from SB3_Extrinsic_Experiments.run_experiment import (
    flatten_action_mask,
    make_vec_env,
    unwrap_base_env,
    validate_action_space,
)
from mini_behavior.register import env_list, register
from sb3_maskable_noveld import MaskableNovelD_PPO

import minigrid  # Ensure dependency is present for the action masking envs

ENV_ID = "MiniGrid-MultiToy-8x8-N2-v0"
ENV_KWARGS = {
    "room_size": 8,
    "max_steps": 100,
    "extrinsic_rewards": {},
}

# Ensure the env is registered for gymnasium.make
if ENV_ID not in env_list:
    register(id=ENV_ID, entry_point="mini_behavior.envs:MultiToyEnv", kwargs=ENV_KWARGS)


def test_action_space_and_masks_match():
    vec_env = make_vec_env(ENV_ID, ENV_KWARGS, seed=123, num_envs=1)
    try:
        validate_action_space(vec_env)
        env_nvec = np.asarray(vec_env.action_space.nvec, dtype=np.int64)
        base_env = unwrap_base_env(vec_env.envs[0])
        expected_nvec = np.asarray(
            [
                len(base_env.manipulation_actions),
                len(base_env.manipulation_actions),
                len(base_env.locomotion_actions),
            ],
            dtype=np.int64,
        )
        assert np.array_equal(env_nvec, expected_nvec)
        assert np.array_equal(env_nvec, np.asarray(base_env.action_space.nvec, dtype=np.int64))

        masks = base_env.get_action_mask()
        assert set(masks.keys()) == {"left", "right", "loco"}
        assert masks["left"].shape[0] == env_nvec[0]
        assert masks["right"].shape[0] == env_nvec[1]
        assert masks["loco"].shape[0] == env_nvec[2]

        noop_idx = base_env.manipulation_actions.noop
        assert masks["left"][noop_idx] and masks["right"][noop_idx]

        flat_mask = flatten_action_mask(masks, vec_env.action_space)
        assert flat_mask.dtype == np.bool_
        assert flat_mask.shape[0] == int(env_nvec.sum())
    finally:
        vec_env.close()


def test_model_is_maskable_with_matching_action_space():
    vec_env = make_vec_env(ENV_ID, ENV_KWARGS, seed=456, num_envs=2)
    try:
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
        assert model.policy_class is MaskableActorCriticPolicy
        assert isinstance(model.policy, MaskableActorCriticPolicy)
        assert model.use_masking
        assert np.array_equal(np.asarray(model.action_space.nvec), np.asarray(vec_env.action_space.nvec))
        model_nvec = np.asarray(getattr(model.action_space, "nvec", None))
        assert np.array_equal(model_nvec, validate_action_space(vec_env))
    finally:
        vec_env.close()


def test_checkpoint_saves_maskable_policy_and_nvec(tmp_path):
    vec_env = make_vec_env(ENV_ID, ENV_KWARGS, seed=789, num_envs=1)
    ckpt_path = tmp_path / "maskable_test"
    try:
        validate_action_space(vec_env)
        model = MaskableNovelD_PPO(
            policy=MaskableActorCriticPolicy,
            env=vec_env,
            learning_rate=1e-3,
            n_steps=4,
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
        model.save(str(ckpt_path))
        ckpt_zip = ckpt_path.with_suffix(".zip")
        assert ckpt_zip.exists()
        nvec, policy_text = checkpoint_metadata(ckpt_zip)
        assert "MaskableActorCriticPolicy" in policy_text
        assert np.array_equal(nvec, np.asarray(vec_env.action_space.nvec, dtype=np.int64))
    finally:
        vec_env.close()
