# SB3 Vanilla PPO Experiments

This directory contains experiments for testing vanilla SB3 PPO with **pure extrinsic rewards only**. This serves as a baseline comparison for NovelD experiments in `SB3_Extrinsic_Experiments/`.

## Overview

Three experiments are configured, each testing a different extrinsic reward signal:

1. **Noise Only** - Rewards only for noise state changes (e.g., rattles making noise)
2. **Location Only** - Rewards only for location changes (e.g., kicking, pushing, pulling objects)
3. **Interaction Only** - Rewards only for interactions (e.g., drop-in, take-out, toggle)

All experiments use:
- **Algorithm**: Vanilla SB3 PPO (no intrinsic rewards)
- **Reward value**: 0.1 for the active reward type

## Directory Structure

```
SB3_PPO_Experiments/
├── configs/
│   ├── noise_only.json       # Noise-only experiment config
│   ├── location_only.json    # Location-only experiment config
│   └── interaction_only.json # Interaction-only experiment config
├── run_experiment.py         # Script to run a single experiment
├── run_all_experiments.py    # Script to run all experiments
└── README.md                 # This file
```

## Usage

### Prerequisites

Make sure you have stable-baselines3 installed:

```bash
pip install stable-baselines3==1.6.2
```

### Run a Single Experiment

```bash
# From the mini_behavior root directory
python SB3_PPO_Experiments/run_experiment.py \
    SB3_PPO_Experiments/configs/noise_only.json \
    --output-dir SB3_PPO_Experiments/results \
    --seed 1
```

### Run All Three Experiments

```bash
# From the mini_behavior root directory
python SB3_PPO_Experiments/run_all_experiments.py \
    --output-dir SB3_PPO_Experiments/results \
    --seed 1
```

Or run specific experiments:

```bash
python SB3_PPO_Experiments/run_all_experiments.py \
    --experiments noise location \
    --output-dir SB3_PPO_Experiments/results
```

### Make Scripts Executable (Optional)

```bash
chmod +x SB3_PPO_Experiments/run_experiment.py
chmod +x SB3_PPO_Experiments/run_all_experiments.py
```

## Configuration Files

Each config file contains:

- **name**: Experiment name
- **description**: Brief description
- **extrinsic_rewards**: Dict specifying which reward types are active
  - `noise`: Reward for noise state changes
  - `location_change`: Reward for object location changes
  - `interaction`: Reward for object interactions
- **hyperparameters**: Standard SB3 PPO hyperparameters
- **env_config**: Environment configuration
- **wandb_config**: Weights & Biases logging configuration

### Example Config (noise_only.json)

```json
{
  "name": "ppo_noise_only",
  "description": "Vanilla SB3 PPO with only noise extrinsic rewards",
  "extrinsic_rewards": {
    "noise": 0.1
  },
  "hyperparameters": {
    "total_timesteps": 2500000,
    "num_envs": 8,
    "n_steps": 125,
    "batch_size": 250,
    "n_epochs": 4,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  },
  ...
}
```

## Output

Results for each experiment are saved to:

```
results/
├── ppo_noise_only/
│   ├── experiment_config.json  # Copy of config used
│   ├── checkpoints/            # Model checkpoints every 500k steps
│   ├── logs/                   # Tensorboard logs
│   ├── final_model.zip         # Final trained model
│   └── SUCCESS                 # Marker file if completed
├── ppo_location_only/
│   └── ...
└── ppo_interaction_only/
    └── ...
```

## Weights & Biases Tracking

If `wandb` is installed, the following metrics are automatically logged:

- **Rewards**:
  - `rollout/ep_rew_mean`: Mean episode reward
  - `rollout/ep_len_mean`: Mean episode length
- **Policy Metrics**:
  - `train/entropy_loss`: Policy entropy
  - `train/policy_gradient_loss`: Policy loss
  - `train/value_loss`: Value function loss
- **Action Distribution**:
  - `action_dist/branch_X/aY`: Frequency of each action per branch
- **Training Info**:
  - `train/learning_rate`: Current learning rate
  - `train/approx_kl`: Approximate KL divergence
  - `train/clip_fraction`: Fraction of clipped actions

All experiments are logged to the `sb3-ppo-extrinsic` project with appropriate tags.

## Comparison with SB3_Extrinsic_Experiments (NovelD)

| Aspect | SB3_PPO_Experiments | SB3_Extrinsic_Experiments |
|--------|---------------------|---------------------------|
| Algorithm | Vanilla PPO | NovelD_PPO |
| Intrinsic rewards | None | Disabled (int_coef=0.0) |
| NovelD networks | None | None (not created when int_coef=0.0) |
| Purpose | Pure PPO baseline | Baseline using NovelD code path |

**Note**: With `int_coef=0.0`, NovelD_PPO should be functionally equivalent to vanilla PPO since no intrinsic rewards are computed and no RND networks are created. This vanilla PPO experiment provides a cleaner baseline without any NovelD overhead.

## Key Hyperparameters

- **Total timesteps**: 2,500,000
- **Num environments**: 8
- **Steps per rollout**: 125
- **Batch size**: 250
- **Learning rate**: 3e-4
- **Gamma**: 0.99
- **GAE Lambda**: 0.95
- **Clip range**: 0.2
- **Entropy coefficient**: 0.01
- **Value function coefficient**: 0.5
- **Max gradient norm**: 0.5

## Expected Results

With sparse extrinsic rewards and no intrinsic motivation:
- Agent may show initial reward spikes from random exploration
- Rewards may decline over time as the agent fails to maintain exploration
- This baseline demonstrates the need for intrinsic motivation in exploration tasks

## Troubleshooting

### Module not found: stable_baselines3

Make sure you have stable-baselines3 installed:

```bash
pip install stable-baselines3
```

### Wandb not available

If wandb is not installed, the experiments will still run but won't log to W&B. Install with:

```bash
pip install wandb
wandb login
```

### CUDA out of memory

Reduce `num_envs` in the config files if you encounter OOM errors.
