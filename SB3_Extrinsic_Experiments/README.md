# SB3 Extrinsic-Only NovelD Experiments

This directory contains experiments for testing the SB3 implementation of NovelD with **pure extrinsic rewards only** (no intrinsic/curiosity rewards).

## Overview

Three experiments are configured, each testing a different extrinsic reward signal:

1. **Noise Only** - Rewards only for noise state changes (e.g., rattles making noise)
2. **Location Only** - Rewards only for location changes (e.g., kicking, pushing, pulling objects)
3. **Interaction Only** - Rewards only for interactions (e.g., drop-in, take-out, toggle)

All experiments use:
- **Intrinsic coefficient (`int_coef`)**: 0.0 (disabled)
- **Extrinsic coefficient (`ext_coef`)**: 1.0 (enabled)
- **Reward value**: 0.1 for the active reward type

## Directory Structure

```
SB3_Extrinsic_Experiments/
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

Make sure you have the SB3 NovelD implementation installed:

```bash
# Install stable_baselines3 with NovelD support
# (Follow your project's installation instructions)
```

### Run a Single Experiment

```bash
# From the mini_behavior root directory
python SB3_Extrinsic_Experiments/run_experiment.py \
    SB3_Extrinsic_Experiments/configs/noise_only.json \
    --output-dir SB3_Extrinsic_Experiments/results \
    --seed 1
```

### Run All Three Experiments

```bash
# From the mini_behavior root directory
python SB3_Extrinsic_Experiments/run_all_experiments.py \
    --output-dir SB3_Extrinsic_Experiments/results \
    --seed 1
```

Or run specific experiments:

```bash
python SB3_Extrinsic_Experiments/run_all_experiments.py \
    --experiments noise location \
    --output-dir SB3_Extrinsic_Experiments/results
```

### Make Scripts Executable (Optional)

```bash
chmod +x SB3_Extrinsic_Experiments/run_experiment.py
chmod +x SB3_Extrinsic_Experiments/run_all_experiments.py
```

## Configuration Files

Each config file contains:

- **name**: Experiment name
- **description**: Brief description
- **extrinsic_rewards**: Dict specifying which reward types are active
  - `noise`: Reward for noise state changes
  - `location_change`: Reward for object location changes
  - `interaction`: Reward for object interactions
- **hyperparameters**: Standard SB3 PPO hyperparameters plus NovelD-specific params
  - `int_coef`: 0.0 (intrinsic rewards disabled)
  - `ext_coef`: 1.0 (extrinsic rewards enabled)
  - `update_proportion`: 0.25 (proportion of novelty buffer to update)
  - `int_gamma`: 0.99 (intrinsic reward discount, not used when int_coef=0)
- **env_config**: Environment configuration
- **wandb_config**: Weights & Biases logging configuration

### Example Config (noise_only.json)

```json
{
  "name": "sb3_noise_only",
  "description": "SB3 NovelD with only noise extrinsic rewards (no intrinsic)",
  "extrinsic_rewards": {
    "noise": 0.1
  },
  "hyperparameters": {
    "total_timesteps": 2500000,
    "int_coef": 0.0,
    "ext_coef": 1.0,
    ...
  }
}
```

## Output

Results for each experiment are saved to:

```
results/
├── sb3_noise_only/
│   ├── experiment_config.json  # Copy of config used
│   ├── checkpoints/            # Model checkpoints every 500k steps
│   ├── logs/                   # Tensorboard logs
│   ├── final_model.zip         # Final trained model
│   └── SUCCESS                 # Marker file if completed
├── sb3_location_only/
│   └── ...
└── sb3_interaction_only/
    └── ...
```

## Weights & Biases Tracking

If `wandb` is installed, the following metrics are automatically logged:

- **Rewards**:
  - `rollout/ep_rew_mean`: Mean episode reward
  - `step_reward`: Step-by-step rewards
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

All experiments are logged to the `sb3-noveld-extrinsic` project with appropriate tags.

## Comparison with Old NovelD_Extrinsic

This directory differs from the old `NovelD_Extrinsic` directory:

| Aspect | Old NovelD_Extrinsic | New SB3_Extrinsic_Experiments |
|--------|---------------------|-------------------------------|
| Implementation | Custom NovelD_PPO from `algorithms/` | SB3's NovelD_PPO |
| Code Base | Legacy implementation | Modern SB3 implementation from `test/test_NovelD.py` |
| Focus | Multiple extrinsic combinations | Pure extrinsic only (3 types) |
| Config Format | Similar JSON structure | Updated for SB3 parameters |

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

## Troubleshooting

### Module not found: stable_baselines3

Make sure you have the custom SB3 implementation with NovelD support installed. Check your project's installation instructions.

### Wandb not available

If wandb is not installed, the experiments will still run but won't log to W&B. Install with:

```bash
pip install wandb
wandb login
```

### CUDA out of memory

Reduce `num_envs` in the config files if you encounter OOM errors.

## Citation

If you use these experiments, please cite the appropriate papers for NovelD and your modifications.
