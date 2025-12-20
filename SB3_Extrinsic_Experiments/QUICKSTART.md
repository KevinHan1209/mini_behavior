# Quick Start Guide

## Run All Three Experiments

From the mini_behavior root directory:

```bash
cd /home/kevin/mini_behavior
python SB3_Extrinsic_Experiments/run_all_experiments.py
```

This will run:
1. `sb3_noise_only` - Only noise rewards
2. `sb3_location_only` - Only location change rewards
3. `sb3_interaction_only` - Only interaction rewards

Results will be saved to `results/` directory.

## Run a Single Experiment

```bash
# Noise only
python SB3_Extrinsic_Experiments/run_experiment.py \
    SB3_Extrinsic_Experiments/configs/noise_only.json

# Location only
python SB3_Extrinsic_Experiments/run_experiment.py \
    SB3_Extrinsic_Experiments/configs/location_only.json

# Interaction only
python SB3_Extrinsic_Experiments/run_experiment.py \
    SB3_Extrinsic_Experiments/configs/interaction_only.json
```

## Key Configuration

All experiments are configured with:
- **int_coef = 0.0** (NO intrinsic rewards)
- **ext_coef = 1.0** (YES extrinsic rewards only)
- **Reward value = 0.1** for the active reward type
- **Total timesteps = 2,500,000**
- **8 parallel environments**

## Check Results

After running, check:
- `results/sb3_noise_only/`
- `results/sb3_location_only/`
- `results/sb3_interaction_only/`

Each contains:
- `experiment_config.json` - Config used
- `checkpoints/` - Model checkpoints
- `logs/` - Tensorboard logs
- `final_model.zip` - Final model
- `SUCCESS` - Completion marker

## Weights & Biases

If wandb is installed and configured, metrics will be logged to:
- **Project**: `sb3-noveld-extrinsic`
- **Groups**: `noise_only`, `location_only`, `interaction_only`

## Metrics Tracked

- Episode rewards
- Policy entropy
- Action distributions (per branch)
- Value function loss
- Policy gradient loss
- Learning rate
- Clip fraction
- Approximate KL divergence

## View Tensorboard Logs

```bash
tensorboard --logdir results/
```

Then open http://localhost:6006
