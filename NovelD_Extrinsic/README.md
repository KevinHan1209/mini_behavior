# NovelD Extrinsic Reward Experiments

This directory contains experiments for ablating extrinsic reward features in the NovelD algorithm.

## Overview

These experiments test the impact of different types of extrinsic rewards on the NovelD algorithm's performance. The experiments use the best hyperparameters discovered from NovelD experiment 006:
- `alpha`: 0.5
- `update_proportion`: 1
- `int_gamma`: 0.99
- `ent_coef`: 0.01

## Extrinsic Reward Categories

Based on the planning document, we test three categories of extrinsic rewards:

1. **Noise/Sound Rewards**: Rewards for any sound produced in the environment
2. **Object Interaction Rewards**: Rewards for actions that cause object interactions (put_in, take_out, brush, assemble, disassemble, open, close, toggle, noise_toggle)
3. **Location Change Rewards**: Rewards for actions that change object locations (kick, push, walk with object, pickup, drop, throw)

## Experiments

The experiments ablate different combinations of these reward types:

1. **Baseline**: No extrinsic rewards (pure intrinsic motivation)
2. **Individual Rewards**: Each reward type alone
3. **Paired Combinations**: All pairs of reward types
4. **All Rewards**: All three reward types combined
5. **Magnitude Sensitivity**: Different reward magnitudes (0.01, 0.1, 1.0)
6. **Mixed Coefficients**: Different balances between intrinsic and extrinsic rewards

## Running Experiments

### Generate Configurations
```bash
python generate_configs.py
```

### Run Single Experiment
```bash
python run_single_experiment.py configs/exp_000_baseline_no_extrinsic.json
```

### Run All Experiments
```bash
python run_all_experiments.py
```

### Analyze Results
```bash
python analyze_results.py
```

## Directory Structure

```
NovelD_Extrinsic/
├── configs/              # Experiment configuration files
├── results/              # Experiment results and checkpoints
├── analysis/             # Analysis outputs and visualizations
├── generate_configs.py   # Generate experiment configurations
├── run_single_experiment.py  # Run a single experiment
├── run_all_experiments.py    # Run all experiments sequentially
├── analyze_results.py    # Analyze and visualize results
└── README.md            # This file
```

## Results

Results for each experiment are saved in the `results/` directory, including:
- Checkpoints at regular intervals
- Activity logs tracking agent behavior
- Experiment configuration
- Success/failure status

The analysis script generates:
- Performance comparisons by reward category
- Heatmaps showing reward combination effects
- Comprehensive analysis report