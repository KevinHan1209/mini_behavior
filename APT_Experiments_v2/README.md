# APT_Experiments_v2

Clean, organized ablation studies for APT-PPO algorithm.

## Experiment Setup

### Default Configuration
- **k**: 12 (number of nearest neighbors)
- **ent_coef**: 0.01 (entropy coefficient)
- All other hyperparameters are held constant

### Ablation Studies

#### k-parameter Ablation
Testing different values of k while keeping ent_coef=0.01:
- `exp_001_k_5`: k=5
- `exp_002_k_25`: k=25
- `exp_003_k_50`: k=50
- `exp_004_k_100`: k=100

#### Entropy Coefficient Ablation
Testing different entropy coefficients while keeping k=12:
- `exp_005_ent_coef_0`: ent_coef=0 (no entropy regularization)
- `exp_006_ent_coef_0.005`: ent_coef=0.005
- `exp_007_ent_coef_0.02`: ent_coef=0.02
- `exp_008_ent_coef_0.05`: ent_coef=0.05
- `exp_009_ent_coef_0.1`: ent_coef=0.1

## Running Experiments

To run a single experiment:
```bash
python run_single_experiment.py configs/exp_000_default.json
```

## Features

### Automatic Post-Processing
After each experiment completes:
1. **Checkpoint Visualizations**: Automatically generates activity heatmaps, exploration metrics, and evolution plots
2. **Intrinsic Rewards Logging**: Saves detailed intrinsic reward progression to CSV

### Output Structure
```
APT_Experiments_v2/
├── configs/           # Experiment configurations
└── results/
    └── exp_XXX_name/
        ├── experiment_config.json    # Copy of config used
        ├── intrinsic_rewards_log.csv # Intrinsic rewards over training
        ├── SUCCESS                    # Marker file if successful
        ├── checkpoints/              # Model checkpoints and activity logs
        │   ├── checkpoint_*.pt
        │   └── activity_logs/
        │       └── checkpoint_*_activity.csv
        └── visualizations/           # Auto-generated visualizations
            ├── activity_heatmaps.png
            ├── exploration_metrics.png
            ├── object_state_breakdown.png
            ├── activity_evolution.png
            └── checkpoint_analysis_report.txt
```

### Wandb Integration
- **Project**: APT_PPO_V2
- **Run Names**: Match experiment names exactly
- All metrics are logged to wandb for easy comparison

## Hyperparameters

### Common Parameters (all experiments)
- `total_timesteps`: 2,500,000
- `learning_rate`: 0.0001
- `num_envs`: 8
- `num_steps`: 125
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `num_minibatches`: 4
- `update_epochs`: 4
- `clip_coef`: 0.1
- `vf_coef`: 0.5
- `max_grad_norm`: 0.5
- `int_coef`: 1.0 (intrinsic reward coefficient)
- `ext_coef`: 0.0 (extrinsic reward coefficient - purely intrinsic)
- `int_gamma`: 0.99
- `c`: 1
- `replay_buffer_size`: 1,000,000
- `apt_batch_size`: 1024
- `aggregation_method`: "mean"

## Environment
- **Environment**: MiniGrid-MultiToy-8x8-N2-v0
- **Room Size**: 8x8
- **Max Steps**: 1000
- **Action Space**: [15, 15, 5] (left arm, right arm, locomotion)

## Analysis
Compare experiments by:
1. Exploration coverage (% of object-state pairs discovered)
2. Activity entropy (diversity of interactions)
3. Intrinsic reward progression
4. Final performance metrics

Use the auto-generated visualizations and wandb dashboard for comprehensive analysis.