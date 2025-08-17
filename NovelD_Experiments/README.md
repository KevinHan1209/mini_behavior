# NovelD Hyperparameter Ablation Experiments

**Updated Version** - This version includes the normalized advantages fix in NovelD.

This directory contains a complete pipeline for running NovelD hyperparameter ablation experiments on virtual machines.

## Overview

The pipeline systematically tests different values for key NovelD hyperparameters while keeping others at default values:

**Default Hyperparameters:**
- alpha: 0.5
- update_proportion: 0.25
- int_gamma: 0.99
- ent_coef: 0.01

**Ablated Parameters:**
- **alpha**: [0, 0.25, 0.5, 1]
- **update_proportion**: [0.25, 0.5, 1]
- **int_gamma**: [0.9, 0.95, 0.98, 0.99]
- **ent_coef**: [0, 0.01, 0.02, 0.05, 0.1, 0.2]

Total: 17 experiments

## Quick Start

### Running All Experiments
```bash
cd NovelD_Experiments
python run_all_experiments.py
```

This will:
1. Generate all experiment configurations
2. Run each experiment sequentially
3. Analyze results and calculate JS divergences
4. Generate a summary report in `experiment_results.md`

### Running Specific Parameter Ablations
To run experiments for a specific parameter only:
```bash
python run_all_experiments.py --subset alpha
```

### Quick Testing
To test the pipeline with reduced timesteps (100k instead of 2.5M):
```bash
python run_all_experiments.py --quick-test
```

## Pipeline Components

### 1. Configuration Generator (`generate_configs.py`)
Generates JSON configuration files for all experiments:
```bash
python generate_configs.py
```

Output:
- `configs/all_experiments.json` - All configurations
- `configs/exp_*.json` - Individual experiment configs
- `configs/experiment_summary.json` - Summary of ablations

### 2. Single Experiment Runner (`run_single_experiment.py`)
Runs a single experiment with specified hyperparameters:
```bash
python run_single_experiment.py configs/exp_000_alpha_0.json
```

Output in `results/exp_*/`:
- `experiment_config.json` - Configuration used
- `checkpoints/` - Model checkpoints
- `checkpoints/activity_logs/` - CSV activity logs
- `SUCCESS` - Marker file if completed successfully

**WandB Integration:** Each experiment run is automatically logged to WandB with:
- Project name: `NovelD_Hyperparameter_Ablation`
- Run name: matches the experiment name (e.g., `exp_000_alpha_0`)
- Full hyperparameter configuration logged

### 3. Results Analyzer (`analyze_results.py`)
Analyzes all experiment results and calculates JS divergences:
```bash
python analyze_results.py
```

Generates `experiment_results.md` with:
- Summary statistics
- Results grouped by parameter
- Best configurations
- Detailed JS divergences by checkpoint

### 4. Master Script (`run_all_experiments.py`)
Orchestrates the entire pipeline:
```bash
python run_all_experiments.py [options]

Options:
  --skip-generation    Skip config generation if already exists
  --skip-experiments   Only analyze existing results
  --subset PARAM       Only run experiments for specific parameter
  --quick-test         Run with reduced timesteps for testing
```

## Output Structure

```
NovelD_Experiments/
├── configs/                    # Generated configurations
│   ├── all_experiments.json
│   ├── experiment_summary.json
│   └── exp_*.json
├── results/                    # Experiment results
│   └── exp_*/
│       ├── experiment_config.json
│       ├── checkpoints/
│       │   ├── checkpoint_*.pt
│       │   └── activity_logs/
│       │       └── checkpoint_*_activity.csv
│       └── SUCCESS
└── experiment_results.md       # Final summary report
```

## Running on Virtual Machines

1. SSH into your VM:
```bash
ssh user@vm-hostname
```

2. Navigate to the repository:
```bash
cd /path/to/mini_behavior
```

3. Install dependencies if needed:
```bash
pip install -r requirements.txt
```

4. Run experiments:
```bash
cd NovelD_Experiments
python run_all_experiments.py
```

5. Monitor progress:
```bash
# In another terminal
tail -f experiment_results.md
```

## Interpreting Results

The `experiment_results.md` file contains:

1. **Summary**: Total experiments and success/failure counts
2. **Results by Parameter**: Tables showing JS divergence for each parameter value
3. **Best Configuration**: The hyperparameter combination with lowest JS divergence
4. **Detailed Results**: Full JS divergence progression for each experiment

Lower JS divergence indicates better alignment with baby behavior data.

## Troubleshooting

### Out of Memory
Reduce `num_envs` in `generate_configs.py` DEFAULTS

### Experiments Failing
Check individual experiment logs in `results/exp_*/`

### Missing Pickle File
Ensure `post_processing/averaged_state_distributions.pkl` exists

### CUDA Issues
The pipeline automatically falls back to CPU if CUDA is unavailable

## Resuming Failed Experiments

To re-run only failed experiments:
1. Delete the `results/exp_*/` directories for failed experiments
2. Run with `--skip-generation` flag:
```bash
python run_all_experiments.py --skip-generation
```