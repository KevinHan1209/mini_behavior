#!/usr/bin/env python3
"""
Generate experiment configurations for APT hyperparameter ablation.
"""

import json
import os
from pathlib import Path

# Default hyperparameters (as per hyperparameters.py)
DEFAULTS = {
    # APT-specific
    "k": 50,
    "int_gamma": 0.99,
    
    # PPO core
    "ent_coef": 0.01,
    
    # Other fixed parameters
    "learning_rate": 0.0001,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 16,
    "total_timesteps": 2500000
}

# Parameters to ablate and their test values
ABLATIONS = {
    "k": [10, 20, 50, 100, 200, 500],
    "int_gamma": [0.9, 0.95, 0.98, 0.99],
    "ent_coef": [0, 0.01, 0.02, 0.05, 0.1, 0.2]
}

def generate_experiment_configs():
    """Generate individual experiment configurations for each hyperparameter ablation."""
    configs = []
    exp_id = 0
    
    # For each parameter to ablate
    for param_name, test_values in ABLATIONS.items():
        # For each test value of this parameter
        for value in test_values:
            # Create config with defaults
            config = DEFAULTS.copy()
            # Override the specific parameter
            config[param_name] = value
            
            # Add metadata
            config["experiment_id"] = f"exp_{exp_id:03d}_{param_name}_{str(value).replace('.', '_')}"
            config["ablated_param"] = param_name
            config["ablated_value"] = value
            
            configs.append(config)
            exp_id += 1
    
    return configs

def save_configs(configs):
    """Save experiment configurations to JSON files."""
    # Create configs directory
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Save individual configs
    for config in configs:
        filename = config_dir / f"{config['experiment_id']}.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Save all configs in one file
    all_configs = {config['experiment_id']: config for config in configs}
    with open(config_dir / "all_experiments.json", 'w') as f:
        json.dump(all_configs, f, indent=2)
    
    # Save experiment summary
    summary = {
        "total_experiments": len(configs),
        "defaults": DEFAULTS,
        "ablations": ABLATIONS,
        "experiments_by_param": {}
    }
    
    for param in ABLATIONS:
        summary["experiments_by_param"][param] = [
            config['experiment_id'] for config in configs 
            if config['ablated_param'] == param
        ]
    
    with open(config_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {len(configs)} experiment configurations")
    print(f"Saved to {config_dir}/")
    
    # Print summary
    print("\nExperiment breakdown:")
    for param, values in ABLATIONS.items():
        print(f"  {param}: {len(values)} experiments")

if __name__ == "__main__":
    configs = generate_experiment_configs()
    save_configs(configs)