#!/usr/bin/env python3
"""
Generate experiment configurations for NovelD hyperparameter ablation
"""
import json
import os

# Default hyperparameters
DEFAULTS = {
    'alpha': 0.5,
    'update_proportion': 0.25,
    'int_gamma': 0.99,
    'ent_coef': 0.01,
    # Other fixed hyperparameters
    'learning_rate': 3e-4,
    'num_envs': 8,
    'num_steps': 125,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_minibatches': 4,
    'update_epochs': 4,
    'clip_coef': 0.2,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'int_coef': 1.0,
    'ext_coef': 0.0,
    'total_timesteps': 2500000
}

# Hyperparameters to ablate
ABLATIONS = {
    'alpha': [0, 0.25, 0.5, 1],
    'update_proportion': [0.25, 0.5, 1],
    'int_gamma': [0.9, 0.95, 0.98, 0.99],
    'ent_coef': [0, 0.01, 0.02, 0.05, 0.1, 0.2]
}

def generate_experiment_configs():
    """Generate all experiment configurations"""
    configs = []
    experiment_id = 0
    
    # For each hyperparameter to ablate
    for param_name, param_values in ABLATIONS.items():
        # For each value of this hyperparameter
        for value in param_values:
            # Create config with this value and defaults for others
            config = DEFAULTS.copy()
            config[param_name] = value
            
            # Create experiment name
            exp_name = f"exp_{experiment_id:03d}_{param_name}_{value}"
            exp_name = exp_name.replace('.', '_')  # Replace dots for filesystem compatibility
            
            experiment = {
                'id': experiment_id,
                'name': exp_name,
                'ablated_param': param_name,
                'ablated_value': value,
                'hyperparameters': config
            }
            
            configs.append(experiment)
            experiment_id += 1
    
    return configs

def save_configs(configs, output_dir='configs'):
    """Save configurations to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all configs in one file
    all_configs_path = os.path.join(output_dir, 'all_experiments.json')
    with open(all_configs_path, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Save individual config files
    for config in configs:
        config_path = os.path.join(output_dir, f"{config['name']}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create summary
    summary = {
        'total_experiments': len(configs),
        'ablated_parameters': list(ABLATIONS.keys()),
        'defaults': DEFAULTS,
        'ablations': ABLATIONS
    }
    
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {len(configs)} experiment configurations")
    print(f"Saved to {output_dir}/")
    
    # Print summary
    print("\nExperiment Summary:")
    for param, values in ABLATIONS.items():
        count = len(values)
        print(f"  {param}: {count} experiments")
    print(f"\nTotal: {len(configs)} experiments")
    
    return all_configs_path

if __name__ == "__main__":
    configs = generate_experiment_configs()
    save_configs(configs)