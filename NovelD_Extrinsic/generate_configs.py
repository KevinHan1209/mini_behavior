#!/usr/bin/env python3
"""
Generate experiment configuration for NovelD with pure extrinsic rewards
"""
import json
import os

# Best hyperparameters from NovelD exp_006
BEST_HYPERPARAMS = {
    'alpha': 0.5,
    'update_proportion': 1,
    'int_gamma': 0.99,
    'ent_coef': 0.01,
    'learning_rate': 0.0003,
    'num_envs': 8,
    'num_steps': 125,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_minibatches': 4,
    'update_epochs': 4,
    'clip_coef': 0.2,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'int_coef': 0.0,  # No intrinsic rewards
    'ext_coef': 1.0,  # Only extrinsic rewards
    'total_timesteps': 2500000
}

# Single experiment with pure extrinsic rewards
EXTRINSIC_REWARD_CONFIG = {
    'name': 'pure_extrinsic_all_rewards',
    'extrinsic_rewards': {
        'noise': 0.1,
        'interaction': 0.1,
        'location_change': 0.1
    },
    'description': 'Pure extrinsic rewards only (no intrinsic motivation)'
}

def generate_experiment_config():
    """Generate single experiment configuration"""
    
    # Start with best hyperparameters
    hyperparams = BEST_HYPERPARAMS.copy()
    
    experiment = {
        'id': 0,
        'name': f"exp_000_{EXTRINSIC_REWARD_CONFIG['name']}",
        'description': EXTRINSIC_REWARD_CONFIG['description'],
        'extrinsic_rewards': EXTRINSIC_REWARD_CONFIG['extrinsic_rewards'],
        'hyperparameters': hyperparams
    }
    
    return [experiment]

def save_configs(configs, output_dir='configs'):
    """Save configuration to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all configs in one file
    all_configs_path = os.path.join(output_dir, 'all_experiments.json')
    with open(all_configs_path, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Save individual config file
    for config in configs:
        config_path = os.path.join(output_dir, f"{config['name']}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create summary
    summary = {
        'total_experiments': len(configs),
        'base_hyperparameters': BEST_HYPERPARAMS,
        'reward_categories': ['noise', 'interaction', 'location_change'],
        'experiments': [
            {
                'name': config['name'],
                'description': config['description'],
                'extrinsic_rewards': config['extrinsic_rewards'],
                'int_coef': config['hyperparameters']['int_coef'],
                'ext_coef': config['hyperparameters']['ext_coef']
            }
            for config in configs
        ]
    }
    
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {len(configs)} experiment configuration")
    print(f"Saved to {output_dir}/")
    
    # Print summary
    print("\nExperiment Summary:")
    for config in configs:
        rewards = config['extrinsic_rewards']
        reward_str = ', '.join([f"{k}={v}" for k, v in rewards.items()])
        print(f"  {config['name']}:")
        print(f"    Description: {config['description']}")
        print(f"    Rewards: {reward_str}")
        print(f"    int_coef: {config['hyperparameters']['int_coef']}")
        print(f"    ext_coef: {config['hyperparameters']['ext_coef']}")
    
    return all_configs_path

if __name__ == "__main__":
    configs = generate_experiment_config()
    save_configs(configs)