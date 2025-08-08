#!/usr/bin/env python3
"""
Generate experiment configurations for NovelD with scaled extrinsic rewards
Ablation study on intrinsic:extrinsic reward ratios
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
    'total_timesteps': 2500000
}

# Scaled values based on max possible per step analysis:
# - Location changes: max 3 per step (2 arms + locomotion)
# - Interactions: max 2 per step (2 arms only)
# - Noise: max ~3 per step (multiple objects)
SCALED_VALUES = {
    'location_change': 0.0276,  # 0.0829/3
    'interaction': 0.0415,      # 0.0829/2
    'noise': 0.0276             # 0.0829/3
}

# Intrinsic:Extrinsic ratios for ablation
# Ratio = intrinsic/extrinsic (how many times more important is intrinsic vs extrinsic)
ABLATION_RATIOS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

def generate_ablation_configs():
    """Generate configurations for ratio ablation study - ONE category at a time"""
    configs = []
    
    # Categories to ablate individually
    categories = [
        ('location', 'location_change', 0.0276, 100),  # 100 series
        ('interaction', 'interaction', 0.0415, 200),    # 200 series
        ('noise', 'noise', 0.0276, 300)                 # 300 series
    ]
    
    # For each category, test all ratios
    for cat_name, cat_key, cat_value, series_start in categories:
        for i, ratio in enumerate(ABLATION_RATIOS):
            # Using Option 1: int_coef=1.0, ext_coef=1/ratio
            ext_coef = 1.0 / ratio if ratio > 0 else 0.0
            
            # Create rewards dict with only this category active
            rewards = {
                'location_change': cat_value if cat_key == 'location_change' else 0.0,
                'interaction': cat_value if cat_key == 'interaction' else 0.0,
                'noise': cat_value if cat_key == 'noise' else 0.0
            }
            
            config = {
                'id': series_start + i,
                'name': f'{cat_name}_ratio_{str(ratio).replace(".", "_")}',
                'extrinsic_rewards': rewards,
                'int_coef': 1.0,
                'ext_coef': ext_coef,
                'description': f'{cat_name.capitalize()} only, I:E ratio = {ratio}:1 (ext_coef={ext_coef:.2f})'
            }
            configs.append(config)
    
    # Add pure intrinsic baseline (experiment 001 already exists but include for reference)
    configs.append({
        'id': 1,
        'name': 'pure_intrinsic',
        'extrinsic_rewards': {},
        'int_coef': 1.0,
        'ext_coef': 0.0,
        'description': 'Pure intrinsic rewards (NovelD only)'
    })
    
    return configs

# Generate all experiment configurations
EXPERIMENT_CONFIGS = generate_ablation_configs()

def generate_experiment_configs():
    """Generate experiment configurations"""
    configs = []
    
    for exp_config in EXPERIMENT_CONFIGS:
        # Start with best hyperparameters
        hyperparams = BEST_HYPERPARAMS.copy()
        
        # Override with experiment-specific coefficients
        hyperparams['int_coef'] = exp_config['int_coef']
        hyperparams['ext_coef'] = exp_config['ext_coef']
        
        experiment = {
            'id': exp_config['id'],
            'name': f"exp_{exp_config['id']:03d}_{exp_config['name']}",
            'description': exp_config['description'],
            'extrinsic_rewards': exp_config['extrinsic_rewards'],
            'hyperparameters': hyperparams
        }
        
        configs.append(experiment)
    
    return configs

def save_configs(configs, output_dir='configs'):
    """Save configuration to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all configs in one file
    all_configs_path = os.path.join(output_dir, 'all_ablation_experiments.json')
    with open(all_configs_path, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Save individual config files
    for config in configs:
        config_path = os.path.join(output_dir, f"{config['name']}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create summary
    summary = {
        'ablation_study': 'Intrinsic:Extrinsic Reward Ratio',
        'ratios_tested': ABLATION_RATIOS,
        'target_mean_reward': 0.0829,
        'scaling_rationale': {
            'location_change': 'Max 3 per step (2 arms + locomotion), scaled to 0.0829/3 = 0.0276',
            'interaction': 'Max 2 per step (2 arms only), scaled to 0.0829/2 = 0.0415',
            'noise': 'Max ~3 per step (multiple objects), scaled to 0.0829/3 = 0.0276'
        },
        'coefficient_strategy': 'Option 1: int_coef=1.0, ext_coef=1/ratio',
        'base_hyperparameters': BEST_HYPERPARAMS,
        'total_experiments': len(configs),
        'experiments': [
            {
                'name': config['name'],
                'description': config['description'],
                'extrinsic_rewards': config['extrinsic_rewards'],
                'int_coef': config['hyperparameters']['int_coef'],
                'ext_coef': config['hyperparameters']['ext_coef'],
                'ratio': config['hyperparameters']['int_coef'] / config['hyperparameters']['ext_coef'] if config['hyperparameters']['ext_coef'] > 0 else 'inf'
            }
            for config in configs
        ]
    }
    
    summary_path = os.path.join(output_dir, 'ablation_experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {len(configs)} ablation experiment configurations")
    print(f"Saved to {output_dir}/")
    
    # Print summary
    print("\n" + "="*90)
    print("INTRINSIC:EXTRINSIC RATIO ABLATION STUDY")
    print("="*90)
    print(f"Target mean reward per step: 0.0829")
    print(f"\nReward Scaling:")
    print(f"  Location change: 0.0276 (max 3/step)")
    print(f"  Interaction:     0.0415 (max 2/step)")
    print(f"  Noise:           0.0276 (max ~3/step)")
    print(f"\nCoefficient Strategy: int_coef=1.0, ext_coef=1/ratio")
    print(f"\n{'Experiment':<25} {'Ratio':<15} {'int_coef':<10} {'ext_coef':<10} {'Description'}")
    print("-"*90)
    
    for config in configs:
        name = config['name'].replace('exp_', '').replace(f"{config['id']}_", '')
        int_c = config['hyperparameters']['int_coef']
        ext_c = config['hyperparameters']['ext_coef']
        ratio_str = f"{int_c/ext_c:.1f}:1" if ext_c > 0 else "inf:1"
        print(f"{name:<25} {ratio_str:<15} {int_c:<10.2f} {ext_c:<10.2f} {config['description'][:35]}")
    
    return all_configs_path

def create_quick_test_config():
    """Create a quick test configuration with shorter timesteps"""
    test_config = {
        'id': 99,
        'name': 'exp_099_quick_test_ratio_1',
        'description': 'Quick test with 1:1 ratio (500k timesteps)',
        'extrinsic_rewards': SCALED_VALUES,
        'hyperparameters': {
            **BEST_HYPERPARAMS,
            'int_coef': 1.0,
            'ext_coef': 1.0,
            'total_timesteps': 500000  # Reduced for quick testing
        }
    }
    
    # Save test config
    os.makedirs('configs', exist_ok=True)
    test_config_path = os.path.join('configs', f"{test_config['name']}.json")
    with open(test_config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"\nCreated quick test config: {test_config_path}")
    print(f"  Ratio: 1:1 (int_coef=1.0, ext_coef=1.0)")
    print(f"  Rewards: location={SCALED_VALUES['location_change']:.4f}, "
          f"interaction={SCALED_VALUES['interaction']:.4f}, "
          f"noise={SCALED_VALUES['noise']:.4f}")
    
    return test_config_path

if __name__ == "__main__":
    # Generate and save configs
    configs = generate_experiment_configs()
    save_configs(configs)
    
    # Create quick test config
    test_config_path = create_quick_test_config()
    
    print("\n" + "="*90)
    print("NEXT STEPS:")
    print("="*90)
    print("1. Run quick test to validate scaling:")
    print(f"   python run_single_experiment.py {test_config_path}")
    print("\n2. Run specific ratio experiment:")
    print("   python run_single_experiment.py configs/exp_103_ratio_1_0.json")
    print("\n3. Run all ablation experiments:")
    print("   python run_all_experiments.py configs/")