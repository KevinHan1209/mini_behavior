#!/usr/bin/env python3
"""
Test script to verify APT experiment setup is working correctly.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_python_version():
    """Check Python version."""
    print("\\n1. Testing Python version...")
    print(f"   Python {sys.version}")
    if sys.version_info < (3, 6):
        print("   ❌ Python 3.6+ required")
        return False
    print("   ✓ Python version OK")
    return True

def test_imports():
    """Test required imports."""
    print("\\n2. Testing required imports...")
    
    modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("gym", "OpenAI Gym"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"   ✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import {display_name}: {e}")
            all_ok = False
    
    return all_ok

def test_cuda():
    """Test CUDA availability."""
    print("\\n3. Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ CUDA version: {torch.version.cuda}")
        else:
            print("   ⚠ CUDA not available (CPU will be used)")
    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")
        return False
    return True

def test_paths():
    """Test required paths exist."""
    print("\\n4. Testing required paths...")
    
    paths = [
        ("../algorithms/APT_PPO.py", "APT_PPO algorithm"),
        ("../algorithms/RND.py", "RND module"),
        ("../post_processing/batch_js_divergence.py", "JS divergence calculator"),
        ("../behavioral_diversity_benchmarks.pkl", "Baseline distributions"),
    ]
    
    all_ok = True
    for path, description in paths:
        full_path = Path(path).resolve()
        if full_path.exists():
            print(f"   ✓ {description}: {path}")
        else:
            print(f"   ❌ Missing {description}: {path}")
            all_ok = False
    
    return all_ok

def test_config_generation():
    """Test config generation."""
    print("\\n5. Testing config generation...")
    
    try:
        from generate_configs import generate_experiment_configs, ABLATIONS
        configs = generate_experiment_configs()
        
        # Count expected configs
        expected = sum(len(values) for values in ABLATIONS.values())
        
        if len(configs) == expected:
            print(f"   ✓ Generated {len(configs)} configurations as expected")
            
            # Check config structure
            sample_config = configs[0]
            required_keys = ['k', 'int_gamma', 'ent_coef', 'experiment_id', 
                           'ablated_param', 'ablated_value']
            
            missing_keys = [key for key in required_keys if key not in sample_config]
            if missing_keys:
                print(f"   ❌ Missing keys in config: {missing_keys}")
                return False
            else:
                print("   ✓ Config structure looks good")
        else:
            print(f"   ❌ Expected {expected} configs, got {len(configs)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to generate configs: {e}")
        return False
    
    return True

def test_apt_import():
    """Test APT_PPO import and instantiation."""
    print("\\n6. Testing APT_PPO import...")
    
    try:
        from algorithms.APT_PPO import APT_PPO
        print("   ✓ APT_PPO imported successfully")
        
        # Try to check class attributes
        if hasattr(APT_PPO, '__init__'):
            print("   ✓ APT_PPO class structure looks good")
        else:
            print("   ❌ APT_PPO class structure issue")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to import APT_PPO: {e}")
        return False
    
    return True

def test_environment():
    """Test environment creation."""
    print("\\n7. Testing environment...")
    
    try:
        import gym
        env = gym.make('MiniGrid-MultiToy-8x8-N2-v0', room_size=8, max_steps=1000)
        print("   ✓ Environment created successfully")
        env.close()
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("APT Experiment Setup Test")
    print("=" * 60)
    
    tests = [
        test_python_version,
        test_imports,
        test_cuda,
        test_paths,
        test_config_generation,
        test_apt_import,
        test_environment,
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to run experiments.")
        print("\\nNext steps:")
        print("1. Run: python generate_configs.py")
        print("2. Run: python run_all_experiments.py")
        print("   Or for a quick test: python run_all_experiments.py --quick-test")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())