#!/usr/bin/env python3
"""
Test script to verify the experiment setup
"""
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("NovelD Experiment Setup Test")
print("="*50)

# Check Python version
print(f"\n1. Python version: {sys.version}")

# Check required modules
print("\n2. Checking required modules:")
required_modules = [
    'torch',
    'numpy',
    'gym',
    'scipy',
    'pandas'
]

for module in required_modules:
    try:
        __import__(module)
        print(f"   ✓ {module}")
    except ImportError:
        print(f"   ✗ {module} - NOT INSTALLED")

# Check CUDA availability
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device = "CUDA" if cuda_available else "CPU"
    print(f"\n3. Device: {device}")
    if cuda_available:
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.get_device_name(0)}")
except:
    print("\n3. Device: Cannot determine")

# Check paths
print("\n4. Checking paths:")
paths_to_check = [
    ("Parent directory", ".."),
    ("Algorithms", "../algorithms/NovelD_PPO.py"),
    ("Post processing", "../post_processing/batch_js_divergence.py"),
    ("Pickle file", "../post_processing/averaged_state_distributions.pkl"),
    ("Test activity logs", "../test/activity_logs/")
]

for name, path in paths_to_check:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"   {status} {name}: {path}")

# Test config generation
print("\n5. Testing config generation:")
try:
    from generate_configs import generate_experiment_configs
    configs = generate_experiment_configs()
    print(f"   ✓ Generated {len(configs)} configurations")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check if we can import the main modules
print("\n6. Testing imports:")
try:
    from algorithms.NovelD_PPO import NovelD_PPO
    print("   ✓ NovelD_PPO")
except Exception as e:
    print(f"   ✗ NovelD_PPO: {e}")

try:
    from post_processing.batch_js_divergence import process_activity_logs
    print("   ✓ batch_js_divergence")
except Exception as e:
    print(f"   ✗ batch_js_divergence: {e}")

# Summary
print("\n" + "="*50)
print("Setup test complete!")
print("\nTo run experiments:")
print("  python run_all_experiments.py --quick-test  # For testing")
print("  python run_all_experiments.py              # For full run")