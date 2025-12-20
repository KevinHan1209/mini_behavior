#!/usr/bin/env python3
"""
Run all three extrinsic-only experiments sequentially.
"""
import os
import sys
import argparse
from pathlib import Path
import subprocess

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIGS_DIR = SCRIPT_DIR / "configs"


def run_experiment(config_path: Path, output_dir: Path, seed: int = 1):
    """Run a single experiment using run_experiment.py"""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_experiment.py"),
        str(config_path),
        "--output-dir", str(output_dir),
        "--seed", str(seed)
    ]

    print(f"\n{'='*80}")
    print(f"Running experiment: {config_path.stem}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Experiment {config_path.stem} failed with code {result.returncode}")
        return False

    print(f"\nSUCCESS: Experiment {config_path.stem} completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run all three extrinsic-only experiments'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed (default: 1)'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['noise', 'location', 'interaction', 'all'],
        default=['all'],
        help='Which experiments to run (default: all)'
    )

    args = parser.parse_args()

    # Determine which experiments to run
    experiments = ['noise', 'location', 'interaction'] if 'all' in args.experiments else args.experiments

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map experiment names to config files
    config_map = {
        'noise': CONFIGS_DIR / 'noise_only.json',
        'location': CONFIGS_DIR / 'location_only.json',
        'interaction': CONFIGS_DIR / 'interaction_only.json',
    }

    # Verify all config files exist
    for exp_name in experiments:
        config_path = config_map[exp_name]
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Running {len(experiments)} experiments:")
    for exp_name in experiments:
        print(f"  - {exp_name}_only")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Run experiments
    results = {}
    for exp_name in experiments:
        config_path = config_map[exp_name]
        success = run_experiment(config_path, output_dir, args.seed)
        results[exp_name] = success

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    for exp_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {exp_name}_only: {status}")
    print(f"{'='*80}\n")

    # Exit with error if any experiment failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
