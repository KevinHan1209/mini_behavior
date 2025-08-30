This folder contains entropy coefficient ablation configs for extrinsic-only experiments.

- Categories: location_change, interaction, noise
- Entropy coefficients ablated over: 0, 0.01, 0.02, 0.05, 0.1, 0.2
- Base hyperparameters copied from NovelD_Experiments/results/exp_003_alpha_1/experiment_config.json, with int_coef=0.0 and ext_coef=1.0 to ensure extrinsic-only optimization.

Run all with:

  ./run_entropy_ablation_experiments.py

Results saved under NovelD_Extrinsic/results/<experiment_name>/

