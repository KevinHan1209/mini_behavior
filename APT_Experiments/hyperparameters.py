"""
APT Hyperparameters Configuration

The default hyperparameters should be already set in @algorithms/APT_PPO.py. However, just to confirm:

k: 50
int_gamma: 0.99 (intrinsic discount factor)
PPO ent_coef: 0.01 (entropy coefficient)


We want to ablate the following hyperparameters for APT:

APT-Specific Parameters
  - k: [10, 20, 50, 100, 200, 500]
  - int_gamma: [0.9, 0.95, 0.98, 0.99] (intrinsic discount factor)

PPO Core Parameters
  - ent_coef: [0, 0.01, 0.02, 0.05, 0.1, 0.2] (entropy coefficient)

TO ABLATE THE HYPERPARAMETERS, SET EVERY OTHER HYPERPARAMETER AT DEFAULT AND THEN ITERATE THROUGH THE VALUES.
"""

# Hyperparameter configurations for ablation studies
APT_HYPERPARAMETERS = {
    "default": {
        "k": 50,
        "int_gamma": 0.99,
        "ent_coef": 0.01
    },
    "ablations": {
        "k": [10, 20, 50, 100, 200, 500],
        "int_gamma": [0.9, 0.95, 0.98, 0.99],
        "ent_coef": [0, 0.01, 0.02, 0.05, 0.1, 0.2]
    }
}