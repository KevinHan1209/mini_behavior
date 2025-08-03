"""
APT Hyperparameters Configuration

The default hyperparameters should be already set in @algorithms/APT_PPO.py. However, just to confirm:

k: 12
int_gamma: 0.99 (intrinsic discount factor)
PPO ent_coef: 0.01 (entropy coefficient)
aggregation_method: 'mean' (how to aggregate k-NN distances: 'mean' or 'sum')
apt_batch_size: 1024 (number of samples from replay buffer for k-NN computation)


We want to ablate the following hyperparameters for APT:

APT-Specific Parameters
  - k: [5, 12, 25, 50, 100]
  - int_gamma: [0.9, 0.95, 0.98, 0.99] (intrinsic discount factor)
  - aggregation_method: ['sum'] (mean is default; sum: sum(log(c + distances)))
  - apt_batch_size: [512, 1024, 2048] (number of samples from replay buffer)

PPO Core Parameters
  - ent_coef: [0, 0.01, 0.02, 0.05, 0.1, 0.2] (entropy coefficient)

TO ABLATE THE HYPERPARAMETERS, SET EVERY OTHER HYPERPARAMETER AT DEFAULT AND THEN ITERATE THROUGH THE VALUES.
"""

# Hyperparameter configurations for ablation studies
APT_HYPERPARAMETERS = {
    "default": {
        "k": 12,
        "int_gamma": 0.99,
        "ent_coef": 0.01,
        "aggregation_method": "mean",
        "apt_batch_size": 1024
    },
    "ablations": {
        "k": [5, 12, 25, 50, 100],
        "int_gamma": [0.9, 0.95, 0.98, 0.99],
        "ent_coef": [0, 0.01, 0.02, 0.05, 0.1, 0.2],
        "aggregation_method": ["sum"],  # Only test 'sum' since 'mean' is default
        "apt_batch_size": [512, 1024, 2048]
    }
}