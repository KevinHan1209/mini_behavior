The default hyperparameters should be already set in @algorithms/NovelD_PPO.py. However, just to confirm:

alpha: 0.5
update_proportion: 0.25 (RND update proportion)
int_gamma: 0.99 (intrinsic discount factor)
PPO ent_coef: 0.01 (entropy coefficient)


We want to ablate the following hyperparameters for NovelD:

NovelD-Specific Parameters
  - alpha: [0, 0.25, 0.5, 1]
  - update_proportion: [0.25, 0.5, 1] (RND update proportion)
  - int_gamma: [0.9, 0.95, 0.98, 0.99] (intrinsic discount factor)

PPO Core Parameters
  - ent_coef: [0, 0.01, 0.02, 0.05, 0.1, 0.2] (entropy coefficient)

TO ABLATE THE HYPERPARAMETERS, SET EVERY OTHER HYPERPARAMETER AT DEFAULT AND THEN ITERATE THROUGH THE VALUES.