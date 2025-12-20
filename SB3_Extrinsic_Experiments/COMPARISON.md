# Comparison: Old vs New Extrinsic Experiments

## Directory Comparison

| Feature | NovelD_Extrinsic (Old) | SB3_Extrinsic_Experiments (New) |
|---------|------------------------|--------------------------------|
| **Location** | `/NovelD_Extrinsic/` | `/SB3_Extrinsic_Experiments/` |
| **Implementation** | Custom `algorithms/NovelD_PPO.py` | SB3 `stable_baselines3.NovelD_PPO` |
| **Base Code** | Legacy PPO implementation | Modern SB3 from `test/test_NovelD.py` |
| **Python Script** | `run_single_experiment.py` | `run_experiment.py` |
| **Batch Runner** | `run_all_experiments.py` | `run_all_experiments.py` |

## Experiment Focus

### Old NovelD_Extrinsic
- Mixed intrinsic + extrinsic experiments
- Entropy ablation studies
- Multiple reward combinations
- Example configs:
  - `exp_000_pure_extrinsic_all_rewards` (all 3 rewards)
  - `exp_001_pure_intrinsic_baseline` (intrinsic only)
  - Various entropy coefficient variations

### New SB3_Extrinsic_Experiments
- **Pure extrinsic only** (NO intrinsic/curiosity)
- Three isolated reward types:
  1. **Noise only** - Reward for noise state changes
  2. **Location only** - Reward for location changes
  3. **Interaction only** - Reward for interactions
- Clean separation for testing individual extrinsic signals

## Key Parameter Differences

| Parameter | Old | New |
|-----------|-----|-----|
| `int_coef` | Varies (0.0 to 1.0) | **0.0** (disabled) |
| `ext_coef` | Varies (0.0 to 1.0) | **1.0** (enabled) |
| `update_proportion` | 1.0 | **0.25** |
| `alpha` | 0.5 | N/A (SB3 default) |
| Wandb project | Multiple projects | `sb3-noveld-extrinsic` |

## Extrinsic Reward Types

Both use the same reward system from the environment:

1. **noise**: Triggered when objects make noise (e.g., rattle shaking)
   - Implemented in: `mini_behavior/minibehavior.py:846-853`

2. **location_change**: Triggered by actions that move objects
   - Actions: kick, push, pull, pickup, drop, throw
   - Implemented in: `mini_behavior/minibehavior.py:948-951`

3. **interaction**: Triggered by object interactions
   - Actions: dropin, takeout, putin, brush, assemble, disassemble, toggle, noise_toggle, mouthing
   - Implemented in: `mini_behavior/minibehavior.py:944-947`

## Config File Format

Both use similar JSON structure:

```json
{
  "name": "experiment_name",
  "description": "...",
  "extrinsic_rewards": {
    "noise": 0.1,           // Old: can have multiple
    "location_change": 0.1,  // New: only ONE active
    "interaction": 0.1
  },
  "hyperparameters": {
    "int_coef": 0.0,  // New: always 0.0
    "ext_coef": 1.0,  // New: always 1.0
    ...
  }
}
```

## When to Use Which?

### Use Old NovelD_Extrinsic when:
- Testing mixed intrinsic + extrinsic rewards
- Running entropy ablation studies
- Using legacy NovelD_PPO implementation
- Comparing with previous results

### Use New SB3_Extrinsic_Experiments when:
- Testing **pure extrinsic** rewards only
- Isolating individual reward signals
- Using modern SB3 implementation
- Starting fresh experiments with SB3

## Migration Path

If migrating from old to new:

1. **Config structure** is similar, but update:
   - Set `int_coef: 0.0` (disable intrinsic)
   - Set `ext_coef: 1.0` (enable extrinsic)
   - Adjust `update_proportion` from 1.0 to 0.25
   - Update `wandb_config` section

2. **Script changes**:
   - Old: `python NovelD_Extrinsic/run_single_experiment.py config.json`
   - New: `python SB3_Extrinsic_Experiments/run_experiment.py config.json`

3. **Import changes**:
   - Old: `from algorithms.NovelD_PPO import NovelD_PPO`
   - New: `from stable_baselines3 import NovelD_PPO`

## Results Compatibility

Both produce similar output structures:
- Checkpoints saved every 500k steps
- Final model saved at end
- Wandb logging (if available)
- SUCCESS marker file

However, model formats differ due to different implementations, so you cannot directly load old models with new code or vice versa.
