# Extrinsic Reward Scaling Plan for NovelD Experiments

## Goal
Scale the maximum possible extrinsic rewards to match the mean intrinsic reward value of **0.0829 per step** from past experiments.

## Current Reward Structure

### Three Reward Categories
1. **Noise Rewards**: Triggered when objects produce noise (state change from non-noise to noise)
2. **Interaction Rewards**: Triggered by object manipulation actions
3. **Location Change Rewards**: Triggered by object movement actions

### Current Values (All Categories)
- **Noise**: 0.1 per activation
- **Interaction**: 0.1 per successful action
- **Location Change**: 0.1 per successful action

## Action-to-Reward Mapping

### Interaction Actions (from minibehavior.py:1135-1136)
- `dropin`, `takeout`, `putin`, `brush`, `assemble`, `disassemble`
- `toggle`, `noise_toggle`, `mouthing`

### Location Change Actions (from minibehavior.py:1138)
- `kick`, `push`, `pull`, `pickup`, `drop`, `throw`

## Factors Contributing to Reward Scaling

### 1. Base Reward Values
- Current: 0.1 for all categories
- **This is indeed too large** compared to the 0.0829 mean intrinsic reward

### 2. Frequency Factors
- **Noise rewards**: Can be triggered multiple times as objects enter/exit noise states
- **Interaction rewards**: Triggered per successful manipulation action
- **Location rewards**: Triggered per successful movement action

### 3. Multiple Object Interactions
- The environment contains **23 different object types** with multiple instances
- Total objects: ~50+ (including 10 coins, 6 cubes, 6 gears, 3 shape toys, etc.)
- Each object can potentially trigger rewards in multiple categories

### 4. Action Success Rate
- Not all actions succeed (depends on agent position, object state, etc.)
- Success rate affects actual reward frequency vs. theoretical maximum

## Maximum Possible Extrinsic Rewards Per Step

### How Each Category Works
1. **Location Change**: Triggered by successful `kick`, `push`, `pull`, `pickup`, `drop`, `throw`, `walk_with_object`
   - Max per step: 2 (one per arm)
   
2. **Interaction**: Triggered by successful `dropin`, `takeout`, `putin`, `brush`, `assemble`, `disassemble`, `toggle`, `noise_toggle`, `mouthing`
   - Max per step: 2 (one per arm)
   
3. **Noise**: Triggered when ANY object's noise state changes from False to True
   - Max per step: N (where N = number of objects that start making noise)

### Important: Categories are NOT mutually exclusive!
- Both arms act independently → up to 2 action rewards per step
- Multiple objects can trigger noise → multiple noise rewards per step
- Same step can have: interaction + noise, location + noise, or even location + interaction (different arms)

### Theoretical Maximum (Current Settings at 0.1)
- **Action rewards only**: 0.2 (both arms successful)
- **With noise**: 0.2 + 0.1×N (actions plus N objects making noise)
- **Practical maximum**: ~0.2-0.3 per step

### Issue Analysis
- Current 0.1 per reward gives 0.2-0.3 typical max per step
- This is **240-360%** of target 0.0829
- Need smaller individual values since multiple rewards can trigger

## Proposed Scaling Strategy

### Option 1: Conservative Scaling (Recommended)
Account for multiple rewards per step:
```json
{
  "noise": 0.02,           // Rare but can trigger multiple times
  "interaction": 0.04,      // Max 2 per step (both arms)
  "location_change": 0.04   // Max 2 per step (both arms)
}
```
**Expected per step**: 0.04-0.08 (1-2 actions) + occasional 0.02 (noise) ≈ 0.08 average

### Option 2: Moderate Scaling
Assumes ~1.5 rewards trigger per step on average:
```json
{
  "noise": 0.03,
  "interaction": 0.055,
  "location_change": 0.055
}
```
**Expected per step**: With 1-2 actions typical ≈ 0.055-0.11, average ~0.0825

### Option 3: Direct Scaling (Not Recommended)
Simple division by expected triggers, but likely too high:
```json
{
  "noise": 0.0829,
  "interaction": 0.0829,
  "location_change": 0.0829
}
```
**Problem**: With 2 arms this gives 0.166 max, double the target!

### Option 4: Differential Scaling
Based on expected frequency and impact:
```json
{
  "noise": 0.015,          // Rarest trigger
  "interaction": 0.035,     // Moderate frequency
  "location_change": 0.05   // Most common (includes walk_with_object)
}
```
**Expected per step**: Weighted by frequency ≈ 0.085

## Recommended Implementation

### Phase 1: Baseline Calibration
1. Run experiment with conservative scaling (0.02/0.04/0.04)
2. Measure actual reward distribution per category
3. Calculate effective rewards per step
4. Verify average is close to 0.0829

### Phase 2: Category Optimization
1. Adjust individual category weights based on Phase 1 results
2. Target: Mean extrinsic reward = 0.0829 per step
3. Consider variance and distribution shape

### Phase 3: Fine-tuning
1. Test boundary conditions (very high/low individual categories)
2. Ensure reward signal remains informative
3. Validate against intrinsic-only baseline

## Additional Considerations

### 1. Reward Accumulation
- Multiple rewards can trigger in single step
- Need to monitor total reward per step distribution

### 2. Curriculum Effects
- Early training: More location changes (exploration)
- Late training: More interactions (exploitation)
- Consider time-varying reward schedules

### 3. Intrinsic-Extrinsic Balance
- `ext_coef` parameter controls extrinsic weight (currently 1.0)
- `int_coef` parameter controls intrinsic weight (varies by experiment)
- May need to adjust these alongside reward values

## Monitoring Metrics
- Mean reward per step (target: 0.0829)
- Reward variance
- Category distribution percentages
- Reward frequency histograms
- Cumulative reward curves

## Next Steps
1. Implement uniform 0.0829 scaling as baseline
2. Run short test (500k steps) to validate
3. Analyze reward logs for actual distribution
4. Adjust category-specific values based on data
5. Run full experiments with optimized values
