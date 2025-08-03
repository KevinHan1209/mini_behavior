#!/usr/bin/env python3
"""
Test script to verify APT observation indexing is correct after the fix.
"""

import numpy as np
import gym
from mini_behavior.envs.multitoy import MultiToyEnv
from env_wrapper import CustomObservationWrapper
from mini_behavior.utils.states_base import RelativeObjectState

def test_apt_observation_indexing():
    """Test that APT's observation indexing matches the actual observation structure."""
    
    # Create environment
    env = MultiToyEnv(test_env=True, max_steps=1000)
    wrapped_env = CustomObservationWrapper(env)
    
    # Get observation
    obs = wrapped_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"First few values: {obs[:10]}")
    
    # Default states to exclude (from env_wrapper.py)
    default_states = [
        'atsamelocation',
        'infovofrobot',
        'inleftreachofrobot',
        'inrightreachofrobot',
        'inside',
        'nextto',
        'inlefthandofrobot',
        'inrighthandofrobot',
    ]
    
    # Manually build what we expect the observation to be
    expected_obs = []
    
    # Agent state (3 values)
    expected_obs.extend([env.agent_pos[0], env.agent_pos[1], env.agent_dir])
    
    # For each object
    for obj_type in env.objs.values():
        for obj in obj_type:
            # Get position
            if obj.cur_pos is None:
                if 'gear' in obj.get_name():
                    cur_pos = env.objs['gear_toy'][0].cur_pos
                elif 'winnie' in obj.get_name():
                    cur_pos = env.objs['winnie_cabinet'][0].cur_pos
                elif 'coin' in obj.get_name():
                    cur_pos = env.objs['piggie_bank'][0].cur_pos
                elif 'cube' in obj.get_name():
                    cur_pos = env.objs['cube_cabinet'][0].cur_pos
            else:
                cur_pos = obj.cur_pos
            
            # Add position (2 values)
            expected_obs.extend(cur_pos)
            
            # Add non-default states
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState) and state_name not in default_states:
                    value = state.get_value(env)
                    expected_obs.append(1 if value else 0)
    
    expected_obs = np.array(expected_obs, dtype=np.float32)
    
    # Compare
    print(f"\nExpected observation length: {len(expected_obs)}")
    print(f"Actual observation length: {len(obs)}")
    
    if len(expected_obs) == len(obs):
        print("✓ Observation lengths match!")
        if np.allclose(expected_obs, obs):
            print("✓ Observation values match!")
        else:
            print("✗ Observation values differ")
            diff_indices = np.where(np.abs(expected_obs - obs) > 0.01)[0]
            print(f"Differences at indices: {diff_indices[:10]}...")  # Show first 10
    else:
        print("✗ Observation lengths don't match!")
    
    # Now test APT's indexing
    print("\n=== Testing APT Indexing ===")
    
    index = 3  # Skip agent state
    object_count = 0
    
    for obj_type in env.objs.values():
        for obj in obj_type:
            object_count += 1
            
            # Count non-default states
            state_count = sum(1 for state_name, state in obj.states.items() 
                            if not isinstance(state, RelativeObjectState) 
                            and state_name not in default_states)
            
            print(f"\nObject {object_count}: {obj.name}")
            print(f"  Position indices: {index}, {index+1}")
            print(f"  State indices: {index+2} to {index+2+state_count-1}")
            print(f"  Number of states: {state_count}")
            
            # Verify we can access these indices
            if index + 2 + state_count <= len(obs):
                pos_values = obs[index:index+2]
                state_values = obs[index+2:index+2+state_count]
                print(f"  Position: {pos_values}")
                print(f"  States: {state_values}")
            else:
                print(f"  ERROR: Indices out of bounds!")
            
            index += 2 + state_count
    
    print(f"\nFinal index: {index}")
    print(f"Observation length: {len(obs)}")
    
    if index == len(obs):
        print("✓ APT indexing matches observation structure!")
    else:
        print("✗ APT indexing does not match observation structure!")
        print(f"  Difference: {len(obs) - index}")

if __name__ == "__main__":
    test_apt_observation_indexing()