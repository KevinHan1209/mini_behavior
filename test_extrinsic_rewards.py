import gym
import numpy as np
from mini_behavior.envs.multitoy import MultiToyEnv

def test_extrinsic_rewards():
    """Test the extrinsic rewards implementation"""
    
    # Define extrinsic rewards
    extrinsic_rewards = {
        'noise': 0.5,          # Reward for any noise state change
        'interaction': 0.3,    # Reward for object interactions (dropin, takeout, etc.)
        'location_change': 0.2, # Reward for location changes (pickup, drop, kick, etc.)
        'pickup': 0.1,         # Additional specific reward for pickup
        'toggle': 0.15,        # Additional specific reward for toggle
    }
    
    # Create environment with extrinsic rewards
    env = MultiToyEnv(
        test_env=True,
        extrinsic_rewards=extrinsic_rewards,
        max_steps=1000
    )
    
    print("Testing extrinsic rewards system...")
    print(f"Extrinsic rewards config: {extrinsic_rewards}")
    print("-" * 50)
    
    # Reset environment
    obs = env.reset()
    total_rewards = 0
    action_count = 0
    
    # Test sequence of actions
    print("\nExecuting test actions...")
    
    # Try to pick up an object
    print("\n1. Testing pickup action...")
    for _ in range(10):
        # Random actions trying to pickup
        action = [1, -1, 2]  # pickup_0 with left arm, no right arm action, move forward
        obs, reward, done, info = env.step(action)
        action_count += 1
        if reward > 0:
            print(f"   Reward received: {reward}")
            total_rewards += reward
            break
    
    # Try to toggle something
    print("\n2. Testing toggle action...")
    for _ in range(10):
        # Random actions trying to toggle
        action = [0, -1, np.random.choice([0, 1, 2])]  # toggle with left arm
        obs, reward, done, info = env.step(action)
        action_count += 1
        if reward > 0:
            print(f"   Reward received: {reward}")
            total_rewards += reward
            break
    
    # Try to make noise
    print("\n3. Testing noise toggle action...")
    for _ in range(10):
        action = [6, -1, np.random.choice([0, 1, 2])]  # noise_toggle with left arm
        obs, reward, done, info = env.step(action)
        action_count += 1
        if reward > 0:
            print(f"   Reward received: {reward}")
            total_rewards += reward
            break
    
    # Test walking with object
    print("\n4. Testing walk with object...")
    # First pickup something
    for _ in range(10):
        action = [1, -1, 0]  # pickup_0 with left arm
        obs, reward, done, info = env.step(action)
        if env.carrying['left']:
            print("   Picked up object")
            break
    
    # Now walk with it
    if env.carrying['left']:
        action = [-1, -1, 2]  # no arm actions, move forward
        obs, reward, done, info = env.step(action)
        action_count += 1
        if reward > 0:
            print(f"   Walk with object reward: {reward}")
            total_rewards += reward
    
    print("\n" + "-" * 50)
    print(f"Total rewards collected: {total_rewards}")
    print(f"Actions executed: {action_count}")
    print(f"Average reward per action: {total_rewards/action_count:.3f}")
    
    # Test with no extrinsic rewards
    print("\n\nTesting with no extrinsic rewards...")
    env_no_rewards = MultiToyEnv(test_env=True, max_steps=1000)
    obs = env_no_rewards.reset()
    
    # Execute same actions
    no_reward_total = 0
    for _ in range(20):
        action = [np.random.choice([0, 1, 2, 6, 7, 8]), 
                  np.random.choice([-1, 0, 1, 2]), 
                  np.random.choice([0, 1, 2])]
        obs, reward, done, info = env_no_rewards.step(action)
        no_reward_total += reward
    
    print(f"Total rewards without extrinsic rewards: {no_reward_total}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_extrinsic_rewards()