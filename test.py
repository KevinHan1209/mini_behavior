# test.py
import gym
import mini_behavior
from NovelD_PPO import NovelD_PPO
import numpy as np
import torch
import os
import time
from array2gif import write_gif
from env_wrapper import CustomObservationWrapper

def train_agent(env_id):
    print("\n=== Starting Agent Training ===")
    try:
        noveld_ppo = NovelD_PPO(env_id)
        noveld_ppo.train()
        print("\nSaving model to noveld_ppo_model.pth")
        noveld_ppo.save_model("noveld_ppo_model.pth")
        return noveld_ppo
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

def test_agent(env_id, noveld_ppo, device, num_episodes=10, max_steps_per_episode=500):
    print(f"\n=== Testing Agent: {num_episodes} Episodes ===")
    
    test_env = gym.make(env_id)
    test_env = CustomObservationWrapper(test_env)
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        novelty_values = []
        frames = []
        
        while not done and steps < max_steps_per_episode:
            frames.append(np.moveaxis(test_env.render(), 2, 0))
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _, ext_value, int_value = noveld_ppo.agent.get_action_and_value(obs_tensor)
            
            obs, reward, done, _ = test_env.step(action.cpu().numpy()[0])
            
            total_reward += reward
            steps += 1
            novelty = noveld_ppo.calculate_novelty(torch.FloatTensor(obs).unsqueeze(0).to(device))
            novelty_values.append(novelty)
            
            novelty_val = novelty.item() if torch.is_tensor(novelty) else novelty
            ext_val = ext_value.item() if torch.is_tensor(ext_value) else ext_value
            int_val = int_value.item() if torch.is_tensor(int_value) else int_value
            
            print(f"\nStep {steps}")
            print(f"Action Taken: {test_env.actions(action.item()).name}")
            print(f"Reward: {reward:.2f}")
            print(f"Novelty Score: {novelty_val:.4f}")
            
            time.sleep(0.1)
        
        write_gif(np.array(frames), f"episode_{episode + 1}.gif", fps=1)
        print(f"\nEpisode {episode + 1} Summary")
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")

def main():
    print("Initializing Environment")
    env_id = 'MiniGrid-MultiToy-8x8-N2-v0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    print("Training New Model")
    noveld_ppo = train_agent(env_id)
    
    if noveld_ppo is not None:
        test_agent(env_id, noveld_ppo, device)
    else:
        print("Failed to train model")

if __name__ == "__main__":
    main()
