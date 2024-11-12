# test.py
import gym
import mini_behavior
from NovelD_PPO import NovelD_PPO
import numpy as np
import torch
import wandb
import time
from array2gif import write_gif
from env_wrapper import CustomObservationWrapper

def train_agent(env_id):
    print("\n=== Starting Agent Training ===")
    noveld_ppo = NovelD_PPO(env_id, "cpu")
    noveld_ppo.train()
    print("\nSaving model to noveld_ppo_model.pth")
    noveld_ppo.save_model("noveld_ppo_model.pth")
    return noveld_ppo

def test_agent(env_id, noveld_ppo, num_episodes=10, max_steps_per_episode=500):
    print(f"\n=== Testing Agent: {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="noveld-ppo-test",
              config={"env_id": env_id,
                     "mode": "testing",
                     "num_episodes": num_episodes,
                     "max_steps": max_steps_per_episode})
    
    test_env = gym.make(env_id)
    test_env = CustomObservationWrapper(test_env)
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = test_env.reset()
        done = False
        steps = 0
        frames = []
        episode_reward = 0
        episode_novelty = []
        
        while not done and steps < max_steps_per_episode:
            frames.append(np.moveaxis(test_env.render(), 2, 0))
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, ext_value, int_value = noveld_ppo.agent.get_action_and_value(obs_tensor)
                novelty = noveld_ppo.calculate_novelty(torch.FloatTensor(obs).unsqueeze(0))
            
            obs, reward, done, _ = test_env.step(action.numpy()[0])
            episode_reward += reward
            episode_novelty.append(novelty.item())
            
            # Log step metrics
            wandb.log({
                "step_reward": reward,
                "step_novelty": novelty.item(),
                "step_ext_value": ext_value.item(),
                "step_int_value": int_value.item(),
                "episode": episode,
                "step": steps
            })
            
            # Print step information
            print(f"Step {steps:3d} | "
                  f"Action: {test_env.actions(action.item()).name:10s} | "
                  f"Reward: {reward:6.2f} | "
                  f"Extrinsic: {ext_value.item():6.2f} | "
                  f"Novelty: {novelty.item():6.4f} | "
                  f"Intrinsic: {int_value.item():6.2f}")
            
            steps += 1
            time.sleep(0.1)
        
        # Log episode metrics
        wandb.log({
            "episode_total_reward": episode_reward,
            "episode_length": steps,
            "episode_mean_novelty": np.mean(episode_novelty),
            "episode": episode
        })
        
        # Save gif as wandb artifact
        gif_path = f"episode_{episode + 1}.gif"
        write_gif(np.array(frames), gif_path, fps=10)
        wandb.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})
    
    wandb.finish()
    test_env.close()

def main():
    print("Initializing Environment")
    env_id = 'MiniGrid-PickingUpARattle-6x6-N2-v0'
    
    # Train model
    print("Training Model")
    noveld_ppo = train_agent(env_id)
    
    # Test model
    print("Testing Model")
    test_agent(env_id, noveld_ppo)

if __name__ == "__main__":
    main()