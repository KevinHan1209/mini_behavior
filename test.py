# test.py
import gym
from NovelD_PPO import NovelD_PPO
import numpy as np
import torch
import time
import wandb
from array2gif import write_gif
from env_wrapper import CustomObservationWrapper

def train_agent(env_id, device):
    print("\n=== Starting Agent Training ===")
    try:
        noveld_ppo = NovelD_PPO(env_id, device)
        noveld_ppo.train()
        noveld_ppo.save_model("noveld_ppo_model.pth")
        return noveld_ppo
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

def test_agent(env_id, noveld_ppo, device, num_episodes=1, max_steps_per_episode=500):
    print(f"\n=== Testing Agent: {num_episodes} Episodes ===")
    
    # Initialize wandb for testing
    wandb.init(project="noveld-ppo-test",
              config={"env_id": env_id,
                     "mode": "testing",
                     "num_episodes": num_episodes,
                     "max_steps": max_steps_per_episode})

    test_env = gym.make(env_id)
    test_env = CustomObservationWrapper(test_env)
    
    noveld_ppo.agent.network.to(device)
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        novelty_values = []
        frames = []
        activity = [0] * len(obs)
        prev_obs = None
        
        while not done and steps < max_steps_per_episode:
            frames.append(np.moveaxis(test_env.render(), 2, 0))
            
            with torch.no_grad():
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

            if prev_obs is not None:
                differences = (obs[3:] != prev_obs[3:]).astype(int)
                activity = [a + d for a, d in zip(activity, differences)]
            prev_obs = obs

            # Log step metrics
            wandb.log({
                "step_reward": reward,
                "step_novelty": novelty.item(),
                "step_ext_value": ext_value.item(),
                "step_int_value": int_value.item(),
                "episode": episode,
                "step": steps
            })
            
            print(f"\nStep {steps}")
            print(f"Action Taken: {test_env.actions(action.item()).name}")
            print(f"Novelty Score: {novelty_val:.4f}")
            
            time.sleep(0.1)
        
        # Log episode metrics
        wandb.log({
            "episode_total_reward": total_reward,
            "episode_length": steps,
            "episode_mean_novelty": np.mean(novelty_values),
            "episode": episode
        })

        gif_path = f"episode_{episode + 1}.gif"
        wandb.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})
        write_gif(np.array(frames), gif_path, fps=1)
        print(f"\nEpisode {episode + 1} Summary")
        print(f"Average Novelty: {np.mean(novelty_values):.4f}")
        print(f"Activity Array: {[int(a) for a in activity]}")

    test_env.close()
    wandb.finish()

def main():
    print("Initializing Environment")
    env_id = 'MiniGrid-MultiToy-8x8-N2-v0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    print("Training Model")
    noveld_ppo = train_agent(env_id, device)
    
    # Test model
    print("Testing Model")
    test_agent(env_id, noveld_ppo, device)

if __name__ == "__main__":
    main()
