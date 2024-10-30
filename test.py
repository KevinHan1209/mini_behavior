import gym
import mini_behavior
from NovelD_PPO import NovelD_PPO
import numpy as np
from mini_behavior.utils.states_base import RelativeObjectState
import torch
import os
import time
from array2gif import write_gif

# === Environment Wrapper ===
class CustomObservationWrapper(gym.ObservationWrapper):
    """Converts environment observations into a flat vector containing agent position and object states"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.get_obs_space()

    def observation(self, obs):
        return self.gen_obs()

    def get_obs_space(self):
        # Count non-relative object states and add agent position (x,y,dir)
        obj_states = []
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        obj_states.append(0)

        obs = [0, 0, 0]  # Agent position (x, y) and direction
        obs += obj_states
        return gym.spaces.Box(low=0, high=max(self.env.width, self.env.height), 
                            shape=(len(obs),), dtype=np.float32)

    def gen_obs(self):
        obj_states = []
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        state = obj.states[state_value].get_value(self.env)
                        obj_states.append(1 if state else 0)

        obs = list(self.env.agent_pos) + [self.env.agent_dir] + obj_states
        return np.array(obs, dtype=np.float32)

def train_agent(env):
    print("\n=== Training New Agent ===")
    noveld_ppo = NovelD_PPO(env)
    noveld_ppo.train()
    noveld_ppo.save_model("noveld_ppo_model.pth")
    return noveld_ppo

def test_agent(env, noveld_ppo, device, num_episodes=1, max_steps_per_episode=100):
    print(f"\n=== Testing Agent ===")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        novelty_values = []
        frames = []
        
        while not done and steps < max_steps_per_episode:
            frames.append(np.moveaxis(env.render(), 2, 0))
            
            action, _, _, ext_value, int_value = noveld_ppo.agent.get_action_and_value(
                torch.FloatTensor(obs).unsqueeze(0).to(device))
            obs, reward, done, _ = env.step(action.cpu().numpy().item())
            
            total_reward += reward
            steps += 1
            novelty = noveld_ppo.calculate_novelty(torch.FloatTensor(obs).unsqueeze(0).to(device))
            novelty_values.append(novelty)
            
            print(f"Step {steps:3d} | Action: {env.actions(action.item()).name:15s}")
            print(f"  Reward: {reward:6.2f} | Novelty: {novelty:6.4f}")
            print(f"  Values - Ext: {ext_value.item():6.2f} | Int: {int_value.item():6.2f}")
            
            time.sleep(0.1)
        
        # Save episode results
        write_gif(np.array(frames), f"episode_{episode + 1}.gif", fps=1)
        print(f"\n=== Episode {episode + 1} Summary ===")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")

def main():
    print("\n=== Initializing Environment ===")
    env = gym.make('MiniGrid-ShakingARattle-6x6-N2-v0')
    env = CustomObservationWrapper(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or train model
    model_path = "noveld_ppo_model.pth"
    try:
        print("\n=== Loading Existing Model ===")
        noveld_ppo = NovelD_PPO(env, device=device)
        noveld_ppo.load_model(model_path)
    except (FileNotFoundError, ValueError):
        print("\n=== No Existing Model Found ===")
        noveld_ppo = train_agent(env)

    test_agent(env, noveld_ppo, device)

if __name__ == "__main__":
    main()
