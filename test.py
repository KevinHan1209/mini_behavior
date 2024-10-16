import gym
import mini_behavior
from NovelD_PPO import NovelD_PPO
import numpy as np
from mini_behavior.utils.states_base import RelativeObjectState
import torch

class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.get_obs_space()

    def observation(self, obs):
        return self.gen_obs()

    def get_obs_space(self):
        obj_states = []
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        obj_states.append(0)

        obs = [0, 0, 0]  # Agent position (x, y) and direction
        obs += obj_states
        return gym.spaces.Box(low=0, high=max(self.env.width, self.env.height), shape=(len(obs),), dtype=np.float32)

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

# Create the environment
env = gym.make('MiniGrid-ShakingARattle-8x8-N2-v0')

# Wrap the environment with the custom observation wrapper
env = CustomObservationWrapper(env)

# Debug prints
print(f"Observation space: {env.observation_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize NovelD_PPO
noveld_ppo = NovelD_PPO(env, device=device)

# Debug prints
print(f"NovelD_PPO initialized with:")
print(f"  Observation dimension: {noveld_ppo.obs_dim}")
print(f"  Action space: {env.action_space.n}")
print(f"  Batch size: {noveld_ppo.batch_size}")
print(f"  Minibatch size: {noveld_ppo.minibatch_size}")
print(f"  Number of iterations: {noveld_ppo.num_iterations}")

# Train the agent
print("Starting training...")
noveld_ppo.train()

# Test the trained agent
print("Testing trained agent...")
obs = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    action, _, _, _, _ = noveld_ppo.agent.get_action_and_value(torch.FloatTensor(obs).unsqueeze(0).to(device))
    obs, reward, done, _ = env.step(action.cpu().numpy().item())
    total_reward += reward
    steps += 1

    if steps % 10 == 0:
        print(f"Step {steps}, Current reward: {total_reward}")

print(f"Episode finished after {steps} steps")
print(f"Total reward: {total_reward}")
