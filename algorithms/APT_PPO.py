import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import wandb
from collections import deque
from array2gif import write_gif

from networks.actor_critic import Agent
from mini_behavior.roomgrid import *
from mini_behavior.utils.utils import RewardForwardFilter, RMS
from env_wrapper import CustomObservationWrapper
from gym.wrappers.normalize import RunningMeanStd


class APT_PPO:
    def __init__(self,
                 env,
                 env_id,
                 env_kwargs,
                 save_dir,
                 device="cpu",
                 save_freq=100,
                 test_steps=500,
                 total_timesteps=2000000,
                 learning_rate=1e-4,
                 num_envs=8,
                 num_steps=125,
                 anneal_lr=True,
                 gamma=0.999,
                 gae_lambda=0.95,
                 num_minibatches=4,
                 update_epochs=4,
                 norm_adv=True,
                 clip_coef=0.1,
                 clip_vloss=True,
                 ent_coef=0.001,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 target_kl=None,
                 int_coef=1.0,
                 ext_coef=2.0,
                 int_gamma=0.99,
                 k=50,
                 c=1):
        self.env = env
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.save_dir = save_dir
        self.device = device
        self.save_freq = save_freq
        self.test_steps = test_steps
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.k = k
        self.c = c

        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_iterations = self.total_timesteps // self.batch_size

        self.rms = RMS(self.device)
        self.total_actions = []
        self.total_obs = []
        self.total_avg_curiosity_rewards = []
        self.model_saves = []
        self.exploration_percentages = []
        self.test_actions = []
        self.exploration_state_occurrences = []

        # Precompute the object-state pattern for distance calculations
        self.objstate_pattern = self.get_object_state_pattern()

    def train(self):
        print("TRAINING PARAMETERS")
        print("-------------------")
        print("Total timesteps:", self.total_timesteps)
        print("Learning rate:", self.learning_rate)
        print("Total updates:", self.total_timesteps // self.batch_size)
        print("Parallel envs:", self.num_envs)
        print("Steps per rollout:", self.num_steps)
        print("Batch size:", self.batch_size)
        print("PPO epochs:", self.update_epochs)
        print("Minibatch size:", self.minibatch_size)
        print("k parameter:", self.k)
        print("-------------------")
        assert self.total_timesteps % self.batch_size == 0
        self.run = wandb.init(project="APT_PPO_Training", config={
            "env_id": self.env_id,
            "Total timesteps": self.total_timesteps,
            "Learning rate": self.learning_rate,
            "Total updates": self.total_timesteps // self.batch_size,
            "Parallel envs": self.num_envs,
            "Steps per rollout": self.num_steps,
            "Batch size": self.batch_size,
            "PPO epochs": self.update_epochs,
            "Minibatch size": self.minibatch_size,
            "k parameter": self.k,
            "Save frequency": self.save_freq
        })
        self.agent = Agent(self.env.action_space[0].nvec, self.env.single_observation_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        combined_params = list(self.agent.parameters())

        reward_rms = RunningMeanStd()
        discounted_reward = RewardForwardFilter(self.int_gamma)

        # Storage for rollout data
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space[0].shape).to(self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + (self.env.observation_space.shape[1],)).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.curiosity_rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        global_step = 0
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = int(self.total_timesteps // self.batch_size)
        for update in range(1, num_updates + 1):
            print(f"UPDATE {update}/{num_updates}")
            if update % self.save_freq == 0:
                print("Saving model and testing...")
                self.model_saves.append([self.agent.state_dict(), self.optimizer.state_dict()])
                self.test_agent(save_episode=update, max_steps_per_episode=self.test_steps)

            if self.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Collect rollout
            for step in range(self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    value_ext, value_int = self.agent.get_value(obs[step])
                    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, done, _ = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            self.total_actions.append(actions.clone())
            self.total_obs.append(obs.clone())

            # Compute intrinsic (curiosity) rewards via kNN on state representations
            sim_matrix = self._compute_similarity_matrix(obs)
            self.curiosity_rewards = self.compute_reward(sim_matrix)

            # Normalize curiosity rewards
            rewards_per_env = np.array([discounted_reward.update(r) for r in self.curiosity_rewards.cpu().numpy().T])
            mean_r, std_r = rewards_per_env.mean(), rewards_per_env.std()
            print("Average intrinsic reward:", mean_r)
            self.total_avg_curiosity_rewards.append(mean_r)
            reward_rms.update_from_moments(mean_r, std_r**2, len(rewards_per_env))
            self.curiosity_rewards /= np.sqrt(reward_rms.var)
            self.run.log({
                "Average Reward": mean_r,
                "Std Reward": std_r,
                "Actions": actions,
                "Observations": obs
            })

            # Compute advantages and returns
            ext_advantages, int_advantages = torch.zeros_like(rewards), torch.zeros_like(self.curiosity_rewards)
            ext_lastgaelam, int_lastgaelam = 0, 0
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        ext_nextnonterminal = 1.0 - next_done
                        int_nextnonterminal = 1.0
                        ext_nextvalues = next_value_ext
                        int_nextvalues = next_value_int
                    else:
                        ext_nextnonterminal = 1.0 - dones[t + 1]
                        int_nextnonterminal = 1.0
                        ext_nextvalues = ext_values[t + 1]
                        int_nextvalues = int_values[t + 1]
                    ext_delta = rewards[t] + self.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                    int_delta = self.curiosity_rewards[t] + self.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                    ext_lastgaelam = ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    int_lastgaelam = int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam
                    ext_advantages[t] = ext_lastgaelam
                    int_advantages[t] = int_lastgaelam

                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

            # Flatten rollout
            b_obs = obs.reshape((-1,) + (self.env.observation_space.shape[1],))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, actions.shape[-1])
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = ext_returns.reshape(-1)
            b_int_returns = int_returns.reshape(-1)
            b_ext_values = ext_values.reshape(-1)
            b_advantages = b_int_advantages * self.int_coef + b_ext_advantages * self.ext_coef

            # PPO update loop
            indices = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = indices[start:end]
                    print('epoch:', epoch)
                    print('b_actions:', b_actions.shape)
                    _, new_logprob, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = new_logprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss = torch.max(-mb_advantages * ratio,
                                        -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)).mean()
                    new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                    if self.clip_vloss:
                        ext_v_loss = 0.5 * torch.max(
                            (new_ext_values - b_ext_returns[mb_inds])**2,
                            (b_ext_values[mb_inds] + torch.clamp(new_ext_values - b_ext_values[mb_inds],
                                                                 -self.clip_coef, self.clip_coef) - b_ext_returns[mb_inds])**2
                        ).mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds])**2).mean()
                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds])**2).mean()
                    loss = pg_loss - self.ent_coef * entropy.mean() + (ext_v_loss + int_v_loss) * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.agent.parameters()), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

    def test_agent(self, save_episode, num_episodes=1, max_steps_per_episode=500):
        """Run test episodes using the current agent policy and log a gif replay."""
        print(f"\n=== Testing Agent: {num_episodes} Episode(s) ===")
        action_log = []
        test_env = gym.make(self.env_id, **self.env_kwargs)
        test_env = CustomObservationWrapper(test_env)

        for ep in range(num_episodes):
            obs = test_env.reset()
            done = False
            steps = 0
            frames = []
            while not done and steps < max_steps_per_episode:
                frames.append(np.moveaxis(test_env.render(), 2, 0))
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                obs, reward, done, _ = test_env.step(action.numpy()[0])
                action_log.append(test_env.manipulation_actions(action[0][0].item()).name)
                action_log.append(test_env.manipulation_actions(action[0][1].item()).name)
                action_log.append(test_env.locomotion_actions(action[0][2].item()).name)
                print(f"Step {steps:3d} | Actions: {test_env.manipulation_actions(action[0][0].item()).name}, {test_env.manipulation_actions(action[0][1].item()).name}, {test_env.locomotion_actions(action[0][2].item()).name}")
                steps += 1
                time.sleep(0.1)

            gif_path = os.path.join(self.save_dir, "test_replays", f"episode_{save_episode}.gif")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            write_gif(np.array(frames), gif_path, fps=10)
            self.run.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})

        test_env.close()
        self.test_actions.append(action_log)

    # ---------------- Helper Methods ----------------

    def get_object_state_pattern(self):
        """Return the number of object states per object (minus three fixed indices)."""
        test_env = gym.make(self.env_id, **self.env_kwargs)
        test_env = CustomObservationWrapper(test_env)
        pattern = []
        for obj_type in test_env.objs.values():
            for obj in obj_type:
                num_states = sum(1 for state in obj.states if not isinstance(obj.states[state], RelativeObjectState))
                pattern.append(num_states - 3)
        test_env.close()
        return pattern

    def compute_distance_matrix(self, env_obs):
        """
        Compute a Hamming-like distance matrix over object-state slices.
        env_obs: [num_steps, obs_dim]
        Returns: [num_steps, num_steps] distance matrix.
        """
        num_steps = env_obs.shape[0]
        total_distance = torch.zeros(num_steps, num_steps, device=env_obs.device)
        start_idx = 3
        for obj_len in self.objstate_pattern:
            state_start = start_idx + 5
            slice_obs = env_obs[:, state_start: state_start + obj_len]
            diff = (slice_obs.unsqueeze(1) != slice_obs.unsqueeze(0)).float()
            total_distance += diff.sum(dim=-1)
            start_idx += obj_len + 5
        return total_distance

    def compute_reward(self, sim_matrix):
        """
        Compute intrinsic rewards as log(c + average kNN distance)
        sim_matrix: [num_steps, num_steps, num_envs]
        Returns: [num_steps, num_envs] rewards.
        """
        num_steps, _, num_envs = sim_matrix.shape
        rewards = torch.zeros(num_steps, num_envs, device=sim_matrix.device)
        for env in range(num_envs):
            env_dist = sim_matrix[:, :, env].clone()
            env_dist.fill_diagonal_(float('inf'))
            k_nearest, _ = torch.topk(env_dist, k=self.k, largest=False)
            avg_distance = k_nearest.mean(dim=1)
            rewards[:, env] = torch.log(self.c + avg_distance)
        return rewards

    def _compute_similarity_matrix(self, obs):
        """
        Build a similarity matrix (distance matrix) for each environment.
        obs: [num_steps, num_envs, obs_dim]
        Returns: [num_steps, num_steps, num_envs]
        """
        sim_matrices = []
        for env_idx in range(self.num_envs):
            env_obs = obs[:, env_idx, :]
            sim_matrices.append(self.compute_distance_matrix(env_obs))
        return torch.stack(sim_matrices, dim=-1)
