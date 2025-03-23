import numpy as np
import torch 
import torch.optim as optim
from networks.actor import Agent
from mini_behavior.roomgrid import *
from mini_behavior.utils.utils import RewardForwardFilter, RMS
from env_wrapper import CustomObservationWrapper
from gym.wrappers.normalize import RunningMeanStd
from collections import deque
import time
from array2gif import write_gif
import torch.nn as nn
import wandb
import gym
import os

class APT_PPO():
    def __init__(self,
                 env,
                 env_id,    
                 save_dir,
                 test_env_kwargs = None,
                 test_env_id = None,
                 device = "cpu",
                 save_freq: int = 100,
                 test_steps = 500,
                 total_timesteps: int = 2000000,
                 learning_rate: float = 1e-4,
                 num_envs: int = 128,
                 num_steps: int = 128,
                 anneal_lr: bool = True,
                 gamma: float = 0.999,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 update_epochs: int = 4,
                 norm_adv: bool = True,
                 clip_coef: float = 0.1,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.001,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None,
                 int_coef: float = 1.0,
                 ext_coef: float = 2.0,
                 int_gamma: float = 0.99,
                 k: int = 50,
                 c: float = 1,
                 ): 
        self.env = env
        self.env_id = env_id
        self.test_env_kwargs = test_env_kwargs
        self.test_env_id = test_env_id
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

        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        self.rms  = RMS(self.device)
        self.exploration_percentages = []
        self.test_actions = []
        self.exploration_state_occurrences = []
        # Get the object-state pattern once (used for computing distances)
        self.objstate_pattern = self.get_object_state_pattern()

    def train(self):
        print("TRAINING PARAMETERS\n---------------------------------------")
        print("Total timesteps:", self.total_timesteps)
        print("Learning rate:", self.learning_rate)
        print("Number of total updates:", int(self.total_timesteps // self.batch_size))
        print("Number of parallel environments:", self.num_envs)
        print("Number of steps per rollout (Used for each kNN update and curiosity reward calculation):", self.num_steps)
        print("Batch size:", self.batch_size)
        print("Number of PPO update epochs:", self.update_epochs)
        print("Minibatch size:", self.minibatch_size)
        print("K parameter:", self.k)
        print("---------------------------------------")
        assert self.total_timesteps % self.batch_size == 0

        self.run = wandb.init(project="APT_PPO_Training", 
                    config={"env_id": self.env_id, 
                     "Mode": "training",
                    "Total timesteps": self.total_timesteps,
                    "Learning rate": self.learning_rate,
                    "Number of total updates": int(self.total_timesteps // self.batch_size),
                    "Number of parallel environments": self.num_envs,
                    "Number of steps per rollout": self.num_steps,
                    "Batch size": self.batch_size,
                    "Number of PPO update epochs": self.update_epochs,
                    "Minibatch size": self.minibatch_size,
                    "K parameter": self.k,
                    "Save frequency": self.save_freq})

        self.agent = Agent(self.env.action_space[0].n, self.env.single_observation_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )

        combined_parameters = list(self.agent.parameters())
        reward_rms = RunningMeanStd()
        discounted_reward = RewardForwardFilter(self.int_gamma)

        # ALGO Logic: Storage setup
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space[0].shape).to(self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + (self.env.observation_space.shape[1],)).to(self.device)                     
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.curiosity_rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        self.total_actions = []
        self.total_obs = []
        self.total_avg_curiosity_rewards = []
        self.model_saves = []

        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        avg_returns = deque(maxlen=20)

        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = int(self.total_timesteps // self.batch_size)
        num_test = 0

        for update in range(1, num_updates + 1):
            print("UPDATE: " + str(update) + "/" + str(num_updates))
            if update % self.save_freq == 0:
                print('Saving model...')
                self.model_saves.append([self.agent.state_dict(), self.optimizer.state_dict()])
                self.test_agent(save_episode=update, max_steps_per_episode=self.test_steps)
                num_test += 1

            # Anneal the learning rate if instructed.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    value_ext, value_int = self.agent.get_value(obs[step])
                    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])
                
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            self.total_actions.append(actions.clone())
            self.total_obs.append(obs.clone())

            # --- Corrected: Vectorized Construction of Similarity Matrices ---
            # obs shape: [num_steps, num_envs, obs_dim]
            # We compute a distance matrix for each environment using a vectorized method.
            similarity_matrices = []
            for env_idx in range(self.num_envs):
                env_obs = obs[:, env_idx, :]  # [num_steps, obs_dim]
                distance_matrix = self.compute_distance_matrix(env_obs)
                similarity_matrices.append(distance_matrix)
            # Stack into a tensor: shape [num_steps, num_steps, num_envs]
            sim_matrix = torch.stack(similarity_matrices, dim=-1)

            self.curiosity_rewards = self.compute_reward(sim_matrix)

            curiosity_reward_per_env = np.array(
                [discounted_reward.update(reward_per_step) for reward_per_step in self.curiosity_rewards.cpu().data.numpy().T]
            )
            mean, std, count = np.mean(curiosity_reward_per_env), np.std(curiosity_reward_per_env), len(curiosity_reward_per_env)
            print("Average reward:", mean)
            self.total_avg_curiosity_rewards.append(mean)

            reward_rms.update_from_moments(mean, std**2, count)
            self.curiosity_rewards /= np.sqrt(reward_rms.var)

            self.run.log({
                "Average Reward": mean,
                "Standard Deviation in Reward": std,
                "Actions": actions,
                "Observations": obs
            })

            # Compute advantages and returns (bootstrap value if not done)
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
                ext_advantages = torch.zeros_like(rewards, device=self.device)
                int_advantages = torch.zeros_like(self.curiosity_rewards, device=self.device)
                ext_lastgaelam = 0
                int_lastgaelam = 0
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
                    ext_advantages[t] = ext_lastgaelam = ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    int_advantages[t] = int_lastgaelam = int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam
                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

            # Flatten the batch.
            b_obs = obs.reshape((-1,) + (self.env.observation_space.shape[1],))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = ext_returns.reshape(-1)
            b_int_returns = int_returns.reshape(-1)
            b_ext_values = ext_values.reshape(-1)

            b_advantages = b_int_advantages * self.int_coef + b_ext_advantages * self.ext_coef

            # PPO update.
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                    if self.clip_vloss:
                        ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                        ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                            new_ext_values - b_ext_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                        ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                        ext_v_loss = 0.5 * ext_v_loss_max.mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                    v_loss = ext_v_loss + int_v_loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef 
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(combined_parameters, self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

    def save(self, path, env_kwargs):
        torch.save({
            'env_kwargs': env_kwargs,
            'model_saves': self.model_saves,
            'final_model_state_dict': self.agent.state_dict(),
            'final_optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'total_timesteps': self.total_timesteps,
            'num_envs': self.num_envs,
            'num_steps': self.num_steps,
            'curiosity_rewards': self.total_avg_curiosity_rewards,
            'actions': self.total_actions,
            'observations': self.total_obs,
            'exploration_percentages': self.exploration_percentages,
            'test_actions': self.test_actions,
            'test_action_heat_map': getattr(self, 'test_action_heat_map', None),
            'state_exploration_graph_per_obj': getattr(self, 'state_exploration_per_obj', None),
            'state_exploration_all_objs': getattr(self, 'state_exploration_all_objs', None),
            'test_exploration_state_occurrences': self.exploration_state_occurrences
        }, path)

    def get_object_state_pattern(self):
        """
        Returns a list of the number of object states per object for computing distance based on object states only.
        """
        test_env = gym.make(self.test_env_id, **self.test_env_kwargs)
        test_env = CustomObservationWrapper(test_env)
        pattern = []
        for obj_type in test_env.objs.values():
            for obj in obj_type:
                num_states = sum(1 for state_value in obj.states if not isinstance(obj.states[state_value], RelativeObjectState))
                pattern.append(num_states - 3)  # subtract three for infovofrobot, inhandofrobot, and inreachofrobot
        test_env.close()
        return pattern

    def compute_distance_matrix(self, env_obs):
        """
        Vectorized computation of a distance matrix for a single environment's observations.
        Uses a Hamming-like distance on the selected object state segments.
        
        Parameters:
            env_obs (torch.Tensor): Tensor of shape [num_steps, obs_dim].
            
        Returns:
            torch.Tensor: Distance matrix of shape [num_steps, num_steps].
        """
        num_steps, obs_dim = env_obs.shape
        total_distance = torch.zeros(num_steps, num_steps, device=env_obs.device)
        start_idx = 3
        for obj_len in self.objstate_pattern:
            state_start = start_idx + 5
            # Extract the relevant slice corresponding to the object states.
            # This gives a tensor of shape [num_steps, obj_len]
            slice_obs = env_obs[:, state_start: state_start + obj_len]
            # Compute pairwise differences: the result is [num_steps, num_steps, obj_len]
            diff = (slice_obs.unsqueeze(1) != slice_obs.unsqueeze(0)).float()
            # Sum over the last dimension to get the count of differences.
            total_distance += diff.sum(dim=-1)
            start_idx += obj_len + 5
        return total_distance

    def compute_reward(self, similarity_matrix):
        """
        Compute intrinsic (entropic) rewards based on k-nearest neighbors.
        Uses the vectorized similarity matrices.
        
        Parameters:
            similarity_matrix (torch.Tensor): Shape [num_steps, num_steps, num_envs]
        
        Returns:
            torch.Tensor: Intrinsic rewards of shape [num_steps, num_envs]
        """
        num_steps, _, num_envs = similarity_matrix.shape
        rewards = torch.zeros(num_steps, num_envs, device=similarity_matrix.device)
        # For each environment and each step, compute the average distance of the k-nearest neighbors.
        for env in range(num_envs):
            # For each step, sort distances and take the first k neighbors (excluding self)
            # Set the diagonal to a very high value so that self-distance is ignored.
            env_dist = similarity_matrix[:, :, env].clone()
            env_dist.fill_diagonal_(float('inf'))
            # Get the k smallest distances.
            k_nearest, _ = torch.topk(env_dist, k=self.k, largest=False)
            # Compute reward as log(c + average distance)
            avg_distance = k_nearest.mean(dim=1)
            rewards[:, env] = torch.log(self.c + avg_distance)
        return rewards

    def test_agent(self, save_episode, num_episodes=1, max_steps_per_episode=500):
        print(f"\n=== Testing Agent: {num_episodes} Episodes ===")
        action_log = []
        test_env = gym.make(self.test_env_id, **self.test_env_kwargs)
        test_env = CustomObservationWrapper(test_env)
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            obs = test_env.reset()
            done = False
            steps = 0
            frames = []
            episode_reward = 0
            
            while not done and steps < max_steps_per_episode:
                frames.append(np.moveaxis(test_env.render(), 2, 0))
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                obs, reward, done, _ = test_env.step(action.numpy()[0])
                episode_reward += reward
                action_log.append(test_env.actions(action.item()).name)
                print(f"Step {steps:3d} | Action: {test_env.actions(action.item()).name:10s} | ")
                steps += 1
                time.sleep(0.1)
            
            gif_path = f"{self.save_dir}/test_replays/episode_{save_episode}.gif"
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)  
            write_gif(np.array(frames), gif_path, fps=10)
            self.run.log({"episode_replay": wandb.Video(gif_path, fps=10, format="gif")})

        exploration_percentages = test_env.get_exploration_statistics2()
        exploration_state_occurrences = test_env.get_state_exploration_counts()
        self.exploration_state_occurrences.append(exploration_state_occurrences)
        self.exploration_percentages.append(exploration_percentages)
        self.test_actions.append(action_log)
        test_env.close()
