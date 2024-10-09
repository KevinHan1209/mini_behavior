# Active Pre-Training without Contrastive Learning for Discrete State and Action Spaces
import numpy as np
import torch 
import torch.optim as optim
from networks.actor import Agent
import numpy as np
from numpy import linalg as LA
from mini_behavior.roomgrid import *
from mini_behavior.utils.utils import RewardForwardFilter, dir_to_rad, schedule, recursive_getattr
import random
from gym.wrappers.normalize import RunningMeanStd
from collections import deque
import time
import torch.nn.functional as F
import torch.nn as nn

class APT_PPO():
    def __init__(self,
                 env,                     
                 device = "cpu",
                 total_timesteps: int = 2000000000,    # total timesteps of the experiments
                 learning_rate: float = 1e-4,          # the learning rate of the optimizer
                 num_envs: int = 128,                  # the number of parallel game environments
                 num_steps: int = 128,                 # the number of steps to run in each environment per policy rollout
                 anneal_lr: bool = True,               # Toggle learning rate annealing for policy and value networks
                 gamma: float = 0.999,                 # the discount factor gamma
                 gae_lambda: float = 0.95,             # the lambda for the general advantage estimation
                 num_minibatches: int = 4,             # the number of mini-batches
                 update_epochs: int = 4,               # the K epochs to update the policy
                 norm_adv: bool = True,                # Toggles advantages normalization
                 clip_coef: float = 0.1,               # the surrogate clipping coefficient
                 clip_vloss: bool = True,              # Toggles whether or not to use a clipped loss for the value function
                 ent_coef: float = 0.001,              # coefficient of the entropy
                 vf_coef: float = 0.5,                 # coefficient of the value function
                 max_grad_norm: float = 0.5,           # the maximum norm for the gradient clipping
                 target_kl: float = None,              # the target kl divergence threshold
                 int_coef: float = 1.0,                # coefficient of intrinsic reward
                 ext_coef: float = 2.0,                # coefficient of extrinsic reward
                 int_gamma: float = 0.99,              # intrinsic reward discount rate
                 th: float = 10,                       # threshold distance for k-nearest neighbors
                 c: float = 1                          # numerical stability constant
                 ): 
        self.env = env
        self.device = device
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
        self.th = th
        self.c = c

        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

    def train(self):

        self.agent = Agent(self.env.action_space.n, self.env.observation_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )
        combined_parameters = list(self.agent.parameters())

        reward_rms = RunningMeanStd()
        obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        discounted_reward = RewardForwardFilter(self.int_gamma)

         # ALGO Logic: Storage setup
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space.shape).to(self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + self.env.observation_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.curiosity_rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        avg_returns = deque(maxlen=20)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        assert self.total_timesteps % self.batch_size == 0
        num_updates = int(self.total_timesteps // self.batch_size)
        for update in range(1, num_updates + 1):
            print("UPDATE: " + str(update) + "/" + str(num_updates))
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    value_ext, value_int = self.agent.get_value(obs[step])
                    ext_values[step], int_values[step] = (
                        value_ext.flatten(),
                        value_int.flatten(),
                    )
                    #print("OBSERVATION:", len(obs[step]))
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])
                
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                #print("ACTION:", action)
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            # MODIFICATION FOR ATP: REPLACE CURIOSITY REWARDS FROM RND TO THAT FROM ATP
            batch_obs = obs[update * self.num_steps - self.num_steps : update * self.num_steps]
            for observation, ind in zip(batch_obs, range(len(batch_obs))):
                env_rewards = []
                for env_idx, env_observation in enumerate(observation):  # `observation` is (num_envs, obs_shape)
                    # Collect other steps in the same environment (from the `obs` tensor)
                    other_steps_in_env = [obs[step, env_idx] for step in range(obs.shape[0])]  # Extract for this environment
                    # Compute curiosity reward for the current environment
                    env_rewards.append(self.compute_reward(main=env_observation, others=[x for x in other_steps_in_env if not torch.equal(x, env_observation)]))
                self.curiosity_rewards[ind] = torch.tensor(env_rewards)

            curiosity_reward_per_env = np.array(
                [discounted_reward.update(reward_per_step) for reward_per_step in self.curiosity_rewards.cpu().data.numpy().T]
            )
            mean, std, count = (
                np.mean(curiosity_reward_per_env),
                np.std(curiosity_reward_per_env),
                len(curiosity_reward_per_env),
            )
            print("Average reward:", mean)

            reward_rms.update_from_moments(mean, std**2, count)
            self.curiosity_rewards /= np.sqrt(reward_rms.var)

            # bootstrap value if not done
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
                    ext_advantages[t] = ext_lastgaelam = (
                        ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    )
                    int_advantages[t] = int_lastgaelam = (
                        int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam
                    )
                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = ext_returns.reshape(-1)
            b_int_returns = int_returns.reshape(-1)
            b_ext_values = ext_values.reshape(-1)

            b_advantages = b_int_advantages * self.int_coef + b_ext_advantages * self.ext_coef
            #obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

            # Optimizing the policy and value network
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
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
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
                        nn.utils.clip_grad_norm_(
                            combined_parameters,
                            self.max_grad_norm,
                        )
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

    def save(self, path):
        """
        Save the model parameters to the specified path.
        """
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'total_timesteps': self.total_timesteps,
            'num_envs': self.num_envs,
            'num_steps': self.num_steps,
            'curiosity_rewards': self.curiosity_rewards
        }, path)

    def compute_distance(self, p1, p2, d1, d2, objs1, objs2):
        """
        Compute distance between two states in the MDP
        """
        p1, p2 = np.array(p1), np.array(p2)
        dp = LA.norm((p1 - p2), 1)
        hd = 0 # hamming distance
        for obj1, obj2 in zip(objs1, objs2):
            if obj1 != obj2:
                hd += 1
        # map direction to radians
        d1_r, d2_r = dir_to_rad(d1), dir_to_rad(d2)
        dd = abs(d1_r - d2_r)
        return dp + dd + hd
    
    def compute_reward(self, main, others):
        """
        Compute the entropic reward based on k-nearest neighbors
        main: The current state containing [position, direction, objs]
        others: A list of other states each containing [position, direction, objs]
        """
        k = 0
        total_distance = 0
        for z in others:
            assert len(main) == len(z)
            distance = self.compute_distance([main[0], main[1]], [z[0], z[1]], main[2], z[2], main[3: len(main)], z[3: len(z)])
            if distance <= self.th:
                total_distance += distance
                k += 1
        if k == 0:
            return 0
        return np.log(self.c + (total_distance) / k)



        
