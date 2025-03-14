# Active Pre-Training without Contrastive Learning for Discrete State and Action Spaces

import numpy as np
import torch 
import torch.optim as optim
from networks.actor import Agent
from numpy import linalg as LA
from mini_behavior.roomgrid import *
from mini_behavior.utils.utils import RewardForwardFilter, RMS, dir_to_rad
from env_wrapper import CustomObservationWrapper
import random
from gym.wrappers.normalize import RunningMeanStd
from collections import deque
import time
import torch.nn.functional as F
from array2gif import write_gif
import torch.nn as nn
import wandb
import gym
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mini_behavior.register import register



class APT_PPO():
    def __init__(self,
                 env,
                 env_id,    
                 save_dir,
                 test_env_kwargs = None,               # kwargs for the test environment  
                 test_env_id = None,                   # id for the test environment                                          
                 device = "cpu",
                 save_freq: int = 100,                 # number of updates per model save         
        	     test_steps = 500,                     # number of steps per test episode
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
                 k: int = 50,                          # number of neighbors
                 c: float = 1,                         # numerical stability constant
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
                    "Number of steps per rollout": self.num_steps,  # Used for each kNN update and curiosity reward calculation
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

        # TRY NOT TO MODIFY: start the game
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
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])
                
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)


            self.total_actions.append(actions.clone())
            self.total_obs.append(obs.clone())

            # Refactor: Construct similarity matrices for each of the parallel environments
            # obs shape: torch.Size([128, 8, 14]) or torch.Size([# of rollout steps, # of parallel envs, dim of 1 observation])
            sim_matrix = self.construct_matrix(obs)
            self.curiosity_rewards = self.compute_reward(sim_matrix)

            curiosity_reward_per_env = np.array(
                [discounted_reward.update(reward_per_step) for reward_per_step in self.curiosity_rewards.cpu().data.numpy().T]
            )
            mean, std, count = (
                np.mean(curiosity_reward_per_env),
                np.std(curiosity_reward_per_env),
                len(curiosity_reward_per_env),
            )

            print("Average reward:", mean)
            self.total_avg_curiosity_rewards.append(mean)

            reward_rms.update_from_moments(mean, std**2, count)
            self.curiosity_rewards /= np.sqrt(reward_rms.var)

            self.run.log({
                    "Average Reward": mean,
                    "Standard Deviation in Reward": std,
                    "Actions": actions,
                    "Observations:": obs
                })

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
            b_obs = obs.reshape((-1,) + (self.env.observation_space.shape[1],))
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

    def save(self, path, env_kwargs):
        """
        Save the model parameters to the specified path.
        """
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
            'test_action_heat_map': self.test_action_heat_map,
            'state_exploration_graph_per_obj': self.state_exploration_per_obj,
            'state_exploration_all_objs': self.state_exploration_all_objs,
            'test_exploration_state_occurrences': self.exploration_state_occurrences
        }, path)

    def construct_matrix(self, observations):
        # Get the dimensions
        num_steps, num_envs, obs_dim = observations.shape

        # Initialize the similarity matrix for each parallel environment
        similarity_matrices = []

        # Loop through each environment
        for env_idx in range(num_envs):
            # Extract the observations for the current environment
            env_obs = observations[:, env_idx, :]  # Shape: [num_steps, obs_dim]
            
            # Initialize an empty matrix to store distances
            similarity_matrix = torch.zeros((num_steps, num_steps))  # Shape: [num_steps, num_steps]
            
            # Fill the similarity matrix
            for i in range(num_steps):
                for j in range(i, num_steps):
                    # Compute distance between observation i and observation j
                    distance = self.compute_distance(np.array(env_obs[i]), np.array(env_obs[j]))
                    similarity_matrix[i, j] = torch.tensor(distance, dtype=similarity_matrix.dtype, device=similarity_matrix.device)
                    similarity_matrix[j, i] = torch.tensor(distance, dtype=similarity_matrix.dtype, device=similarity_matrix.device) # symmetric property
            
            # Add the similarity matrix for this environment to the list
            similarity_matrices.append(similarity_matrix)

        # Stack all similarity matrices along a new dimension for the environments
        return torch.stack(similarity_matrices, dim=-1)  # Shape: [num_steps, num_steps, num_envs]

    def get_object_state_pattern(self):
        '''
        Returns a list of the number of object states per object for computing distance based on object states only
        '''
        test_env = gym.make(self.test_env_id, **self.test_env_kwargs)
        test_env = CustomObservationWrapper(test_env)
        pattern = []
        for obj_type in test_env.objs.values():
                    for obj in obj_type:
                        num_states = 0
                        for state_value in obj.states:
                            if not isinstance(obj.states[state_value], RelativeObjectState):
                                num_states += 1
                        pattern.append(num_states - 3) # subtract three for infovofrobot, inhandofrobot, and inreachofrobot
        test_env.close()
        return pattern
                        

    def compute_distance(self, obs1, obs2):
            """
            Compute distance between two states in the MDP
            """
            #p1, p2 = np.array([obs1[0], obs1[1]]), np.array(obs2[0], obs2[1])
            #dp = LA.norm((p1 - p2), 1)
            hd = 0

            #### ONLY IF USING OBJECT STATES ONLY ####
            start_idx = 3  # Initial start index for obs1 and obs2

            for obj_len in self.objstate_pattern:
                # Skip the two position elements for each object
                state_start = start_idx + 5
                
                # Loop through the object state entries
                for obj1, obj2 in zip(obs1[state_start: state_start + obj_len],
                                    obs2[state_start: state_start + obj_len]):
                    if obj1 != obj2:
                        hd += 1

                start_idx += obj_len + 5
            ##########################################

            ###### NORMAL CALCULATION ######
            #for obj1, obj2 in zip(obs1[3: len(obs1)], obs2[3: len(obs2)]):

            #    if obj1 != obj2:
            #        hd += 1
            #################################

            # map direction to radians
            #d1_r, d2_r = dir_to_rad(obs1[2]), dir_to_rad(obs2[2])
            #dd = abs(d1_r - d2_r)

            return hd # Modify when changing reward calculation 
    
    def compute_reward(self, similarity_matrix):
        """
        Compute entropic rewards based on k-nearest neighbors for each observation across environments.
        
        Parameters:
            similarity_matrix (torch.Tensor): Similarity matrix of shape [# of rollout steps, # of rollout steps, # of parallel envs].
        
        Returns:
            torch.Tensor: A tensor of rewards with shape [# of rollout steps, # of parallel envs].
        """
        # Get the number of steps and environments
        num_steps, _, num_envs = similarity_matrix.shape
        
        # Initialize a tensor to store rewards for each step and environment
        rewards = torch.zeros(num_steps, num_envs)
        
        for step in range(num_steps):
            for env in range(num_envs):
                # Get the distances from the current step to all other steps in the current environment
                distances = similarity_matrix[step, :, env].tolist()
                
                # Remove the self-distance (usually zero at step index)
                distances.pop(step)
                
                # Sort distances to find the k-nearest neighbors
                distances.sort()
                k_nearest = distances[:self.k] if len(distances) >= self.k else distances
                
                # Compute reward based on the average distance of k-nearest neighbors
                if len(k_nearest) == 0:
                    reward = 0
                else:
                    total_distance = sum(k_nearest)
                    reward = np.log(self.c + total_distance / len(k_nearest))
                
                # Store the computed reward for the current step and environment
                rewards[step, env] = reward
        
        return rewards  # Shape: [num_steps, num_envs]
    
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
                
                # Print step information
                print(f"Step {steps:3d} | "
                    f"Action: {test_env.actions(action.item()).name:10s} | ")
                
                steps += 1
                time.sleep(0.1)
            
            # Save gif as wandb artifact
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

    def evaluate_training(self):
        total_tests = len(self.exploration_percentages)
        objects = list(self.exploration_percentages[0].keys())  
        print(f"\n=== Final State Exploration Percentages ===")
        final_test = self.exploration_percentages[total_tests - 1]
        for obj, percentage in zip(objects, final_test.values()):
            print(str(obj) + ": " + str(percentage) + "% of total state")

        ################## GRAPH EVOLUTION OF STATE SPACE EXPLORATION FOR EACH OBJECT ##################
        data = {obj: [float(timestep[obj]) for timestep in self.exploration_percentages] for obj in objects}
        fig, ax = plt.subplots(figsize=(14, 8))
        for obj in objects:
            ax.plot(range(1, total_tests + 1), data[obj], marker='o', linestyle='-', label=obj)
        ax.set_xlabel('Test Stage')
        ax.set_ylabel('Percentage of State Space Explored')
        ax.set_title('State Space Exploration Over Test Stages')
        ax.legend(title='Objects', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
        plt.tight_layout()
        self.state_exploration_per_obj = fig

        ################## GRAPH EVOLUTION OF STATE SPACE EXPLORATION FOR ALL OBJECTS ##################
        aggregated_percentages = []
        for timestep in self.exploration_percentages:
            total_percentage = sum(float(value) for value in timestep.values())  # Sum of all object percentages
            num_objects = len(timestep)  # Total number of objects
            avg_percentage = total_percentage / num_objects  # Average percentage explored
            aggregated_percentages.append(avg_percentage)
        # Plot the evolution of state space exploration
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(range(1, total_tests + 1), aggregated_percentages, marker='o', linestyle='-', color='b')

        # Configure the plot
        ax2.set_xlabel('Test Stage')
        ax2.set_ylabel('Average Percentage of State Space Explored')
        ax2.set_title('Evolution of State Space Exploration for All Objects')
        ax2.grid(True)
        plt.tight_layout()
        self.state_exploration_all_objs = fig2

        ################## GRAPH HEAT MAP OF ACTIONS TAKEN PER TEST STAGE ##################
        unique_actions = sorted(set(action for actions in self.test_actions for action in actions))
        data = pd.DataFrame(0, index=range(1, len(self.test_actions) + 1), columns=unique_actions)

        for i, actions in enumerate(self.test_actions):
            for action in actions:
                data.loc[i + 1, action] += 1

        # Plot the heatmap
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.heatmap(data, annot=True, cmap="Blues", fmt='d', ax=ax3) 
        ax3.set_xlabel('Actions')
        ax3.set_ylabel('Testing Stages')
        ax3.set_title('Heatmap of Actions Taken at Each Testing Stage')
        plt.tight_layout()
        self.test_action_heat_map = fig3

        import io

        # For the first figure
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image = Image.open(buffer)
        self.run.log({"chart_1": wandb.Image(image, caption="State Space Exploration for Each Object")})
        buffer.close()

        # For the second figure
        buffer2 = io.BytesIO()
        fig2.savefig(buffer2, format='png')
        buffer2.seek(0)
        image2 = Image.open(buffer2)
        self.run.log({"chart_2": wandb.Image(image2, caption="Evolution of State Space Exploration for All Objects")})
        buffer2.close()

        # For the third figure (Heatmap)
        buffer3 = io.BytesIO()
        fig3.savefig(buffer3, format='png')
        buffer3.seek(0)
        image3 = Image.open(buffer3)
        self.run.log({"chart_3": wandb.Image(image3, caption="Heatmap of Actions Taken")})
        buffer3.close()

