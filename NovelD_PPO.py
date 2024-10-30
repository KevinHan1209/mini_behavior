import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import gym
import os

class NovelD_PPO:
    def __init__(self, env, device="cpu", total_timesteps=1000, learning_rate=3e-4,
                 num_envs=1, num_steps=100, gamma=0.99, gae_lambda=0.95,
                 num_minibatches=4, update_epochs=4, clip_coef=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None,
                 int_coef=1.0, ext_coef=2.0, int_gamma=0.99, alpha=0.5,
                 update_proportion=0.25):
        # Initialize hyperparameters and environment
        self.env = env
        self.device = torch.device(device)
        self.total_timesteps = total_timesteps  # Changed default value to 20000
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.alpha = alpha
        self.update_proportion = update_proportion
        self.anneal_lr = True  # Add this line to define the anneal_lr attribute

        # Calculate batch sizes
        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        # Initialize running statistics
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)

        # Use the environment's observation space
        self.obs_space = env.observation_space
        self.obs_dim = self.obs_space.shape[0]

        # Initialize agent and RND model
        self.agent = Agent(self.obs_dim, env.action_space.n).to(self.device).float()
        self.rnd_model = RNDModel(self.obs_dim).to(self.device).float()

    def train(self):
        print(f"Starting training: {self.total_timesteps} timesteps, {self.batch_size} batch size")
        
        # Initialize optimizer
        optimizer = optim.Adam(list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()),
                               lr=self.learning_rate, eps=1e-5)

        # Initialize tensors for storing episode data
        next_obs = torch.FloatTensor(self.env.reset()).to(self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space.shape, dtype=torch.float32).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space.shape, dtype=torch.float32).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        curiosity_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)

        # Main training loop
        global_step = 0
        start_time = time.time()
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size

        prev_novelty = None
        for update in range(1, num_updates + 1):
            # Learning rate annealing
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Collect episode data
            for step in range(self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Get action and value
                with torch.no_grad():
                    value_ext, value_int = self.agent.get_value(obs[step])
                    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])

                actions[step] = action
                logprobs[step] = logprob

                # Execute action in environment
                action = action.cpu().numpy().squeeze()
                next_obs, reward, done, info = self.env.step(action.item())
                next_obs = torch.FloatTensor(next_obs).to(self.device)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_done = torch.FloatTensor([done]).to(self.device)

                # Calculate NovelD intrinsic reward
                novelty_curr = self.calculate_novelty(next_obs)
                if prev_novelty is None:
                    prev_novelty = novelty_curr
                
                # Ensure novelty_curr and prev_novelty are scalars
                if isinstance(novelty_curr, torch.Tensor):
                    novelty_curr = novelty_curr.item()
                if isinstance(prev_novelty, torch.Tensor):
                    prev_novelty = prev_novelty.item()
                
                noveld_reward = max(novelty_curr - self.alpha * prev_novelty, 0) * (info.get('first_visit', True))
                prev_novelty = novelty_curr
                curiosity_rewards[step] = noveld_reward

            # Update observation normalization
            self.obs_rms.update(obs.cpu().numpy().reshape(-1, self.obs_dim))

            # Normalize curiosity rewards
            curiosity_rewards_np = curiosity_rewards.cpu().numpy()
            flattened_rewards = curiosity_rewards_np.reshape(-1)  # Flatten to 1D array

            # Skip normalization if rewards are empty or invalid
            if flattened_rewards.size == 0 or np.any(np.isnan(flattened_rewards)):
                print("Warning: Invalid rewards detected. Skipping normalization.")
                continue

            # Update running statistics with flattened rewards
            self.reward_rms.update(flattened_rewards)

            # Normalize the rewards using running statistics
            # Convert running statistics to tensor and move to correct device
            rms_var_tensor = torch.FloatTensor([self.reward_rms.var]).to(self.device)
            curiosity_rewards = curiosity_rewards / torch.sqrt(rms_var_tensor + 1e-8)

            # Optional debugging prints
            if update % 10 == 0:  # Only print every 10 updates to avoid spam
                print(f"Reward statistics:")
                print(f"  Mean: {self.reward_rms.mean:.3f}")
                print(f"  Var: {self.reward_rms.var:.3f}")
                print(f"  Normalized rewards range: [{curiosity_rewards.min():.3f}, {curiosity_rewards.max():.3f}]")

            # Compute advantages and returns
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                ext_advantages, int_advantages = self.compute_advantages(next_value_ext, next_value_int, rewards, curiosity_rewards, ext_values, int_values, dones, next_done)

            # Flatten the batch
            b_obs = obs.reshape((-1,) + (self.obs_dim,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = (b_ext_advantages + ext_values.reshape(-1))
            b_int_returns = (b_int_advantages + int_values.reshape(-1))
            b_advantages = b_int_advantages * self.int_coef + b_ext_advantages * self.ext_coef

            # Optimize policy and value networks
            self.optimize(b_obs, b_logprobs, b_actions, b_advantages, b_ext_returns, b_int_returns, optimizer, global_step)

            if update % 10 == 0:  # Log periodically
                print(f"Update {update}/{num_updates}")
                print(f"Average reward: {rewards.mean():.3f}")
                print(f"Average novelty: {curiosity_rewards.mean():.3f}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 30)

        print(f"Training completed in {time.time() - start_time:.2f}s")

    def compute_advantages(self, next_value_ext, next_value_int, rewards, curiosity_rewards, ext_values, int_values, dones, next_done):
        # Compute GAE for extrinsic and intrinsic rewards
        ext_advantages = torch.zeros_like(rewards, device=self.device)
        int_advantages = torch.zeros_like(curiosity_rewards, device=self.device)
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
            int_delta = curiosity_rewards[t] + self.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]

            ext_advantages[t] = ext_lastgaelam = ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
            int_advantages[t] = int_lastgaelam = int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam

        return ext_advantages, int_advantages

    def optimize(self, b_obs, b_logprobs, b_actions, b_advantages, b_ext_returns, b_int_returns, optimizer, global_step):
        # Optimize policy and value networks
        rnd_next_obs = self.normalize_obs(b_obs)

        clipfracs = []

        for epoch in range(self.update_epochs):
            inds = np.arange(self.batch_size)
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                # RND Loss
                predict_next_state_feature, target_next_state_feature = self.rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(predict_next_state_feature, target_next_state_feature.detach(), reduction="none").mean(-1)
                mask = (torch.rand(len(forward_loss), device=self.device) < self.update_proportion).float()
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1], device=self.device, dtype=torch.float32))

                # Get new values and logprob
                _, newlogprob, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                # Policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values = new_ext_values.view(-1)
                new_int_values = new_int_values.view(-1)
                ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss

                # Combined loss
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()), self.max_grad_norm)
                optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

    def calculate_novelty(self, obs):
        # Calculate novelty using RND model
        normalized_obs = self.normalize_obs(obs)
        target_feature = self.rnd_model.target(normalized_obs)
        predict_feature = self.rnd_model.predictor(normalized_obs)
        
        # Ensure the tensors have the correct shape
        if target_feature.dim() == 1:
            target_feature = target_feature.unsqueeze(0)
        if predict_feature.dim() == 1:
            predict_feature = predict_feature.unsqueeze(0)
        
        # Calculate novelty
        novelty = ((target_feature - predict_feature) ** 2).sum(1) / 2
        
        # print("Calculated novelty:", novelty)  # Add this line
        
        # If novelty is a single value, return it as a scalar
        if novelty.numel() == 1:
            return novelty.item()
        else:
            return novelty

    def normalize_obs(self, obs):
        return ((obs - torch.FloatTensor(self.obs_rms.mean).to(self.device)) / 
                torch.sqrt(torch.FloatTensor(self.obs_rms.var).to(self.device) + 1e-8)).clip(-5, 5)

    def save_model(self, filename):
        try:
            model_state = {
                'agent': self.agent.state_dict(),
                'rnd_model': self.rnd_model.state_dict(),
                'obs_rms_mean': self.obs_rms.mean,
                'obs_rms_var': self.obs_rms.var,
                'reward_rms_mean': self.reward_rms.mean,
                'reward_rms_var': self.reward_rms.var,
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'int_coef': self.int_coef,
                    'ext_coef': self.ext_coef
                }
            }
            torch.save(model_state, filename)
            print(f"Model successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No model file found at {filename}")
        
        try:
            model_state = torch.load(filename, map_location=self.device)
            
            # Verify model components
            required_keys = ['agent', 'rnd_model', 'obs_rms_mean', 'obs_rms_var', 
                            'reward_rms_mean', 'reward_rms_var']
            if not all(k in model_state for k in required_keys):
                raise ValueError("Model file is missing required components")
            
            # Verify observation space matches
            if model_state['obs_rms_mean'].shape != self.obs_rms.mean.shape:
                raise ValueError(f"Model observation space {model_state['obs_rms_mean'].shape} "
                               f"does not match environment {self.obs_rms.mean.shape}")
            
            # Load state dictionaries
            self.agent.load_state_dict(model_state['agent'])
            self.rnd_model.load_state_dict(model_state['rnd_model'])
            self.obs_rms.mean = model_state['obs_rms_mean']
            self.obs_rms.var = model_state['obs_rms_var']
            self.reward_rms.mean = model_state['reward_rms_mean']
            self.reward_rms.var = model_state['reward_rms_var']
            
            # Optionally load hyperparameters if they exist
            if 'hyperparameters' in model_state:
                print("Loaded hyperparameters:", model_state['hyperparameters'])
            
            print(f"Model successfully loaded from {filename}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        if x.size == 0:
            return x

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        if np.any(np.isnan(batch_mean)) or np.any(np.isnan(batch_var)):
            return x

        self.update_from_moments(batch_mean, batch_var, batch_count)
        return x

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class Agent(nn.Module):
    # Agent class implementing the policy and value functions
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 448),
            nn.ReLU(),
            nn.Linear(448, 448),
            nn.ReLU()
        )
        self.actor = nn.Linear(448, action_dim)
        self.critic_ext = nn.Linear(448, 1)
        self.critic_int = nn.Linear(448, 1)
        
        # Ensure all parameters are float32
        self.float()

    def get_value(self, x):
        hidden = self.network(x)
        return self.critic_ext(hidden), self.critic_int(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_ext(hidden), self.critic_int(hidden)

class RNDModel(nn.Module):
    # Random Network Distillation model
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Ensure all parameters are float32
        self.float()

    def forward(self, x):
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature
