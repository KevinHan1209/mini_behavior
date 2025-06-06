# RND_PPO.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mini_behavior.utils.utils import RewardForwardFilter, RMS #added
from networks.actor_critic import Agent
import gym
import wandb
from env_wrapper import CustomObservationWrapper
#from env_wrapper_no_position import CustomObservationWrapper #rnd without agent position

def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = CustomObservationWrapper(env)
        if hasattr(env, 'seed'):
            env.seed(seed + idx)
        return env
    return thunk

class RND_PPO:
    """
    PPO implementation with Random Network Distillation
    """
    def __init__(
            self,
            env, #added
            env_id,
            device="cpu",
            total_timesteps=500000,
            learning_rate=3e-4,
            num_envs=8,
            num_steps=125,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            #target_kl=None,
            #int_coef=1.0,
            #ext_coef=0.0,
            #int_gamma=0.99,
            #alpha=0.5,
            #update_proportion=0.25,
            rnd_reward_scale=1.0, #added
            seed=1
            ):
        
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, seed, i) for i in range(num_envs)]
        )
        self.env = env #new addition
        self.env_id = env_id
        self.device = torch.device(device)
        self.total_timesteps = total_timesteps
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
        #self.target_kl = target_kl
        #self.int_coef = int_coef
        #self.ext_coef = ext_coef
        #self.int_gamma = int_gamma
        #self.alpha = alpha
        #self.update_proportion = update_proportion
        self.rnd_reward_scale = rnd_reward_scale #added
        self.anneal_lr = True

        # Calculate batch sizes
        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        # Initialize running statistics
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=self.envs.single_observation_space.shape)
        self.rms = RMS(self.device)

        # Use the environment's observation space
        self.obs_space = self.envs.single_observation_space
        self.obs_dim = self.obs_space.shape[0]

        # Initialize agent and RND model
        #self.agent = Agent(self.obs_dim, self.envs.single_action_space.n).to(self.device).float()
        action_dims = self.envs.single_action_space.nvec  # [5, 2, 3]
        obs_dim = self.envs.single_observation_space.shape[0]

        self.agent = Agent(obs_dim=obs_dim, action_dims=action_dims).to(self.device)

        #action_dim = int(np.prod(self.envs.single_action_space.nvec))
        #self.agent = Agent(obs_dim=self.envs.single_observation_space.shape[0],action_dims=action_dim).to(self.device)
        #self.agent = Agent(self.env.action_space[0].nvec, self.env.single_observation_space.shape[0]).to(self.device)
        self.rnd_model = RNDModel(self.obs_dim).to(self.device).float()

        # Add validation for environment compatibility
        if not hasattr(self.envs.single_observation_space, 'shape'):
            raise ValueError("Environment must have an observation space with a shape attribute")
        
        # Additional reward scaling parameters
        #self.reward_scale = 1.0
        #self.novelty_scale = 1.0
        #self.ext_reward_scale = 1.0
        #self.int_reward_scale = 1.0

    def train(self, save_freq=None, save_path=None):

        wandb.init(
            project="rnd-ppo-train",
            config={
                "env_id": self.env_id,
                "total_timesteps": self.total_timesteps,
                "learning_rate": self.learning_rate,
                "num_envs": self.num_envs,
                "num_steps": self.num_steps,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "num_minibatches": self.num_minibatches,
                "update_epochs": self.update_epochs,
                "clip_coef": self.clip_coef,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                #"int_coef": self.int_coef,
                #"ext_coef": self.ext_coef,
                "device": str(self.device)
            }
        )

        # Watch models for gradients and parameter histograms.
        wandb.watch(self.agent, log="all", log_freq=100)
        wandb.watch(self.rnd_model, log="all", log_freq=100)
        
        print("\n=== Training Configuration ===")
        print(f"Env: {self.env_id} | Device: {self.device}")
        print(f"Total Steps: {self.total_timesteps:,} | Batch Size: {self.batch_size} | Minibatch Size: {self.minibatch_size}")
        print(f"Learning Rate: {self.learning_rate}\n")
        
        optimizer = optim.Adam(
            list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()),
            lr=self.learning_rate, eps=1e-5
        )
        
        next_obs = torch.FloatTensor(self.envs.reset()).to(self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space.shape, dtype=torch.float32).to(self.device)
        #actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs, 3), dtype=torch.long).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        extrinsic_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        intrinsic_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        #curiosity_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        #ext_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)

        # Training loop variables
        global_step = 0
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size

        last_save_step = 0

        for update in range(1, num_updates + 1):

            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Collect rollout data for one update
            for step in range(self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Get actions from the agent
                with torch.no_grad():
                    action, logprob, entropy, value_ext, value = self.agent.get_action_and_value(obs[step])
                    #print("Sampled action shape:", action.shape) 
                    #print("Sampled actions (first env) =", action[0])
                    #action, logprob, _, value = self.agent.get_action_and_value(obs[step])
                actions[step] = action
                logprobs[step] = logprob
                #ext_values[step] = value_ext.flatten()
                values[step] = value.flatten()

                action_list = action.detach().cpu().tolist()
                next_obs, reward, done, info = self.envs.step(action_list)

                with torch.no_grad():
                    next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
                    rnd_reward = self.calculate_novelty(next_obs_tensor)
                    reward_tensor = torch.tensor(reward, device=self.device)
                    rnd_reward_scaled = self.rnd_reward_scale * rnd_reward
                    combined_reward = reward_tensor + rnd_reward_scaled
                    #novelty = self.calculate_novelty(next_obs)
                    #curiosity_rewards[step] = self.normalize_rewards(novelty)

                #rewards[step] = torch.FloatTensor(reward).to(self.device)
                rewards[step] = combined_reward #training on both extrinsic and intrinsic
                next_obs = torch.FloatTensor(next_obs).to(self.device)
                next_done = torch.FloatTensor(done).to(self.device)

            if save_freq is not None and save_path is not None and (global_step - last_save_step) >= save_freq:
                save_file = f"{save_path}/model_step_{global_step}.pt"
                self.save(save_file)
                print(f"\n[Checkpoint] Model saved to {save_file} at step {global_step}")
            
                last_save_step = global_step
            
                wandb.log({"checkpoint_saved": True}, step=global_step)

            # Compute advantages rewards
            with torch.no_grad():
                next_value_ext, next_value = self.agent.get_value(next_obs) # added next_value_ext, 
                advantages = self.compute_gae(
                    next_value, rewards, dones, values, next_done
                )
            

            # Flatten the rollout data.
            b_obs = obs.reshape((-1,) + (self.obs_dim,))
            b_logprobs = logprobs.reshape(-1)
            #print("actions.shape before flatten:", actions.shape)
            b_actions = actions.reshape(-1, 3)
            #print("b_actions.shape after flatten:", b_actions.shape)
            #print("First row of b_actions =", b_actions[0])
            b_advantages = advantages.reshape(-1)
            b_returns = (b_advantages + values.reshape(-1))

            # Combine advantages using the intrinsic coefficient.
            #b_advantages = b_int_advantages * self.int_coef

            # Log the histogram of actions taken in this batch.
            wandb.log({
                "action_distribution": wandb.Histogram(b_actions.cpu().numpy())
            }, step=global_step)

            # Optimize policy and value networks while collecting metrics.
            opt_metrics = self.optimize(
                b_obs, b_logprobs, b_actions, b_advantages,
                b_returns, optimizer
            )

            # Log update-level training metrics.
            if update % 10 == 0:
                wandb.log({
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    #"novelty": novelty.mean().item(),
                    #"curiosity_reward": curiosity_rewards.mean().item(),
                    #"extrinsic_reward": rewards.mean().item(),
                    "global_step": global_step,
                    "rnd_reward": rnd_reward.mean().item(),
                    #"updates": update,
                    **opt_metrics
                }, step=global_step)

                print(f"\n[Update {update}/{num_updates}] Step: {global_step:,}")
                print(f"RND Reward: {rnd_reward_scaled.mean().item():.4f}")
                print(f"Policy Loss: {opt_metrics['pg_loss']:.4f} | Value Loss: {opt_metrics['v_loss']:.4f} | Entropy: {opt_metrics['entropy']:.4f}")
                print(f"RND Loss: {opt_metrics['rnd_loss']:.4f} | ClipFrac: {opt_metrics['clipfrac']:.4f}")
                print("-" * 50)

        wandb.finish()
        self.envs.close()

    def compute_gae(self, next_value, rewards, dones, values, next_done):
        #Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards, device=self.device)

        if next_value.dim() > 1:  # If shape is [8, 1]
            next_value = next_value.squeeze(-1)

        lastgaelam = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]

            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        return advantages

    def optimize(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, optimizer):
        # Normalize observations for the RND loss.
        metrics = {
            "pg_loss": [],
            "v_loss": [],
            "entropy": [],
            "rnd_loss": [],
            "total_loss": [],
            "clipfrac": []
        }

        for epoch in range(self.update_epochs):
            # Shuffle data
            inds = np.arange(self.batch_size)
            np.random.shuffle(inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                # New log probabilities, entropy and value predictions.
                _, newlogprob, entropy, new_value_ext, new_values = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Policy loss.
                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                metrics["pg_loss"].append(pg_loss.item())

                # Value loss.
                v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()
                metrics["v_loss"].append(v_loss.item())

                # RND loss.
                normalized_obs = self.normalize_obs(b_obs[mb_inds]) #added to remove agent position from rnd
                masked_obs = self.mask_agent_position(normalized_obs) #added to remove agent position from rnd
                predict_feature, target_feature = self.rnd_model(masked_obs)
                rnd_loss = F.mse_loss(predict_feature, target_feature.detach())
                metrics["rnd_loss"].append(rnd_loss.item())

                # Combined loss.
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef + rnd_loss
                metrics["entropy"].append(entropy_loss.item())
                metrics["total_loss"].append(loss.item())

                clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                metrics["clipfrac"].append(clip_fraction)

                optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()), self.max_grad_norm)
                optimizer.step()

        # Average the collected metrics.
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics
    
    def mask_agent_position(self, obs):
        obs_masked = obs.clone()
        obs_masked[:, 0:3] = 0.0  # Zero out agent_x, agent_y, agent_dir
        return obs_masked

    def calculate_novelty(self, obs):
        with torch.no_grad():
            normalized_obs = self.normalize_obs(obs)
            masked_obs = self.mask_agent_position(normalized_obs) #added to remove agent position from rnd
            target_feature = self.rnd_model.target(masked_obs) #added to remove agent position from rnd
            predict_feature = self.rnd_model.predictor(masked_obs) #added to remove agent position from rnd
            novelty = ((target_feature - predict_feature) ** 2).sum(1) / 2 + 1e-8
            return novelty.clamp(0, 10)

    def normalize_obs(self, obs):
        original_dim = len(obs.shape)
        if torch.isnan(obs).any():
            raise ValueError("NaN values detected in observations")
        if obs.requires_grad:
            obs = obs.detach()
        if original_dim == 1:
            obs = obs.unsqueeze(0)
        normalized = ((obs - torch.FloatTensor(self.obs_rms.mean).to(self.device)) / 
                    torch.sqrt(torch.FloatTensor(self.obs_rms.var).to(self.device) + 1e-8)).clip(-5, 5)
        return normalized.squeeze(0) if original_dim == 1 else normalized
    
    def save(self, path):
        torch.save({
            'agent': self.agent.state_dict(),
            'rnd_model': self.rnd_model.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent'])
        self.rnd_model.load_state_dict(checkpoint['rnd_model'])

class RunningMeanStd:
    """Tracks running mean and standard deviation of input data"""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.size == 0 or np.any(np.isnan(x)):
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

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

'''class Agent(nn.Module):
    """Neural network for policy and value functions"""
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
        self.critic = nn.Linear(448, 1)
        self.float()

    def get_value(self, x):
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)'''

class RNDModel(nn.Module):
    """Random Network Distillation model for novelty detection"""
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
        self.float()

        #initialize target network-remains fixed
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature
