# NovelD_PPO.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import wandb
from env_wrapper import CustomObservationWrapper
import os

def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk

class NovelD_PPO:
    """
    PPO implementation with Random Network Distillation for novelty detection
    """
    def __init__(
            self,
            env_id,
            device="cpu",
            total_timesteps=2000000,
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
            target_kl=None,
            int_coef=1.0,
            ext_coef=0.0,
            int_gamma=0.99,
            alpha=0.5,
            update_proportion=0.25,
            seed=1
            ):
        
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, seed, i) for i in range(num_envs)]
        )
        
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
        self.target_kl = target_kl
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.alpha = alpha
        self.update_proportion = update_proportion
        self.anneal_lr = True

        # Calculate batch sizes
        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        # Initialize running statistics
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=self.envs.single_observation_space.shape)

        # Use the environment's observation space
        self.obs_space = self.envs.single_observation_space
        self.obs_dim = self.obs_space.shape[0]

        # Initialize agent and RND model
        self.agent = Agent(self.obs_dim, self.envs.single_action_space.n).to(self.device).float()
        self.rnd_model = RNDModel(self.obs_dim).to(self.device).float()

        # Add validation for environment compatibility
        if not hasattr(self.envs.single_observation_space, 'shape'):
            raise ValueError("Environment must have an observation space with a shape attribute")
        
        # Additional reward scaling parameters
        self.reward_scale = 1.0
        self.novelty_scale = 1.0
        self.ext_reward_scale = 1.0
        self.int_reward_scale = 1.0

    def train(self):

        wandb.init(
            project="noveld-ppo-train",
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
                "int_coef": self.int_coef,
                "ext_coef": self.ext_coef,
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
        actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        curiosity_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)

        # Training loop variables
        global_step = 0
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size

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

                with torch.no_grad():
                    action, logprob, _, value_ext, value_int = self.agent.get_action_and_value(obs[step])
                actions[step] = action
                logprobs[step] = logprob
                ext_values[step] = value_ext.flatten()
                int_values[step] = value_int.flatten()

                next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
                rewards[step] = torch.FloatTensor(reward).to(self.device)
                next_obs = torch.FloatTensor(next_obs).to(self.device)
                next_done = torch.FloatTensor(done).to(self.device)

                with torch.no_grad():
                    novelty = self.calculate_novelty(next_obs)
                    curiosity_rewards[step] = self.normalize_rewards(novelty)

            # Compute advantages for extrinsic and intrinsic rewards.
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                ext_advantages, int_advantages = self.compute_advantages(
                    next_value_ext, next_value_int, rewards, curiosity_rewards,
                    ext_values, int_values, dones, next_done
                )

            # Flatten the rollout data.
            b_obs = obs.reshape((-1,) + (self.obs_dim,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = (b_ext_advantages + ext_values.reshape(-1))
            b_int_returns = (b_int_advantages + int_values.reshape(-1))
            # Combine advantages using the intrinsic coefficient.
            b_advantages = b_int_advantages * self.int_coef

            # Log the histogram of actions taken in this batch.
            wandb.log({
                "action_distribution": wandb.Histogram(b_actions.cpu().numpy())
            }, step=global_step)

            # Optimize policy and value networks while collecting metrics.
            opt_metrics = self.optimize(
                b_obs, b_logprobs, b_actions, b_advantages,
                b_ext_returns, b_int_returns, optimizer, global_step
            )

            # Log update-level training metrics.
            if update % 10 == 0:
                wandb.log({
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "novelty": novelty.mean().item(),
                    "curiosity_reward": curiosity_rewards.mean().item(),
                    "extrinsic_reward": rewards.mean().item(),
                    "global_step": global_step,
                    "updates": update,
                    **opt_metrics
                }, step=global_step)

                print(f"\n[Update {update}/{num_updates}] Step: {global_step:,}")
                print(f"Novelty: {novelty.mean().item():.4f} | Curiosity Reward: {curiosity_rewards.mean().item():.4f}")
                print(f"Policy Loss: {opt_metrics['pg_loss']:.4f} | Value Loss: {opt_metrics['v_loss']:.4f} | Entropy: {opt_metrics['entropy']:.4f}")
                print(f"KL: {opt_metrics['approx_kl']:.4f} | ClipFrac: {opt_metrics['clipfrac']:.4f}")
                print("-" * 50)

            if global_step % 500000 < self.num_envs:
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    'agent_state_dict': self.agent.state_dict(),
                    'rnd_predictor_state_dict': self.rnd_model.predictor.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint at {global_step} timesteps to {checkpoint_path}")

        wandb.finish()
        self.envs.close()

    def compute_advantages(self, next_value_ext, next_value_int, rewards, curiosity_rewards, ext_values, int_values, dones, next_done):
        next_value_ext = next_value_ext.flatten()
        next_value_int = next_value_int.flatten()
        
        ext_advantages = torch.zeros_like(rewards, device=self.device)
        int_advantages = torch.zeros_like(curiosity_rewards, device=self.device)
        ext_lastgaelam = torch.zeros(self.num_envs, device=self.device)
        int_lastgaelam = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                ext_nextnonterminal = 1.0 - next_done
                int_nextnonterminal = torch.ones_like(next_done)
                ext_nextvalues = next_value_ext
                int_nextvalues = next_value_int
            else:
                ext_nextnonterminal = 1.0 - dones[t + 1]
                int_nextnonterminal = torch.ones_like(dones[t + 1])
                ext_nextvalues = ext_values[t + 1]
                int_nextvalues = int_values[t + 1]

            ext_delta = (rewards[t] * self.ext_reward_scale) + \
                        self.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
            int_delta = (curiosity_rewards[t] * self.int_reward_scale) + \
                        self.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]

            ext_advantages[t] = ext_lastgaelam = ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
            int_advantages[t] = int_lastgaelam = int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam

        return ext_advantages, int_advantages

    def optimize(self, b_obs, b_logprobs, b_actions, b_advantages, b_ext_returns, b_int_returns, optimizer, global_step):
        # Normalize observations for the RND loss.
        rnd_next_obs = self.normalize_obs(b_obs)
        metrics = {
            "pg_loss": [],
            "v_loss": [],
            "entropy": [],
            "rnd_forward_loss": [],
            "total_loss": [],
            "approx_kl": [],
            "clipfrac": []
        }

        for epoch in range(self.update_epochs):
            inds = np.arange(self.batch_size)
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                # RND loss.
                predict_next_state_feature, target_next_state_feature = self.rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(predict_next_state_feature, target_next_state_feature.detach(), reduction="none").mean(-1)
                mask = (torch.rand(len(forward_loss), device=self.device) < self.update_proportion).float()
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1], device=self.device, dtype=torch.float32))
                metrics["rnd_forward_loss"].append(forward_loss.item())

                # New log probabilities, entropy and value predictions.
                _, newlogprob, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    metrics["approx_kl"].append(approx_kl.item())
                    clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    metrics["clipfrac"].append(clip_frac)

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                metrics["pg_loss"].append(pg_loss.item())

                # Value loss.
                new_ext_values = new_ext_values.view(-1)
                new_int_values = new_int_values.view(-1)
                ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = int_v_loss
                metrics["v_loss"].append(v_loss.item())

                # Combined loss.
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef + forward_loss
                metrics["entropy"].append(entropy_loss.item())
                metrics["total_loss"].append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()), self.max_grad_norm)
                optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

        # Average the collected metrics.
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics

    def calculate_novelty(self, obs):
        with torch.no_grad():
            normalized_obs = self.normalize_obs(obs)
            target_feature = self.rnd_model.target(normalized_obs)
            predict_feature = self.rnd_model.predictor(normalized_obs)
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

    def normalize_reward(self, reward):
        return reward / torch.sqrt(torch.FloatTensor([self.reward_rms.var]).to(self.device) + 1e-8)

    def normalize_rewards(self, rewards):
        if torch.isnan(rewards).any():
            print("Warning: NaN rewards detected")
            rewards = torch.nan_to_num(rewards, 0.0)
        rewards_np = rewards.detach().cpu().numpy()
        self.reward_rms.update(rewards_np.reshape(-1))
        normalized_rewards = rewards / torch.sqrt(
            torch.FloatTensor([self.reward_rms.var]).to(self.device) + 1e-8
        )
        return normalized_rewards.clamp(-10, 10)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.rnd_model.predictor.load_state_dict(checkpoint['rnd_predictor_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

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

class Agent(nn.Module):
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
        self.critic_ext = nn.Linear(448, 1)
        self.critic_int = nn.Linear(448, 1)
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

    def forward(self, x):
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature
