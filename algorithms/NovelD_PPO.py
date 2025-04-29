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
        env.action_space.seed(seed + idx)
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
            learning_rate=1e-4,
            num_envs=8,
            num_steps=200,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            clip_coef=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            ext_coef=0.0,
            int_coef=1.0,
            int_gamma=0.99,
            update_proportion=0.4,
            seed=1,
            log_interval=10,
            use_wandb=True
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
        self.ext_coef = ext_coef
        self.int_coef = int_coef
        self.int_gamma = int_gamma
        self.update_proportion = update_proportion
        self.anneal_lr = True
        self.log_interval = log_interval
        self.use_wandb = use_wandb

        # Calculate batch sizes
        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        # Running statistics for obs & rewards
        self.obs_rms = RunningMeanStd(shape=self.envs.single_observation_space.shape)
        self.reward_rms = RunningMeanStd()

        # Observation dimensions
        self.obs_space = self.envs.single_observation_space
        self.obs_dim = self.obs_space.shape[0]

        # Agent and RND
        self.agent = Agent(self.obs_dim, self.envs.single_action_space.nvec).to(self.device).float()
        self.rnd_model = RNDModel(self.obs_dim).to(self.device).float()
        self.seed = seed

        if not hasattr(self.envs.single_observation_space, 'shape'):
            raise ValueError("Environment must have an observation space with a shape attribute")

    def train(self):
        # Configure wandb if enabled
        if self.use_wandb:
            run = wandb.init(
                project="noveld-ppo-train",
                config={
                    "env_id": self.env_id,
                    "total_timesteps": self.total_timesteps,
                    "learning_rate": self.learning_rate,
                    "num_envs": self.num_envs,
                    "num_steps": self.num_steps,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "ext_coef": self.ext_coef,
                    "int_coef": self.int_coef,
                    "int_gamma": self.int_gamma,
                    "seed": self.seed,
                },
                mode="online"
            )
        
        # Log a short summary of training configuration
        print(f"📋 NovelD-PPO | {self.env_id} | {self.device}")
        print(f"📈 Steps: {self.total_timesteps:,} | LR: {self.learning_rate}")
        
        optimizer = optim.Adam(
            list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()),
            lr=self.learning_rate, eps=1e-5
        )
        next_obs = torch.FloatTensor(self.envs.reset()).to(self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space.shape, dtype=torch.float32).to(self.device)

        action_dims = self.envs.single_action_space.nvec
        action_shape = (len(action_dims),)
        actions = torch.zeros((self.num_steps, self.num_envs) + action_shape, dtype=torch.long).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        curiosity_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)

        global_step = 0
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size

        for update in range(1, num_updates + 1):
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * self.learning_rate

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

                next_obs_np, reward_np, done_np, _ = self.envs.step(action.cpu().numpy())
                # Update obs RMS
                self.obs_rms.update(next_obs_np)
                next_obs = torch.FloatTensor(next_obs_np).to(self.device)
                next_done = torch.FloatTensor(done_np).to(self.device)
                rewards[step] = torch.FloatTensor(reward_np).to(self.device)

                with torch.no_grad():
                    novelty = self.calculate_novelty(next_obs)
                    curiosity_rewards[step] = self.normalize_rewards(novelty)

            # Compute advantages
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                ext_adv, int_adv = self.compute_advantages(
                    next_value_ext, next_value_int, rewards, curiosity_rewards,
                    ext_values, int_values, dones, next_done
                )

            # Flatten rollout
            b_obs = obs.reshape((-1,) + (self.obs_dim,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, action_shape[0])
            b_ext_adv = ext_adv.reshape(-1)
            b_int_adv = int_adv.reshape(-1)
            b_ext_ret = b_ext_adv + ext_values.reshape(-1)
            b_int_ret = b_int_adv + int_values.reshape(-1)
            # Combine advantages
            b_advantages = self.ext_coef * b_ext_adv + self.int_coef * b_int_adv

            opt_metrics = self.optimize(
                b_obs, b_logprobs, b_actions, b_advantages,
                b_ext_ret, b_int_ret, optimizer, global_step
            )

            # Log metrics
            if update % self.log_interval == 0:
                # Collect metrics
                metrics = {
                    "train/update": update,
                    "train/step": global_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/novelty_mean": novelty.mean().item(),
                    "train/curiosity_reward_mean": curiosity_rewards.mean().item(),
                    "train/policy_loss": opt_metrics["pg_loss"],
                    "train/value_loss": opt_metrics["v_loss"],
                    "train/entropy": opt_metrics["entropy"],
                    "train/approx_kl": opt_metrics["approx_kl"],
                    "train/clipfrac": opt_metrics["clipfrac"],
                    "train/rnd_loss": opt_metrics["rnd_forward_loss"],
                }
                
                # Print concise summary
                print(f"⏱️  Update {update}/{num_updates} | Step {global_step:,}")
                print(f"🔍 Novelty: {metrics['train/novelty_mean']:.4f} | Reward: {metrics['train/curiosity_reward_mean']:.4f}")
                print(f"📉 Loss: {metrics['train/policy_loss']:.4f} | KL: {metrics['train/approx_kl']:.4f}")
                
                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log(metrics)
        
            if global_step % 400000 < self.num_envs or global_step < self.num_envs:
                os.makedirs("checkpoints", exist_ok=True)
                path = f"checkpoints/checkpoint_{global_step}.pt"
                torch.save({
                    'agent_state_dict': self.agent.state_dict(),
                    'rnd_predictor_state_dict': self.rnd_model.predictor.state_dict(),
                }, path)
                print(f"💾 Checkpoint saved: {path}")

        # Cleanup
        if self.use_wandb:
            wandb.finish()
        self.envs.close()
        print(f"✅ Training completed after {global_step:,} steps")

    def compute_advantages(self, next_val_ext, next_val_int, rewards, curiosity_rewards, ext_vals, int_vals, dones, next_done):
        next_ext = next_val_ext.flatten()
        next_int = next_val_int.flatten()
        ext_adv = torch.zeros_like(rewards, device=self.device)
        int_adv = torch.zeros_like(curiosity_rewards, device=self.device)
        last_ext = torch.zeros(self.num_envs, device=self.device)
        last_int = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nonterm_ext = 1.0 - next_done
                nonterm_int = torch.ones_like(next_done)
                val_ext = next_ext
                val_int = next_int
            else:
                nonterm_ext = 1.0 - dones[t + 1]
                nonterm_int = torch.ones_like(dones[t + 1])
                val_ext = ext_vals[t + 1]
                val_int = int_vals[t + 1]

            delta_ext = rewards[t] * self.ext_coef + self.gamma * val_ext * nonterm_ext - ext_vals[t]
            delta_int = curiosity_rewards[t] * self.int_coef + self.int_gamma * val_int * nonterm_int - int_vals[t]

            last_ext = delta_ext + self.gamma * self.gae_lambda * nonterm_ext * last_ext
            last_int = delta_int + self.int_gamma * self.gae_lambda * nonterm_int * last_int
            ext_adv[t] = last_ext
            int_adv[t] = last_int

        return ext_adv, int_adv

    def optimize(self, b_obs, b_logp, b_actions, b_adv, b_ext_ret, b_int_ret, optimizer, global_step):
        rnd_in = self.normalize_obs(b_obs)
        metrics = {k: [] for k in ["pg_loss","v_loss","entropy","rnd_forward_loss","total_loss","approx_kl","clipfrac"]}
        early_stop = False

        for epoch in range(self.update_epochs):
            idxs = np.arange(self.batch_size)
            np.random.shuffle(idxs)
            for start in range(0, self.batch_size, self.minibatch_size):
                mb = idxs[start:start + self.minibatch_size]

                # RND loss
                pred_feat, targ_feat = self.rnd_model(rnd_in[mb])
                fwd_loss = F.mse_loss(pred_feat, targ_feat.detach(), reduction="none").mean(-1)
                mask = (torch.rand(len(fwd_loss), device=self.device) < self.update_proportion).float()
                fwd_loss = (fwd_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                metrics["rnd_forward_loss"].append(fwd_loss.item())

                _, newlogp, entropy, new_ext_val, new_int_val = self.agent.get_action_and_value(b_obs[mb], b_actions.long()[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clipfrac"].append(clipfrac)

                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                metrics["pg_loss"].append(pg_loss.item())

                new_ext_val = new_ext_val.view(-1)
                new_int_val = new_int_val.view(-1)
                v_loss = 0.5 * ((new_int_val - b_int_ret[mb]) ** 2).mean()
                metrics["v_loss"].append(v_loss.item())

                ent_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * ent_loss + self.vf_coef * v_loss + fwd_loss
                metrics["entropy"].append(ent_loss.item())
                metrics["total_loss"].append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters()), self.max_grad_norm)
                optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        return {k: np.mean(v) for k, v in metrics.items()} 

    def calculate_novelty(self, obs):
        with torch.no_grad():
            normalized = self.normalize_obs(obs)
            target_feat = self.rnd_model.target(normalized)
            predict_feat = self.rnd_model.predictor(normalized)
            novelty = ((target_feat - predict_feat) ** 2).sum(1) / 2 + 1e-8
            return novelty.clamp(0, 10)

    def normalize_obs(self, obs):
        if obs.requires_grad:
            obs = obs.detach()
        original_dim = len(obs.shape)
        if original_dim == 1:
            obs = obs.unsqueeze(0)
        mean = torch.tensor(self.obs_rms.mean, device=self.device, dtype=torch.float32)
        var = torch.tensor(self.obs_rms.var, device=self.device, dtype=torch.float32)
        normalized = ((obs - mean) / (torch.sqrt(var + 1e-8))).clip(-5, 5)
        return normalized.squeeze(0) if original_dim == 1 else normalized

    def normalize_rewards(self, rewards):
        if torch.isnan(rewards).any():
            rewards = torch.nan_to_num(rewards, 0.0)
        arr = rewards.detach().cpu().numpy().reshape(-1)
        self.reward_rms.update(arr)
        var = torch.tensor(self.reward_rms.var, device=self.device, dtype=torch.float32)
        normed = rewards / torch.sqrt(var + 1e-8)
        return normed.clamp(-10, 10)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt['agent_state_dict'])
        self.rnd_model.predictor.load_state_dict(ckpt['rnd_predictor_state_dict'])
        print(f"Loaded checkpoint from {path}")

class RunningMeanStd:
    """Tracks running mean and variance"""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x)
        if x.size == 0 or np.isnan(x).any():
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, bm, bv, bc):
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + delta**2 * self.count * bc / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 448), nn.ReLU(),
            nn.Linear(448, 448), nn.ReLU()
        )
        self.action_dims = action_dim
        self.actor = nn.ModuleList([nn.Linear(448, d) for d in action_dim])
        self.critic_ext = nn.Linear(448, 1)
        self.critic_int = nn.Linear(448, 1)
        self.float()

    def get_value(self, x):
        h = self.network(x)
        return self.critic_ext(h), self.critic_int(h)

    def get_action_and_value(self, x, action=None):
        h = self.network(x)
        logits = [a(h) for a in self.actor]
        cats = [torch.distributions.Categorical(logits=lg) for lg in logits]
        if action is None:
            action = torch.stack([c.sample() for c in cats], dim=1)
        logp = torch.stack([c.log_prob(action[:, i]) for i, c in enumerate(cats)], dim=1).sum(1)
        ent = torch.stack([c.entropy() for c in cats], dim=1).sum(1)
        return action, logp, ent, self.critic_ext(h), self.critic_int(h)

class RNDModel(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.float()

    def forward(self, x):
        return self.predictor(x), self.target(x)
