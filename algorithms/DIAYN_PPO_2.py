import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import wandb
import random
from collections import namedtuple, OrderedDict, deque
import collections
from torch.distributions import Categorical


Transition = namedtuple('Transition', ('state', 'z', 'action', 'next_state', 'done'))

class RolloutMemory:
    """Minimal on-policy buffer to store transitions for a single rollout."""
    def __init__(self):
        self.storage = []
    
    def add(self, state, z, action, next_state, done):
        self.storage.append(Transition(state, z, action, next_state, done))
    
    def get_all(self):
        return self.storage
    
    def clear(self):
        self.storage = []


class DIAYNObservationWrapper(gym.ObservationWrapper):
    """Appends skill ID to dictionary observations"""
    def __init__(self, env, skill_z=None):
        super().__init__(env)
        self.skill_z = skill_z
        self.n_skills = 8  # matches N_SKILLS
        
        spaces = dict(env.observation_space.spaces)
        spaces['skill'] = gym.spaces.Box(
            low=0, high=1,
            shape=(self.n_skills,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
    
    def observation(self, observation):
        skill_one_hot = np.zeros(self.n_skills)
        if 0 <= self.skill_z < self.n_skills:
            skill_one_hot[self.skill_z] = 1.0
        
        augmented_obs = dict(observation)
        augmented_obs['skill'] = skill_one_hot
        return augmented_obs


class RunningMeanStd:
    """Tracks running mean and stdev of input data"""
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
        
    def normalize(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RollingRewardNormalizer:
    """
    Maintains a rolling window of intrinsic rewards for normalization.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, rewards):
        if isinstance(rewards, (list, tuple)):
            self.buffer.extend(rewards)
        else:
            self.buffer.extend(rewards.tolist())

    def normalize(self, rewards):
        if len(self.buffer) < 10:
            return rewards  # not enough data to normalize

        mean = np.mean(self.buffer)
        std = np.std(self.buffer) + 1e-8

        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
            normalized = (rewards_np - mean) / std
            return torch.tensor(normalized, dtype=rewards.dtype, device=rewards.device)
        else:
            return (rewards - mean) / std


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
        self.critic = nn.Linear(448, 1)
        
        # Orthogonal initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1)
        nn.init.constant_(self.critic.bias, 0)

    def get_value(self, x):
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class Discriminator(nn.Module):
    """Discriminator network for skill classification"""
    def __init__(self, obs_dim, num_skills):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_skills)
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.classifier.weight, gain=1)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        hidden = self.network(x)
        return self.classifier(hidden)


class DIAYN:
    def __init__(
            self,
            env_id,
            device="cpu",
            total_timesteps=500000,
            learning_rate=3e-4,
            num_envs=8,
            num_steps=125,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=10,  # was 4
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_skills=8,
            seed=1
            ):

        self.n_skills = n_skills
        self.skills = np.arange(n_skills)
        self.current_skills = np.random.choice(self.skills, size=num_envs)
        
        def make_env(env_id, seed, idx, skill_z=None):
            def thunk():
                env = gym.make(env_id)
                env.seed(seed + idx)
                if hasattr(env, 'observation_space'):
                    env = DIAYNObservationWrapper(env, skill_z)
                return env
            return thunk
        
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, seed, i, self.current_skills[i]) for i in range(num_envs)]
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
        self.seed = seed
        self.anneal_lr = True

        self.reward_normalizer = RollingRewardNormalizer(window_size=1000)

        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        #observation dimensions
        self.obs_space = self.envs.single_observation_space
        if isinstance(self.obs_space, gym.spaces.Dict):
            self.obs_dim = 0
            #assume keys other than 'skill' are flattened
            for key, space in self.obs_space.spaces.items():
                if key != 'skill':
                    self.obs_dim += int(np.prod(space.shape))
        else:
            self.obs_dim = int(np.prod(self.obs_space.shape))

        #Observations + skill
        self.aug_obs_dim = self.obs_dim + self.n_skills
        
        self.agent = Agent(self.aug_obs_dim, self.envs.single_action_space.n).to(self.device)
        self.discriminator = Discriminator(self.obs_dim, self.n_skills).to(self.device)
        
        #rollout buffer for on-policy transitions that we just collected.
        self.disc_rollout = RolloutMemory()

    def create_optimizers(self):
        policy_optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        return policy_optimizer, disc_optimizer

    def train(self, save_freq=None, save_path=None):
        wandb.init(
            project="diayn-train",
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
                "n_skills": self.n_skills,
                "device": str(self.device)
            }
        )

        wandb.watch(self.agent, log="all", log_freq=100)
        wandb.watch(self.discriminator, log="all", log_freq=100)

        print("\n=== Training Configuration ===")
        print(f"Env: {self.env_id} | Device: {self.device}")
        print(f"Total Steps: {self.total_timesteps:,} | Batch Size: {self.batch_size} | Minibatch Size: {self.minibatch_size}")
        print(f"Learning Rate: {self.learning_rate} | Skills: {self.n_skills}")
        
        policy_optimizer, disc_optimizer = self.create_optimizers()

        def process_obs(obs_dict):
            flattened_obs = []
            if isinstance(obs_dict, dict):
                for key, value in sorted(obs_dict.items()):
                    if key != 'skill':
                        flattened_obs.append(value.reshape(value.shape[0], -1))
                processed = np.concatenate(flattened_obs, axis=1)
            else:
                processed = obs_dict.reshape(obs_dict.shape[0], -1)
            return processed

        reset_obs = self.envs.reset()
        processed_obs = process_obs(reset_obs)
        next_obs_base = torch.FloatTensor(processed_obs).to(self.device)

        skills_onehot = torch.zeros(self.num_envs, self.n_skills, device=self.device)
        for i, skill in enumerate(self.current_skills):
            skills_onehot[i, skill] = 1.0
        next_obs_aug = torch.cat([next_obs_base, skills_onehot], dim=1)

        obs_aug = torch.zeros((self.num_steps, self.num_envs, self.aug_obs_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        intrinsic_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)

        next_done = torch.zeros(self.num_envs).to(self.device)
        global_step = 0
        num_updates = self.total_timesteps // self.batch_size

        last_save_step = 0
        
        for update in range(1, num_updates + 1):
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                policy_optimizer.param_groups[0]["lr"] = lrnow
                disc_optimizer.param_groups[0]["lr"] = lrnow

            #clear the on-policy discriminator buffer each iteration
            self.disc_rollout.clear()

            for step in range(self.num_steps):
                global_step += self.num_envs
                obs_aug[step] = next_obs_aug
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs_aug)
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()

                action_list = action.detach().cpu().tolist()
                next_obs_dict, _, done, info = self.envs.step(action_list)

                next_obs_processed = process_obs(next_obs_dict)
                next_obs_base = torch.FloatTensor(next_obs_processed).to(self.device)

                # Create new augmented observation with skill
                skills_onehot = torch.zeros(self.num_envs, self.n_skills, device=self.device)
                for i, skill in enumerate(self.current_skills):
                    skills_onehot[i, skill] = 1.0
                next_obs_aug = torch.cat([next_obs_base, skills_onehot], dim=1)

                #store transitions for the discriminator using *next_obs_base*,
                # the same one we use for intrinsic reward:
                for i in range(self.num_envs):
                    self.disc_rollout.add(
                        # store current next_obs_base for classification
                        next_obs_base[i].cpu().clone(),
                        torch.tensor(self.current_skills[i], dtype=torch.long),
                        action[i].cpu(),
                        None,
                        done[i]
                    )

                #compute log p(z|s) using next_obs_base
                with torch.no_grad():
                    disc_logits = self.discriminator(next_obs_base)
                    log_q_z_given_s = F.log_softmax(disc_logits, dim=1)

                    # log p(z) for uniform distribution
                    log_p_z = torch.log(torch.tensor(1.0 / self.n_skills)).to(self.device)

                    # gather log q for each environment's skill
                    skill_indices = torch.LongTensor(self.current_skills).to(self.device)
                    skill_indices_expanded = skill_indices.unsqueeze(1)
                    
                    # raw_reward can easily be < -2. We'll keep only a mild clamp, say [-10, 0].
                    raw_reward = log_q_z_given_s.gather(1, skill_indices_expanded).squeeze() - log_p_z
                    clipped_reward = torch.clamp(raw_reward, min=-10.0, max=0.0)

                    self.reward_normalizer.update(clipped_reward.cpu().numpy())
                    intrinsic_r = self.reward_normalizer.normalize(clipped_reward)

                intrinsic_rewards[step] = intrinsic_r
                next_done = torch.FloatTensor(done).to(self.device)

                # If an env is done, resample skill
                for i in range(self.num_envs):
                    if done[i]:
                        new_skill = np.random.randint(self.n_skills)
                        self.current_skills[i] = new_skill
                        self.envs.envs[i].skill_z = new_skill

            if save_freq is not None and save_path is not None and (global_step - last_save_step) >= save_freq:
                save_file = f"{save_path}/model_step_{global_step}.pt"
                self.save(save_file)
                print(f"[Checkpoint] Model saved to {save_file} at step {global_step}")
                last_save_step = global_step
                wandb.log({"checkpoint_saved": True}, step=global_step)

            # Compute advantages/returns using only intrinsic_rewards
            with torch.no_grad():
                # bootstrap
                next_value = self.agent.get_value(next_obs_aug).flatten()
                advantages = self.compute_gae(next_value, intrinsic_rewards, dones, values, next_done)
                returns = advantages + values.reshape(-1)

            b_obs_aug = obs_aug.reshape(-1, self.aug_obs_dim)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            wandb.log({
                "intrinsic_reward_mean": intrinsic_rewards.mean().item(),
                "intrinsic_reward_std": intrinsic_rewards.std().item(),
            }, step=global_step)

            # Update PPO
            pg_loss, v_loss, entropy_loss = self.update_policy(
                b_obs_aug, b_logprobs, b_actions, b_advantages, b_returns, b_values, policy_optimizer
            )

            #train the discriminator ON-POLICY with data just collected
            disc_loss = self.update_discriminator(disc_optimizer)

            if update % 1 == 0:
                wandb.log({
                    "learning_rate": policy_optimizer.param_groups[0]["lr"],
                    "disc_learning_rate": disc_optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                    "policy_loss": pg_loss,
                    "value_loss": v_loss,
                    "entropy": entropy_loss,
                    "discriminator_loss": disc_loss,
                }, step=global_step)

                print(f"[Update {update}/{num_updates}] Step: {global_step:,}")
                print(f"Intrinsic R: {intrinsic_rewards.mean().item():.4f} | Policy Loss: {pg_loss:.4f} | Value Loss: {v_loss:.4f}")
                print(f"Discriminator Loss: {disc_loss:.4f} | Entropy: {entropy_loss:.4f}")

        wandb.finish()
        self.envs.close()

    def compute_gae(self, next_value, rewards, dones, values, next_done):
        advantages = torch.zeros_like(rewards, device=self.device)
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
            
        return advantages.reshape(-1)

    def update_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, optimizer):
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        value_clip_param = 0.2
        for epoch in range(self.update_epochs):
            inds = np.arange(self.batch_size)
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]
                
                _, newlogprob, entropy, new_values = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_values = new_values.view(-1)
                value_pred_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -value_clip_param,
                    value_clip_param
                )
                value_losses = (new_values - b_returns[mb_inds]).pow(2)
                value_losses_clipped = (value_pred_clipped - b_returns[mb_inds]).pow(2)
                v_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                total_entropy += entropy_loss.item()
        
        num_updates = self.update_epochs * (self.batch_size // self.minibatch_size)
        return (
            total_pg_loss / num_updates,
            total_value_loss / num_updates,
            total_entropy / num_updates
        )

    def update_discriminator(self, optimizer):
        """Train discriminator on the on-policy data from self.disc_rollout"""
        total_disc_loss = 0

        #do a few epochs
        transitions = self.disc_rollout.get_all()
        if len(transitions) < self.minibatch_size:
            return 0.0

        states = []
        skills = []
        for t in transitions:
            #stored next_obs_base in 'state'
            states.append(t.state.unsqueeze(0))  # shape [1, obs_dim]
            skills.append(t.z.unsqueeze(0))      # shape [1,]

        states = torch.cat(states).to(self.device)
        skills = torch.cat(skills).to(self.device)

        dataset_size = states.shape[0]
        for epoch in range(self.update_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = indices[start:end]

                mb_states = states[mb_inds]
                mb_skills = skills[mb_inds]

                logits = self.discriminator(mb_states)
                disc_loss = F.cross_entropy(logits, mb_skills)

                optimizer.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                optimizer.step()

                total_disc_loss += disc_loss.item()

        num_batches = self.update_epochs * (dataset_size // self.minibatch_size)
        if num_batches == 0:
            return float(total_disc_loss)  # in case dataset_size < minibatch_size

        return total_disc_loss / num_batches

    def save(self, path):
        torch.save({
            'agent': self.agent.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

    def test(self, skill_id=None, episodes=10, render=False):
        if skill_id is None:
            skill_id = np.random.choice(self.skills)
        print(f"Testing skill: {skill_id}")

        env = gym.make(self.env_id)
        if hasattr(env, 'observation_space'):
            env = DIAYNObservationWrapper(env, np.array(skill_id))
        
        self.agent.eval()
        rewards = []
        for ep in range(episodes):
            state_dict = env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    env.render()
                # process dict
                base_obs_list = []
                for key, val in sorted(state_dict.items()):
                    if key != 'skill':
                        base_obs_list.append(val.reshape(-1))
                base_obs = np.concatenate(base_obs_list, axis=0)
                base_obs_t = torch.FloatTensor(base_obs).unsqueeze(0).to(self.device)

                skill_one_hot = np.zeros(self.n_skills)
                if 0 <= skill_id < self.n_skills:
                    skill_one_hot[skill_id] = 1.0
                skill_t = torch.FloatTensor(skill_one_hot).unsqueeze(0).to(self.device)

                obs_aug_t = torch.cat([base_obs_t, skill_t], dim=1)

                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(obs_aug_t)
                action = action.item()

                next_state_dict, reward, done, _ = env.step(action)
                state_dict = next_state_dict
                total_reward += reward
            
            rewards.append(total_reward)
            print(f"Episode {ep+1}: reward={total_reward}")
        
        avg_reward = sum(rewards) / len(rewards)
        print(f"Average reward skill={skill_id}: {avg_reward}")
        env.close()
        self.agent.train()
        return avg_reward
