# DIAYN_PPO.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import wandb
import random
from collections import namedtuple, OrderedDict
import collections
from torch.distributions import Categorical
from env_wrapper import CustomObservationWrapper

# Memory replay implementation
Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state'))

class Memory:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        self.buffer = []
        self.seed = seed
        self.priorities = None
        random.seed(self.seed)
        
    def add(self, *transition):
        self.buffer.append(Transition(*transition))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.buffer_size

    def sample(self, size):
        if hasattr(self, 'priorities') and self.priorities is not None and len(self.priorities) == len(self.buffer):
            indices = np.random.choice(
                len(self.buffer),
                size=min(size, len(self.buffer)),
                replace=False,
                p=self.priorities
            )
            return [self.buffer[i] for i in indices]
        else:
            return random.sample(self.buffer, min(size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
    def process_priorities(self):
        """Update priorities based on skill diversity"""
        if len(self.buffer) < 100:
            self.priorities = None
            return
        
        skill_counts = {}
        for transition in self.buffer:
            skill = transition.z.item()
            if skill not in skill_counts:
                skill_counts[skill] = 0
            skill_counts[skill] += 1
        
        total = sum(skill_counts.values())
        skill_weights = {skill: total / count for skill, count in skill_counts.items()}
        
        self.priorities = []
        for transition in self.buffer:
            skill = transition.z.item()
            self.priorities.append(skill_weights[skill])
        
        sum_weights = sum(self.priorities)
        self.priorities = [p / sum_weights for p in self.priorities]
    
    @staticmethod
    def get_rng_state():
        return random.getstate()
    
    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)

def make_env(env_id, seed, idx, skill_z=None):
    def thunk():
        env = gym.make(env_id)
        env.seed(seed + idx)
        if hasattr(env, 'observation_space'):
            env = DIAYNObservationWrapper(env, skill_z)
        return env
    return thunk

class DIAYNObservationWrapper(gym.ObservationWrapper):
    """Appends skill ID to dictionary observations"""
    def __init__(self, env, skill_z=None):
        super().__init__(env)
        self.skill_z = skill_z
        self.n_skills = 8  #matches N_SKILLS
        
        #add skill to dictionary observation space
        spaces = dict(env.observation_space.spaces)
        spaces['skill'] = gym.spaces.Box(
            low=0, high=1, 
            shape=(self.n_skills,), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)
        #print(self.observation_space)
    
    def observation(self, observation):
        # Create skill one-hot encoding
        skill_one_hot = np.zeros(self.n_skills)
        if 0 <= self.skill_z < self.n_skills:
            skill_one_hot[self.skill_z] = 1.0
        
        # Add to observation dictionary
        augmented_obs = dict(observation)
        augmented_obs['skill'] = skill_one_hot
        return augmented_obs

class DIAYN:
    """
    DIAYN implementation
    """
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
            update_epochs=8, #incresed from 4
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            disc_coef=0.1, #unused since no external reward
            n_skills=8, #new
            seed=1
            ):
        
        # Initialize skills
        self.n_skills = n_skills
        self.skills = np.arange(n_skills)
        self.current_skills = np.random.choice(self.skills, size=num_envs)
        
        # Create vector environment with skill encoding
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
        self.disc_coef = disc_coef
        self.seed = seed
        self.anneal_lr = True

        self.reward_normalizer = RunningMeanStd()

        # Calculate batch sizes
        self.batch_size = int(num_envs * num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = total_timesteps // self.batch_size

        # Determine observation dimensionality
        self.obs_space = self.envs.single_observation_space
        #print("Observation space:", self.obs_space)
        if isinstance(self.obs_space, gym.spaces.Dict):
        #calculate the flattened observation dimension without the skill
            self.obs_dim = 0
            for key, space in self.obs_space.spaces.items():
                if key != 'skill':  # Skip the skill part which we'll handle separately
                    self.obs_dim += int(np.prod(space.shape))
        else:
            self.obs_dim = int(np.prod(self.obs_space.shape))
        #print(f"Base observation dimension: {self.obs_dim}")

        self.aug_obs_dim = self.obs_dim + self.n_skills
        
        #initialize agent and discriminator
        self.agent = Agent(self.aug_obs_dim, self.envs.single_action_space.n).to(self.device)
        self.discriminator = Discriminator(self.obs_dim, self.n_skills).to(self.device)
        
        #memory buffer for experience replay
        self.memory = Memory(buffer_size=1000, seed=self.seed)

        #debugging
        if not hasattr(self.envs.single_observation_space, 'shape'):
            raise ValueError("Environment must have an observation space with a shape attribute")
        
    def create_optimizers(self):
        """Create separate optimizers for policy and discriminator"""
        policy_optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, #* 0.1,  #lower learning rate for discriminator
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
                "discriminator_coef": self.disc_coef,
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
            """Process observations correctly from dictionary or array format."""
            flattened_obs = []
            
            if isinstance(obs_dict, dict):
                # For each environment, extract and flatten all observation parts except skill
                for key, value in sorted(obs_dict.items()):  # Sort to ensure consistent ordering
                    if key != 'skill':  # Don't include skill in the base observation
                        if isinstance(value, np.ndarray):
                            flattened_obs.append(value.reshape(value.shape[0], -1))
                
                # Concatenate all parts
                processed = np.concatenate(flattened_obs, axis=1)
            else:
                # If not a dict, flatten
                processed = obs_dict.reshape(obs_dict.shape[0], -1)
                
            return processed
        
        #reset environments and get initial observations
        reset_obs = self.envs.reset()
        processed_obs = process_obs(reset_obs)
        
        #create base observation (without skill)
        next_obs_base = torch.FloatTensor(processed_obs).to(self.device)
        
        #create one-hot encoded skills tensor
        skills_onehot = torch.zeros(self.num_envs, self.n_skills, device=self.device)
        for i, skill in enumerate(self.current_skills):
            skills_onehot[i, skill] = 1.0
        
        #combine observation with skills
        next_obs_aug = torch.cat([next_obs_base, skills_onehot], dim=1)
        
        #initialize tensors for rollout collection
        obs_base = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), dtype=torch.float32).to(self.device)
        obs_aug = torch.zeros((self.num_steps, self.num_envs, self.aug_obs_dim), dtype=torch.float32).to(self.device)
        
        actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        intrinsic_rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32).to(self.device)
        skills = torch.LongTensor(self.current_skills).to(self.device)

        global_step = 0
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = self.total_timesteps // self.batch_size
        
        last_save_step = 0
        #track mean intrinsic reward for logging
        raw_reward_buffer = []

        for update in range(1, num_updates + 1):
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                policy_optimizer.param_groups[0]["lr"] = lrnow
                disc_optimizer.param_groups[0]["lr"] = lrnow

            for step in range(self.num_steps):
                global_step += self.num_envs
                obs_base[step] = next_obs_base
                obs_aug[step] = next_obs_aug
                dones[step] = next_done

                #get actions from the agent using aug obs with skill
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs_aug[step])
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()

                #step environment
                action_list = action.detach().cpu().tolist()
                next_obs_dict, reward, done, info = self.envs.step(action_list)

                # For each environment that finished an episode:
                for i in range(self.num_envs):
                    if done[i]:
                        #resample a new skill
                        new_skill = np.random.randint(self.n_skills)
                        self.current_skills[i] = new_skill

                        #update observation wrapper so that environment’s next reset or next obs will reflect this skill
                        self.envs.envs[i].skill_z = new_skill
                
                #process new observation correctly
                next_obs_processed = process_obs(next_obs_dict)
                next_obs_base = torch.FloatTensor(next_obs_processed).to(self.device)
                
                #create new augmented observation with skill
                skills_onehot = torch.zeros(self.num_envs, self.n_skills, device=self.device)
                for i, skill in enumerate(self.current_skills):
                    skills_onehot[i, skill] = 1.0
                next_obs_aug = torch.cat([next_obs_base, skills_onehot], dim=1)
                
                #store transitions in memory
                for i in range(self.num_envs):
                    try:
                        #extract the state without skill for discriminator
                        raw_obs = obs_base[step, i].cpu()
                        raw_next_obs = next_obs_base[i].cpu()
                        
                        #store transition with skill - making sure skill is a single value, not a list
                        skill_value = int(self.current_skills[i])
                        
                        # Create a 1D tensor with a single element for the skill
                        skill_tensor = torch.tensor([skill_value], dtype=torch.long)
                        
                        self.memory.add(
                            raw_obs,
                            skill_tensor,  # Store as 1D tensor with shape [1]
                            torch.BoolTensor([done[i]]),
                            action[i].cpu(),
                            raw_next_obs
                        )
                    except Exception as e:
                        print(f"Error adding to memory: {e}")
                        print(f"Shapes - obs: {obs_base[step, i].shape}, next_obs: {next_obs_base[i].shape}")
                        print(f"Skill: {self.current_skills[i]}, type: {type(self.current_skills[i])}")
                
                
                #calculate intrinsic rewards using discriminator on BASE observations
                with torch.no_grad():
                    disc_logits = self.discriminator(next_obs_base)
                    
                    #calculate log q(z|s) using discriminator
                    log_q_z_given_s = F.log_softmax(disc_logits, dim=1)
                    
                    #calculate log p(z) (uniform prior)
                    log_p_z = torch.log(torch.tensor(1.0 / self.n_skills)).to(self.device)
                    
                    #intrinsic reward is log p(z|s) - log p(z)
                    skill_indices = torch.LongTensor(self.current_skills).to(self.device)
                    skill_indices_expanded = skill_indices.unsqueeze(1)
                    #clip intrinsic rewards to prevent extreme negative values
                    raw_reward = log_q_z_given_s.gather(1, skill_indices_expanded).squeeze() - log_p_z
                    #clipped_reward = torch.clamp(raw_reward, min=-2.0, max=2.0)
                    clipped_reward = raw_reward
                    
                    #collect raw rewards for normalization
                    raw_reward_buffer.extend(clipped_reward.cpu().numpy())
                    
                    #if collected enough rewards, update the normalizer
                    if len(raw_reward_buffer) >= 100:
                        self.reward_normalizer.update(np.array(raw_reward_buffer))
                        raw_reward_buffer = []
                    
                    #apply normalization if we have enough data
                    if self.reward_normalizer.count > 100:
                        normalized_reward = self.reward_normalizer.normalize(clipped_reward.cpu().numpy())
                        intrinsic_r = torch.FloatTensor(normalized_reward).to(self.device)
                    else:
                        intrinsic_r = clipped_reward
                    
                rewards[step] = torch.FloatTensor(reward).to(self.device)
                intrinsic_rewards[step] = intrinsic_r
                
                next_done = torch.FloatTensor(done).to(self.device)

            if save_freq is not None and save_path is not None and (global_step - last_save_step) >= save_freq:
                save_file = f"{save_path}/model_step_{global_step}.pt"
                self.save(save_file)
                print(f"\n[Checkpoint] Model saved to {save_file} at step {global_step}")
                last_save_step = global_step
                wandb.log({"checkpoint_saved": True}, step=global_step)

            # Compute advantages and returns
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs_aug)  # Use augmented obs for value
                #use intrinsic rewards for learning policy
                advantages = self.compute_gae(next_value, intrinsic_rewards, dones, values, next_done)
                returns = advantages + values.reshape(-1)

            #flatten rollout data
            b_obs_aug = obs_aug.reshape(-1, self.aug_obs_dim)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            
            wandb.log({
                "action_distribution": wandb.Histogram(b_actions.cpu().numpy()),
                "intrinsic_rewards": intrinsic_rewards.mean().item(),
                "extrinsic_rewards": rewards.mean().item(),
                "raw_intrinsic_reward_mean": clipped_reward.mean().item(),
                "raw_intrinsic_reward_std": clipped_reward.std().item(),
                "normalized_intrinsic_reward_mean": intrinsic_r.mean().item(),
                "normalized_intrinsic_reward_std": intrinsic_r.std().item(),
                "skill_counts": {f"skill_{i}": (self.current_skills == i).sum() for i in range(self.n_skills)}
            }, step=global_step)
            
            # Optimize policy and value networks
            policy_loss, value_loss, entropy_loss = self.update_policy(
                b_obs_aug, b_logprobs, b_actions, b_advantages, b_returns, b_values, policy_optimizer
            )

            #update the memory priorities to balance skill sampling
            #self.memory.process_priorities()
            
            #train discriminator to classify skills from states
            #if update % 2 == 0 and len(self.memory) >= self.minibatch_size:
            disc_loss = self.update_discriminator(disc_optimizer)
            #else:
                #disc_loss = 0.0

            if update % 10 == 0:
                wandb.log({
                    "learning_rate": policy_optimizer.param_groups[0]["lr"],
                    "disc_learning_rate": disc_optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_loss,
                    "discriminator_loss": disc_loss,
                }, step=global_step)

                print(f"\n[Update {update}/{num_updates}] Step: {global_step:,}")
                print(f"Intrinsic Reward: {intrinsic_rewards.mean().item():.4f} | Extrinsic Reward: {rewards.mean().item():.4f}")
                print(f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Entropy: {entropy_loss:.4f}")
                print(f"Discriminator Loss: {disc_loss:.4f}")
                print("-" * 50)

        wandb.finish()
        self.envs.close()

    def compute_gae(self, next_value, rewards, dones, values, next_done):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards, device=self.device)
        
        if next_value.dim() > 1:
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
            
        return advantages.reshape(-1)

    def update_policy(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, optimizer):
        """Update the policy network using PPO"""
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Value function clipping as in PPO paper
        value_clip_param = 0.2
        
        for epoch in range(self.update_epochs):
            # Shuffle data
            inds = np.arange(self.batch_size)
            np.random.shuffle(inds)
            
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]
                
                # Get new policy and value predictions
                _, newlogprob, entropy, new_values = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                # Compute policy loss (clipped PPO objective)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss with clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                new_values = new_values.view(-1)

                # Clipped value function objective
                value_pred_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -value_clip_param,
                    value_clip_param
                )
                value_losses = (new_values - b_returns[mb_inds]).pow(2)
                value_losses_clipped = (value_pred_clipped - b_returns[mb_inds]).pow(2)
                v_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Scale down value loss to avoid overwhelming policy
                v_loss = v_loss * 0.5  # Reduce the impact of value loss
                #v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Calculate total loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                # Perform optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                total_entropy += entropy_loss.item()
        
        avg_pg_loss = total_pg_loss / (self.update_epochs * (self.batch_size // self.minibatch_size))
        avg_value_loss = total_value_loss / (self.update_epochs * (self.batch_size // self.minibatch_size))
        avg_entropy = total_entropy / (self.update_epochs * (self.batch_size // self.minibatch_size))
        
        return avg_pg_loss, avg_value_loss, avg_entropy
    
    def update_discriminator(self, optimizer):
        """Update the discriminator network to predict which skill is being executed from state information"""
        total_disc_loss = 0
        
        for _ in range(self.update_epochs):
            #sample batch from memory only when memory is large enough
            if len(self.memory) < self.minibatch_size:
                return 0.0
                
            batch = self.memory.sample(self.minibatch_size) #retrieved transitions
            
            #unpack batch
            #states = torch.cat([item.state.unsqueeze(0) for item in batch]).to(self.device)
            states = torch.cat([item.next_state.unsqueeze(0) for item in batch]).to(self.device)
            
            #convert each to a tensor with shape [1] before concatenating
            skills = []
            for item in batch:
                #if z is scalar unsqueeze it to make it a 1D tensor with one element
                if item.z.dim() == 0:
                    skills.append(item.z.unsqueeze(0))
                else:
                    #if already a tensor with shape [1] or more, use it directly
                    skills.append(item.z)
            
            #concatenate skills
            skills = torch.cat(skills).to(self.device)
            
            #forward pass to predict skill probabilities from state observations 
            logits = self.discriminator(states)
            
            #cross-entropy loss
            disc_loss = F.cross_entropy(logits, skills)
            
            #backward pass
            optimizer.zero_grad()
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            optimizer.step()
            
            total_disc_loss += disc_loss.item()
        
        return total_disc_loss / self.update_epochs

    def save(self, path):
        """Save model checkpoints"""
        torch.save({
            'agent': self.agent.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)

    def load(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

    def test(self, skill_id=None, episodes=10, render=False):
        """Test a specific skill"""
        if skill_id is None:
            skill_id = np.random.choice(self.skills)
            
        print(f"Testing skill: {skill_id}")
        
        # Create test environment
        env = gym.make(self.env_id)
        
        if hasattr(env, 'observation_space'):
            env = DIAYNObservationWrapper(env, np.array(skill_id))
        
        self.agent.eval()
        
        rewards = []
        for ep in range(episodes):
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            done = False
            total_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(state)
                
                action = action.item()
                next_state, reward, done, _ = env.step(action)
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                total_reward += reward
                state = next_state
                
            rewards.append(total_reward)
            print(f"Episode {ep+1}: Total reward = {total_reward}")
            
        avg_reward = sum(rewards) / len(rewards)
        print(f"Average reward for skill {skill_id}: {avg_reward}")
        
        env.close()
        self.agent.train()
        
        return avg_reward
    
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
        
    def normalize(self, x):
        """Normalizes data using running statistics"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
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
        
        # Initialize weights
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
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.classifier.weight, gain=0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        hidden = self.network(x)
        return self.classifier(hidden)