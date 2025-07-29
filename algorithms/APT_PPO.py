import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
# import wandb
from array2gif import write_gif
import csv

from networks.actor_critic import Agent
from mini_behavior.roomgrid import *
from mini_behavior.utils.utils import RewardForwardFilter, RMS
from env_wrapper import CustomObservationWrapper
from gym.wrappers.normalize import RunningMeanStd
from mini_behavior.utils.states_base import RelativeObjectState


class APT_PPO:
    def __init__(self,
                 env,
                 env_id,
                 env_kwargs,
                 save_dir,
                 device="cpu",
                 save_freq=500,
                 test_steps=500,
                 total_timesteps=2000000,
                 learning_rate=1e-4,
                 num_envs=8,
                 num_eps=5,
                 num_steps=125,
                 anneal_lr=True,
                 gamma=0.99,
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
                 ext_coef=0.0,
                 int_gamma=0.99,
                 k=50,
                 c=1):
        self.env = env
        self.envs = env  # Alias for compatibility
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.save_dir = save_dir
        self.device = torch.device(device)
        self.save_freq = save_freq
        self.test_steps = test_steps
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_eps = num_eps
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

        # Precompute object-state pattern for distance calculations
        self.objstate_pattern = self.get_object_state_pattern()

    def train(self):
        """Train the agent using PPO."""
        print("\n=== Training Configuration ===")
        print(f"Env: {self.env_id} | Device: {self.device}")
        print(f"Total Steps: {self.total_timesteps:,} | Batch Size: {self.batch_size} | Minibatch Size: {self.minibatch_size}")
        print(f"Learning Rate: {self.learning_rate} | k parameter: {self.k}\n")
        assert self.total_timesteps % self.batch_size == 0

        
        # wandb.init(project="APT_PPO", config={
        #     "env_id": self.env_id,
        #     "Total timesteps": self.total_timesteps,
        #     "Learning rate": self.learning_rate,
        #     "Total updates": self.total_timesteps // self.batch_size,
        #     "Parallel envs": self.num_envs,
        #     "Steps per rollout": self.num_steps,
        #     "Batch size": self.batch_size,
        #     "PPO epochs": self.update_epochs,
        #     "Minibatch size": self.minibatch_size,
        #     "k parameter": self.k,
        #     "Save frequency": self.save_freq
        # })

        # Use the single environment observation space if available.
        obs_shape = getattr(self.env, "single_observation_space", self.env.observation_space).shape
        # Agent expects (action_dims, obs_dim) as positional arguments
        self.agent = Agent(self.envs.single_action_space.nvec, obs_shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        # wandb.watch(self.agent, self.optimizer)

        reward_rms = RunningMeanStd()
        discounted_reward = RewardForwardFilter(self.int_gamma)

        # Rollout storage
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape, device=self.device)
        obs = torch.zeros((self.num_steps, self.num_envs) + obs_shape, device=self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        curiosity_rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        ext_values = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        int_values = torch.zeros((self.num_steps, self.num_envs), device=self.device)

        global_step = 0
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs, device=self.device)
        num_updates = self.total_timesteps // self.batch_size

        # Training loop over updates
        for update in range(1, num_updates + 1):
            
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
                    ext_values[step] = value_ext.flatten()
                    int_values[step] = value_int.flatten()
                    action, logprob, _, _, _ = self.agent.get_action_and_value(obs[step])
                actions[step] = action
                logprobs[step] = logprob

                try:
                    next_obs_np, reward, done, _ = self.env.step(action.cpu().numpy())
                except Exception as e:
                    print(f"\n[ERROR] Environment step failed at global_step {global_step}, step {step}")
                    print(f"Error type: {type(e).__name__}: {e}")
                    
                    # First, try to identify which environment caused the error by stepping through them individually
                    failed_env_idx = None
                    for env_idx in range(self.num_envs):
                        try:
                            # Try to step each environment individually
                            single_action = action[env_idx].cpu().numpy()
                            test_obs, test_reward, test_done, test_info = self.env.envs[env_idx].step(single_action)
                        except Exception as test_e:
                            failed_env_idx = env_idx
                            print(f"\n!!! FOUND: Error occurred in environment {env_idx} !!!")
                            break
                    
                    if failed_env_idx is not None:
                        # Print detailed info only for the failing environment
                        env = self.env.envs[failed_env_idx]
                        actual_env = env.env if hasattr(env, 'env') else env
                        action_vals = action[failed_env_idx].cpu().numpy()
                        
                        print(f"\n--- Detailed info for failing Environment {failed_env_idx} ---")
                        print(f"Action values: {action_vals}")
                        print(f"Action: left_arm={action_vals[0]}, right_arm={action_vals[1]}, locomotion={action_vals[2]}")
                        
                        # Map action indices to names
                        if action_vals[0] < len(actual_env.manipulation_actions):
                            left_action = actual_env.manipulation_actions(action_vals[0])
                            print(f"Left arm action: {left_action.name} (index {action_vals[0]})")
                        if action_vals[1] < len(actual_env.manipulation_actions):
                            right_action = actual_env.manipulation_actions(action_vals[1])
                            print(f"Right arm action: {right_action.name} (index {action_vals[1]})")
                        if action_vals[2] < len(actual_env.locomotion_actions):
                            loco_action = actual_env.locomotion_actions(action_vals[2])
                            print(f"Locomotion action: {loco_action.name} (index {action_vals[2]})")
                        
                        # Print what agent is carrying
                        print(f"\nAgent state:")
                        print(f"  Carrying left: {actual_env.carrying['left']}")
                        print(f"  Carrying right: {actual_env.carrying['right']}")
                        print(f"  Position: {actual_env.agent_pos}")
                        print(f"  Direction: {actual_env.agent_dir}")
                        
                        # Print nearby objects following the action sequence
                        print(f"\nChecking surrounding cells:")
                        
                        # Define the sequences for left and right arms
                        front_pos = actual_env.front_pos
                        upper_left_pos = front_pos + np.array([-1, 0])
                        upper_right_pos = front_pos + np.array([1, 0])
                        left_pos = actual_env.agent_pos + np.array([-1, 0])
                        right_pos = actual_env.agent_pos + np.array([1, 0])
                        
                        left_seq_positions = [("front", front_pos), ("upper-left", upper_left_pos), ("left", left_pos)]
                        right_seq_positions = [("front", front_pos), ("upper-right", upper_right_pos), ("right", right_pos)]
                        
                        # Check cells for the failing arm
                        if action_vals[0] == 9:  # Left arm assemble
                            print("Left arm sequence:")
                            for name, pos in left_seq_positions:
                                if 0 <= pos[0] < actual_env.width and 0 <= pos[1] < actual_env.height:
                                    cell = actual_env.grid.get_all_items(*pos)
                                    print(f"  {name} position {pos}: {cell}")
                                    for dim, item in enumerate(cell):
                                        if item is not None and hasattr(item, 'name'):
                                            print(f"    Dim {dim}: {item.name}")
                                            if hasattr(item, 'states'):
                                                if 'attached' in item.states:
                                                    print(f"      attached: {item.states['attached'].get_value(actual_env)}")
                                                if 'contains' in item.states:
                                                    contained = item.states['contains'].get_contained_objs()
                                                    print(f"      contains: {[o.name for o in contained] if contained else 'empty'}")
                        
                        if action_vals[1] == 10:  # Right arm disassemble
                            print("Right arm sequence:")
                            for name, pos in right_seq_positions:
                                if 0 <= pos[0] < actual_env.width and 0 <= pos[1] < actual_env.height:
                                    cell = actual_env.grid.get_all_items(*pos)
                                    print(f"  {name} position {pos}: {cell}")
                                    for dim, item in enumerate(cell):
                                        if item is not None and hasattr(item, 'name'):
                                            print(f"    Dim {dim}: {item.name}")
                                            if hasattr(item, 'states'):
                                                if 'attached' in item.states:
                                                    print(f"      attached: {item.states['attached'].get_value(actual_env)}")
                                                if 'contains' in item.states:
                                                    contained = item.states['contains'].get_contained_objs()
                                                    print(f"      contains: {[o.name for o in contained] if contained else 'empty'}")
                    else:
                        print("Could not identify which specific environment caused the error")
                        print(f"All action values: {action.cpu().numpy()}")
                    
                    raise
                rewards[step] = torch.tensor(reward, device=self.device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

            self.total_actions.append(actions.clone())
            self.total_obs.append(obs.clone())
            
            # Check if we should save a checkpoint (every 500k steps)
            if global_step % 500000 < self.num_envs:
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
                print(f"Saving checkpoint at {global_step} timesteps to {checkpoint_path}")
                torch.save({
                    'agent_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, checkpoint_path)
                
                # Test the agent with 10 episodes of 200 steps each
                # Create a separate CSV file for this checkpoint in a dedicated directory
                csv_dir = os.path.join(checkpoint_dir, "activity_logs")
                os.makedirs(csv_dir, exist_ok=True)
                checkpoint_csv_path = os.path.join(csv_dir, f"checkpoint_{global_step}_activity.csv")
                self.test_agent(num_episodes=self.num_eps, max_steps_per_episode=self.test_steps,
                                checkpoint_path=checkpoint_path, checkpoint_id=global_step, 
                                save_episode=False, csv_path=checkpoint_csv_path)

            # Compute intrinsic (curiosity) rewards via kNN on state representations
            sim_matrix = self._compute_similarity_matrix(obs)
            curiosity_rewards = self.compute_reward(sim_matrix)

            # Normalize intrinsic rewards using running mean and std (computed from variance)
            reward_rms.update(curiosity_rewards.cpu().numpy())
            rms_mean = torch.tensor(reward_rms.mean, device=self.device)
            rms_std = torch.tensor(np.sqrt(reward_rms.var), device=self.device) + 1e-8
            curiosity_rewards = (curiosity_rewards - rms_mean) / rms_std

            avg_intrinsic = curiosity_rewards.mean().item()
            std_intrinsic = curiosity_rewards.std().item()
            self.total_avg_curiosity_rewards.append(avg_intrinsic)
            # wandb.log({
            #     "update": update,
            #     "global_step": global_step,
            #     "avg_intrinsic_reward": avg_intrinsic,
            #     "std_intrinsic_reward": std_intrinsic,
            #     "learning_rate": self.optimizer.param_groups[0]["lr"]
            # })

            # Compute advantages and returns
            ext_advantages = torch.zeros_like(rewards)
            int_advantages = torch.zeros_like(curiosity_rewards)
            ext_lastgaelam, int_lastgaelam = 0, 0
            with torch.no_grad():
                next_value_ext, next_value_int = self.agent.get_value(next_obs)
                next_value_ext = next_value_ext.reshape(1, -1)
                next_value_int = next_value_int.reshape(1, -1)
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
                    ext_lastgaelam = ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    int_lastgaelam = int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam
                    ext_advantages[t] = ext_lastgaelam
                    int_advantages[t] = int_lastgaelam

                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

            # Flatten rollout for PPO update
            b_obs = obs.reshape((-1,) + obs.shape[2:])
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
            total_pg_loss = []
            total_v_loss = []
            total_entropy = []
            total_approx_kl = []
            total_clipfrac = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = indices[start:end]
                    _, new_logprob, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = new_logprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                    total_approx_kl.append(approx_kl.item())
                    total_clipfrac.append(clipfrac.item())
                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    ).mean()
                    new_ext_values = new_ext_values.view(-1)
                    new_int_values = new_int_values.view(-1)
                    if self.clip_vloss:
                        ext_v_loss = 0.5 * torch.max(
                            (new_ext_values - b_ext_returns[mb_inds])**2,
                            (b_ext_values[mb_inds] + torch.clamp(new_ext_values - b_ext_values[mb_inds],
                                                                 -self.clip_coef, self.clip_coef) - b_ext_returns[mb_inds])**2
                        ).mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds])**2).mean()
                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds])**2).mean()
                    v_loss = ext_v_loss + int_v_loss
                    loss = pg_loss - self.ent_coef * entropy.mean() + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_pg_loss.append(pg_loss.item())
                    total_v_loss.append(v_loss.item())
                    total_entropy.append(entropy.mean().item())

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break
            
            # Log update-level training metrics every 10 updates
            if update % 10 == 0:
                avg_pg_loss = np.mean(total_pg_loss)
                avg_v_loss = np.mean(total_v_loss)
                avg_entropy = np.mean(total_entropy)
                avg_approx_kl = np.mean(total_approx_kl)
                avg_clipfrac = np.mean(total_clipfrac)
                
                print(f"\n[Update {update}/{num_updates}] Step: {global_step:,}")
                print(f"Intrinsic Reward: {avg_intrinsic:.4f} | Std: {std_intrinsic:.4f}")
                print(f"Policy Loss: {avg_pg_loss:.4f} | Value Loss: {avg_v_loss:.4f} | Entropy: {avg_entropy:.4f}")
                print(f"KL: {avg_approx_kl:.4f} | ClipFrac: {avg_clipfrac:.4f}")
                print("-" * 50)

    def test_agent(self, num_episodes, max_steps_per_episode, checkpoint_path, checkpoint_id, save_episode=False, csv_path=None):
        """
        Run test episodes using the current agent policy. If a checkpoint_path is provided, load it before testing.
        Save CSV logs for each episode in a folder named after the checkpoint and log that folder to wandb.
        """
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path} for testing.")
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            self.agent.load_state_dict(checkpoint_data['agent_state_dict'])
        
        print(f"\n=== Testing Agent on Checkpoint {checkpoint_id}: {num_episodes} Episode(s) ===")
        action_log = []
        test_env = gym.make(self.env_id, **self.env_kwargs)
        test_env = CustomObservationWrapper(test_env)

        # Create a dedicated folder for this checkpoint's activity logs if csv_path not provided
        if csv_path is None:
            activity_dir = os.path.join('item_interaction', f"checkpoint_{checkpoint_id}")
            os.makedirs(activity_dir, exist_ok=True)

        def count_binary_flags(env):
            """
            Count the total number of non-relative (binary) state flags in the environment.
            """
            default_states = [
                'atsamelocation',
                'infovofrobot',
                'inleftreachofrobot',
                'inrightreachofrobot',
                'inside',
                'nextto',
                'inlefthandofrobot',
                'inrighthandofrobot',
            ]
            
            num_flags = 0
            for obj_list in env.objs.values():
                for obj in obj_list:
                    for state_name, state in obj.states.items():
                        if not isinstance(state, RelativeObjectState):
                            if state_name not in default_states:
                                num_flags += 1
            return num_flags

        def generate_flag_mapping(env):
            """
            Generate a mapping that tells which binary flag corresponds to which object's state.
            Skip default states that are not included in observations.
            """
            default_states = [
                'atsamelocation',
                'infovofrobot',
                'inleftreachofrobot',
                'inrightreachofrobot',
                'inside',
                'nextto',
                'inlefthandofrobot',
                'inrighthandofrobot',
            ]
            
            mapping = []
            for obj_type, obj_list in env.objs.items():
                for idx, obj in enumerate(obj_list):
                    for state_name, state in obj.states.items():
                        if not isinstance(state, RelativeObjectState):
                            if state_name not in default_states:
                                mapping.append({
                                    "object_type": obj_type,
                                    "object_index": idx,
                                    "state_name": state_name
                                })
            return mapping

        def extract_binary_flags(obs, env):
            """
            Extract only the binary flags from an observation vector.
            Skip position values and default states that are not included.
            """
            default_states = [
                'atsamelocation',
                'infovofrobot',
                'inleftreachofrobot',
                'inrightreachofrobot',
                'inside',
                'nextto',
                'inlefthandofrobot',
                'inrighthandofrobot',
            ]
            
            flags = []
            index = 3  # skip agent state (x, y, direction)
            
            for obj_list in env.objs.values():
                for obj in obj_list:
                    # Skip the object's position (2 values)
                    index += 2
                    
                    for state_name, state in obj.states.items():
                        if not isinstance(state, RelativeObjectState):
                            if state_name not in default_states:
                                # This is a binary state that should be in the observation
                                if index < len(obs):
                                    flags.append(obs[index])
                                else:
                                    flags.append(0)
                                index += 1
                            # Skip default states as they're not included in observation
            return np.array(flags)

        num_binary_flags = count_binary_flags(test_env.env if hasattr(test_env, 'env') else test_env)
        flag_mapping = generate_flag_mapping(test_env.env if hasattr(test_env, 'env') else test_env)
        
        # Track aggregated activity counts across all episodes
        total_activity_counts = np.zeros(num_binary_flags)

        for ep in range(num_episodes):
            obs = test_env.reset()
            done = False
            steps = 0
            frames = []
            activity = [0] * num_binary_flags
            prev_flags = None

            while not done and steps < max_steps_per_episode:
                frame = test_env.render()
                frames.append(np.moveaxis(frame, 2, 0))
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                action_value = action.cpu().numpy()[0]
                obs, _, done, _ = test_env.step(action_value)

                current_flags = extract_binary_flags(obs, test_env.env if hasattr(test_env, 'env') else test_env)
                if prev_flags is not None:
                    differences = (current_flags != prev_flags).astype(int)
                    activity = [a + d for a, d in zip(activity, differences)]
                    # Also add to total activity counts
                    total_activity_counts += differences
                prev_flags = current_flags

                action_name = test_env.actions(action_value).name
                action_log.append(action_name)
                print(f"Step {steps:3d} | Action: {action_name}")
                steps += 1

            # Only write individual episode CSVs if no csv_path provided
            if csv_path is None:
                csv_file = os.path.join(activity_dir, f'episode_{ep+1}.csv')
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['flag_id', 'object_type', 'object_index', 'state_name', 'activity_count'])
                    for idx, count in enumerate(activity):
                        mapping = flag_mapping[idx]
                        writer.writerow([idx, mapping['object_type'], mapping['object_index'], mapping['state_name'], count])

            if save_episode:
                gif_path = os.path.join(self.save_dir, f"episode_{ep+1}_checkpoint_{checkpoint_id}.gif")
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                write_gif(np.array(frames), gif_path, fps=10)

        # Write aggregated activity counts to CSV if path provided
        if csv_path is not None:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                header = ['checkpoint_id']
                for mapping in flag_mapping:
                    header.append(f"{mapping['object_type']}_{mapping['object_index']}_{mapping['state_name']}")
                writer.writerow(header)
                
                # Write the activity counts for this checkpoint
                row = [checkpoint_id]
                row.extend(total_activity_counts.astype(int))
                writer.writerow(row)
                
            print(f"\nActivity counts saved to {csv_path}")
            print(f"Total state changes observed across {num_episodes} episodes:")
            for idx, count in enumerate(total_activity_counts):
                if count > 0:
                    mapping = flag_mapping[idx]
                    print(f"  {mapping['object_type']}_{mapping['object_index']}_{mapping['state_name']}: {int(count)}")

        test_env.close()
        self.test_actions.append(action_log)

    def get_object_state_pattern(self):
        """
        Precompute the object state pattern for distance calculations.
        """
        test_env = gym.make(self.env_id, **self.env_kwargs)
        test_env = CustomObservationWrapper(test_env)
        pattern = []
        
        # Access the unwrapped environment's objs
        unwrapped_env = test_env.env if hasattr(test_env, 'env') else test_env
        for obj_type in unwrapped_env.objs.values():
            for obj in obj_type:
                num_states = sum(1 for state in obj.states if not isinstance(obj.states[state], RelativeObjectState))
                pattern.append(num_states - 3)
        test_env.close()
        return pattern

    def compute_distance_matrix(self, env_obs):
        """
        Compute a Hamming-like distance matrix over object-state slices.
        """
        num_steps = env_obs.shape[0]
        total_distance = torch.zeros((num_steps, num_steps), device=env_obs.device)
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
        Compute intrinsic rewards as log(c + average kNN distance).
        """
        num_steps, _, num_envs = sim_matrix.shape
        rewards = torch.zeros((num_steps, num_envs), device=sim_matrix.device)
        for env in range(num_envs):
            env_dist = sim_matrix[:, :, env].clone()
            env_dist.fill_diagonal_(float('inf'))
            # Use min(k, num_steps-1) to avoid out of range error
            k_use = min(self.k, num_steps - 1)
            if k_use > 0:
                k_nearest, _ = torch.topk(env_dist, k=k_use, largest=False)
                avg_distance = k_nearest.mean(dim=1)
                rewards[:, env] = torch.log(self.c + avg_distance)
            else:
                rewards[:, env] = 0.0
        return rewards

    def _compute_similarity_matrix(self, obs):
        """
        Build a similarity (distance) matrix for each environment.
        """
        sim_matrices = []
        for env_idx in range(self.num_envs):
            env_obs = obs[:, env_idx, :]
            sim_matrices.append(self.compute_distance_matrix(env_obs))
        return torch.stack(sim_matrices, dim=-1)
