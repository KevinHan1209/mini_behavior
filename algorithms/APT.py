# Active Pre-Training without Contrastive Learning for Discrete State and Action Spaces
import numpy as np
import torch 
from networks.actor import ActorNetwork
from networks.critic import Critic
import numpy as np
from numpy import linalg as LA
from mini_behavior.roomgrid import *
from mini_behavior.utils.utils import dir_to_rad, schedule

class APT():
    def __init__(
            self,
            env,
            device,
            N, # mini-batch size
            th, # threshold distance for neighbors
            obs_dim,
            action_dim = 15,
            actor_hidden_dims = [256, 128, 64],
            critic_hidden_dims = [32,32],
            lr = 0.99,
            c = 1, # constant for numerical stability
            num_episodes = 10,
            num_timesteps = 100
    ):
        self.env = env
        self.N = N
        self.th = th
        self.lr = lr
        self.c = c
        self.obs_dim = obs_dim,
        self.action_dim = action_dim
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.device = device
        self.num_episodes = num_episodes
        self.num_timesteps = num_timesteps

        self.actor = ActorNetwork(self.obs_dim, self.action_dim, actor_hidden_dims).to(device)

        self.critic = Critic(self.obs_dim, self.action_dim, critic_hidden_dims).to(device)

    def train(self):
        for e in self.num_episodes:
            print('Episode ' + str(e))
            state, _ = self.env.reset(type = 'APT')
            t = 0
            while t < self.num_timesteps:
                action, _ = self.actor(state)
                obs = self.env.step(action, type = 'APT')



    def compute_distance(self, p1, p2, d1, d2, objs1, objs2):
        """
        Compute distance between two states in the MDP
        """
        dp = LA.norm((p1 - p2), 1)
        hd = 0 # hamming distance
        for obj_type1, obj_type2 in zip(objs1.values(), objs2.values()):
            for obj1, obj2 in zip(obj_type1, obj_type2):
                for state1, state2 in zip(obj1.states, obj2.states):
                    if isinstance(obj1.states[state1], RelativeObjectState):
                        continue
                    if obj1.states[state1].get_value() != obj2.states[state2].get_value():
                        hd += 1
        # map direction to radians
        d1_r, d2_r = dir_to_rad(d1), dir_to_rad(d2)
        dd = abs(d1_r - d2_r)
        return dp + dd + hd
    
    def compute_reward(self, main, neighbors):
        """
        Compute the entropic reward based on k-nearest neighbors
        main: The current state containing [position, direction, objs]
        neighbors: A list of neighbor states each containing [position, direction, objs]
        """
        n = len(neighbors)
        total_distance = 0
        for z in neighbors:
            distance = self.compute_distance(main[0], z[0], main[1], z[1], main[2], z[2])
            total_distance += distance
        return np.log(self.c + (total_distance ** n) / self.k)



        
