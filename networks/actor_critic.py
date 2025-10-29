import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.categorical import Categorical
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union

class MultiCategorical:
    def __init__(self, logits, action_dims):
        """
        :param logits: Tensor of shape [batch, sum(action_dims)]
        :param action_dims: List of integers for each discrete action space.
        """
        self.action_dims = action_dims
        # Split the logits for each action dimension.
        self.logits_split = torch.split(logits, tuple(action_dims), dim=-1)
        # Create a Categorical distribution for each split.
        self.distributions = [Categorical(logits=logit) for logit in self.logits_split]

    def sample(self):
        # Sample an action from each distribution and stack them into a single tensor.
        samples = [dist.sample() for dist in self.distributions]
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions):
        # Unbind the actions along the last dimension so we get one tensor per action dimension.
        actions_unbind = torch.unbind(actions, dim=-1)
        # Calculate the log prob for each action and sum them.
        log_probs = [dist.log_prob(a) for dist, a in zip(self.distributions, actions_unbind)]
        return sum(log_probs)

    def entropy(self):
        # Calculate the entropy for each distribution and sum them.
        entropies = [dist.entropy() for dist in self.distributions]
        return sum(entropies)

    def mode(self):
        # Take the most likely action for each categorical component.
        modes = [dist.logits.argmax(dim=-1) for dist in self.distributions]
        return torch.stack(modes, dim=-1)

def layer_init(layer, std=torch.sqrt(torch.tensor(2.0)), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, action_dims, obs_dim):
        """
        :param action_dims: List of sizes for each discrete action space.
        :param obs_dim: Dimension of the observation space.
        """
        super().__init__()
        self.action_dims = action_dims
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),  # Input layer
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),       # Hidden layer
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1),
            nn.ReLU()
        )
        # Actor network split: a hidden layer and a final linear layer outputting logits.
        self.actor_hidden = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
        )
        # The final layer outputs logits for all discrete action spaces.
        self.actor_logits = layer_init(nn.Linear(448, sum(action_dims)), std=0.01)
        
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        actor_h = self.actor_hidden(hidden)
        logits = self.actor_logits(actor_h)
        
        # Build our custom multi-categorical distribution from the logits.
        multi_dist = MultiCategorical(logits, self.action_dims)
        
        if action is None:
            action = multi_dist.sample()
        #print('ACTION: ', action)
        log_prob = multi_dist.log_prob(action)
        entropy = multi_dist.entropy()
        
        features = self.extra_layer(hidden)
        critic_input = features + hidden
        return (
            action,
            log_prob,
            entropy,
            self.critic_ext(critic_input),
            self.critic_int(critic_input),
        )

    def get_value(self, x):
        hidden = self.network(x)
        features = self.extra_layer(hidden)
        critic_input = features + hidden
        return self.critic_ext(critic_input), self.critic_int(critic_input)
