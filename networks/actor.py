import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.categorical import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list):
        super(ActorNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Create the list of layers based on hidden_dims
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Combine layers into a sequential model
        self.hidden_layers = nn.Sequential(*layers)
        
        # Define the output layer that produces action probabilities
        self.output_layer = nn.Linear(input_dim, action_dim)
        
    def forward(self, obs):
        # Pass the observation through the hidden layers
        x = self.hidden_layers(obs)
        
        # Get the raw action scores (logits)
        logits = self.output_layer(x)
        
        # Apply softmax to convert logits to probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        # Sample an action from the action distribution
        action_dist = dist.Categorical(action_probs)
        action = action_dist.sample()
        
        return action, action_probs

# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, action_dim, obs_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),  # Input layer matching obs_dim
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),  # Hidden layer
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1), 
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, action_dim), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)  # No need to divide by 255 as this is a general tensor
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)
