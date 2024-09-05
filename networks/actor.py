import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

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
