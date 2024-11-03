import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list):
        super(Critic, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Create the list of layers based on hidden_dims
        layers = []
        input_dim = obs_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Combine layers into a sequential model
        self.hidden_layers = nn.Sequential(*layers)
        
        # Define the output layer that produces the Q-value
        self.output_layer = nn.Linear(input_dim, 1)
        
    def forward(self, obs, action):
        # Concatenate the observation and action
        x = torch.cat([obs, action], dim=-1)
        
        # Pass through the hidden layers
        x = self.hidden_layers(x)
        
        # Produce the Q-value
        q_value = self.output_layer(x)
        
        return q_value
