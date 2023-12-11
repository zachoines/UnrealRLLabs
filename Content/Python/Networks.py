import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import torch
import torch.nn as nn
from typing import List

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super(DiscretePolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)
    
class MultiDiscretePolicyNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: List[int], hidden_size: int):
        super(DiscretePolicyNetwork, self).__init__()

        # Initial layers are common for all branches
        self.common_layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
        )

        # Create a separate output layer for each branch
        self.branches = nn.ModuleList([nn.Linear(hidden_size, out_feature) for out_feature in out_features])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.common_layers(x)
        # Apply each branch separately and apply softmax
        return [self.softmax(branch(x)) for branch in self.branches]

class ContinousPolicyNetwork(nn.Module):
    """
    A Gaussian policy network that outputs the mean and standard deviation of a Gaussian distribution for each action.
    """
    def __init__(self, in_features : int, out_features : int, hidden_size : int ):
        super(ContinousPolicyNetwork, self).__init__()

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
        )

        # Network to output the mean of the action distribution
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Tanh()
        )

        # Network to output the log standard deviation of the action distribution
        self.log_std = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
        )

        self.eps = 1e-8

    def forward(self, state):
        """
        Forward pass through the network.

        :param state: The current state.
        :return: The mean and standard deviation of the action distribution.
        """
        shared_features = self.shared_net(state.to(self.device))
        means = self.mean(shared_features)
        log_stds = self.log_std(shared_features)
        stds = F.softplus(log_stds)
        return means, stds
  
class ValueNetwork(nn.Module):
    def __init__(self, in_features : int, hidden_size : int):
        super(ValueNetwork, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.value_net(x)
        return x
        
class RSA(nn.Module):
    def __init__(self, embed_size, heads):
        super(RSA, self).__init__()
        
        # Embedding for Q, K, V
        self.query_embed = self.linear_layer(embed_size, embed_size)
        self.key_embed = self.linear_layer(embed_size, embed_size)
        self.value_embed = self.linear_layer(embed_size, embed_size)
        
        # Layer normalization
        self.input_norm = nn.LayerNorm(embed_size)
        self.output_norm = nn.LayerNorm(embed_size)
        
        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        
    def forward(self, x):
        x = self.input_norm(x)

        # Further embedding into Q, K, V
        Q = self.query_embed(x)
        K = self.key_embed(x)
        V = self.value_embed(x)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(Q, K, V)
        
        # Residual connection
        x = x + attn_output
        
        # Layer normalization after residual addition
        x = self.output_norm(x)
        
        return x
    
    def linear_layer(self, input_size: int, output_size: int) -> torch.nn.Module:
        layer = torch.nn.Linear(input_size, output_size)
        kernel_gain = (0.125 / input_size) ** 0.5

        # Apply normal distribution to weights and scale with kernel_gain
        torch.nn.init.normal_(layer.weight.data)
        layer.weight.data *= kernel_gain

        # Initialize biases to zero
        torch.nn.init.zeros_(layer.bias.data)

        return layer
    
class LayerNorm(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
        return (x - mean) / (torch.sqrt(var + 1e-5))
    
class StatesEncoder2d(nn.Module):
    def __init__(self, grid_size, embed_size):
        super(StatesEncoder2d, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(32 * grid_size, embed_size)
        self.sqrt_grid = int(grid_size**0.5)
        self.embed_size = embed_size

    def forward(self, x):
        # Reshape to combine batch dimensions
        original_shape = x.shape
        x = x.view(original_shape[0] * original_shape[1], 1, self.sqrt_grid, self.sqrt_grid)  # Reshape to [combined_batch_size, channels, height, width]

        # Apply convolutional layers
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # Reshape and apply fully connected layer
        x = x.view(original_shape[0], original_shape[1], -1)
        x = F.relu(self.fc(x))

        return x
    
class StatesActionsEncoder2d(nn.Module):
    def __init__(self, grid_size, action_dim, embed_size):
        super(StatesActionsEncoder2d, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.linear = nn.Linear(32 * grid_size + action_dim, embed_size)

        self.sqrt_grid = int(grid_size**0.5)
        self.embed_size = embed_size
        self.action_dim = action_dim

    def forward(self, observation, action):
        # Reshape observation to combine batch dimensions
        original_shape = observation.shape
        observation = observation.view(original_shape[0] * original_shape[1], 1, self.sqrt_grid, self.sqrt_grid)  # Reshape to [combined_batch_size, channels, height, width]
        
        # Apply convolutional layers to observation
        x = F.leaky_relu(self.conv1(observation))
        x = F.leaky_relu(self.conv2(x))

        # Flatten conv output
        x = x.view(original_shape[0], original_shape[1], -1)

        # Concatenate flattened observation and action
        x = torch.cat([x, action], dim=-1)

        # Apply fully connected layers
        x = F.leaky_relu(self.linear(x))
        
        return x

class StatesEncoder(nn.Module):
    def __init__(self, state_dim, embed_size):
        super(StatesEncoder, self).__init__()
        self.fc = nn.Linear(state_dim, embed_size)
        
    def forward(self, x):
        return self.fc(x)

class StatesActionsEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, embed_size):
        super(StatesActionsEncoder, self).__init__()
        self.fc = nn.Linear(state_dim + action_dim, embed_size)
        
    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.fc(x)

   
