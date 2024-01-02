import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.init as init

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int, dropout_rate=0.0):
        super(DiscretePolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Softmax(dim=-1)
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)

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
    def __init__(self, in_features: int, hidden_size: int, dropout_rate=0.0):
        super(ValueNetwork, self).__init__()

        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)

        )

        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.value_net(x)

class RSA(nn.Module):
    def __init__(self, embed_size, heads, dropout_rate=0.0):
        super(RSA, self).__init__()
        # Embedding for Q, K, V
        self.query_embed = self.linear_layer(embed_size, embed_size)
        self.key_embed = self.linear_layer(embed_size, embed_size)
        self.value_embed = self.linear_layer(embed_size, embed_size)
        
        self.input_norm = nn.LayerNorm(embed_size)
        self.output_norm = nn.LayerNorm(embed_size)
        
        self.multihead_attn = nn.MultiheadAttention(embed_size, heads, batch_first=True, dropout=dropout_rate)
        
    def forward(self, x: torch.Tensor, apply_residual: bool=True):
        x = self.input_norm(x)
        output, _ = self.multihead_attn(
            self.query_embed(x), 
            self.key_embed(x), 
            self.value_embed(x)
        )

        if apply_residual:
            output = x + output
        
        return self.output_norm(output)
    
    def linear_layer(self, input_size: int, output_size: int) -> torch.nn.Module:
        layer = torch.nn.Linear(input_size, output_size)
        kernel_gain = (0.125 / input_size) ** 0.5

        # Apply normal distribution to weights and scale with kernel_gain
        torch.nn.init.normal_(layer.weight.data)
        layer.weight.data *= kernel_gain

        # Initialize biases to zero
        torch.nn.init.zeros_(layer.bias.data)

        return layer

class StatesEncoder2d(nn.Module):
    def __init__(self, state_size, embed_size, dropout_rate=0.0):
        super(StatesEncoder2d, self).__init__()
        self.conv1_size = 16
        self.conv2_size = 32
        self.n = int(state_size**0.5)

        conv1_output_size = self.n // 2  # Due to max pooling with kernel_size=2 and stride=2 twice
        conv2_output_size = conv1_output_size // 2
        self.output_size = self.conv2_size * conv2_output_size * conv2_output_size

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.conv1_size, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(self.output_size, embed_size)

    def forward(self, x):
        x = x.view(-1, 1, self.n, self.n)  # Reshape to [combined_batch_size, channels, height, width]
        x = self.conv_layers(x)
        x = x.view(-1, self.output_size)
        x = F.relu(self.fc(x))
        return x

class StatesActionsEncoder2d(nn.Module):
    def __init__(self, state_size, action_dim, embed_size, dropout_rate=0.0):
        super(StatesActionsEncoder2d, self).__init__()
        self.conv1_size = 16
        self.conv2_size = 32
        self.n = int(state_size**0.5)

        conv1_output_size = self.n // 2  # Due to max pooling with kernel_size=2 and stride=2 twice
        conv2_output_size = conv1_output_size // 2
        self.output_size = self.conv2_size * conv2_output_size * conv2_output_size

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.conv1_size, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Linear(self.output_size + action_dim, embed_size)
        self.action_dim = action_dim

    def forward(self, observation, action):
        original_shape = observation.shape
        observation = observation.view(-1, 1, self.n, self.n)  # Reshape to [combined_batch_size, channels, height, width]

        x = observation
        x = self.conv_layers(x)
        x = x.view(original_shape[0], original_shape[1], -1)
        x = torch.cat([x, action], dim=-1)
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
   
class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, embed_size: int, dropout_rate: float):
        super(ICM, self).__init__()
        
        self.feature_network = nn.Sequential(
            StatesEncoder(state_dim, embed_size),
            nn.Dropout(dropout_rate)
        )

        self.forward_model = nn.Sequential(
            StatesActionsEncoder(embed_size, action_dim, embed_size),
            nn.Dropout(dropout_rate)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_size, action_dim)
        )

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, n: float = 0.5, beta: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Feature Network
        state_features = self.feature_network(states)
        next_state_features = self.feature_network(next_states)

        # Forward Model
        predicted_next_state_feature = self.forward_model(state_features, actions)

        # Inverse Model
        combined_features = torch.cat([state_features, next_state_features], dim=1)
        predicted_action = self.inverse_model(combined_features)

        forward_loss_per_state = F.mse_loss(predicted_next_state_feature, next_state_features.detach(), reduction='none')
        forward_loss = beta * forward_loss_per_state.mean()
        inverse_loss = (1 - beta) * F.mse_loss(predicted_action, actions.detach())  # Directly comparing to actions

        # Intrinsic reward is the forward loss for each state
        intrinsic_reward = n * forward_loss_per_state.mean(-1).squeeze(-1).detach()

        return forward_loss, inverse_loss, intrinsic_reward

