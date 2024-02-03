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
        init.xavier_normal_(layer.weight)
        # kernel_gain = (0.125 / input_size) ** 0.5

        # # Apply normal distribution to weights and scale with kernel_gain
        # torch.nn.init.normal_(layer.weight.data)
        # layer.weight.data *= kernel_gain

        # # Initialize biases to zero
        # torch.nn.init.zeros_(layer.bias.data)

        return layer

class StatesEncoder2d(nn.Module):
    def __init__(self, state_size, embed_size, conv_sizes=[8, 16, 32], 
                 kernel_sizes=[3, 3, 3], strides=[1, 1, 1], dilations=[1, 2, 3], 
                 pooling_kernel_size=3, pooling_stride=2, dropout_rate=0.0):  # Updated pooling parameters
        super(StatesEncoder2d, self).__init__()

        self.n = int(state_size**0.5)
        
        # Validate lengths
        if not (len(conv_sizes) == len(kernel_sizes) == len(strides) == len(dilations)):
            raise ValueError("conv_sizes, kernel_sizes, strides, and dilations must be of the same length")

        self.paddings = [(kernel_sizes[i] + (kernel_sizes[i] - 1) * (dilations[i] - 1)) // 2 for i in range(len(conv_sizes))]

        # Define convolutional layers dynamically
        layers = []
        in_channels = 1
        for i, out_channels in enumerate(conv_sizes):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], 
                                    stride=strides[i], padding=self.paddings[i], dilation=dilations[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_channels = out_channels

        # Add a single pooling layer at the end
        layers.append(nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride))

        self.conv_layers = nn.Sequential(*layers)

        # Dynamically calculate the output size
        self.output_size = self.calculate_output_size(self.n, conv_sizes, kernel_sizes, strides, dilations, pooling_kernel_size, pooling_stride)

        # Define the fully connected layer
        self.fc = nn.Linear(self.output_size, embed_size)

    def calculate_output_size(self, n, conv_sizes, kernel_sizes, strides, dilations, pooling_kernel_size, pooling_stride):
        size = n
        for i in range(len(conv_sizes)):
            size = (size + 2 * self.paddings[i] - dilations[i] * (kernel_sizes[i] - 1) - 1) // strides[i] + 1
        # Pooling layer applied once at the end
        size = (size - pooling_kernel_size) // pooling_stride + 1
        return conv_sizes[-1] * size * size

    def forward(self, x):
        x = x.view(-1, 1, self.n, self.n)  # Reshape to [combined_batch_size, channels, height, width]
        x = self.conv_layers(x)
        x = x.view(-1, self.output_size)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc(x))
        return x

class StatesActionsEncoder2d(nn.Module):
    def __init__(self, state_size, action_dim, embed_size, conv_sizes=[8, 16, 32], 
                 kernel_sizes=[3, 3, 3], strides=[1, 1, 1], dilations=[1, 2, 3], 
                 pooling_kernel_size=3, pooling_stride=2, dropout_rate=0.0):  # Updated pooling parameters
        super(StatesActionsEncoder2d,self).__init__()

        self.n = int(state_size**0.5)
        self.action_dim = action_dim

        # Validate lengths
        if not (len(conv_sizes) == len(kernel_sizes) == len(strides) == len(dilations)):
            raise ValueError("conv_sizes, kernel_sizes, strides, and dilations must be of the same length")

        self.paddings = [(kernel_sizes[i] + (kernel_sizes[i] - 1) * (dilations[i] - 1)) // 2 for i in range(len(conv_sizes))]

        # Define convolutional layers dynamically
        layers = []
        in_channels = 1
        for i, out_channels in enumerate(conv_sizes):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], 
                                    stride=strides[i], padding=self.paddings[i], dilation=dilations[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_channels = out_channels

        # Add a single pooling layer at the end
        layers.append(nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride))

        self.conv_layers = nn.Sequential(*layers)

        # Dynamically calculate the output size
        self.output_size = self.calculate_output_size(self.n, conv_sizes, kernel_sizes, strides, dilations, pooling_kernel_size, pooling_stride)

        # Define the linear layer that combines conv output and actions
        self.linear = nn.Linear(self.output_size + action_dim, embed_size)

    def calculate_output_size(self, n, conv_sizes, kernel_sizes, strides, dilations, pooling_kernel_size, pooling_stride):
        size = n
        for i in range(len(conv_sizes)):
            size = (size + 2 * self.paddings[i] - dilations[i] * (kernel_sizes[i] - 1) - 1) // strides[i] + 1
        # Pooling layer applied once at the end
        size = (size - pooling_kernel_size) // pooling_stride + 1
        return conv_sizes[-1] * size * size

    def forward(self, observation, action):
        original_shape = observation.shape
        observation = observation.view(-1, 1, self.n, self.n)  # Reshape to [combined_batch_size, channels, height, width]

        x = self.conv_layers(observation)
        x = x.view(original_shape[0], original_shape[1], self.output_size)
        x = torch.cat([x, action], dim=-1)  # Concatenate action to the flattened conv output
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
    
class LinearNetwork(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int, dropout_rate=0.0):
        super(LinearNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_features)
        )

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)
   