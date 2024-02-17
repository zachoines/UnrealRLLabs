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
                init.kaiming_normal_(layer.weight)

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
        init.kaiming_normal_(layer.weight)
        return layer

class StatesEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super(StatesEncoder, self).__init__()
        self.fc = LinearNetwork(input_size, output_size, dropout_rate, activation=True)
        
    def forward(self, x):
        return self.fc(x)

class StatesActionsEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, output_size, dropout_rate=0.0):
        super(StatesActionsEncoder, self).__init__()
        self.fc = LinearNetwork(state_dim + action_dim, output_size, dropout_rate, activation=True)
        
    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.fc(x)
    
class LinearNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate=0.0, activation=True):
        super(LinearNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU() if activation else {},
        )

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)

class LSTMNetwork(nn.Module):
    def __init__(self, in_features: int, output_size: int, num_layers: int = 1, dropout: float = 1.0):
        super(LSTMNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout
        )
        self.hidden_size = output_size
        self.num_layers = num_layers
        self.prev_hidden = None
        self.prev_cell = None
        self.hidden = None
        self.cell = None

    def init_hidden(self, batch_size: int):
        """Initialize the hidden state and cell state for a new batch."""
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device).contiguous()
        self.cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device).contiguous()

    def get_hidden(self):
        """Return the current hidden and cell states concatenated."""
        if self.hidden == None or self.cell == None:
            return None
        return torch.cat((self.hidden, self.cell), dim=0) # type: ignore

    def get_prev_hidden(self):
        """Return the previous hidden and cell states concatenated."""
        return torch.cat((self.prev_hidden, self.prev_cell), dim=0) # type: ignore

    def set_hidden(self, hidden):
        """Set the hidden state and cell state to a specific value."""
        self.hidden, self.cell = torch.split(hidden.clone(), hidden.size(0) // 2, dim=0)

    def forward(self, x, dones=None, input_hidden=None):
        seq_length, batch_size, *_ = x.size() 

        # Enforce internal hidden state
        if self.hidden is None or self.hidden.size(1) != batch_size or self.cell is None or self.cell.size(1) != batch_size:
            self.init_hidden(batch_size)

        # Process each sequence step, taking dones into consideration
        lstm_outputs = []

        # Set the hidden and cell states
        if input_hidden is not None:
            self.set_hidden(input_hidden)

        hidden_outputs = torch.zeros(seq_length, self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        for t in range(seq_length):
            
            # Reset hidden and cell states for environments that are done
            if dones is not None:
                mask = dones[t].to(dtype=torch.bool, device=self.device).view(1, batch_size, 1)
                self.hidden = self.hidden * (~mask)
                self.cell = self.cell * (~mask)

            self.prev_hidden = self.hidden
            self.prev_cell = self.cell
            hidden_outputs[t] = self.get_hidden()
            lstm_output, (self.hidden, self.cell) = self.lstm(x[t].unsqueeze(0), (self.hidden, self.cell)) # type: ignore
            lstm_outputs.append(lstm_output)

        lstm_outputs = torch.cat(lstm_outputs, dim=0)
        return lstm_outputs, hidden_outputs

    @property
    def device(self) -> torch.device:
        """Device property to ensure tensors are on the same device as the model."""
        return next(self.parameters()).device