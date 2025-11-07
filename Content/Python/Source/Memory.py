# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class RecurrentMemoryNetwork(nn.Module):
    """Abstract base for recurrent memory modules."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, **kwargs):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward_step(self, features: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single step of features and return the next hidden state."""
        raise NotImplementedError("Subclasses must implement forward_step")

    def forward_sequence(self, features_seq: torch.Tensor, h_initial: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a feature sequence and return sequence outputs plus final hidden state."""
        raise NotImplementedError("Subclasses must implement forward_sequence")

class GRUSequenceMemory(RecurrentMemoryNetwork):
    """GRU-based memory module with optional gated residual connections."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 bias: bool = True, dropout: float = 0.0,
                 use_residual: bool = True, **kwargs):
        """Initialize the GRU memory module."""
        super().__init__(input_size, hidden_size, num_layers)
        self.use_residual = use_residual
        # Ensure dropout is only applied if num_layers > 1, as per nn.GRU behavior
        actual_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True, # Crucial: Expects (Batch, Seq, Feature) input
            dropout=actual_dropout
        )

        # Projection layer if residual connection dimensions differ
        self.residual_projection = None
        if self.use_residual and input_size != hidden_size:
            self.residual_projection = nn.Linear(input_size, hidden_size)
            nn.init.xavier_uniform_(self.residual_projection.weight, gain=0.1)
            nn.init.zeros_(self.residual_projection.bias)

        # Learnable gate controlling residual contribution
        if self.use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.1))

        print(f"GRUSequenceMemory: Initialized nn.GRU with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={actual_dropout}, batch_first=True, use_residual={self.use_residual}")

    def forward_step(self, features: torch.Tensor, h_prev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single timestep with the GRU."""
        # Reshape features for GRU: (N, H_in) -> (N, 1, H_in) as sequence length L=1
        features_unsqueezed = features.unsqueeze(1) 
        
        # Initialize hidden state if not provided
        if h_prev is None:
            batch_size_n = features.shape[0]
            h_prev = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, 
                                 device=features.device, dtype=features.dtype)

        # GRU forward pass
        # gru_out_seq shape: (N, 1, H_out)
        # h_next shape: (D, N, H_out)
        gru_out_seq, h_next = self.gru(features_unsqueezed, h_prev)

        # Squeeze the sequence length dimension from GRU output
        processed_features = gru_out_seq.squeeze(1) # Shape: (N, H_out)

        # Apply residual connection if enabled
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(features)
            else:
                residual = features
            gate_value = torch.sigmoid(self.residual_gate)
            processed_features = (1 - gate_value) * processed_features + gate_value * residual

        return processed_features, h_next

    def forward_sequence(self, features_seq: torch.Tensor, h_initial: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence with the GRU and apply the optional residual connection."""
        # Initialize hidden state if not provided
        if h_initial is None:
            batch_size_n = features_seq.shape[0] # N = NumBatchSequences * NumAgents
            h_initial = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, 
                                    device=features_seq.device, dtype=features_seq.dtype)
        
        # GRU forward pass for sequences
        # processed_features_seq shape: (N, L, H_out)
        # h_final shape: (D, N, H_out)
        processed_features_seq, h_final = self.gru(features_seq, h_initial)

        # Apply residual connection if enabled
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(features_seq)
            else:
                residual = features_seq
            gate_value = torch.sigmoid(self.residual_gate)
            processed_features_seq = (1 - gate_value) * processed_features_seq + gate_value * residual

        return processed_features_seq, h_final

