# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class RecurrentMemoryNetwork(nn.Module):
    """
    Abstract base class for recurrent memory modules (e.g., GRU, LSTM).
    Defines the interface for processing single steps and sequences of features.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, **kwargs):
        """
        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward_step(self, features: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single step of features. This is typically used during action selection.

        Args:
            features (torch.Tensor): Input features for the current step. 
                                     Expected shape: (BatchSize_N, InputSize_H_in), 
                                     where N = NumEnvs * NumAgents.
            h_prev (torch.Tensor): Previous hidden state. 
                                   Expected shape: (NumLayers_D * NumDirections, BatchSize_N, HiddenSize_H_out).
                                   NumDirections is 1 for unidirectional, 2 for bidirectional.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - processed_features (torch.Tensor): Output features from the memory module for this step.
                                                     Shape: (BatchSize_N, HiddenSize_H_out).
                - h_next (torch.Tensor): Next hidden state.
                                         Shape: (NumLayers_D * NumDirections, BatchSize_N, HiddenSize_H_out).
        """
        raise NotImplementedError("Subclasses must implement forward_step")

    def forward_sequence(self, features_seq: torch.Tensor, h_initial: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of features. This is typically used during agent updates.

        Args:
            features_seq (torch.Tensor): Input features for the entire sequence.
                                         Expected shape: (BatchSize_N, SeqLen_L, InputSize_H_in)
                                         if batch_first=True, where N = NumBatchSequences * NumAgents.
            h_initial (Optional[torch.Tensor]): Initial hidden state for the sequence.
                                                Expected shape: (NumLayers_D * NumDirections, BatchSize_N, HiddenSize_H_out).
                                                If None, it will be initialized to zeros.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - processed_features_seq (torch.Tensor): Output features for the entire sequence.
                                                         Shape: (BatchSize_N, SeqLen_L, HiddenSize_H_out).
                - h_final (torch.Tensor): Final hidden state of the sequence.
                                          Shape: (NumLayers_D * NumDirections, BatchSize_N, HiddenSize_H_out).
        """
        raise NotImplementedError("Subclasses must implement forward_sequence")

class GRUSequenceMemory(RecurrentMemoryNetwork):
    """
    GRU-based recurrent memory module.
    Wraps nn.GRU and provides the RecurrentMemoryNetwork interface.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 bias: bool = True, dropout: float = 0.0, **kwargs):
        """
        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            bias (bool): If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of each GRU layer 
                             except the last layer, with dropout probability equal to dropout. Default: 0.
        """
        super().__init__(input_size, hidden_size, num_layers)
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
        print(f"GRUSequenceMemory: Initialized nn.GRU with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={actual_dropout}, batch_first=True")

    def forward_step(self, features: torch.Tensor, h_prev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single step of features using the GRU.
        Args:
            features (torch.Tensor): Input features for the current step. 
                                     Shape: (N, H_in), where N = NumEnvs * NumAgents.
            h_prev (Optional[torch.Tensor]): Previous hidden state. 
                                   Shape: (D, N, H_out), where D = num_layers.
                                   If None, initialized to zeros.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - processed_features (torch.Tensor): Output features. Shape: (N, H_out).
                - h_next (torch.Tensor): Next hidden state. Shape: (D, N, H_out).
        """
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
        return processed_features, h_next

    def forward_sequence(self, features_seq: torch.Tensor, h_initial: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of features using the GRU.
        Args:
            features_seq (torch.Tensor): Input features for the sequence. 
                                         Shape: (N, L, H_in), where N = NumBatchSequences * NumAgents.
            h_initial (Optional[torch.Tensor]): Initial hidden state for the sequence. 
                                      Shape: (D, N, H_out). If None, initialized to zeros.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - processed_features_seq (torch.Tensor): Output features for the sequence. Shape: (N, L, H_out).
                - h_final (torch.Tensor): Final hidden state. Shape: (D, N, H_out).
        """
        # Initialize hidden state if not provided
        if h_initial is None:
            batch_size_n = features_seq.shape[0] # N = NumBatchSequences * NumAgents
            h_initial = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, 
                                    device=features_seq.device, dtype=features_seq.dtype)
        
        # GRU forward pass for sequences
        # processed_features_seq shape: (N, L, H_out)
        # h_final shape: (D, N, H_out)
        processed_features_seq, h_final = self.gru(features_seq, h_initial)
        return processed_features_seq, h_final

# Example of how you might add an LSTM later:
# class LSTMSequenceMemory(RecurrentMemoryNetwork):
#     def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, dropout: float = 0.0, **kwargs):
#         super().__init__(input_size, hidden_size, num_layers)
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bias=bias,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0
#         )
#         print(f"LSTMSequenceMemory: Initialized nn.LSTM with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, batch_first=True")

#     def forward_step(self, features: torch.Tensor, h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         # LSTM hidden state is a tuple (h_n, c_n)
#         features_unsqueezed = features.unsqueeze(1)
#         if h_prev is None:
#             batch_size_n = features.shape[0]
#             h0 = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, device=features.device, dtype=features.dtype)
#             c0 = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, device=features.device, dtype=features.dtype)
#             h_prev = (h0, c0)
        
#         lstm_out_seq, (h_next, c_next) = self.lstm(features_unsqueezed, h_prev)
#         processed_features = lstm_out_seq.squeeze(1)
#         return processed_features, (h_next, c_next)

#     def forward_sequence(self, features_seq: torch.Tensor, h_initial: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         if h_initial is None:
#             batch_size_n = features_seq.shape[0]
#             h0 = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, device=features_seq.device, dtype=features_seq.dtype)
#             c0 = torch.zeros(self.num_layers, batch_size_n, self.hidden_size, device=features_seq.device, dtype=features_seq.dtype)
#             h_initial = (h0, c0)
            
#         processed_features_seq, (h_final, c_final) = self.lstm(features_seq, h_initial)
#         return processed_features_seq, (h_final, c_final)

