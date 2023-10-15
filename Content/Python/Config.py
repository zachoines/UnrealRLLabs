import numpy as np
from typing import Dict, Any
import torch
from enum import Enum

class ActionType(Enum):
    CONTINUOUS = "Continuous"
    DISCRETE = "Discrete"

class ActionSpace:
    def __init__(self, action_type: ActionType, low: float, high: float):
        self.action_type = action_type
        self.low = low
        self.high = high

    def is_continuous(self):
        return self.action_type == ActionType.CONTINUOUS

    def is_discrete(self):
        return self.action_type == ActionType.DISCRETE

class EnvParams:
    def __init__(self, num_environments: int, num_actions: int, state_size, action_space: ActionSpace):
        self.num_environments = num_environments
        self.num_actions = num_actions
        self.state_size = state_size
        self.action_space = action_space

class TrainParams:
    def __init__(self, training_batch_size: int, device: torch.device):
        self.training_batch_size = training_batch_size
        self.device = device

class NetworkParams:
    """
    A class to hold different network parameters 

    Attributes:
        hidden_size (int): The size of the hidden layers in the agent's networks.
    """
    def __init__(self,
            hidden_size: int = 256,
        ):
        self.hidden_size = hidden_size
        
class Config:
    def __init__(self,
            envParams: EnvParams,
            trainParams: TrainParams,
            networkParams: NetworkParams
        ):
        self.envParams = envParams
        self.trainParams = trainParams
        self.networkParams = networkParams    
