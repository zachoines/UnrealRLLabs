import torch
import torch.nn as nn
from typing import Dict, Tuple

class Agent(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super(Agent, self).__init__()
        self.config = config
        self.device = device
        self.optimizers = {}

    def save(self, location: str) -> None:
        torch.save(self.state_dict(), location)

    def load(self, location: str) -> None:
        state = torch.load(location, map_location=self.device)
        self.load_state_dict(state)

    def get_actions(self, states: torch.Tensor, dones=None, truncs=None, eval: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def get_average_lr(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Returns a Tensor containing the average learning rate
        across all param groups in the given optimizer.
        """
        # Collect the 'lr' from each param_group
        lr_list = [param_group["lr"] for param_group in optimizer.param_groups]
        # Convert to a float32 Tensor
        lr_tensor = torch.tensor(lr_list, dtype=torch.float32)
        # Return the mean as a 0-dimensional Tensor
        return torch.mean(lr_tensor)