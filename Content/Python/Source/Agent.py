import torch
import torch.nn as nn
from typing import Dict, Tuple

class Agent(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super(Agent, self).__init__()
        self.config = config
        self.device = device
        self.optimizers = {}

    def save(self, location: str, include_optimizers: bool = False) -> None:
        """Save model parameters and optionally optimizer states."""
        if include_optimizers:
            checkpoint = {
                "model": self.state_dict(),
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            }
            torch.save(checkpoint, location)
        else:
            torch.save(self.state_dict(), location)

    def load(self, location: str, load_optimizers: bool = False) -> None:
        """Load model parameters and optionally optimizer states."""
        state = torch.load(location, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            self.load_state_dict(state["model"])
            if load_optimizers and "optimizers" in state:
                for name, opt_state in state["optimizers"].items():
                    if name in self.optimizers:
                        self.optimizers[name].load_state_dict(opt_state)
        else:
            self.load_state_dict(state)

    def get_actions(self, states: torch.Tensor, dones=None, truncs=None, eval: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def total_grad_norm(self, params):
        # compute the global L2 norm across all param grads
        total = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total += param_norm.item() ** 2
        return torch.tensor(total**0.5)
        