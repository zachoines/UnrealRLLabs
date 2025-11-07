import torch
import numpy as np

from Source.Agent import Agent


class RandomAgent(Agent):
    """Agent that samples uniformly random actions from the configured action space."""

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        self.config = config
        self.device = device

        env_shape = self.config["environment"]["shape"]
        if "agent" in env_shape["action"]:
            self.is_multi_agent = True
            action_cfg = env_shape["action"]["agent"]
        else:
            self.is_multi_agent = False
            action_cfg = env_shape["action"]["central"]

        self.discrete_count_list = [
            item["num_choices"] for item in action_cfg.get("discrete", [])
        ]

        self.continuous_ranges = [
            tuple(rng) for rng in action_cfg.get("continuous", [])
        ]

    def get_actions(
        self,
        states: torch.Tensor,
        dones=None,
        truncs=None,
        eval: bool = False,
        **kwargs,
    ):
        """Return random actions plus zero log-probs/entropy placeholders."""
        S, E, NA = 1, 1, 1
        if isinstance(states, dict):
            if "agent" in states:
                S, E, NA, _ = states["agent"].shape
            elif "central" in states:
                S, E, _ = states["central"].shape
                NA = 1

        total_branches = len(self.discrete_count_list) + len(self.continuous_ranges)
        batch_elems = S * E * NA
        action_components = []

        for num_choices in self.discrete_count_list:
            action_components.append(
                np.random.randint(low=0, high=num_choices, size=(batch_elems,))
            )

        for low, high in self.continuous_ranges:
            action_components.append(
                np.random.uniform(low=low, high=high, size=(batch_elems,))
            )

        if action_components:
            stacked = np.stack(action_components, axis=-1).astype(np.float32)
        else:
            stacked = np.zeros((batch_elems, 1), dtype=np.float32)

        actions = torch.from_numpy(stacked).view(S, E, NA, -1).to(self.device)
        log_probs = torch.zeros((S, E, NA), device=self.device)
        entropy = torch.zeros((S, E, NA), device=self.device)
        return actions, (log_probs, entropy)

    def update(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncs: torch.Tensor,
    ) -> dict:
        """Random agent performs no learning and returns an empty diagnostics dict."""
        return {}
