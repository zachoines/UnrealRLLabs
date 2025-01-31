import torch
import numpy as np

from Source.Agent import Agent

class RandomAgent(Agent):
    """
    A simple agent that produces random actions using the new config structure:
      environment.shape.action.[agent|central].{ discrete:[...], continuous:[...] }

    The 'update' method returns an empty dict, meaning it does no training.
    """
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        self.config = config
        self.device = device

        # We'll parse the same structure you used for MAPOCAAgent or environment
        #  environment -> shape -> action -> agent or central
        # Then we produce random actions accordingly.

        # 1) Read out if multi-agent or not
        env_shape = self.config["environment"]["shape"]
        if "agent" in env_shape["action"]:
            # multi-agent
            self.is_multi_agent = True
            action_cfg = env_shape["action"]["agent"]
        else:
            # single-agent
            self.is_multi_agent = False
            # If "central" exists
            action_cfg = env_shape["action"]["central"]

        # 2) discrete => read array of { "num_choices": 4, ... }
        self.discrete_count_list = []
        if "discrete" in action_cfg:
            discrete_arr = action_cfg["discrete"]  # e.g. [ {"num_choices":4}, {"num_choices":4} ]
            for item in discrete_arr:
                self.discrete_count_list.append(item["num_choices"])

        # 3) continuous => read array for each dimension range
        #   e.g. [ [-1.0,1.0], [0.0,5.0] ]
        # But from the new code, you have something like "continuous":[ ??? ],
        # If not present, we skip.
        self.continuous_ranges = []
        if "continuous" in action_cfg:
            # e.g. an array of arrays
            # but your current code does not define the exact structure
            # We'll assume each entry is a 2-element [low, high] range
            for rng in action_cfg["continuous"]:
                # rng => [low, high]
                self.continuous_ranges.append(rng)

    def get_actions(self, states: torch.Tensor, dones=None, truncs=None, eval: bool = False, **kwargs):
        """
        states => can be dictionary or any shape. But we only need to produce random actions
        Return shape => same as MAPOCAAgent => (S,E,NA, action_dim?), 
                        plus optional (log_probs, entropies) or so.

        We'll produce random actions on CPU or device, with shape matching
        multi-discrete or continuous. We'll also produce an empty log_probs, entropy if you want.
        """
        # states shape might be (S,E,NA,...). We'll produce the same shaped actions in the final dimension
        # We can examine states shape or rely on any leftover info. 
        # Let's just do a simple approach:
        S,E,NA = 1, 1, 1
        if isinstance(states, dict):
            # If multi-agent => states["agent"].shape => (S, E, NA, agentObsSize)
            # We'll guess from that
            if "agent" in states:
                S, E, NA, _ = states["agent"].shape
            elif "central" in states:
                # then shape => (S,E,obsSize)
                # let's do S=1, E = shape[1], NA=1
                S, E, obsDim = states["central"].shape
                NA = 1
            else:
                # fallback
                pass
        else:
            # fallback, e.g. shape => (S,E,NA,obsDim) ??? 
            pass

        # Now let's build random actions => shape => (S,E,NA, total_action_dims)
        total_branches = len(self.discrete_count_list) + len(self.continuous_ranges)
        # We'll store them in a final np array => shape => (S*E*NA, total_branches) for easier assembling
        B = S*E*NA
        action_array = []

        # 1) discrete branches
        for num_choices in self.discrete_count_list:
            # random choice in [0..num_choices)
            a_i = np.random.randint(low=0, high=num_choices, size=(B,))
            action_array.append(a_i)

        # 2) continuous branches
        for rng in self.continuous_ranges:
            low, high = rng[0], rng[1]
            c_i = np.random.uniform(low=low, high=high, size=(B,))
            action_array.append(c_i)

        if len(action_array) == 0:
            # no branches => no actions
            out_np = np.zeros((B,1), dtype=np.float32)
        else:
            # stack => shape => (len_branches, B)
            stacked = np.stack(action_array, axis=-1)  # => shape (B, total_branches)
            out_np = stacked.astype(np.float32)

        out_t = torch.from_numpy(out_np).view(S,E,NA,-1).to(self.device)

        # Return actions plus placeholders for log_probs, entropies
        # MAPOCAAgent returns (actions, (log_probs, entropy)) presumably
        log_probs = torch.zeros((S,E,NA), device=self.device)
        entropy   = torch.zeros((S,E,NA), device=self.device)

        return out_t, (log_probs, entropy)

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, 
               rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> dict:
        """
        The random agent does no training, so just return empty logs.
        """
        return {}