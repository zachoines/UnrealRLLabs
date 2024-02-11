import torch
from typing import Dict
from Config import BaseConfig

import torch
import torch.nn as nn
from typing import Dict, Tuple

class Agent(nn.Module):
    def __init__(self, config: BaseConfig):
        super(Agent, self).__init__()
        self.config: BaseConfig

        """
        Base class representing a reinforcement learning agent.

        Attributes:
            config (BaseConfig): Contains various network, training, and environment parameters
        """
        self.optimizers = {}

    def save(self, location: str) -> None:
        pass

    def load(self, location: str) -> None:
        pass

    def state_dict(self) -> Dict[str, Dict]:
        # This method is already provided by nn.Module, so you might not need to override it unless you have specific requirements.
        pass

    def get_actions(self, states: torch.Tensor, dones=None, truncs=None, eval: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def rescaleAction(self, action: torch.Tensor, min: float, max: float) -> torch.Tensor:
        return min + (0.5 * (action + 1.0) * (max - min))
    
    def calc_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        num_envs, num_steps = rewards.shape
        running_returns = torch.zeros(num_envs, dtype=torch.float32)
        returns = torch.zeros_like(rewards)
        
        for i in range(num_steps - 1, -1, -1):
            running_returns = rewards[:, i] + (1 - dones[:, i]) * self.config.gamma * running_returns
            returns[:, i] = running_returns

        return returns
    
    def eligibility_trace_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        num_steps, _, _ = rewards.shape
        td_lambda_returns = torch.zeros_like(rewards)
        
        # Initialize eligibility traces
        eligibility_traces = torch.zeros_like(rewards)

        # Initialize the next_value for the calculation of returns at T + 1
        next_value = next_values[-1]
        
        for t in reversed(range(num_steps - 1)):
            # Clamp the sum of dones and truncs to ensure it doesn't exceed 1
            masks = 1.0 - torch.clamp(dones[t] + truncs[t], 0, 1)
            rewards_t = rewards[t]
            values_t = values[t]
            
            # Update the eligibility traces
            eligibility_traces[t] = self.config.gamma * self.config.lambda_ * eligibility_traces[t + 1] * masks + 1.0
            
            # Calculate TD Error
            td_error = rewards_t + self.config.gamma * next_value * masks - values_t
            
            # Calculate TD(λ) return for time t
            td_lambda_returns[t] = td_lambda_returns[t] + eligibility_traces[t] * td_error
            
            # Update the next_value for the next iteration of the loop
            next_value = values_t

        # Handle the last step separately as it does not have a next step
        td_lambda_returns[-1] = rewards[-1] + self.config.gamma * next_value * (1.0 - torch.clamp(dones[-1] + truncs[-1], 0, 1)) - values[-1]

        return td_lambda_returns

    
    def compute_gae_and_targets(self, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor):
        """
        Compute GAE and bootstrapped targets for PPO.

        :param rewards: (torch.Tensor) Rewards.
        :param dones: (torch.Tensor) Done flags.
        :param truncs: (torch.Tensor) Truncation flags.
        :param values: (torch.Tensor) State values.
        :param next_values: (torch.Tensor) Next state values.
        :param gamma: (float) Discount factor.
        :param lambda_: (float) GAE smoothing factor.
        :return: Bootstrapped targets and advantages.

        The λ parameter in the Generalized Advantage Estimation (GAE) algorithm acts as a trade-off between bias and variance in the advantage estimation.
        When λ is close to 0, the advantage estimate is more biased, but it has less variance. It would rely more on the current reward and less on future rewards. This could be useful when your reward signal is very noisy because it reduces the impact of that noise on the advantage estimate.
        On the other hand, when λ is close to 1, the advantage estimate has more variance but is less biased. It will take into account a longer sequence of future rewards.
        """
        batch_size = rewards.shape[0]  # Determine batch size from the rewards tensor
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(batch_size)):
            non_terminal = 1.0 - torch.clamp(dones[t, :] + truncs[t, :], 0.0, 1.0)
            delta = (rewards[t, :] + (self.config.gamma * next_values[t, :] * non_terminal)) - values[t, :]
            last_gae_lam = delta + (self.config.gamma * self.config.lambda_ * non_terminal * last_gae_lam)
            advantages[t, :] = last_gae_lam * non_terminal

        # Compute bootstrapped targets by adding unnormalized advantages to values 
        targets = values + advantages
        return targets, advantages
    
    def compute_targets(self, rewards, terminals, values, values_next):
            """
            Compute the target value for each state for the value function update,
            using Generalized Advantage Estimation (GAE) for smoothing.

            :param rewards: Tensor of rewards received from the environment.
            :param terminals: Tensor indicating terminal states.
            :param values: Tensor of value estimates V(s) for each state.
            :param values_next: Tensor of value estimates V(s') for the next states.
            :param gamma: Discount factor for future rewards.
            :param lambda_: Smoothing parameter for GAE.
            :return: Tensor of target values for each state.
            """
            num_steps, _, _ = rewards.shape
            gae = torch.zeros_like(rewards).to(rewards.device)
            returns = torch.zeros_like(rewards).to(rewards.device)

            # Handle the last timestep separately as a special case
            t = num_steps - 1
            gae[t] = rewards[t] + self.config.gamma * values_next[t] * (1 - terminals[t]) - values[t]  # No gae[t + 1] to accumulate from
            returns[t] = gae[t] + values[t]

            for t in reversed(range(t)):
                delta = rewards[t] + self.config.gamma * values_next[t] * (1 - terminals[t]) - values[t]
                gae[t] = delta + self.config.gamma * self.config.lambda_ * (1 - terminals[t]) * gae[t + 1]
                returns[t] = gae[t] + values[t]

            return returns

    
    def save_train_state(self):
        # Implementation for saving the training state
        return None

    def restore_train_state(self, state):
        # Implementation for restoring the training state
        pass

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def forward(self):
        pass
