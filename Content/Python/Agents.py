import torch
import torch.nn as nn
import torch.optim as optim
from Networks import *

class Config:
    def __init__(self, num_environments: int, num_actions: int, state_size: int, training_batch_size: int):
        self.num_environments = num_environments
        self.num_actions = num_actions
        self.state_size = state_size
        self.training_batch_size = training_batch_size

class Learner:
    def __init__(self, config: Config):
        self.config = config

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        pass

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        return torch.rand((states.shape[0], self.config.num_actions))

class REINFORCELearner(Learner):
    def __init__(self, config: Config, learning_rate=1e-3):
        super().__init__(config)
        self.policy = PolicyNetwork(config.state_size, config.num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.saved_log_probs = []
        self.rewards = []

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        # Calculate the loss
        mu, sigma = self.policy(states)
        dist = torch.distributions.Normal(mu, sigma)
        log_probs = dist.log_prob(actions)
        loss = -torch.sum(log_probs * rewards)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear the saved rewards and log probabilities
        self.rewards = []
        self.saved_log_probs = []

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.policy(states)
        dist = torch.distributions.Normal(mu, sigma)
        actions = dist.sample()
        return actions
