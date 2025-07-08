import os
import sys
import torch

# Add path to import MAPOCAAgent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Agents.MAPOCAAgent import MAPOCAAgent


def test_bootstrapped_returns_uses_bootstrap_value():
    agent = MAPOCAAgent.__new__(MAPOCAAgent)
    agent.gamma = 0.9
    agent.lmbda = 1.0
    agent.enable_popart = False
    agent.device = torch.device('cpu')

    rewards = torch.tensor([[[1.0]]])
    values = torch.tensor([[[0.5]]])
    dones = torch.tensor([[[0.0]]])
    truncs = torch.tensor([[[0.0]]])
    bootstrap_value = torch.tensor([0.7])

    returns = agent.compute_bootstrapped_returns(rewards, values, dones, truncs, bootstrap_value)
    expected = rewards.squeeze() + agent.gamma * bootstrap_value
    assert torch.allclose(returns.squeeze(), expected)


def test_bootstrapped_returns_multi_agent_vector():
    agent = MAPOCAAgent.__new__(MAPOCAAgent)
    agent.gamma = 0.9
    agent.lmbda = 1.0
    agent.enable_popart = False
    agent.device = torch.device('cpu')

    rewards = torch.tensor([[[1.0, 2.0]]])
    values = torch.tensor([[[0.5, 0.5]]])
    dones = torch.tensor([[[0.0]]])
    truncs = torch.tensor([[[0.0]]])
    bootstrap_value = torch.tensor([0.7, 1.0])

    returns = agent.compute_bootstrapped_returns(
        rewards, values, dones, truncs, bootstrap_value.view(1, -1)
    )
    expected = rewards.squeeze() + agent.gamma * bootstrap_value
    assert torch.allclose(returns.squeeze(), expected)


def test_bootstrapped_returns_truncated_no_done():
    agent = MAPOCAAgent.__new__(MAPOCAAgent)
    agent.gamma = 0.9
    agent.lmbda = 1.0
    agent.enable_popart = False
    agent.device = torch.device('cpu')

    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[0.5], [0.6]]])
    dones = torch.tensor([[[0.0], [0.0]]])
    truncs = torch.tensor([[[0.0], [1.0]]])
    bootstrap_value = torch.tensor([0.7])

    returns = agent.compute_bootstrapped_returns(rewards, values, dones, truncs, bootstrap_value)
    expected_last = rewards[0, -1, 0] + agent.gamma * bootstrap_value
    assert torch.allclose(returns[0, -1, 0], expected_last)


def test_compute_returns_resets_on_truncation():
    agent = MAPOCAAgent.__new__(MAPOCAAgent)
    agent.gamma = 0.9
    agent.lmbda = 1.0
    agent.enable_popart = False
    agent.device = torch.device("cpu")

    rewards = torch.tensor([[[1.0], [2.0]]])
    values = torch.tensor([[[0.5], [0.6]]])
    dones = torch.tensor([[[0.0], [0.0]]])
    truncs = torch.tensor([[[0.0], [1.0]]])

    returns = agent.compute_returns(rewards, values, dones, truncs)
    expected = torch.tensor([[[1.0], [2.0]]])
    assert torch.allclose(returns, expected)
