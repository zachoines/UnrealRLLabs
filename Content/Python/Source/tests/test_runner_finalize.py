import sys
import types
import torch

# RLRunner imports win32event on Windows; provide a stub for tests.
# Minimal stubs for Windows-only modules used in Runner/Environment
sys.modules.setdefault("win32event", types.SimpleNamespace(SetEvent=lambda *a, **k: None))
sys.modules.setdefault("win32api", types.SimpleNamespace(CloseHandle=lambda *a, **k: None))

from Source.Runner import RLRunner, TrajectorySegment
from Agents.MAPOCAAgent import MAPOCAAgent


def _make_runner(num_agents: int) -> RLRunner:
    runner = RLRunner.__new__(RLRunner)
    runner.num_agents_cfg = num_agents
    runner.device = torch.device("cpu")
    runner.pad_trajectories = True
    runner.sequence_length = 8
    runner.enable_memory = False
    runner.current_memory_hidden_states = None

    agent = MAPOCAAgent.__new__(MAPOCAAgent)
    agent.gamma = 0.9
    agent.lmbda = 1.0
    agent.enable_popart = False
    agent.device = torch.device("cpu")
    runner.agent = agent

    runner.current_segments = [TrajectorySegment(num_agents, runner.device, True, 8)]
    runner.current_episode_segments = [[]]
    runner.completed_segments = [[]]
    return runner


def _populate_segment(seg: TrajectorySegment, steps: int = 2):
    for _ in range(steps):
        seg.add_step({}, torch.tensor([0.0]), torch.tensor([1.0]), {},
                     torch.tensor([0.0]), torch.tensor([0.0]),
                     torch.tensor([0.0]), torch.tensor([0.5]))


def test_finalize_rollout_single_agent():
    runner = _make_runner(1)
    _populate_segment(runner.current_segments[0])
    runner._finalize_rollout(0, torch.tensor([0.7]))
    assert len(runner.completed_segments[0]) == 1
    seg = runner.completed_segments[0][0]
    # Two return values should be stored
    assert len(seg.returns) == 2


def test_finalize_rollout_multi_agent_bootstrap_reshape():
    runner = _make_runner(3)
    _populate_segment(runner.current_segments[0])
    # bootstrap_value is a single scalar like get_actions would return
    runner._finalize_rollout(0, torch.tensor([0.7]))
    assert len(runner.completed_segments[0]) == 1
    seg = runner.completed_segments[0][0]
    assert len(seg.returns) == 2
