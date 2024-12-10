import torch
from typing import Tuple
from Source.Environment import SharedMemoryInterface
from Source.Runner import RLRunner
from Source.Agent import Agent
from Agents.MAPOCAAgent import MAPOCAAgent

class AgentEnvFactory:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_agent_and_environment(self) -> Tuple[Agent, SharedMemoryInterface]:
        # Create environment communication interface
        # For now we use SharedMemoryInterface as is
        env_comm = SharedMemoryInterface()

        # Parse config to determine agent type
        agent_type = self.config['agent']['type']

        if agent_type == 'MA_POCA':
            agent = MAPOCAAgent(self.config, self.device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent, env_comm

    def create_runner(self, agent: Agent, agentComm: SharedMemoryInterface) -> RLRunner:
        train_params = self.config['train']
        # Extract runner params if needed. Otherwise just pass agentComm and agent.
        normalizeStates = train_params.get('normalize_states', False)
        saveFrequency = train_params.get('saveFrequency', 10)

        runner = RLRunner(agent=agent, agentComm=agentComm, normalizeStates=normalizeStates, saveFrequency=saveFrequency)
        return runner
