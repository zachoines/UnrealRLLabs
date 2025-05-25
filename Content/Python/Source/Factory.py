import torch
from typing import Tuple
from Source.Environment import SharedMemoryInterface
from Source.Runner import RLRunner
from Source.Agent import Agent
from Agents.MAPOCAAgent import MAPOCAAgent
from Agents.RandomAgent import RandomAgent

class AgentEnvFactory:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_agent_and_environment(self) -> Tuple[Agent, SharedMemoryInterface]:
        env_comm = SharedMemoryInterface(self.config)

        agent_type = self.config['agent']['type']

        if agent_type == 'MA_POCA':
            agent = MAPOCAAgent(self.config, self.device)
        elif agent_type == 'RND':  # <-- new condition
            agent = RandomAgent(self.config, self.device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent, env_comm

    def create_runner(self, agent: Agent, agentComm: SharedMemoryInterface) -> RLRunner:
        runner = RLRunner(
            agent=agent, 
            agentComm=agentComm, 
            cfg=self.config
        )
        return runner
