from Agents import Agent
from Environment import EnvCommunicationInterface, EventType
import torch
from torch.utils.tensorboard.writer import SummaryWriter

class RLRunner:
    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface) -> None:
        self.current_update = 0
        self.writer = SummaryWriter()
        self.agent = agent
        self.agentComm = agentComm

    def start(self):
        while True:
            event = self.agentComm.wait_for_event()
            if event == EventType.GET_ACTIONS:
                states = self.agentComm.get_states()
                actions, _ = self.agent.get_actions(states)
                self.agentComm.send_actions(actions)
            elif event == EventType.UPDATE:
                self.current_update += 1
                states, next_states, actions, rewards, dones, truncs = self.agentComm.get_experiences()
                logs = self.agent.update(states, next_states, actions, rewards, dones, truncs)
                self.log_step(logs)

    def log_step(self, train_results: dict[str, torch.Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.current_update)

    def end(self) -> None:
        self.agentComm.cleanup()