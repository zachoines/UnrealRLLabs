from Agents.Agent import Agent
from Utility import RunningMeanStdNormalizer
from Environment import EnvCommunicationInterface, EventType
import torch
from torch.utils.tensorboard.writer import SummaryWriter

class RLRunner:
    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface, normalizeStates: bool = False, saveFrequency: int = 10) -> None:
        self.currentUpdate = 0
        self.saveFrequency = saveFrequency
        self.writer = SummaryWriter()
        self.agent = agent
        self.agentComm = agentComm
        self.stateNormalizer = None
        self.normalizeStates = normalizeStates
        if self.normalizeStates:
            self.stateNormalizer = RunningMeanStdNormalizer(
                device=agentComm.device
            )

    def start(self):
        while True:
            event = self.agentComm.wait_for_event()
            if event == EventType.GET_ACTIONS:
                
                states =  self.agentComm.get_states()
                if self.normalizeStates:
                    self.stateNormalizer.update(states)
                    states = self.stateNormalizer.normalize(states)
                actions, *other = self.agent.get_actions(states)
                self.agentComm.send_actions(actions)
            elif event == EventType.UPDATE:
                self.currentUpdate += 1
                states, next_states, actions, rewards, dones, truncs = self.agentComm.get_experiences()
                if self.normalizeStates:
                    states = self.stateNormalizer.normalize(states)
                    next_states = self.stateNormalizer.normalize(next_states) 
                logs = self.agent.update(
                    states, 
                    next_states, 
                    actions, 
                    rewards, 
                    dones, 
                    truncs
                )
                self.log_step(logs)
                if self.currentUpdate % self.saveFrequency == 0:
                    torch.save(self.agent.state_dict(), 'model_state.pth')

    def log_step(self, train_results: dict[str, torch.Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.currentUpdate)

    def end(self) -> None:
        self.agentComm.cleanup()