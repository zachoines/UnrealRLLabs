from Agents.Agent import Agent
from Utility import RunningMeanStd
from Environment import EnvCommunicationInterface, EventType
import torch
from torch.utils.tensorboard.writer import SummaryWriter

class RLRunner:
    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface) -> None:
        self.currentUpdate = 0
        self.writer = SummaryWriter()
        self.agent = agent
        self.agentComm = agentComm
        # self.stateNormalizer = RunningMeanStd(
        #     shape = (agentComm.config[ConfigData.num_environments.value], self.agent.config.state_size),
        #     device=agentComm.device
        # )

    def start(self):
        while True:
            event = self.agentComm.wait_for_event()
            if event == EventType.GET_ACTIONS:
                states = self.agentComm.get_states()
                # states = self.stateNormalizer.update(states)
                actions, *other = self.agent.get_actions(states)
                self.agentComm.send_actions(actions)
            elif event == EventType.UPDATE:
                self.currentUpdate += 1
                states, next_states, actions, rewards, dones, truncs = self.agentComm.get_experiences()
                logs = self.agent.update(
                    states, # self.stateNormalizer.normalize(states), 
                    next_states, # self.stateNormalizer.normalize(next_states), 
                    actions, 
                    rewards, 
                    dones, 
                    truncs
                )
                self.log_step(logs)

    def log_step(self, train_results: dict[str, torch.Tensor])->None:
        for metric, value in train_results.items():
            self.writer.add_scalar(tag=metric, scalar_value=value, global_step=self.currentUpdate)

    def end(self) -> None:
        self.agentComm.cleanup()