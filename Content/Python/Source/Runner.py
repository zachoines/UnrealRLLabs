from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer
from Source.Environment import EnvCommunicationInterface, EventType
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

class RLRunner:
    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface, config: Dict) -> None:
        train_params = config['train']
        state_normalization_config = train_params.get('states_normalizer', None)
        saveFrequency  = train_params.get('saveFrequency', 10)
        
        self.currentUpdate = 0
        self.saveFrequency = saveFrequency
        self.writer = SummaryWriter()
        self.agent = agent
        self.agentComm = agentComm
        self.state_normalizer = None
        if state_normalization_config:
            self.state_normalizer = RunningMeanStdNormalizer(
                **state_normalization_config,
                device=agentComm.device
            )

    def start(self):
        while True:
            event = self.agentComm.wait_for_event()
            if event == EventType.GET_ACTIONS:
                states, dones, truncs = self.agentComm.get_states()
                if self.state_normalizer:
                    self.state_normalizer.update(states)
                    states = self.state_normalizer.normalize(states)
                actions, *other = self.agent.get_actions(states, dones=dones, truncs=truncs)
                self.agentComm.send_actions(actions)
            elif event == EventType.UPDATE:
                self.currentUpdate += 1
                states, next_states, actions, rewards, dones, truncs = self.agentComm.get_experiences()
                # self.agentComm.write_experiences_to_file(
                #     states, next_states, actions, rewards, dones, truncs,
                #     "C:\\Users\\zachoines\\Documents\\Unreal\\UnrealRLLabs\\Content\\Python\\TEST\\PythonTransitions.csv"
                # )
                if self.state_normalizer:
                    states = self.state_normalizer.normalize(states)
                    next_states = self.state_normalizer.normalize(next_states) 
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