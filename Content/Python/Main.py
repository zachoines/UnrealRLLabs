import win32event
import win32api
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import shared_memory
import torch
from Agents import A2C
from Config import Config, EnvParams, TrainParams, NetworkParams
from Config import ActionSpace, ActionType
from typing import Dict, List, Tuple

MUTEX_ALL_ACCESS = 0x1F0001
SLEEP_INTERVAL = 1.0  # seconds

class SharedMemoryInterface:
    def __init__(self):
        # Initialize shared memory for configuration
        self.config_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="ConfigSharedMemory")
        self.config_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ConfigReadyEvent")
        self.config_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "ConfigDataMutex")

        # Initialize shared memories for actions, states, and updates based on the fetched configuration
        self.action_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="ActionsSharedMemory")
        self.states_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="StatesSharedMemory")
        self.update_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="UpdateSharedMemory")

        # Initialize synchronization events
        self.action_ready_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReadyEvent")
        self.action_received_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReceivedEvent")
        self.update_ready_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReadyEvent")
        self.update_received_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReceivedEvent")

        # Initialize synchronization mutexes
        self.actions_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "ActionsDataMutex")
        self.states_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "StatesDataMutex")
        self.update_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "UpdateDataMutex")

        self.config = self._fetch_config_from_shared_memory()
        self.device = torch.device(
            "mps" if torch.has_mps else (
                "cuda" if torch.has_cuda else 
                "cpu"
            )
        )

    def _wait_for_resource(self, func, *args, **kwargs):
        """Utility function to wait and retry until the resource is available."""
        dot_count = 1
        pbar = tqdm(total=4, desc="Waiting for Unreal Engine", position=0, leave=True, bar_format='{desc}')
        
        while True:
            try:
                resource = func(*args, **kwargs)
                if resource:
                    pbar.close()
                    return resource
            except Exception as _:
                pbar.set_description(f"Waiting for Unreal Engine {'.' * dot_count}")
                dot_count = (dot_count % 4) + 1  # Cycle between 1 and 4 dots
                pbar.update(dot_count - pbar.n)
                time.sleep(SLEEP_INTERVAL)

    def _fetch_config_from_shared_memory(self) -> Config:
        win32event.WaitForSingleObject(self.config_event, win32event.INFINITE)
        win32event.WaitForSingleObject(self.config_mutex, win32event.INFINITE)

        # Extract the configuration from the shared memory
        config_array = np.frombuffer(self.config_shared_memory.buf, dtype=np.int32)  # Assuming int32 for all config values
        num_environments = config_array[0]
        num_actions = config_array[1]
        state_size = config_array[2]
        training_batch_size = config_array[3]
        win32event.ReleaseMutex(self.config_mutex)

        return Config(
            EnvParams(
                num_environments=num_environments, 
                num_actions=num_actions, 
                state_size=state_size,
                action_space=ActionSpace(ActionType.CONTINUOUS, -1.0, 1.0)
            ),
            TrainParams(
                training_batch_size=training_batch_size
            ),
            NetworkParams(
                hidden_size=128
            ),
            AgentParams()
        )

    def wait_for_event(self):
        result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)

        if result == win32event.WAIT_OBJECT_0:
            return "GET_ACTION"
        elif result == win32event.WAIT_OBJECT_0 + 1:
            return "UPDATE"

    def get_states(self):
        win32event.WaitForSingleObject(self.states_mutex, win32event.INFINITE)
        state_array = np.ndarray(shape=(self.config.envParams.num_environments, self.config.envParams.state_size), dtype=np.float32, buffer=self.states_shared_memory.buf)
        win32event.ReleaseMutex(self.states_mutex)
        return torch.tensor(state_array)

    def send_actions(self, actions: torch.Tensor):
        win32event.WaitForSingleObject(self.actions_mutex, win32event.INFINITE)
        action_array = np.ndarray(shape=(self.config.envParams.num_environments, self.config.envParams.num_actions), dtype=np.float32, buffer=self.action_shared_memory.buf)
        np.copyto(action_array, actions.numpy())
        win32event.ReleaseMutex(self.actions_mutex)
        win32event.SetEvent(self.action_received_event)

    def get_experiences(self):

        win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)
        
        # Assuming the shared memory is flattened
        update_array = np.frombuffer(self.update_shared_memory.buf, dtype=np.float32)
        
        # Calculate the total length of a single transition
        transition_length = (2 * self.config.envParams.state_size) + self.config.envParams.num_actions + 2  # +2 for reward and done
        
        # Reshape the array to shape (num_steps, num_envs, transition_length)
        update_array = update_array.reshape(self.config.trainParams.training_batch_size, self.config.envParams.num_environments, transition_length)
        
        # Split the array into states, next_states, actions, rewards, and dones
        states = update_array[:, :, :self.config.envParams.state_size]
        next_states = update_array[:, :, self.config.envParams.state_size:2 * self.config.envParams.state_size]
        actions = update_array[:, :, 2 * self.config.envParams.state_size:2 * self.config.envParams.state_size + self.config.envParams.num_actions]
        rewards = update_array[:, :, -2]
        dones = update_array[:, :, -1]
        
        # Convert numpy arrays to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        truncs = torch.zeros_like(dones, dtype=torch.float32, device=self.device) # TODO: Read signal from Unreal Engine
        
        win32event.ReleaseMutex(self.update_mutex)
        win32event.SetEvent(self.update_received_event)
        
        return states, next_states, actions, rewards, dones, truncs

    def cleanup(self):
        # Clean up shared memories
        self.action_shared_memory.close()
        self.states_shared_memory.close()
        self.update_shared_memory.close()

        # Clean up the synchronization events and mutexes
        for handle in [self.action_ready_event, self.action_received_event, self.update_ready_event, 
                       self.update_received_event, self.actions_mutex, self.states_mutex, self.update_mutex]:
            win32api.CloseHandle(handle)

if __name__ == "__main__":
    memory_interface = SharedMemoryInterface()

    agent = A2C(
        memory_interface.config
    )

    try:
        while True:
            event = memory_interface.wait_for_event()
            if event == "GET_ACTION":
                states = memory_interface.get_states()
                actions, _ = agent.get_actions(states)
                memory_interface.send_actions(actions)
            elif event == "UPDATE":
                agent.update(*memory_interface.get_experiences())

    except KeyboardInterrupt:
        memory_interface.cleanup()
