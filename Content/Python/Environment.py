import win32event
from ctypes import sizeof, c_float
import win32api
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import shared_memory
from enum import Enum
from typing import Tuple, Dict
import torch
import json

MUTEX_ALL_ACCESS = 0x1F0001
SLEEP_INTERVAL = 1.0  # seconds
    
class EventType(Enum):
    UPDATE = 0
    GET_ACTIONS = 1

class EnvCommunicationInterface:
    def __init__(self) -> None:
        pass

    def get_states(self) -> torch.Tensor:
        pass

    def send_actions(self, actions: torch.Tensor) -> None:
        pass

    def get_experiences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def wait_for_event(self) -> EventType:
        pass

    def cleanup(self) -> None:
        pass

class SharedMemoryInterface(EnvCommunicationInterface):
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
        
        self.device: torch.device = torch.device(
            "mps" if torch.has_mps else (
                "cuda" if torch.has_cuda else 
                "cpu"
            )
        )
        self.config = self._fetch_config_from_shared_memory()
        
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

    def _fetch_config_from_shared_memory(self) -> np.ndarray:
        win32event.WaitForSingleObject(self.config_event, win32event.INFINITE)
        win32event.WaitForSingleObject(self.config_mutex, win32event.INFINITE)

        # Extract the configuration from the shared memory
        byte_array = np.frombuffer(self.config_shared_memory.buf, dtype=np.uint8) 
        
        # Convert the byte array to a string
        json_str = byte_array.tobytes().split(b'\x00', 1)[0].decode('utf-8')
        
        # Parse the string as JSON
        json_data = json.loads(json_str)
        
        win32event.ReleaseMutex(self.config_mutex)
        return json_data

    def wait_for_event(self):
        result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)

        if result == win32event.WAIT_OBJECT_0:
            return EventType.GET_ACTIONS
        elif result == win32event.WAIT_OBJECT_0 + 1:
            return EventType.UPDATE

    def get_states(self) -> Tuple[torch.Tensor]:
        win32event.WaitForSingleObject(self.states_mutex, win32event.INFINITE)

        # Extract the first six floats
        info_offset = 6
        byte_array = np.frombuffer(self.states_shared_memory.buf, dtype=np.float32)
        _, _, NumEnvironments, CurrentAgents, StateSize, _ = self.info = np.frombuffer(byte_array[:info_offset], dtype=np.float32).astype(int)

        # Use the rest of the buffer to create a numpy array for the state
        states_offset = info_offset + (NumEnvironments * StateSize)
        dones_offset = states_offset + NumEnvironments
        truncs_offset = dones_offset + NumEnvironments

        state_data = byte_array[info_offset:states_offset]
        dones_data = byte_array[states_offset:dones_offset]
        truncs_data = byte_array[dones_offset:truncs_offset] 
        if CurrentAgents > 0: # Multi-agent Environments
            state_array = np.ndarray(shape=(1, NumEnvironments, CurrentAgents, int(StateSize / CurrentAgents)), dtype=np.float32, buffer=state_data)
        else:
            state_array = np.ndarray(shape=(NumEnvironments, StateSize), dtype=np.float32, buffer=state_data)

        dones_array = np.ndarray(shape=(1, NumEnvironments, 1), dtype=np.float32, buffer=dones_data)
        truncs_array = np.ndarray(shape=(1, NumEnvironments, 1), dtype=np.float32, buffer=truncs_data)
        
        win32event.ReleaseMutex(self.states_mutex)
        return (
            torch.tensor(state_array, device=self.device), 
            torch.tensor(dones_array, device=self.device), 
            torch.tensor(truncs_array, device=self.device)
        )

    def send_actions(self, actions: torch.Tensor) -> None:
        win32event.WaitForSingleObject(self.actions_mutex, win32event.INFINITE)
        _, _, NumEnvironments, _, _, ActionSize = self.info.astype(int)
        action_array = np.ndarray(shape=(NumEnvironments, ActionSize), dtype=np.float32, buffer=self.action_shared_memory.buf)
        actions = actions.view(NumEnvironments, ActionSize).cpu().numpy()
        np.copyto(action_array, actions)
        win32event.ReleaseMutex(self.actions_mutex)
        win32event.SetEvent(self.action_received_event)

    def get_experiences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)
        
        # Assuming the shared memory is flattened
        _, BatchSize, NumEnvironments, CurrentAgents, StateSize, ActionSize = self.info.astype(int)
        raw_update_array = np.frombuffer(self.update_shared_memory.buf, dtype=np.float32)
        
        # Calculate the total length of a single transition
        transition_length = (2 * StateSize) + ActionSize + 3  # +3 for reward, trunc, and done
        update_array_trunc = raw_update_array[:(transition_length * NumEnvironments * BatchSize)]
        update_array = update_array_trunc.reshape(NumEnvironments, BatchSize, transition_length)
        
        # Split the array into states, next_states, actions, rewards, and dones
        states = update_array[:, :, :StateSize]
        next_states = update_array[:, :, StateSize:2 * StateSize]
        actions = update_array[:, :, 2 * StateSize:2 * StateSize + ActionSize]
        rewards = update_array[:, :, -3]
        truncs = update_array[:, :, -2]
        dones = update_array[:, :, -1]

        win32event.ReleaseMutex(self.update_mutex)
        win32event.SetEvent(self.update_received_event)

        if CurrentAgents > 0: # Multi-agent Environments
            return (
                torch.tensor(states, dtype=torch.float32, device=self.device).contiguous().permute((1, 0, 2)).view(BatchSize, NumEnvironments, CurrentAgents, int(StateSize / CurrentAgents)).contiguous(),
                torch.tensor(next_states, dtype=torch.float32, device=self.device).contiguous().permute((1, 0, 2)).view(BatchSize, NumEnvironments, CurrentAgents, int(StateSize / CurrentAgents)).contiguous(),
                torch.tensor(actions, dtype=torch.float32, device=self.device).contiguous().permute((1, 0, 2)).view(BatchSize, NumEnvironments, CurrentAgents, int(ActionSize / CurrentAgents)).contiguous(),
                torch.tensor(rewards, dtype=torch.float32, device=self.device).contiguous().permute((1, 0)).view(BatchSize, NumEnvironments, 1).contiguous(),
                torch.tensor(dones, dtype=torch.float32, device=self.device).contiguous().permute((1, 0)).view(BatchSize, NumEnvironments, 1).contiguous(),
                torch.tensor(truncs, dtype=torch.float32, device=self.device).contiguous().permute((1, 0)).view(BatchSize, NumEnvironments, 1).contiguous()
            )
        else:
            return (
                torch.tensor(states, dtype=torch.float32, device=self.device).contiguous().permute((1, 0, 2)),
                torch.tensor(next_states, dtype=torch.float32, device=self.device).contiguous().permute((1, 0, 2)),
                torch.tensor(actions, dtype=torch.float32, device=self.device).contiguous().permute((1, 0, 2)),
                torch.tensor(rewards, dtype=torch.float32, device=self.device).contiguous().permute((1, 0)),
                torch.tensor(dones, dtype=torch.float32, device=self.device).contiguous().permute((1, 0)),
                torch.tensor(truncs, dtype=torch.float32, device=self.device).contiguous().permute((1, 0))
            )

    def cleanup(self):
        # Clean up shared memories
        self.action_shared_memory.close()
        self.states_shared_memory.close()
        self.update_shared_memory.close()

        # Clean up the synchronization events and mutexes
        for handle in [self.action_ready_event, self.action_received_event, self.update_ready_event, 
                       self.update_received_event, self.actions_mutex, self.states_mutex, self.update_mutex]:
            win32api.CloseHandle(handle)
