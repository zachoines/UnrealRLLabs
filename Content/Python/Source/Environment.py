import win32event
from ctypes import sizeof, c_float
import win32api
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import shared_memory
from enum import Enum
from typing import Tuple, Dict, Any
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

    def get_states(self) -> Dict[str, torch.Tensor]:
        pass

    def send_actions(self, actions: torch.Tensor) -> None:
        pass

    def get_experiences(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                       torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def wait_for_event(self) -> EventType:
        pass

    def cleanup(self) -> None:
        pass


class SharedMemoryInterface(EnvCommunicationInterface):
    def __init__(self, config: Dict[str, Any]):
        """
        :param config: The JSON config loaded from TerraShift.json.
                       We check environment/shape/state for 'central' or 'agent'
        """
        super().__init__()

        self.config = config
        self.device: torch.device = torch.device(
            "mps" if hasattr(torch, 'has_mps') and torch.has_mps else (
                "cuda" if torch.cuda.is_available() else 
                "cpu"
            )
        )

        # Check presence of central & agent
        shape_cfg = config.get("environment", {}).get("shape", {})
        state_cfg = shape_cfg.get("state", {})
        self.hasCentral = "central" in state_cfg
        self.hasAgent   = "agent" in state_cfg

        self.centralObsSize = 0
        self.agentObsSize   = 0
        if self.hasCentral:
            self.centralObsSize = state_cfg["central"]["obs_size"]
        if self.hasAgent:
            self.agentObsSize   = state_cfg["agent"]["obs_size"]

        # 1) Resource acquisition
        self.action_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="ActionsSharedMemory")
        self.states_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="StatesSharedMemory")
        self.update_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="UpdateSharedMemory")

        # 2) Events
        self.action_ready_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReadyEvent")
        self.action_received_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReceivedEvent")
        self.update_ready_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReadyEvent")
        self.update_received_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReceivedEvent")

        # 3) Mutexes
        self.actions_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "ActionsDataMutex")
        self.states_mutex  = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "StatesDataMutex")
        self.update_mutex  = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "UpdateDataMutex")

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

    def wait_for_event(self) -> EventType:
        result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)

        if result == win32event.WAIT_OBJECT_0:
            return EventType.GET_ACTIONS
        elif result == win32event.WAIT_OBJECT_0 + 1:
            return EventType.UPDATE

    def get_states(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Reads the shared memory for states. 
        Returns (states_dict, dones, truncs)
          states_dict can contain:
              states_dict['central'] => shape (1, NumEnv, centralObsSize) if hasCentral
              states_dict['agent']   => shape (1, NumEnv, CurrentAgents, agentObsSize) if hasAgent
          dones => shape (1, NumEnv, 1)
          truncs => shape (1, NumEnv, 1)
        """
        win32event.WaitForSingleObject(self.states_mutex, win32event.INFINITE)

        byte_array = np.frombuffer(self.states_shared_memory.buf, dtype=np.float32)
        # The first 6 floats: [BufferSize, BatchSize, NumEnvironments, CurrentAgents, SingleEnvStateSize, SingleEnvActionSize]
        info_offset = 6
        self.info = np.frombuffer(byte_array[:info_offset], dtype=np.float32).astype(int)
        _, _, NumEnvironments, CurrentAgents, SingleEnvStateSize, _ = self.info

        # After that, we have:
        #   states portion => NumEnvironments * SingleEnvStateSize
        #   dones portion  => NumEnvironments
        #   truncs portion => NumEnvironments
        states_offset = info_offset
        states_end    = states_offset + (NumEnvironments * SingleEnvStateSize)

        dones_offset  = states_end
        dones_end     = dones_offset + NumEnvironments

        truncs_offset = dones_end
        truncs_end    = truncs_offset + NumEnvironments

        state_data = byte_array[states_offset:states_end]
        dones_data = byte_array[dones_offset:dones_end]
        truncs_data= byte_array[truncs_offset:truncs_end]

        # Construct dictionaries
        states_dict = {}
        # parse the large state_data into "central" + "agent" if they exist
        # total we have SingleEnvStateSize = centralObsSize + agentObsSize * CurrentAgents (if both exist)
        # shape the data => (NumEnvironments, SingleEnvStateSize)
        full_state_np = state_data.reshape(NumEnvironments, SingleEnvStateSize)

        # We'll convert them to shape => [1, NumEnvironments, ...] for consistency with your code
        # so final shapes => (1, NumEnvironments, X)

        idx_start = 0
        if self.hasCentral:
            c_size = self.centralObsSize
            central_data = full_state_np[:, idx_start: idx_start + c_size]
            idx_start += c_size
            # => shape => (1, NumEnvironments, c_size)
            states_dict["central"] = torch.from_numpy(central_data).unsqueeze(0).to(self.device)

        if self.hasAgent:
            # We expect agentObsSize * CurrentAgents
            a_size_total = self.agentObsSize * CurrentAgents
            agent_data = full_state_np[:, idx_start: idx_start + a_size_total]
            idx_start += a_size_total
            # reshape => (NumEnvironments, CurrentAgents, agentObsSize)
            agent_data = agent_data.reshape(NumEnvironments, CurrentAgents, self.agentObsSize)
            # => final => (1, NumEnvironments, CurrentAgents, agentObsSize)
            states_dict["agent"] = torch.from_numpy(agent_data).unsqueeze(0).to(self.device)

        # Now parse dones/truncs => shape => (1, NumEnvironments, 1)
        dones_array = torch.from_numpy(dones_data.reshape(1, NumEnvironments, 1)).float().to(self.device)
        truncs_array= torch.from_numpy(truncs_data.reshape(1, NumEnvironments, 1)).float().to(self.device)

        win32event.ReleaseMutex(self.states_mutex)
        return (states_dict, dones_array, truncs_array)

    def send_actions(self, actions: torch.Tensor) -> None:
        """
        Writes the chosen actions to shared memory in the shape => (NumEnvironments, ActionSize).
        Then signals ActionReceivedEvent.
        """
        win32event.WaitForSingleObject(self.actions_mutex, win32event.INFINITE)
        # self.info => [BufSize, BatchSize, NumEnv, CurrAgents, SingleEnvStateSize, SingleEnvActionSize]
        _, _, NumEnvironments, _, _, ActionSize = self.info

        # shape => (NumEnvironments, ActionSize)
        action_array = np.ndarray(shape=(NumEnvironments, ActionSize),
                                  dtype=np.float32,
                                  buffer=self.action_shared_memory.buf)

        # flatten actions to match the shape
        # user code: expects 'actions' => shape => (NumEnvironments, ActionSize) or maybe (1,NumEnv,ActionSize)
        actions_np = actions.cpu().numpy()
        if actions_np.shape[0] != NumEnvironments:
            # try to reshape if e.g. shape is (1,NumEnv,ActionSize)
            actions_np = actions_np.reshape(NumEnvironments, ActionSize)

        np.copyto(action_array, actions_np)
        win32event.ReleaseMutex(self.actions_mutex)
        win32event.SetEvent(self.action_received_event)

    def get_experiences(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                       torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reads transitions from the 'UpdateSharedMemory'.
         We have [state + next_state + action + reward + trunc + done]
         But we want to parse 'state' into a dict => { "central":..., "agent":... }.
         Similarly for next_state.
         Then we shape the result as needed for training: (BatchSize, NumEnvironments, ...) or so.
        """
        win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)

        raw_data = np.frombuffer(self.update_shared_memory.buf, dtype=np.float32)

        # We re-use the same 6 floats from self.info => [BufSize, BatchSize, NumEnv, CurrAgents, SingleEnvStateSize, SingleEnvActionSize]
        # but read them from self.info set previously in get_states(). Alternatively we can store them again if needed.
        # We'll assume they haven't changed:
        BufSize, BatchSize, NumEnvironments, CurrentAgents, SingleEnvStateSize, ActionSize = self.info

        # Single transition = (2 * SingleEnvStateSize) + ActionSize + 3 (reward, trunc, done)
        # But we also need to consider if multi-agent => we read 1 'done' or maybe we read multiple. 
        # For now, we keep consistent with the updated Unreal side, which uses single floats for done/trunc
        transition_length = (2 * SingleEnvStateSize) + ActionSize + 3

        # We'll only read up to the max needed
        needed_count = transition_length * NumEnvironments * BatchSize
        data_subset = raw_data[:needed_count]
        transitions_np = data_subset.reshape(NumEnvironments, BatchSize, transition_length)

        # Now let's parse out [ state, next_state, action, reward, trunc, done ]
        #  state => (SingleEnvStateSize) 
        #  next_state => (SingleEnvStateSize)
        #  action => (ActionSize)
        #  reward => 1 
        #  trunc => 1
        #  done => 1

        # Sizing:
        #  states => (NumEnv,Batch, SingleEnvStateSize)
        #  next_states => (NumEnv,Batch, SingleEnvStateSize)
        #  actions => (NumEnv,Batch, ActionSize)
        #  rewards => (NumEnv,Batch)
        #  truncs  => (NumEnv,Batch)
        #  dones   => (NumEnv,Batch)

        states_np      = transitions_np[:, :, :SingleEnvStateSize]
        next_states_np = transitions_np[:, :, SingleEnvStateSize : 2*SingleEnvStateSize]
        actions_np     = transitions_np[:, :, 2*SingleEnvStateSize : 2*SingleEnvStateSize + ActionSize]
        reward_np      = transitions_np[:, :, -3]
        trunc_np       = transitions_np[:, :, -2]
        done_np        = transitions_np[:, :, -1]

        # Convert each to Torch
        # But we want states => dict for "central"/"agent"
        # shape => (NumEnv,Batch, SingleEnvStateSize)
        # we might reorder => (Batch, NumEnv, SingleEnvStateSize)
        states_np      = np.transpose(states_np, (1, 0, 2))  # => shape (Batch, NumEnv, SingleEnvStateSize)
        next_states_np = np.transpose(next_states_np, (1, 0, 2))
        actions_np     = np.transpose(actions_np, (1, 0, 2)) # => shape (Batch, NumEnv, ActionSize)

        reward_np      = np.transpose(reward_np, (1, 0))     # => shape (Batch, NumEnv)
        done_np        = np.transpose(done_np, (1, 0))
        trunc_np       = np.transpose(trunc_np, (1, 0))

        # We'll parse states_np => (Batch,NumEnv, SingleEnvStateSize) => separate 'central' + 'agent'
        states_dict_list = []
        next_states_dict_list = []

        B = states_np.shape[0]

        # For each batch row => shape is (NumEnv, SingleEnvStateSize)
        # We'll do "for i in range(B): parse states, next_states => dict"
        for i in range(B):
            s_dict = self._split_central_agent(states_np[i], NumEnvironments, CurrentAgents, SingleEnvStateSize)
            ns_dict= self._split_central_agent(next_states_np[i], NumEnvironments, CurrentAgents, SingleEnvStateSize)
            states_dict_list.append(s_dict)
            next_states_dict_list.append(ns_dict)

        # We'll store them in lists of dict => but you may want to keep them as "S x {...}" 
        # Instead, let's combine them into a single structure => 
        #    states_dict["central"] => shape (B,NumEnv, centralObsSize)
        #    states_dict["agent"] => shape (B,NumEnv, CurrentAgents, agentObsSize) if agent
        # We'll do that by stacking each sub-dict's data across batch dimension.

        # Combine them:
        final_states_dict = self._stack_state_dicts(states_dict_list)
        final_next_states_dict = self._stack_state_dicts(next_states_dict_list)

        # Make them Torch
        rewards  = torch.from_numpy(reward_np).float().to(self.device)  # shape => (Batch, NumEnv)
        done_t   = torch.from_numpy(done_np).float().to(self.device).unsqueeze(-1)   # => (Batch, NumEnv,1)
        trunc_t  = torch.from_numpy(trunc_np).float().to(self.device).unsqueeze(-1)  # => (Batch, NumEnv,1)
        acts_t   = torch.from_numpy(actions_np).float().to(self.device)  # => (Batch, NumEnv, ActionSize)

        win32event.ReleaseMutex(self.update_mutex)
        win32event.SetEvent(self.update_received_event)

        return (final_states_dict, final_next_states_dict, 
                acts_t, rewards, done_t, trunc_t)

    def cleanup(self):
        # Clean up shared memories
        self.action_shared_memory.close()
        self.states_shared_memory.close()
        self.update_shared_memory.close()

        # Clean up the synchronization events and mutexes
        for handle in [self.action_ready_event, self.action_received_event,
                       self.update_ready_event, self.update_received_event,
                       self.actions_mutex, self.states_mutex, self.update_mutex]:
            win32api.CloseHandle(handle)

    # ------------------------------------------------
    #   INTERNAL HELPER: parse (NumEnv,SingleEnvStateSize) => dict
    # ------------------------------------------------
    def _split_central_agent(self, single_batch_np: np.ndarray,
                             NumEnvironments: int, CurrentAgents: int, SingleEnvStateSize: int
                             ) -> Dict[str, torch.Tensor]:
        """
        single_batch_np => shape (NumEnv, SingleEnvStateSize)
        We'll parse out central portion and agent portion,
        returning a dict e.g. { "central":(NumEnv, centralObsSize), "agent":(NumEnv,CurrentAgents,agentObsSize) }
        as Tensors on self.device.
        """
        # shape => (NumEnv, SingleEnvStateSize)
        # We'll do same approach used in get_states.
        states_dict = {}
        idx_start = 0

        # we have the same self.hasCentral, self.centralObsSize, self.hasAgent, self.agentObsSize
        # for each environment row in single_batch_np
        # We'll do row by row for each environment
        # But it's simpler to do a single slicing approach for each environment:
        # We'll do it in a vectorized manner.

        # final shape => we want "central" => (NumEnv, centralObsSize)
        # "agent" => (NumEnv, CurrentAgents, agentObsSize)

        # Flatten approach:
        # Actually, we can do it once for the entire array. Each row => [ 0..centralObsSize + agentObsSize*CurrentAgents ]
        # We'll slice columns. For row in [0..NumEnv)...

        # 1) central
        if self.hasCentral:
            c_size = self.centralObsSize
            central_data = single_batch_np[:, idx_start: idx_start + c_size]
            idx_start += c_size
            # => shape => (NumEnv, c_size)
            states_dict["central"] = torch.from_numpy(central_data).float().to(self.device)

        if self.hasAgent:
            a_size_total = self.agentObsSize * CurrentAgents
            agent_data = single_batch_np[:, idx_start: idx_start + a_size_total]
            idx_start += a_size_total
            # => shape => (NumEnv, a_size_total)
            # reshape => (NumEnv, CurrentAgents, agentObsSize)
            agent_data = agent_data.reshape(NumEnvironments, CurrentAgents, self.agentObsSize)
            states_dict["agent"] = torch.from_numpy(agent_data).float().to(self.device)

        return states_dict

    def _stack_state_dicts(self, dict_list: list) -> Dict[str, torch.Tensor]:
        """
        Takes a list of dicts, each with optional 'central'/'agent' keys,
        each value => shape (NumEnv, X) or (NumEnv,AgentCount, X).
        We stack them along dim=0 => produce shape => (B, NumEnv, X)...

        This transforms from "list of B dicts" => "dict of Tensors with shape (B,...)"
        """
        if not dict_list:
            return {}

        # We'll check what keys exist in the first dict
        keys = dict_list[0].keys()
        out_dict = {}

        for k in keys:
            # gather each item => shape (NumEnv, X...) => stack => (B, NumEnv, X...)
            arrs = [d[k] for d in dict_list]
            out_dict[k] = torch.stack(arrs, dim=0)  # => shape => (B, NumEnv, ...)

        return out_dict
