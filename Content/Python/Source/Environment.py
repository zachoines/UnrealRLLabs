import win32event
import win32api
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import shared_memory
from enum import Enum
from typing import Tuple, Dict, Any
import torch

MUTEX_ALL_ACCESS = 0x1F0001
SLEEP_INTERVAL = 1.0  # seconds

class EventType(Enum):
    UPDATE = 0
    GET_ACTIONS = 1

class EnvCommunicationInterface:
    def __init__(self) -> None:
        pass

    def get_states(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        pass

    def send_actions(self, actions: torch.Tensor) -> None:
        pass

    def get_experiences(self) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        pass

    def wait_for_event(self) -> EventType:
        pass

    def cleanup(self) -> None:
        pass


class SharedMemoryInterface(EnvCommunicationInterface):
    """
    SharedMemoryInterface that reads/writes from three shared memory blocks
    (Actions, States, Update) along with associated synchronization events and mutexes.

    This version is updated to handle multi-agent actions with shape
    (1, NumEnvironments, NumAgents, ActionDim) by flattening the last two dims
    to (NumAgents*ActionDim). Then on reading experiences, we unflatten back to
    shape (Batch, NumEnv, NumAgents, ActionDim).
    """
    def __init__(self, config: Dict[str, Any]):
        """
        :param config: The JSON config loaded from TerraShift.json.
                       We'll check environment/shape/state for 'central' or 'agent'
        """
        super().__init__()

        self.config = config
        self.device: torch.device = torch.device(
            "mps" if (hasattr(torch, 'has_mps') and torch.has_mps) else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        # Check presence of 'central'/'agent' in config (for states)
        shape_cfg = config.get("environment", {}).get("shape", {})
        state_cfg = shape_cfg.get("state", {})
        self.hasCentral = ("central" in state_cfg)
        self.hasAgent   = ("agent" in state_cfg)

        self.centralObsSize = 0
        self.agentObsSize   = 0
        if self.hasCentral:
            self.centralObsSize = state_cfg["central"]["obs_size"]
        if self.hasAgent:
            self.agentObsSize   = state_cfg["agent"]["obs_size"]

        # Acquire references to shared memory
        self.action_shared_memory = self._wait_for_resource(
            shared_memory.SharedMemory,
            name="ActionsSharedMemory"
        )
        self.states_shared_memory = self._wait_for_resource(
            shared_memory.SharedMemory,
            name="StatesSharedMemory"
        )
        self.update_shared_memory = self._wait_for_resource(
            shared_memory.SharedMemory,
            name="UpdateSharedMemory"
        )

        # Events
        self.action_ready_event = self._wait_for_resource(
            win32event.OpenEvent,
            win32event.EVENT_ALL_ACCESS,
            False,
            "ActionReadyEvent"
        )
        self.action_received_event = self._wait_for_resource(
            win32event.OpenEvent,
            win32event.EVENT_ALL_ACCESS,
            False,
            "ActionReceivedEvent"
        )
        self.update_ready_event = self._wait_for_resource(
            win32event.OpenEvent,
            win32event.EVENT_ALL_ACCESS,
            False,
            "UpdateReadyEvent"
        )
        self.update_received_event = self._wait_for_resource(
            win32event.OpenEvent,
            win32event.EVENT_ALL_ACCESS,
            False,
            "UpdateReceivedEvent"
        )

        # Mutexes
        self.actions_mutex = self._wait_for_resource(
            win32event.OpenMutex,
            MUTEX_ALL_ACCESS,
            False,
            "ActionsDataMutex"
        )
        self.states_mutex = self._wait_for_resource(
            win32event.OpenMutex,
            MUTEX_ALL_ACCESS,
            False,
            "StatesDataMutex"
        )
        self.update_mutex = self._wait_for_resource(
            win32event.OpenMutex,
            MUTEX_ALL_ACCESS,
            False,
            "UpdateDataMutex"
        )

        # Will be read inside get_states() => 6 integer fields
        # [ buffer_size, batch_size, NumEnv, CurrAgents, SingleEnvStateSize, SingleEnvActionSize ]
        self.info = None

    def _wait_for_resource(self, func, *args, **kwargs):
        """
        Utility function to wait and retry until the resource is available.
        Repeatedly tries to open the shared memory, event, or mutex.
        """
        dot_count = 1
        pbar = tqdm(total=4, desc="Waiting for Unreal Engine", position=0, leave=True, bar_format='{desc}')
        
        while True:
            try:
                resource = func(*args, **kwargs)
                if resource:
                    pbar.close()
                    return resource
            except Exception:
                pbar.set_description(f"Waiting for Unreal Engine {'.' * dot_count}")
                dot_count = (dot_count % 4) + 1
                pbar.update(dot_count - pbar.n)
                time.sleep(SLEEP_INTERVAL)

    def wait_for_event(self) -> EventType:
        result = win32event.WaitForMultipleObjects(
            [self.action_ready_event, self.update_ready_event],
            False,
            win32event.INFINITE
        )
        if result == win32event.WAIT_OBJECT_0:
            return EventType.GET_ACTIONS
        elif result == win32event.WAIT_OBJECT_0 + 1:
            return EventType.UPDATE

    # ------------------------------------------------
    #   GET_STATES
    # ------------------------------------------------
    def get_states(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Reads the shared memory for states. 
        Returns (states_dict, dones, truncs):
          states_dict can contain:
            'central' => shape (1, NumEnv, cObsSize)   (if hasCentral)
            'agent'   => shape (1, NumEnv, Agents, aObsSize) (if hasAgent)
          dones => shape (1, NumEnv, 1)
          truncs => shape (1, NumEnv, 1)
        """
        win32event.WaitForSingleObject(self.states_mutex, win32event.INFINITE)

        byte_array = np.frombuffer(self.states_shared_memory.buf, dtype=np.float32)

        # The first 6 floats => info => cast to int
        info_offset = 6
        self.info = byte_array[:info_offset].astype(int)
        # => [ BufSize, BatchSize, NumEnv, CurrAgents, SingleEnvStateSize, SingleEnvActionSize]
        _, _, NumEnvironments, CurrentAgents, SingleEnvStateSize, _ = self.info

        # Next chunk => states => (NumEnvironments * SingleEnvStateSize) floats
        # Then dones => NumEnvironments floats
        # Then truncs => NumEnvironments floats
        states_offset = info_offset
        states_end    = states_offset + (NumEnvironments * SingleEnvStateSize)
        dones_offset  = states_end
        dones_end     = dones_offset + NumEnvironments
        truncs_offset = dones_end
        truncs_end    = truncs_offset + NumEnvironments

        state_data = byte_array[states_offset:states_end]
        dones_data = byte_array[dones_offset:dones_end]
        truncs_data= byte_array[truncs_offset:truncs_end]

        # Now reshape => (NumEnv, SingleEnvStateSize)
        full_state_np = np.reshape(state_data, (NumEnvironments, SingleEnvStateSize)).copy()

        # Build states_dict
        states_dict: Dict[str, torch.Tensor] = {}
        idx_start = 0

        if self.hasCentral:
            c_size = self.centralObsSize
            central_data = full_state_np[:, idx_start : idx_start + c_size]
            idx_start += c_size

            # => shape => (NumEnv, c_size) => then => (1, NumEnv, c_size)
            central_t = torch.from_numpy(central_data).float().contiguous().unsqueeze(0).to(self.device)
            states_dict["central"] = central_t

        if self.hasAgent:
            a_size_total = self.agentObsSize * CurrentAgents
            agent_data = full_state_np[:, idx_start : idx_start + a_size_total]
            idx_start += a_size_total

            # => shape => (NumEnv, a_size_total) => (NumEnv, CurrAgents, agentObsSize)
            agent_data = np.reshape(agent_data, (NumEnvironments, CurrentAgents, self.agentObsSize)).copy()
            agent_t = torch.from_numpy(agent_data).float().contiguous().unsqueeze(0).to(self.device)
            states_dict["agent"] = agent_t

        # Dones => shape => (1, NumEnv, 1)
        dones_np  = np.reshape(dones_data, (1, NumEnvironments, 1)).copy()
        truncs_np = np.reshape(truncs_data, (1, NumEnvironments, 1)).copy()

        dones_t  = torch.from_numpy(dones_np).float().contiguous().to(self.device)
        truncs_t = torch.from_numpy(truncs_np).float().contiguous().to(self.device)

        win32event.ReleaseMutex(self.states_mutex)
        return (states_dict, dones_t, truncs_t)

    # ------------------------------------------------
    #   SEND_ACTIONS
    # ------------------------------------------------
    def send_actions(self, actions: torch.Tensor) -> None:
        """
        Writes the chosen actions to shared memory. The layout in memory is (NumEnvironments, ActionSize).

        If multi-agent => the 'ActionSize' is actually (numAgents * perAgentActionDim).
        We'll flatten the last 2 dims of actions => shape (numEnv, ActionSize).
        Then we copy to the shared memory.

        Example shapes:
         - single-agent discrete => (1, env, 1, 1) or (1, env, 1, #branches)
         - multi-agent => (1, env, agents, actionDim)
        """
        win32event.WaitForSingleObject(self.actions_mutex, win32event.INFINITE)

        if self.info is None:
            # haven't called get_states() => not sure of shape => do nothing or raise
            win32event.ReleaseMutex(self.actions_mutex)
            return

        # info => [ _, _, NumEnvironments, CurrentAgents, _, ActionSize ]
        _, _, NumEnvironments, CurrentAgents, _, ActionSize = self.info

        # Create a numpy array over the shared memory => shape=(NumEnvironments, ActionSize)
        action_array = np.ndarray(
            shape=(NumEnvironments, ActionSize),
            dtype=np.float32,
            buffer=self.action_shared_memory.buf
        )

        # Convert actions to CPU
        actions_cpu = actions.detach().cpu()

        # The typical shape if multi-agent is (1, NumEnv, Agents, actDim).
        # Flatten => (NumEnv, Agents*actDim).
        # If single agent => (1, NumEnv, 1, actDim) or (1, NumEnv, actDim).
        # We'll unify the approach:

        if actions_cpu.ndim == 4:
            # shape => (batch=1, NumEnv, Agents, actDim)
            b, E, A, D = actions_cpu.shape
            if b != 1:
                raise ValueError(f"Multi-agent actions expected batch=1, got batch={b}")
            flat_actions = actions_cpu.view(E, A * D).numpy()  # => (NumEnv, A*D)
        elif actions_cpu.ndim == 3:
            # shape => (1, NumEnv, ActionSize) or (NumEnv, ActionSize)
            # if shape[0]==1 => flatten => (NumEnv, ActionSize)
            if actions_cpu.shape[0] == 1:
                # => (1,NumEnv, ActionDim) => flatten => (NumEnv,ActionDim)
                flat_actions = actions_cpu.squeeze(0).numpy()
            else:
                # => (NumEnv, ActionSize) already
                flat_actions = actions_cpu.numpy()
        else:
            # fallback => just reshape to (NumEnvironments, ActionSize)
            flat_actions = actions_cpu.reshape(NumEnvironments, ActionSize).numpy()

        np.copyto(action_array, flat_actions)

        win32event.ReleaseMutex(self.actions_mutex)
        win32event.SetEvent(self.action_received_event)

    # ------------------------------------------------
    #   GET_EXPERIENCES
    # ------------------------------------------------
    def get_experiences(self) -> Tuple[Dict[str, torch.Tensor],
                                       Dict[str, torch.Tensor],
                                       torch.Tensor,
                                       torch.Tensor,
                                       torch.Tensor,
                                       torch.Tensor]:
        """
        Reads transitions from UpdateSharedMemory:
         => [ state + next_state + action + reward + trunc + done ]

        The 'actions' portion is (NumEnv, ActionSize), but in multi-agent
        that means (NumEnv, Agents*perAgentActionDim). We reorder to (Batch, NumEnv, ...).
        Then if multi-agent => reshape => (Batch, NumEnv, Agents, actDim).

        Returns:
          states_dict => shape (B, NumEnv, ...)
          next_states_dict => shape (B, NumEnv, ...)
          actions => shape => (B, NumEnv, CurrAgents, actionDim) if multi-agent
          rewards => (B, NumEnv, 1)
          dones =>   (B, NumEnv, 1)
          truncs =>  (B, NumEnv, 1)
        """
        win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)

        raw_data = np.frombuffer(self.update_shared_memory.buf, dtype=np.float32)
        if self.info is None:
            # no shape => bail out
            win32event.ReleaseMutex(self.update_mutex)
            win32event.SetEvent(self.update_received_event)
            return ({}, {}, torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

        BufSize, BatchSize, NumEnvironments, CurrAgents, SingleEnvStateSize, ActionSize = self.info

        # single transition length => (2 * SingleEnvStateSize) + ActionSize + 3
        transition_length = (2 * SingleEnvStateSize) + ActionSize + 3
        needed_count = transition_length * NumEnvironments * BatchSize

        data_subset = raw_data[:needed_count]
        # transitions_np => shape => (NumEnv, Batch, transition_length)
        transitions_np = np.reshape(data_subset, (NumEnvironments, BatchSize, transition_length)).copy()

        # slice out => states, next_states, actions, reward, trunc, done
        states_np      = transitions_np[:, :, :SingleEnvStateSize]
        next_states_np = transitions_np[:, :, SingleEnvStateSize : 2*SingleEnvStateSize]
        actions_np     = transitions_np[:, :, 2*SingleEnvStateSize : 2*SingleEnvStateSize + ActionSize]
        reward_np      = transitions_np[:, :, -3]
        trunc_np       = transitions_np[:, :, -2]
        done_np        = transitions_np[:, :, -1]

        # reorder => (Batch, NumEnv, ...)
        states_np      = np.transpose(states_np, (1,0,2)).copy()
        next_states_np = np.transpose(next_states_np, (1,0,2)).copy()
        actions_np     = np.transpose(actions_np, (1,0,2)).copy()

        reward_np = np.transpose(reward_np, (1,0)).copy()  # => (B,NumEnv)
        trunc_np  = np.transpose(trunc_np,  (1,0)).copy()
        done_np   = np.transpose(done_np,   (1,0)).copy()

        # parse states into dicts
        B = states_np.shape[0]
        states_dict_list     = []
        next_states_dict_list= []

        for b in range(B):
            s_dict  = self._split_central_agent(states_np[b], NumEnvironments, CurrAgents, SingleEnvStateSize)
            ns_dict = self._split_central_agent(next_states_np[b], NumEnvironments, CurrAgents, SingleEnvStateSize)
            states_dict_list.append(s_dict)
            next_states_dict_list.append(ns_dict)

        final_states_dict      = self._stack_state_dicts(states_dict_list)
        final_next_states_dict = self._stack_state_dicts(next_states_dict_list)

        # Convert to torch
        rewards_t = torch.from_numpy(reward_np).float().contiguous().unsqueeze(-1).to(self.device)
        dones_t   = torch.from_numpy(done_np).float().contiguous().unsqueeze(-1).to(self.device)
        trunc_t   = torch.from_numpy(trunc_np).float().contiguous().unsqueeze(-1).to(self.device)
        acts_t    = torch.from_numpy(actions_np).float().contiguous().to(self.device)  # => (B, Env, ActionSize)

        # If multi-agent => reshape => (B, Env, CurrAgents, perAgentDim)
        if CurrAgents > 1:
            # Typically you'd do something like:
            #   perAgentDim = ActionSize // CurrAgents
            # but if you have multi-discrete, it might be more complicated.
            perAgentDim = ActionSize // CurrAgents
            acts_t = acts_t.view(B, NumEnvironments, CurrAgents, perAgentDim)

        win32event.ReleaseMutex(self.update_mutex)
        win32event.SetEvent(self.update_received_event)

        return (final_states_dict, final_next_states_dict,
                acts_t, rewards_t, dones_t, trunc_t)

    def cleanup(self):
        """
        Clean up shared memories, events, and mutexes.
        """
        self.action_shared_memory.close()
        self.states_shared_memory.close()
        self.update_shared_memory.close()

        for handle in [
            self.action_ready_event, self.action_received_event,
            self.update_ready_event, self.update_received_event,
            self.actions_mutex, self.states_mutex, self.update_mutex
        ]:
            win32api.CloseHandle(handle)

    # -----------------------------------------------
    #   INTERNAL HELPERS
    # -----------------------------------------------
    def _split_central_agent(self,
                             single_batch_np: np.ndarray,
                             NumEnvironments: int,
                             CurrentAgents: int,
                             SingleEnvStateSize: int
                             ) -> Dict[str, torch.Tensor]:
        """
        single_batch_np => shape (NumEnv, SingleEnvStateSize)
        We'll parse out 'central' portion and 'agent' portion, building a dictionary:
           { "central":(NumEnv, cSize), "agent":(NumEnv, CurrAgents,aSize) }
        """
        states_dict: Dict[str, torch.Tensor] = {}
        idx_start = 0

        if self.hasCentral:
            c_size = self.centralObsSize
            central_data = single_batch_np[:, idx_start: idx_start + c_size]
            idx_start += c_size
            c_t = torch.from_numpy(central_data.copy()).float().contiguous().to(self.device)
            states_dict["central"] = c_t

        if self.hasAgent:
            a_size_total = self.agentObsSize * CurrentAgents
            agent_data = single_batch_np[:, idx_start : idx_start + a_size_total]
            idx_start += a_size_total

            # => shape => (NumEnv, CurrAgents, agentObsSize)
            agent_data = np.reshape(agent_data, (NumEnvironments, CurrentAgents, self.agentObsSize)).copy()
            a_t = torch.from_numpy(agent_data).float().contiguous().to(self.device)
            states_dict["agent"] = a_t

        return states_dict

    def _stack_state_dicts(self, dict_list: list) -> Dict[str, torch.Tensor]:
        """
        Given a list of dicts (each with 'central'/'agent' keys), 
        each value => shape (NumEnv, X...) => we stack along dim=0 => (B,NumEnv,X...)
        """
        if not dict_list:
            return {}

        keys = dict_list[0].keys()
        out_dict = {}

        for k in keys:
            # each => shape (NumEnv,X...)
            arrs = [d[k] for d in dict_list]
            # stack => shape (B,NumEnv, X...)
            out_dict[k] = torch.stack(arrs, dim=0).contiguous()

        return out_dict

    def write_experiences_to_file(self,
        states_dict: Dict[str, torch.Tensor],
        next_states_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncs: torch.Tensor,
        file_path: str
    ):
        """
        Dump transitions in the same format as the C++ side:
         [ state (flattened), nextState (flattened), action (flattened), reward, trunc, done ]
        We flatten 'central' then 'agent' in that order, if present,
        then do the same for next_states, then actions, reward/trunc/done.
        We'll produce one line for each (B x NumEnv).
        """
        with open(file_path, 'w') as f:
            B = rewards.shape[0]     # batch
            NE= rewards.shape[1]     # numEnv

            # If multi-agent => (B, NE, Agents, actDim), else (B, NE, actionDim)
            if actions.ndim == 4:
                _, _, A, D = actions.shape
                actionSize = A*D
            else:
                actionSize = actions.shape[-1] if actions.ndim == 3 else 0

            for e in range(NE):
                for b in range(B):
                    line_elems = []

                    # Flatten state => central then agent
                    if "central" in states_dict:
                        c_slice = states_dict["central"][b,e,:]
                        line_elems.extend(str(v) for v in c_slice.cpu().numpy().tolist())

                    if "agent" in states_dict:
                        a_slice = states_dict["agent"][b,e,:,:]  # (CurrAgents, aObsSize)
                        line_elems.extend(str(v) for v in a_slice.flatten().cpu().numpy().tolist())

                    # next state => same approach
                    if "central" in next_states_dict:
                        cns_slice = next_states_dict["central"][b,e,:]
                        line_elems.extend(str(v) for v in cns_slice.cpu().numpy().tolist())

                    if "agent" in next_states_dict:
                        ans_slice = next_states_dict["agent"][b,e,:,:]
                        line_elems.extend(str(v) for v in ans_slice.flatten().cpu().numpy().tolist())

                    # actions => shape => (B,NE, [Agents,] actionDim)
                    # Flatten last two dims if multi-agent
                    if actions.ndim == 4:  # (B,NE,A,actDim)
                        act_slice = actions[b,e,:,:]  # => shape (A,actDim)
                    else:  # (B,NE,actDim)
                        act_slice = actions[b,e,:]    # => shape (actDim)
                    line_elems.extend(str(v) for v in act_slice.flatten().cpu().numpy().tolist())

                    # reward, trunc, done
                    line_elems.append(str(rewards[b,e].item()))
                    line_elems.append(str(truncs[b,e].item()))
                    line_elems.append(str(dones[b,e].item()))

                    line = ",".join(line_elems)
                    f.write(line + "\n")

        print(f"** Wrote experiences to {file_path}")
