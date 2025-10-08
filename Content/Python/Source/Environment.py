# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
import win32event
import win32api
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import shared_memory
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional 
import torch

MUTEX_ALL_ACCESS = 0x1F0001
SLEEP_INTERVAL = 1.0  # seconds

class EventType(Enum):
    UPDATE = 0
    GET_ACTIONS = 1

class EnvCommunicationInterface:
    def __init__(self) -> None:
        pass

    def get_states(self) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:  # Return Dict[str, Any]
        pass

    def send_actions(self, actions: torch.Tensor) -> None:
        pass

    def get_experiences(self) -> Tuple[
        Dict[str, Any], Dict[str, Any], # Return Dict[str, Any]
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        pass

    def wait_for_event(self) -> EventType:
        pass

    def cleanup(self) -> None:
        pass


class SharedMemoryInterface(EnvCommunicationInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device: torch.device = torch.device(
            "mps" if (hasattr(torch, 'has_mps') and torch.has_mps) else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        shape_cfg = config.get("environment", {}).get("shape", {})
        state_cfg = shape_cfg.get("state", {})
        
        self.central_state_components_defs: List[Dict[str, Any]] = []
        self.parsed_central_obs_size = 0
        self.hasAgent = ("agent" in state_cfg)
        self.agentObsSize = 0

        if "central" in state_cfg:
            central_defs_raw = state_cfg["central"]
            if isinstance(central_defs_raw, list): 
                for comp_def in central_defs_raw:
                    if comp_def.get("enabled", False):
                        name = comp_def.get("name")
                        comp_type = comp_def.get("type")
                        shape_info = comp_def.get("shape", {}) 
                        if not name or not comp_type:
                            print(f"Warning: Central state component definition missing name or type: {comp_def}")
                            continue
                        
                        comp_size = 0
                        comp_specific_shape = {} 
                        if comp_type == "matrix2d":
                            h = shape_info.get("h", 0)
                            w = shape_info.get("w", 0)
                            c = shape_info.get("c", 1) # Assume c=1 if not specified, C++ StateManager produces HxW flat
                            comp_size = h * w * c 
                            comp_specific_shape = {"h": h, "w": w, "c": c}
                        elif comp_type == "vector":
                            comp_size = shape_info.get("size", 0)
                            comp_specific_shape = {"size": comp_size}
                        elif comp_type == "sequence": 
                            max_len = shape_info.get("max_length", 0)
                            feat_dim = shape_info.get("feature_dim", 0)
                            comp_size = max_len * feat_dim
                            comp_specific_shape = {"max_length": max_len, "feature_dim": feat_dim}
                        
                        if comp_size > 0:
                            self.central_state_components_defs.append({
                                "name": name,
                                "type": comp_type,
                                "shape_dict": comp_specific_shape, 
                                "size": comp_size 
                            })
                            self.parsed_central_obs_size += comp_size
                        else:
                            print(f"Warning: Central component '{name}' enabled but has size 0 or invalid shape.")
            elif isinstance(central_defs_raw, dict) and "obs_size" in central_defs_raw: 
                print("Warning: Central state is defined with old 'obs_size' format. Consider updating to new array format.")
                self.parsed_central_obs_size = central_defs_raw["obs_size"]
                if self.parsed_central_obs_size > 0:
                    self.central_state_components_defs.append({
                        "name": "legacy_central",
                        "type": "vector_legacy", 
                        "shape_dict": {"size": self.parsed_central_obs_size},
                        "size": self.parsed_central_obs_size
                    })
            else:
                print("Warning: 'central' state definition is not a list or a valid old format object. Central state will be empty.")
        
        if self.hasAgent:
            self.agentObsSize = state_cfg.get("agent",{}).get("obs_size", 0)

        self.action_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="ActionsSharedMemory")
        self.states_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="StatesSharedMemory")
        self.update_shared_memory = self._wait_for_resource(shared_memory.SharedMemory, name="UpdateSharedMemory")
        self.action_ready_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReadyEvent")
        self.action_received_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReceivedEvent")
        self.update_ready_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReadyEvent")
        self.update_received_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReceivedEvent")
        self.begin_test_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "BeginTestEvent")
        self.end_test_event = self._wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "EndTestEvent")
        self.actions_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "ActionsDataMutex")
        self.states_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "StatesDataMutex")
        self.update_mutex = self._wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "UpdateDataMutex")
        self.info = None

    def _wait_for_resource(self, func, *args, **kwargs):
        dot_count = 1
        pbar = tqdm(total=4, desc="Waiting for Unreal Engine", position=0, leave=True, bar_format='{desc}')
        while True:
            try:
                resource = func(*args, **kwargs)
                if resource:
                    pbar.close(); return resource
            except Exception:
                pbar.set_description(f"Waiting for Unreal Engine {'.' * dot_count}")
                dot_count = (dot_count % 3) + 1; pbar.update(1)
                time.sleep(SLEEP_INTERVAL)

    def wait_for_event(self) -> EventType:
        result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)
        return EventType.GET_ACTIONS if result == win32event.WAIT_OBJECT_0 else EventType.UPDATE

    def _parse_flat_central_state(self, flat_central_data_np: np.ndarray, num_environments: int) -> Dict[str, torch.Tensor]:
        central_components_dict: Dict[str, torch.Tensor] = {}
        current_offset = 0
        for comp_def in self.central_state_components_defs:
            name = comp_def["name"]
            comp_type = comp_def["type"]
            comp_size = comp_def["size"] 
            shape_dict = comp_def["shape_dict"]

            if comp_size == 0: continue 

            component_data_flat_all_envs = flat_central_data_np[:, current_offset : current_offset + comp_size]
            
            reshaped_component_data = None
            if comp_type == "matrix2d":
                h = shape_dict.get("h", 0)
                w = shape_dict.get("w", 0)
                c = shape_dict.get("c", 1) # Get channel, default to 1
                expected_comp_elements = h * w * c
                if expected_comp_elements == comp_size and num_environments > 0 and h > 0 and w > 0:
                    if c > 1: # Multi-channel, store as (NumEnv, C, H, W) or (NumEnv, H, W, C)
                        # Assuming Caffe/PyTorch format (NumEnv, C, H, W) for CNNs
                        reshaped_component_data = component_data_flat_all_envs.reshape(num_environments, c, h, w)
                    else: # Single-channel
                        reshaped_component_data = component_data_flat_all_envs.reshape(num_environments, h, w)
                else:
                    print(f"Warning: Shape mismatch for 'matrix2d' component '{name}'. Expected {c}x{h}x{w}={expected_comp_elements}, got flat size per env {comp_size}. Skipping reshape.")
            elif comp_type == "vector":
                if component_data_flat_all_envs.shape == (num_environments, comp_size):
                    reshaped_component_data = component_data_flat_all_envs
                else:
                    print(f"Warning: Shape mismatch for 'vector' component '{name}'. Expected ({num_environments}, {comp_size}), got {component_data_flat_all_envs.shape}. Skipping.")
            elif comp_type == "sequence": 
                max_len = shape_dict.get("max_length", 0)
                feat_dim = shape_dict.get("feature_dim", 0)
                expected_comp_elements = max_len * feat_dim
                if expected_comp_elements == comp_size and num_environments > 0 and max_len > 0 and feat_dim > 0:
                    reshaped_component_data = component_data_flat_all_envs.reshape(num_environments, max_len, feat_dim)
                    
                    # Generate padding mask: True where all features in the sequence are 0.
                    padding_mask = torch.all(torch.from_numpy(reshaped_component_data) == 0, dim=-1)
                    # Backward-compat key and standardized key used by networks
                    padding_mask_t = padding_mask.contiguous().to(self.device)
                    central_components_dict[f"{name}_padding_mask"] = padding_mask_t
                    central_components_dict[f"{name}_mask"] = padding_mask_t
                else:
                     print(f"Warning: Shape mismatch for 'sequence' component '{name}'. Expected {max_len}x{feat_dim}={expected_comp_elements}, got size {comp_size}. Skipping reshape.")
            elif comp_type == "vector_legacy": 
                 reshaped_component_data = component_data_flat_all_envs 
            
            if reshaped_component_data is not None:
                central_components_dict[name] = torch.from_numpy(reshaped_component_data.copy()).float().contiguous().to(self.device)
            
            current_offset += comp_size
        
        return central_components_dict

    def get_states(self) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        win32event.WaitForSingleObject(self.states_mutex, win32event.INFINITE)
        try:
            byte_array = np.frombuffer(self.states_shared_memory.buf, dtype=np.float32)

            info_offset = 6
            if len(byte_array) < info_offset:
                print("Error: Shared memory for states is smaller than info block size.")
                # Return empty/default values to avoid crashing, signal UE might be stuck
                empty_states_dict: Dict[str, Any] = {"central": {}, "agent": torch.empty(0, device=self.device)}
                d_empty = torch.empty((1,0,1), device=self.device)  # (S,E,1) with E=0
                return (empty_states_dict, d_empty, d_empty, d_empty)

            self.info = byte_array[:info_offset].astype(int)
            _, _, NumEnvironments, CurrentAgents, SingleEnvStateSize_from_ue, _ = self.info
            
            if NumEnvironments == 0: # No environments, so no state data to read beyond info
                win32event.ReleaseMutex(self.states_mutex) # Release mutex before returning
                empty_states_dict_ne0: Dict[str, Any] = {"central": {}, "agent": torch.empty(0, device=self.device)}
                d_empty_ne0 = torch.empty((1,0,1), device=self.device)
                return (empty_states_dict_ne0, d_empty_ne0, d_empty_ne0, d_empty_ne0)

            expected_agent_size_this_call = self.agentObsSize * CurrentAgents if self.hasAgent else 0
            calculated_total_state_size_this_call = self.parsed_central_obs_size + expected_agent_size_this_call

            if SingleEnvStateSize_from_ue != calculated_total_state_size_this_call:
                raise ValueError(
                    "Shared memory state size mismatch: UE reports {} but Python expects {} (central {}, agent {}).".format(
                        SingleEnvStateSize_from_ue,
                        calculated_total_state_size_this_call,
                        self.parsed_central_obs_size,
                        expected_agent_size_this_call,
                    )
                )
            
            actual_total_state_data_size = NumEnvironments * SingleEnvStateSize_from_ue
            
            states_offset = info_offset
            states_end    = states_offset + actual_total_state_data_size
            dones_offset  = states_end
            dones_end     = dones_offset + NumEnvironments
            truncs_offset = dones_end
            truncs_end    = truncs_offset + NumEnvironments
            needs_offset  = truncs_end
            needs_end     = needs_offset + NumEnvironments

            expected_min_len_for_buf = needs_end
            if len(byte_array) < expected_min_len_for_buf:
                 print(f"Error: Shared memory for states is too small. Expected at least {expected_min_len_for_buf} floats, got {len(byte_array)}.")
                 empty_states_dict_len: Dict[str, Any] = {"central": {}, "agent": torch.empty(0, device=self.device)}
                 d_empty_len = torch.empty((1,0,1), device=self.device)
                 return (empty_states_dict_len, d_empty_len, d_empty_len, d_empty_len)


            state_data_flat_all_envs = byte_array[states_offset:states_end] if actual_total_state_data_size > 0 else np.array([], dtype=np.float32)
            dones_data = byte_array[dones_offset:dones_end]
            truncs_data = byte_array[truncs_offset:truncs_end]
            needs_data  = byte_array[needs_offset:needs_end]

            full_state_np_all_envs = state_data_flat_all_envs.reshape(NumEnvironments, SingleEnvStateSize_from_ue) if actual_total_state_data_size > 0 else np.empty((NumEnvironments, 0), dtype=np.float32)

            states_dict: Dict[str, Any] = {} 
            current_slice_offset_in_state = 0

            if self.parsed_central_obs_size > 0 and full_state_np_all_envs.shape[1] >= self.parsed_central_obs_size :
                central_data_slice_np = full_state_np_all_envs[:, current_slice_offset_in_state : current_slice_offset_in_state + self.parsed_central_obs_size]
                states_dict["central"] = self._parse_flat_central_state(central_data_slice_np, NumEnvironments)
                current_slice_offset_in_state += self.parsed_central_obs_size

            if self.hasAgent and self.agentObsSize > 0 and CurrentAgents > 0:
                agent_data_size_total_for_all_agents = self.agentObsSize * CurrentAgents
                if full_state_np_all_envs.shape[1] >= current_slice_offset_in_state + agent_data_size_total_for_all_agents:
                    agent_data_slice_np = full_state_np_all_envs[:, current_slice_offset_in_state : current_slice_offset_in_state + agent_data_size_total_for_all_agents]
                    agent_data_reshaped_np = agent_data_slice_np.reshape(NumEnvironments, CurrentAgents, self.agentObsSize)
                    states_dict["agent"] = torch.from_numpy(agent_data_reshaped_np.copy()).float().contiguous().to(self.device)
            
            if "central" in states_dict: # Add S=1 batch dim
                for k_central in states_dict["central"]:
                    if states_dict["central"][k_central].numel() > 0 : # Check if tensor is not empty
                        states_dict["central"][k_central] = states_dict["central"][k_central].unsqueeze(0)
                    else: # If a component ended up empty (e.g. due to parsing warning), ensure it's an empty tensor with a batch dim
                        states_dict["central"][k_central] = torch.empty((1, *states_dict["central"][k_central].shape), device=self.device)
            if "agent" in states_dict: # Add S=1 batch dim
                if states_dict["agent"].numel() > 0:
                    states_dict["agent"] = states_dict["agent"].unsqueeze(0)
                else: # Ensure it's an empty tensor with a batch dim
                    states_dict["agent"] = torch.empty((1, *states_dict["agent"].shape), device=self.device)


            dones_t  = torch.from_numpy(dones_data.copy().reshape(1, NumEnvironments, 1)).float().contiguous().to(self.device)
            truncs_t = torch.from_numpy(truncs_data.copy().reshape(1, NumEnvironments, 1)).float().contiguous().to(self.device)
            needs_t  = torch.from_numpy(needs_data.copy().reshape(1, NumEnvironments, 1)).float().contiguous().to(self.device)

            return (states_dict, dones_t, truncs_t, needs_t)
        finally:
            win32event.ReleaseMutex(self.states_mutex)


    def send_actions(self, actions: torch.Tensor) -> None: 
        win32event.WaitForSingleObject(self.actions_mutex, win32event.INFINITE)
        try:
            if self.info is None: return

            _, _, NumEnvironments, _, _, ActionSize_from_ue = self.info
            if ActionSize_from_ue == 0: 
                win32event.SetEvent(self.action_received_event)
                return

            action_array = np.ndarray(shape=(NumEnvironments, ActionSize_from_ue), dtype=np.float32, buffer=self.action_shared_memory.buf)
            actions_cpu = actions.detach().cpu() 

            expected_shape = (NumEnvironments, ActionSize_from_ue)
            if actions_cpu.shape != expected_shape:
                print(f"CRITICAL WARNING in send_actions: Action tensor shape mismatch. Expected {expected_shape}, got {actions_cpu.shape}. Check agent's output action dimension.")
                try:
                    flat_actions = actions_cpu.reshape(expected_shape).numpy() # Attempt reshape
                except RuntimeError as e:
                    print(f"CRITICAL ERROR: Cannot reshape actions for shared memory. {e}")
                    win32event.SetEvent(self.action_received_event) 
                    return 
            else:
                flat_actions = actions_cpu.numpy()
                
            np.copyto(action_array, flat_actions)
        finally:
            win32event.ReleaseMutex(self.actions_mutex)
            win32event.SetEvent(self.action_received_event)


    def get_experiences(self) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)
        try:
            raw_data = np.frombuffer(self.update_shared_memory.buf, dtype=np.float32)

            if self.info is None:
                return ({}, {}, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

            _, BatchSize, NumEnvironments, CurrentAgents, SingleEnvStateSize_from_ue, ActionSize_from_ue = self.info
            
            if NumEnvironments == 0 or BatchSize == 0 : # No data to read
                return ({}, {}, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

            expected_agent_size = self.agentObsSize * CurrentAgents if self.hasAgent else 0
            calculated_total_state_size = self.parsed_central_obs_size + expected_agent_size
            if SingleEnvStateSize_from_ue != calculated_total_state_size:
                print(f"CRITICAL WARNING in get_experiences: SingleEnvStateSize_from_ue ({SingleEnvStateSize_from_ue}) "
                    f"does not match Python's calculated size ({calculated_total_state_size}). Check JSON.")
            
            transition_length = (2 * SingleEnvStateSize_from_ue) + ActionSize_from_ue + 3
            needed_count = transition_length * NumEnvironments * BatchSize
            
            if needed_count == 0 or len(raw_data) < needed_count: 
                print(f"Warning in get_experiences: Not enough data in buffer. Expected {needed_count} floats, got {len(raw_data)}. Or needed_count is 0.")
                return ({}, {}, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))


            data_subset = raw_data[:needed_count]
            transitions_np = data_subset.reshape(NumEnvironments, BatchSize, transition_length)

            states_np_env_batch_feat = transitions_np[:, :, :SingleEnvStateSize_from_ue]
            next_states_np_env_batch_feat = transitions_np[:, :, SingleEnvStateSize_from_ue : 2*SingleEnvStateSize_from_ue]
            actions_np_env_batch_feat = transitions_np[:, :, 2*SingleEnvStateSize_from_ue : 2*SingleEnvStateSize_from_ue + ActionSize_from_ue]
            reward_np_env_batch = transitions_np[:, :, -3]
            trunc_np_env_batch = transitions_np[:, :, -2]
            done_np_env_batch = transitions_np[:, :, -1]

            states_np = np.transpose(states_np_env_batch_feat, (1, 0, 2)).copy()
            next_states_np = np.transpose(next_states_np_env_batch_feat, (1, 0, 2)).copy()
            actions_np = np.transpose(actions_np_env_batch_feat, (1, 0, 2)).copy()
            reward_np = np.transpose(reward_np_env_batch, (1, 0)).copy()
            trunc_np = np.transpose(trunc_np_env_batch, (1, 0)).copy()
            done_np = np.transpose(done_np_env_batch, (1, 0)).copy()
            
            final_states_dict: Dict[str, Any] = {}
            final_next_states_dict: Dict[str, Any] = {}
            
            current_slice_offset_in_state = 0
            if self.parsed_central_obs_size > 0 and states_np.shape[2] >= self.parsed_central_obs_size:
                s_central_flat_all = states_np[:, :, current_slice_offset_in_state : current_slice_offset_in_state + self.parsed_central_obs_size]
                ns_central_flat_all = next_states_np[:, :, current_slice_offset_in_state : current_slice_offset_in_state + self.parsed_central_obs_size]
                
                s_central_parsed_dict = self._parse_flat_central_state(s_central_flat_all.reshape(-1, self.parsed_central_obs_size), BatchSize * NumEnvironments)
                ns_central_parsed_dict = self._parse_flat_central_state(ns_central_flat_all.reshape(-1, self.parsed_central_obs_size), BatchSize * NumEnvironments)
                
                final_states_dict["central"] = { k: v.view(BatchSize, NumEnvironments, *v.shape[1:]) for k, v in s_central_parsed_dict.items() if v.numel() > 0}
                final_next_states_dict["central"] = { k: v.view(BatchSize, NumEnvironments, *v.shape[1:]) for k, v in ns_central_parsed_dict.items() if v.numel() > 0}
                current_slice_offset_in_state += self.parsed_central_obs_size

            if self.hasAgent and self.agentObsSize > 0 and CurrentAgents > 0:
                agent_data_size_total_for_all_agents = self.agentObsSize * CurrentAgents
                if states_np.shape[2] >= current_slice_offset_in_state + agent_data_size_total_for_all_agents:
                    s_agent_flat_all = states_np[:, :, current_slice_offset_in_state : current_slice_offset_in_state + agent_data_size_total_for_all_agents]
                    ns_agent_flat_all = next_states_np[:, :, current_slice_offset_in_state : current_slice_offset_in_state + agent_data_size_total_for_all_agents]
                    s_agent_reshaped = s_agent_flat_all.reshape(BatchSize, NumEnvironments, CurrentAgents, self.agentObsSize)
                    ns_agent_reshaped = ns_agent_flat_all.reshape(BatchSize, NumEnvironments, CurrentAgents, self.agentObsSize)
                    final_states_dict["agent"] = torch.from_numpy(s_agent_reshaped.copy()).float().contiguous().to(self.device)
                    final_next_states_dict["agent"] = torch.from_numpy(ns_agent_reshaped.copy()).float().contiguous().to(self.device)

            rewards_t = torch.from_numpy(reward_np).float().contiguous().unsqueeze(-1).to(self.device) 
            dones_t = torch.from_numpy(done_np).float().contiguous().unsqueeze(-1).to(self.device)     
            truncs_t = torch.from_numpy(trunc_np).float().contiguous().unsqueeze(-1).to(self.device)   
            acts_t = torch.from_numpy(actions_np).float().contiguous().to(self.device) 
            
            if self.hasAgent and CurrentAgents > 0 and ActionSize_from_ue > 0: 
                per_agent_action_dim = ActionSize_from_ue // CurrentAgents
                if ActionSize_from_ue % CurrentAgents == 0 and per_agent_action_dim > 0:
                    acts_t = acts_t.view(BatchSize, NumEnvironments, CurrentAgents, per_agent_action_dim)
                else:
                    print(f"Warning: Cannot evenly divide ActionSize_from_ue ({ActionSize_from_ue}) by CurrentAgents ({CurrentAgents}). Actions tensor shape: {acts_t.shape}")
            
            return final_states_dict, final_next_states_dict, acts_t, rewards_t, dones_t, truncs_t
        finally:
            win32event.ReleaseMutex(self.update_mutex)
            win32event.SetEvent(self.update_received_event)

    def cleanup(self): 
        if hasattr(self, 'action_shared_memory') and self.action_shared_memory: self.action_shared_memory.close()
        if hasattr(self, 'states_shared_memory') and self.states_shared_memory: self.states_shared_memory.close()
        if hasattr(self, 'update_shared_memory') and self.update_shared_memory: self.update_shared_memory.close()
        
        handles_names = [
            "action_ready_event", "action_received_event", "update_ready_event",
            "update_received_event", "begin_test_event", "end_test_event",
            "actions_mutex", "states_mutex", "update_mutex"
        ]
        for name in handles_names:
            handle = getattr(self, name, None)
            if handle: 
                try:
                    win32api.CloseHandle(handle)
                except Exception as e:
                    print(f"Error closing handle {name}: {e}")
            setattr(self, name, None) # Set to None after closing or if it was already None
