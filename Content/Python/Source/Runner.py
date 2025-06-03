# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import win32event

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer 
from Source.Environment import EnvCommunicationInterface, EventType
# Assuming StateRecorder is in the same directory or accessible via Source.
from Source.StateRecorder import StateRecorder 

# -----------------------------------------------------------------------------
class TrajectorySegment:
    """Accumulates one variable‑length segment until cut or padded."""

    def __init__(self, num_agents_cfg: int, device: torch.device, pad: bool, max_len: int):
        self.device = device
        self.num_agents_cfg = num_agents_cfg
        self.enable_padding = pad
        self.max_segment_length = max_len
        self.true_sequence_length = 0
        # storages
        self.observations: List[Dict[str, Any]] = [] 
        self.next_observations: List[Dict[str, Any]] = [] 
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.truncs: List[torch.Tensor] = []
        self.initial_hidden_state: Optional[torch.Tensor] = None

    def is_full(self) -> bool:
        return self.enable_padding and self.true_sequence_length >= self.max_segment_length

    def add_step(self, obs: Dict[str, Any], act: torch.Tensor, rew: torch.Tensor, 
                 next_obs: Dict[str, Any], done: torch.Tensor, trunc: torch.Tensor):
        if self.is_full():
            print("RLRunner warn: add_step called on full segment")
            return
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.dones.append(done)
        self.truncs.append(trunc)
        self.true_sequence_length += 1

    def tensors_for_collation(self) -> Tuple[
        List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor],
        List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor],
        Optional[torch.Tensor], int
    ]:
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.dones,
            self.truncs,
            self.initial_hidden_state,
            self.true_sequence_length,
        )

# -----------------------------------------------------------------------------
class RLRunner:
    """Handles UE→Python experience pipeline and agent updates."""

    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface, cfg: Dict):
        self.agent, self.agentComm, self.config = agent, agentComm, cfg
        self.device = agentComm.device 

        trn_cfg = cfg["train"]
        ag_cfg = cfg["agent"]["params"]
        env_shape = cfg["environment"]["shape"]
        env_params = cfg["environment"]["params"]
        
        # State Recorder Initialization
        recorder_config = cfg.get("StateRecorder", None) # Get StateRecorder config block
        if cfg.get("StateRecorder_disabled", None) is not None : # Check if "StateRecorder_disabled" key exists
             print("StateRecorder is explicitly disabled via 'StateRecorder_disabled' key in config.")
             self.state_recorder = None
        elif recorder_config:
            print("Initializing StateRecorder...")
            self.state_recorder = StateRecorder(recorder_config)
        else:
            print("StateRecorder config not found or disabled.")
            self.state_recorder = None


        self.sequence_length = trn_cfg.get("sequence_length", 64)
        self.pad_trajectories = trn_cfg.get("pad_trajectories", True)
        self.timesteps_from_ue_batch = trn_cfg.get("batch_size", 256) 
        self.num_envs = trn_cfg.get("num_environments", 1)
        self.save_freq = trn_cfg.get("saveFrequency", 50)

        self.enable_memory = ag_cfg.get("enable_memory", False)
        mem_cfg = ag_cfg.get(ag_cfg.get("memory_type", "gru"), {})
        self.memory_hidden_size = mem_cfg.get("hidden_size", 128)
        self.memory_layers = mem_cfg.get("num_layers", 1)

        self.num_agents_cfg = env_params.get("MaxAgents", 1)
        if "agent" in env_shape.get("state", {}): 
            self.num_agents_cfg = env_shape["state"]["agent"].get("max", self.num_agents_cfg)

        self.current_memory_hidden_states = (
            torch.zeros(
                self.num_envs,
                self.num_agents_cfg,
                self.memory_layers,
                self.memory_hidden_size,
                device=self.device,
            )
            if self.enable_memory
            else None
        )

        self.current_segments = [
            TrajectorySegment(self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length)
            for _ in range(self.num_envs)
        ]
        if self.enable_memory and self.current_memory_hidden_states is not None:
            for e in range(self.num_envs):
                self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()
        
        self.completed_segments: List[List[TrajectorySegment]] = [[] for _ in range(self.num_envs)]

        norm_cfg = trn_cfg.get("states_normalizer", None)
        self.state_normalizer = RunningMeanStdNormalizer(**norm_cfg, device=self.device) if norm_cfg else None

        self.writer = SummaryWriter()
        self.update_idx = 0
        print(f"RLRunner initialised (envs: {self.num_envs}, device: {self.device})")


    def start(self):
        while True:
            evt = self.agentComm.wait_for_event()
            if evt == EventType.GET_ACTIONS:
                self._handle_get_actions()
            elif evt == EventType.UPDATE:
                self._handle_update()
            else:
                print(f"RLRunner: Received unknown event type {evt}. Assuming shutdown or error.")
                break # Break the loop on unknown event
        self.end()


    def _handle_get_actions(self):
        states_from_comm, dones_from_comm, truncs_from_comm = self.agentComm.get_states()
        
        if not states_from_comm and not ("central" in states_from_comm and states_from_comm["central"]) and \
           not ("agent" in states_from_comm and states_from_comm["agent"]):
            # This can happen if initial shared memory read is empty or parsing fails in agentComm
            print("RLRunner: Received empty or invalid states_from_comm in _handle_get_actions. Skipping.")
            # Still need to signal UE to avoid deadlock if it's waiting for actions
            # Assuming a dummy action send might be needed or a more robust error signal to UE
            # For now, let's try to send zero actions if action_dim is known from Agent
            if hasattr(self.agent, 'total_action_dim_per_agent') and self.agent.total_action_dim_per_agent > 0:
                num_total_actions_per_env = self.num_agents_cfg * self.agent.total_action_dim_per_agent
                dummy_actions = torch.zeros((self.num_envs, num_total_actions_per_env), device=self.device)
                self.agentComm.send_actions(dummy_actions)
            else: # Fallback if action dim isn't readily available
                self.agentComm.send_actions(torch.empty(0, device=self.device)) # Or handle error more gracefully
            return

        current_states_dict: Dict[str, Any] = {}
        has_central_state = "central" in states_from_comm and states_from_comm["central"]
        has_agent_state = "agent" in states_from_comm and states_from_comm["agent"] is not None

        if not has_central_state and not has_agent_state:
            raise ValueError("RLRunner: states_from_comm must contain at least 'central' or 'agent' key with data.")

        if has_central_state:
            current_states_dict["central"] = {
                k: v.squeeze(0) for k, v in states_from_comm["central"].items()
            }
        if has_agent_state:
            current_states_dict["agent"] = states_from_comm["agent"].squeeze(0)

        dones = dones_from_comm.squeeze(0).squeeze(-1) 
        truncs = truncs_from_comm.squeeze(0).squeeze(-1)

        if self.enable_memory and self.current_memory_hidden_states is not None:
            for e in range(self.num_envs):
                if (dones[e] > 0.5) or (truncs[e] > 0.5): 
                    self.current_memory_hidden_states[e].zero_() 
                    if self.current_segments[e].true_sequence_length > 0:
                        self.completed_segments[e].append(self.current_segments[e])
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()
        
        states_for_agent_action = current_states_dict 
        if self.state_normalizer:
            normalized_states_for_agent_action: Dict[str, Any] = {}
            if has_central_state:
                normalized_states_for_agent_action["central"] = {}
                for comp_name, comp_tensor in current_states_dict["central"].items():
                    self.state_normalizer.update(comp_tensor, key=f"central_{comp_name}")
                    normalized_states_for_agent_action["central"][comp_name] = \
                        self.state_normalizer.normalize(comp_tensor, key=f"central_{comp_name}")
            
            if has_agent_state:
                self.state_normalizer.update(current_states_dict["agent"], key="agent")
                normalized_states_for_agent_action["agent"] = \
                    self.state_normalizer.normalize(current_states_dict["agent"], key="agent")
            
            states_for_agent_action = normalized_states_for_agent_action

        states_for_agent_action_batched: Dict[str, Any] = {}
        if has_central_state: # Check if 'central' key exists after potential normalization
            if "central" in states_for_agent_action and states_for_agent_action["central"]:
                 states_for_agent_action_batched["central"] = {
                    k: v.unsqueeze(0) for k, v in states_for_agent_action["central"].items()
                }
        if has_agent_state: # Check if 'agent' key exists after potential normalization
            if "agent" in states_for_agent_action and states_for_agent_action["agent"] is not None:
                 states_for_agent_action_batched["agent"] = states_for_agent_action["agent"].unsqueeze(0)

        # Ensure at least one state component is present for the agent
        if not states_for_agent_action_batched or \
           (not ("central" in states_for_agent_action_batched and states_for_agent_action_batched["central"]) and \
            not ("agent" in states_for_agent_action_batched and states_for_agent_action_batched["agent"] is not None)):
            print("RLRunner Error: No valid state data to pass to agent.get_actions().")
            # Decide how to handle - potentially send dummy actions or raise error
            # For now, similar to empty states_from_comm:
            if hasattr(self.agent, 'total_action_dim_per_agent') and self.agent.total_action_dim_per_agent > 0:
                num_total_actions_per_env = self.num_agents_cfg * self.agent.total_action_dim_per_agent
                dummy_actions = torch.zeros((self.num_envs, num_total_actions_per_env), device=self.device)
                self.agentComm.send_actions(dummy_actions)
            else:
                self.agentComm.send_actions(torch.empty(0, device=self.device))
            return


        actions_ue_flat, (log_probs, entropies), next_h = self.agent.get_actions(
            states_for_agent_action_batched, 
            dones=dones_from_comm, 
            truncs=truncs_from_comm,
            h_prev_batch=self.current_memory_hidden_states if self.enable_memory else None,
        ) 

        if self.enable_memory and next_h is not None:
            self.current_memory_hidden_states = next_h 

        self.agentComm.send_actions(actions_ue_flat)
        
        # Pass the unbatched, potentially normalized state dict to StateRecorder
        if self.state_recorder:
            # Assuming StateRecorder will be updated to handle this dictionary:
            # current_states_dict is {"central": {"comp": tensor_EHMW}, "agent": tensor_E_NA_Obs}
            # StateRecorder might only care about specific components or the first env.
            # For now, passing the whole dict for the first environment for simplicity.
            # The StateRecorder needs adaptation to process this.
            first_env_states_for_recorder: Dict[str, Any] = {}
            if "central" in current_states_dict and current_states_dict["central"]:
                first_env_states_for_recorder["central"] = {
                    k: v[0].cpu().numpy() for k, v in current_states_dict["central"].items() # Data for env 0
                }
            if "agent" in current_states_dict and current_states_dict["agent"] is not None:
                first_env_states_for_recorder["agent"] = current_states_dict["agent"][0].cpu().numpy() # Data for env 0
            
            if first_env_states_for_recorder: # Only record if there's something to record
                 self.state_recorder.record_frame(first_env_states_for_recorder)


    def _handle_update(self):
        self.update_idx += 1
        print(f"RLRunner update {self.update_idx}")

        # s_dict/ns_dict from get_experiences:
        # {"central": {"comp": tensor_BNEHW}, "agent": tensor_BNE_NA_Obs}
        # actions_tensor: (B, NE, NA, ActDim) or (B, NE, ActDim)
        # rewards_tensor, dones_tensor, truncs_tensor: (B, NE, 1)
        s_dict, ns_dict, actions_tensor, rewards_tensor, dones_tensor, truncs_tensor = self.agentComm.get_experiences()
        
        if not rewards_tensor.numel(): # No experiences received
            print("RLRunner: no experiences received in update. Skipping.")
            # Still need to signal UE that Python is done with this "empty" update cycle.
            if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
                 win32event.SetEvent(self.agentComm.update_received_event)
            return

        B_ue, NumEnv, _ = rewards_tensor.shape

        # Permute and then process for each environment
        s_dict_permuted: Dict[str, Any] = {}
        has_central_s = "central" in s_dict and s_dict["central"]
        has_agent_s = "agent" in s_dict and s_dict["agent"] is not None

        if has_central_s:
            s_dict_permuted["central"] = {
                k: v.permute(1, 0, *range(2,v.ndim)).contiguous() for k,v in s_dict["central"].items()
            } 
        if has_agent_s:
            s_dict_permuted["agent"] = s_dict["agent"].permute(1, 0, 2, 3).contiguous() 

        ns_dict_permuted: Dict[str, Any] = {}
        has_central_ns = "central" in ns_dict and ns_dict["central"]
        has_agent_ns = "agent" in ns_dict and ns_dict["agent"] is not None

        if has_central_ns:
            ns_dict_permuted["central"] = {
                k: v.permute(1, 0, *range(2,v.ndim)).contiguous() for k,v in ns_dict["central"].items()
            }
        if has_agent_ns:
            ns_dict_permuted["agent"] = ns_dict["agent"].permute(1, 0, 2, 3).contiguous()

        actions_tensor_p = actions_tensor.permute(1, 0, *range(2,actions_tensor.ndim)).contiguous() 
        rewards_tensor_p = rewards_tensor.permute(1, 0, 2).contiguous() 
        dones_tensor_p = dones_tensor.permute(1, 0, 2).contiguous()     
        truncs_tensor_p = truncs_tensor.permute(1, 0, 2).contiguous()  

        for e in range(NumEnv): 
            for t in range(B_ue): 
                obs_step: Dict[str, Any] = {}
                if has_central_s and "central" in s_dict_permuted:
                    obs_step["central"] = {
                        k: v_tensor[e, t] for k,v_tensor in s_dict_permuted["central"].items() # Corrected v to v_tensor
                    }
                if has_agent_s and "agent" in s_dict_permuted:
                    obs_step["agent"] = s_dict_permuted["agent"][e,t]
                
                next_obs_step: Dict[str, Any] = {}
                if has_central_ns and "central" in ns_dict_permuted:
                    next_obs_step["central"] = {
                        k: v_tensor[e, t] for k,v_tensor in ns_dict_permuted["central"].items() # Corrected v to v_tensor
                    }
                if has_agent_ns and "agent" in ns_dict_permuted:
                    next_obs_step["agent"] = ns_dict_permuted["agent"][e,t]

                action_step = actions_tensor_p[e,t] 
                reward_step = rewards_tensor_p[e,t] 
                done_step = dones_tensor_p[e,t]     
                trunc_step = truncs_tensor_p[e,t]  

                self.current_segments[e].add_step(obs_step, action_step, reward_step, next_obs_step, done_step, trunc_step)
                
                term_now = (done_step.item() > 0.5) or (trunc_step.item() > 0.5)
                if self.current_segments[e].is_full() or term_now:
                    self.completed_segments[e].append(self.current_segments[e])
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    if self.enable_memory and self.current_memory_hidden_states is not None:
                        self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        if not self.pad_trajectories:
            for e in range(NumEnv):
                seg = self.current_segments[e]
                if seg.true_sequence_length > 0:
                    self.completed_segments[e].append(seg)
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    if self.enable_memory and self.current_memory_hidden_states is not None:
                         self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()
        
        all_completed_segments = [seg for env_segments in self.completed_segments for seg in env_segments]
        for env_segments in self.completed_segments: 
            env_segments.clear()

        if not all_completed_segments:
            print("RLRunner: no completed segments to update with – skipping agent update.")
            if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
                 win32event.SetEvent(self.agentComm.update_received_event) # Corrected call
            return

        batch_obs_update, batch_act_update, batch_rew_update, batch_nobs_update, \
        batch_done_update, batch_trunc_update, batch_init_h_update, batch_attn_mask_update = \
            self._collate_and_pad_sequences(all_completed_segments)

        if not batch_rew_update.numel(): 
            print("RLRunner: collation resulted in empty batch. Skipping agent update.")
            if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
                 win32event.SetEvent(self.agentComm.update_received_event) # Corrected call
            return
            
        logs = self.agent.update(
            batch_obs_update, batch_act_update, batch_rew_update, batch_nobs_update,
            batch_done_update, batch_trunc_update, batch_init_h_update, batch_attn_mask_update
        )
        self._log_step(logs)

        if self.update_idx % self.save_freq == 0:
            self.agent.save(f"model_update_{self.update_idx}.pth") 
            print("Model checkpoint saved.")
        
        if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
            win32event.SetEvent(self.agentComm.update_received_event)
        else:
            print("RLRunner Warning: agentComm.update_received_event not found or is None in _handle_update.")

    def _collate_and_pad_sequences(self, segments: List[TrajectorySegment]) -> Tuple[
        Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, Any],
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor
    ]:
        if not segments: 
            empty_dict: Dict[str, Any] = {}
            empty_tensor = torch.empty(0, device=self.device)
            return empty_dict, empty_tensor, empty_tensor, empty_dict, empty_tensor, empty_tensor, None, empty_tensor

        target_seq_len = self.sequence_length if self.pad_trajectories else max(s.true_sequence_length for s in segments)
        
        batch_observations: Dict[str, Any] = {}
        batch_next_observations: Dict[str, Any] = {}
        
        # Check if "central" and "agent" keys are expected based on the first segment.
        # This assumes consistent structure across segments.
        first_obs_example = segments[0].observations[0] if segments[0].observations else {}
        expect_central = "central" in first_obs_example and isinstance(first_obs_example["central"], dict)
        expect_agent = "agent" in first_obs_example and first_obs_example["agent"] is not None

        central_component_keys: List[str] = []
        if expect_central:
            batch_observations["central"] = {}
            batch_next_observations["central"] = {}
            central_component_keys = list(first_obs_example["central"].keys())
            for key in central_component_keys:
                batch_observations["central"][key] = []
                batch_next_observations["central"][key] = []
        
        if expect_agent:
            batch_observations["agent"] = []
            batch_next_observations["agent"] = []
        
        A_list, R_list, D_list, TR_list, H_list, lens_list = [], [], [], [], [], []

        def _get_dummy_shape_and_dtype(obs_example_dict: Dict[str, Any], key_path: List[str]) -> Tuple[Tuple, torch.dtype]:
            # Helper to get shape and dtype from a potentially nested example observation
            current_level = obs_example_dict
            for k_part in key_path[:-1]: # Navigate to parent dict
                current_level = current_level.get(k_part, {})
            
            example_tensor = current_level.get(key_path[-1])
            if example_tensor is not None and isinstance(example_tensor, torch.Tensor):
                return example_tensor.shape, example_tensor.dtype
            
            # Fallback if specific component is missing in the first example (should ideally not happen if structure is consistent)
            # This fallback is very basic and might need more sophisticated handling based on expected types/shapes
            if key_path[0] == "agent": return (self.num_agents_cfg, 1), torch.float32 # Guess agent obs dim as 1
            if key_path[0] == "central" and len(key_path) > 1: # e.g. central_some_component
                # Try to find shape from config if possible or use a default
                return (1,1), torch.float32 # Guess H,W as 1,1 for a central component
            return tuple(), torch.float32


        def _pad_sequence_of_tensors(tensor_list: List[torch.Tensor], pad_val: float = 0.0, 
                                     ref_obs_for_shape: Optional[Dict[str, Any]] = None,
                                     ref_key_path: Optional[List[str]] = None) -> torch.Tensor:
            if not tensor_list:
                if target_seq_len > 0 and ref_obs_for_shape and ref_key_path:
                    dummy_shape, dummy_dtype = _get_dummy_shape_and_dtype(ref_obs_for_shape, ref_key_path)
                    if dummy_shape: # Ensure dummy_shape is not empty
                         return torch.full((target_seq_len, *dummy_shape), pad_val, device=self.device, dtype=dummy_dtype)
                print(f"Warning: _pad_sequence_of_tensors received an empty list and could not determine dummy shape for key path {ref_key_path}.")
                return torch.empty((target_seq_len, 0), device=self.device, dtype=torch.float32) # Fallback to a possibly problematic empty tensor

            current_sequence = torch.stack(tensor_list, dim=0) 
            actual_seq_len = current_sequence.shape[0]

            if actual_seq_len == target_seq_len: return current_sequence
            elif actual_seq_len > target_seq_len: return current_sequence[:target_seq_len]
            else:
                pad_len = target_seq_len - actual_seq_len
                padding_shape = (pad_len, *current_sequence.shape[1:])
                pad_tensor = torch.full(padding_shape, pad_val, device=self.device, dtype=current_sequence.dtype)
                return torch.cat([current_sequence, pad_tensor], dim=0)

        first_segment_first_obs_example = segments[0].observations[0] if segments and segments[0].observations else {}

        for seg in segments:
            obs_steps, act_steps, rew_steps, next_obs_steps, done_steps, trunc_steps, h0_step, len_step = seg.tensors_for_collation()
            lens_list.append(min(len_step, target_seq_len))

            if expect_agent:
                agent_obs_sequence = [s.get("agent") for s in obs_steps if s.get("agent") is not None]
                batch_observations["agent"].append(_pad_sequence_of_tensors(agent_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["agent"]))
                
                agent_next_obs_sequence = [ns.get("agent") for ns in next_obs_steps if ns.get("agent") is not None]
                batch_next_observations["agent"].append(_pad_sequence_of_tensors(agent_next_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["agent"]))

            if expect_central:
                for comp_key in central_component_keys:
                    central_comp_obs_sequence = [s.get("central", {}).get(comp_key) for s in obs_steps if s.get("central", {}).get(comp_key) is not None]
                    batch_observations["central"][comp_key].append(_pad_sequence_of_tensors(central_comp_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["central", comp_key]))

                    central_comp_next_obs_sequence = [ns.get("central", {}).get(comp_key) for ns in next_obs_steps if ns.get("central", {}).get(comp_key) is not None]
                    batch_next_observations["central"][comp_key].append(_pad_sequence_of_tensors(central_comp_next_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["central", comp_key]))
            
            A_list.append(_pad_sequence_of_tensors(act_steps, ref_obs_for_shape=None)) # Actions usually don't need shape ref from obs
            R_list.append(_pad_sequence_of_tensors(rew_steps, ref_obs_for_shape=None))
            D_list.append(_pad_sequence_of_tensors(done_steps, pad_val=1.0, ref_obs_for_shape=None)) 
            TR_list.append(_pad_sequence_of_tensors(trunc_steps, pad_val=1.0, ref_obs_for_shape=None))
            
            if self.enable_memory and h0_step is not None: H_list.append(h0_step)

        final_batch_obs: Dict[str, Any] = {}
        if expect_agent and batch_observations["agent"] and any(t.numel() > 0 for t in batch_observations["agent"]):
            final_batch_obs["agent"] = torch.stack([t for t in batch_observations["agent"] if t.numel() > 0], dim=0)
        
        if expect_central:
            final_batch_obs["central"] = {}
            for comp_key in central_component_keys:
                if batch_observations["central"][comp_key] and any(t.numel() > 0 for t in batch_observations["central"][comp_key]):
                    final_batch_obs["central"][comp_key] = torch.stack([t for t in batch_observations["central"][comp_key] if t.numel() > 0], dim=0)
        
        final_batch_nobs: Dict[str, Any] = {}
        if expect_agent and batch_next_observations["agent"] and any(t.numel() > 0 for t in batch_next_observations["agent"]):
            final_batch_nobs["agent"] = torch.stack([t for t in batch_next_observations["agent"] if t.numel() > 0], dim=0)
        
        if expect_central:
            final_batch_nobs["central"] = {}
            for comp_key in central_component_keys:
                if batch_next_observations["central"][comp_key] and any(t.numel() > 0 for t in batch_next_observations["central"][comp_key]):
                    final_batch_nobs["central"][comp_key] = torch.stack([t for t in batch_next_observations["central"][comp_key] if t.numel() > 0], dim=0)
        
        def stack_if_not_empty(tensor_list):
            filtered_list = [t for t in tensor_list if t.numel() > 0]
            return torch.stack(filtered_list, dim=0) if filtered_list else torch.empty(0, device=self.device)

        batch_actions = stack_if_not_empty(A_list)
        batch_rewards = stack_if_not_empty(R_list)
        batch_dones = stack_if_not_empty(D_list)
        batch_truncs = stack_if_not_empty(TR_list)
        
        batch_initial_h = torch.stack(H_list, dim=0) if (self.enable_memory and H_list) else None
        
        num_segments_in_batch = len(lens_list)
        attention_mask = torch.zeros(num_segments_in_batch, target_seq_len, device=self.device)
        for i, l_eff in enumerate(lens_list):
            attention_mask[i, :l_eff] = 1.0
            
        return (final_batch_obs, batch_actions, batch_rewards, final_batch_nobs,
                batch_dones, batch_truncs, batch_initial_h, attention_mask)

    def _log_step(self, logs: Dict[str, Any]):
        for k, v_val in logs.items():
            if isinstance(v_val, torch.Tensor):
                scalar_v = v_val.item() if v_val.numel() == 1 else v_val.mean().item() 
            elif isinstance(v_val, (list, np.ndarray)):
                 scalar_v = np.mean(v_val).item() if len(v_val) > 0 else 0.0 
            elif isinstance(v_val, (int, float)):
                scalar_v = float(v_val)
            else:
                try: scalar_v = float(v_val)
                except (TypeError, ValueError):
                    print(f"RLRunner Log: Skipping log for key '{k}' due to unconvertible type: {type(v_val)}")
                    continue
            self.writer.add_scalar(k, scalar_v, self.update_idx)

    def end(self):
        if hasattr(self, 'agentComm') and self.agentComm:
            self.agentComm.cleanup()
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()
        # Save StateRecorder video if it exists and has frames
        if self.state_recorder and hasattr(self.state_recorder, 'frames') and len(self.state_recorder.frames) > 0:
            print("RLRunner ending: Saving any remaining StateRecorder frames...")
            self.state_recorder.save_video()
            self.state_recorder.frames.clear() # Clear frames after saving
        print("RLRunner ended.")