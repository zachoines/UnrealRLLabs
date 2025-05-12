from Source.Agent import Agent 
from Source.Utility import RunningMeanStdNormalizer
from Source.Environment import EnvCommunicationInterface, EventType 
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Any, Optional 
import numpy as np

class TrajectorySegment:
    """
    Stores a segment of a trajectory, including observations, actions, rewards,
    termination signals, and the initial hidden state for recurrent networks.
    """
    def __init__(self, num_agents_config: int, device: torch.device, 
                 enable_padding: bool, max_segment_len: int):
        # List to store observation dictionaries for each step in the segment
        self.observations: List[Dict[str, torch.Tensor]] = []
        # List to store action tensors for each step
        self.actions: List[torch.Tensor] = []
        # List to store reward tensors for each step
        self.rewards: List[torch.Tensor] = []
        # List to store next_observation dictionaries for each step
        self.next_observations: List[Dict[str, torch.Tensor]] = []
        # List to store done tensors (true termination) for each step
        self.dones: List[torch.Tensor] = []
        # List to store truncation tensors (timeout) for each step
        self.truncs: List[torch.Tensor] = []
        # Initial hidden state for the memory module (e.g., GRU) at the start of this segment
        self.initial_hidden_state: Optional[torch.Tensor] = None 
        # Actual number of steps collected in this segment so far
        self.true_sequence_length: int = 0
        # Maximum number of agents the agent's network is configured for (for tensor shape consistency)
        self.num_agents_config = num_agents_config
        # PyTorch device for tensor operations
        self.device = device
        # Flag indicating if padding should be applied to this segment during collation
        self.enable_padding = enable_padding 
        # The target maximum length for this segment if padding/segmentation is active
        self.max_segment_length = max_segment_len

    def add_step(self, obs: Dict[str, torch.Tensor], act: torch.Tensor, rew: torch.Tensor, 
                 next_obs: Dict[str, torch.Tensor], done: torch.Tensor, trunc: torch.Tensor):
        # Adds a single timestep of experience (s_t, a_t, r_t, s_{t+1}, d_t, tr_t) to the segment.
        if self.is_full(): 
            # This check is more of a safeguard if called incorrectly from outside.
            # The main logic in RLRunner should prevent adding to an already full segment when padding.
            print("RLRunner Warning: Attempted to add step to a full TrajectorySegment.")
            return

        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.truncs.append(trunc)
        self.true_sequence_length += 1

    def is_full(self) -> bool:
        # Checks if the segment has reached its maximum defined length.
        # This is primarily used when self.enable_padding is True to enforce fixed-length segments.
        if self.enable_padding: # If padding is enabled, segments are "full" at max_segment_length.
            return self.true_sequence_length >= self.max_segment_length
        return False # If padding is not enabled, segments are finalized only at episode end.

    def get_tensors_for_collation(self) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor], 
                                           List[Dict[str, torch.Tensor]], List[torch.Tensor], List[torch.Tensor], 
                                           Optional[torch.Tensor], int]:
        # Returns the collected lists of raw (unpadded) tensors and the initial hidden state for this segment.
        # Padding (if enabled) will be handled by the RLRunner's _collate_and_pad_sequences method.
        return (
            self.observations, self.actions, self.rewards,
            self.next_observations, self.dones, self.truncs,
            self.initial_hidden_state, self.true_sequence_length
        )

class RLRunner:
    """
    Orchestrates the reinforcement learning loop. It handles:
    - Communication with the C++ Unreal Engine environments.
    - Management of recurrent memory hidden states.
    - Collection of experiences into trajectory segments.
    - Preparation of batched and (optionally) padded data for agent updates.
    - Triggering agent learning steps.
    - Logging training metrics.
    """
    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface, config: Dict) -> None:
        self.agent = agent
        self.agentComm = agentComm
        self.config = config # Store the full configuration dictionary
        
        # Parse relevant sections from the main configuration dictionary
        train_params = config['train']
        agent_params = config['agent']['params']
        env_shape_params = config['environment']['shape']
        env_config_params = config['environment']['params']

        # --- Training Hyperparameters ---
        self.saveFrequency = train_params.get('saveFrequency', 10) # How often to save the agent model
        # sequence_length: Target length for trajectory segments for BPTT if padding is enabled.
        self.sequence_length = train_params.get('sequence_length', 64) 
        self.num_environments = train_params.get('num_environments', 1) # Number of parallel C++ environments
        # timesteps_from_ue_batch: Number of timesteps C++ sends per environment when EventType.UPDATE occurs.
        # This corresponds to 'batch_size' in the C++ Runner's context and 'batch_size' in train_params JSON.
        self.timesteps_from_ue_batch = train_params.get('batch_size', 256) 
        # pad_trajectories: Boolean flag to enable/disable padding of sequences to self.sequence_length.
        self.pad_trajectories = train_params.get('pad_trajectories', True)
        
        # --- Agent Memory (e.g., GRU, LSTM) Properties ---
        self.enable_memory = agent_params.get('enable_memory', False)
        # memory_type: Specifies the type of recurrent memory module used by the agent (e.g., 'gru', 'lstm').
        self.memory_type = agent_params.get('memory_type', 'gru' if self.enable_memory else 'none') 
        self.memory_hidden_size = 0 
        self.memory_num_layers = 0  
        if self.enable_memory:
            # Get config for the specified memory type (e.g., agent_cfg['gru'] or agent_cfg['lstm'])
            memory_cfg_key = self.memory_type if self.memory_type in agent_params else 'memory_module' # Fallback to generic key
            memory_cfg = agent_params.get(memory_cfg_key, {}) 
            self.memory_hidden_size = memory_cfg.get('hidden_size', 128)
            self.memory_num_layers = memory_cfg.get('num_layers', 1)

        # Max number of agents the agent's network architecture is built for.
        # This ensures consistent tensor shapes for hidden states, etc.
        self.num_agents_max_config = env_config_params.get('MaxAgents', 1)
        if "agent" in env_shape_params["state"]: # Override if specified in environment's state shape config
             self.num_agents_max_config = env_shape_params["state"]["agent"].get("max", self.num_agents_max_config)

        # Initialize persistent hidden states for the recurrent memory module (if enabled).
        # Shape: (NumEnvironments, MaxConfiguredAgents, NumMemoryLayers, MemoryHiddenSize)
        if self.enable_memory:
            self.current_memory_hidden_states = torch.zeros(
                self.num_environments,
                self.num_agents_max_config, 
                self.memory_num_layers,
                self.memory_hidden_size,
                device=self.agentComm.device # Ensure tensors are on the correct device
            )
        else:
            self.current_memory_hidden_states = None

        # --- Buffers for Trajectory Segments ---
        # trajectory_segment_buffers_per_env: Stores lists of completed TrajectorySegment objects for each environment.
        self.trajectory_segment_buffers_per_env: List[List[TrajectorySegment]] = [[] for _ in range(self.num_environments)]
        # current_segments_per_env: Stores the TrajectorySegment currently being built for each environment.
        self.current_segments_per_env: List[TrajectorySegment] = [
            TrajectorySegment(self.num_agents_max_config, self.agentComm.device, self.pad_trajectories, self.sequence_length) 
            for _ in range(self.num_environments)
        ]
        # Initialize initial_hidden_state for the very first segments if memory is enabled.
        if self.enable_memory:
            for i in range(self.num_environments):
                self.current_segments_per_env[i].initial_hidden_state = self.current_memory_hidden_states[i].clone()

        # --- State Normalization (Optional) ---
        state_normalization_config = train_params.get('states_normalizer', None)
        self.state_normalizer = None
        if state_normalization_config:
            self.state_normalizer = RunningMeanStdNormalizer(
                **state_normalization_config,
                device=self.agentComm.device
            )
        
        self.currentUpdate = 0 # Counter for agent update steps.
        self.writer = SummaryWriter() # For TensorBoard logging.
        print("RLRunner initialized.")


    def start(self):
        # Main training loop: alternates between action selection and agent updates.
        print("RLRunner starting main loop...")
        while True:
            # Wait for a signal from the C++ environment.
            event = self.agentComm.wait_for_event() 

            if event == EventType.GET_ACTIONS:
                # --- Action Selection Phase ---
                # 1. Receive current states, dones, and truncations from the environment interface.
                states_dict, dones_ue, truncs_ue = self.agentComm.get_states()
                # Squeeze UE's batch dim (1) and last dim (1) for dones/truncs: (1, NumEnv, 1) -> (NumEnv,).
                dones_ue = dones_ue.squeeze(0).squeeze(-1) 
                truncs_ue = truncs_ue.squeeze(0).squeeze(-1) 

                # 2. Manage hidden states and trajectory segments if memory is enabled.
                if self.enable_memory:
                    for env_idx in range(self.num_environments):
                        # If an environment terminated (done or truncated) in the previous C++ step:
                        if dones_ue[env_idx].item() > 0.5 or truncs_ue[env_idx].item() > 0.5:
                            # Reset the persistent hidden state for this environment for the *next* action selection.
                            self.current_memory_hidden_states[env_idx].zero_() 
                            
                            # Finalize the segment that just ended due to termination.
                            if self.current_segments_per_env[env_idx].true_sequence_length > 0:
                                self.trajectory_segment_buffers_per_env[env_idx].append(self.current_segments_per_env[env_idx])
                            
                            # Start a new segment, capturing the (now reset) hidden state.
                            self.current_segments_per_env[env_idx] = TrajectorySegment(self.num_agents_max_config, self.agentComm.device, self.pad_trajectories, self.sequence_length)
                            self.current_segments_per_env[env_idx].initial_hidden_state = self.current_memory_hidden_states[env_idx].clone()
                
                # 3. Normalize states if a normalizer is configured.
                states_dict_for_agent = states_dict
                if self.state_normalizer:
                    self.state_normalizer.update(states_dict) 
                    states_dict_for_agent = self.state_normalizer.normalize(states_dict)
                
                # 4. Get actions from the agent.
                actions_out, agent_internals, next_hidden_states_batch = self.agent.get_actions(
                    states_dict_for_agent, 
                    dones=dones_ue,       # Pass (NumEnv,) tensor.
                    truncs=truncs_ue,     # Pass (NumEnv,) tensor.
                    h_prev_batch=self.current_memory_hidden_states if self.enable_memory else None
                )
                
                # 5. Update persistent hidden states for the next iteration.
                if self.enable_memory:
                    self.current_memory_hidden_states = next_hidden_states_batch

                # 6. Send actions back to the C++ environment.
                self.agentComm.send_actions(actions_out) 

            elif event == EventType.UPDATE:
                # --- Experience Collection and Agent Update Phase ---
                self.currentUpdate += 1
                print(f"RLRunner: Starting Agent Update Cycle {self.currentUpdate}")
                
                # 1. Get a batch of experiences from C++ (s, a, r, s', d, tr).
                #    Expected shapes: (timesteps_from_ue_batch, num_environments, ...).
                raw_states_batch_dict, raw_next_states_batch_dict, actions_batch, \
                rewards_batch, dones_batch, truncs_batch = self.agentComm.get_experiences()

                # 2. Determine the number of timesteps per environment in this received batch.
                #    This should match self.timesteps_from_ue_batch.
                timesteps_in_received_batch = 0
                if raw_states_batch_dict: 
                    example_key = next(iter(raw_states_batch_dict), None) 
                    if example_key and raw_states_batch_dict[example_key] is not None and raw_states_batch_dict[example_key].ndim > 0:
                        timesteps_in_received_batch = raw_states_batch_dict[example_key].shape[0]
                
                if timesteps_in_received_batch == 0:
                    print(f"RLRunner Update {self.currentUpdate}: Received batch with 0 timesteps or invalid states. Skipping agent update.")
                    continue 
                
                # Critical: Ensure the loop iterates over the correct number of timesteps.
                # If C++ sends a different number of steps than configured in `batch_size` (timesteps_from_ue_batch),
                # this could lead to errors or missed data. For robustness, we use timesteps_in_received_batch.
                # However, it's expected that timesteps_in_received_batch == self.timesteps_from_ue_batch.
                if timesteps_in_received_batch != self.timesteps_from_ue_batch:
                    print(f"RLRunner Warning: Mismatch! Configured 'batch_size' (timesteps_from_ue_batch) = {self.timesteps_from_ue_batch}, "
                          f"but C++ sent a batch with {timesteps_in_received_batch} timesteps per env. "
                          f"Processing based on received {timesteps_in_received_batch} timesteps.")
                
                # 3. Process experiences: add to trajectory segments for each environment.
                for env_idx in range(self.num_environments):
                    # Iterate up to the actual number of timesteps received in this batch from C++.
                    for t_idx in range(timesteps_in_received_batch): 
                        try:
                            # Safely extract step data for the current environment and timestep.
                            current_s = {key: val[t_idx, env_idx] for key, val in raw_states_batch_dict.items() if val is not None and val.ndim > 1 and val.shape[0] > t_idx and val.shape[1] > env_idx}
                            current_a = actions_batch[t_idx, env_idx]       # Expected shape: (NumAgents, ActionDim)
                            current_r = rewards_batch[t_idx, env_idx]       # Expected shape: (1,)
                            current_ns = {key: val[t_idx, env_idx] for key, val in raw_next_states_batch_dict.items() if val is not None and val.ndim > 1 and val.shape[0] > t_idx and val.shape[1] > env_idx}
                            current_d = dones_batch[t_idx, env_idx]         # Expected shape: (1,)
                            current_tr = truncs_batch[t_idx, env_idx]       # Expected shape: (1,)
                        except IndexError as e: 
                            print(f"RLRunner Error: IndexError while accessing C++ batch data at t_idx={t_idx}, env_idx={env_idx}. Error: {e}. Skipping this step.")
                            # This might happen if tensors in the batch have inconsistent first dimensions.
                            continue 

                        # Add step to the current segment for this environment.
                        self.current_segments_per_env[env_idx].add_step(
                            current_s, current_a, current_r, current_ns, current_d, current_tr
                        )
                        
                        # If segment is full (based on sequence_length if padding) or episode ended, finalize and start a new one.
                        if self.current_segments_per_env[env_idx].is_full() or \
                           current_d.item() > 0.5 or current_tr.item() > 0.5: # Episode terminated
                            self.trajectory_segment_buffers_per_env[env_idx].append(self.current_segments_per_env[env_idx])
                            # Start a new segment
                            self.current_segments_per_env[env_idx] = TrajectorySegment(self.num_agents_max_config, self.agentComm.device, self.pad_trajectories, self.sequence_length)
                            if self.enable_memory:
                                # New segment's initial hidden state is the current hidden state for this env
                                # (which would have been updated/reset after the last action selection).
                                self.current_segments_per_env[env_idx].initial_hidden_state = self.current_memory_hidden_states[env_idx].clone()
                
                # 4. Collect all completed segments from all environments for this update.
                all_trajectory_data_for_update: List[TrajectorySegment] = []
                for env_buffer in self.trajectory_segment_buffers_per_env:
                    all_trajectory_data_for_update.extend(env_buffer)
                
                # Clear the per-environment buffers after collecting segments.
                for env_idx in range(self.num_environments):
                    self.trajectory_segment_buffers_per_env[env_idx].clear()

                if not all_trajectory_data_for_update:
                    print(f"RLRunner Update {self.currentUpdate}: No complete trajectory segments available for agent update. Skipping.")
                    continue 

                # 5. Collate, Pad (if enabled), and Create Masks for the batch of sequences.
                padded_obs_seqs, padded_act_seqs, padded_rew_seqs, \
                padded_next_obs_seqs, padded_done_seqs, padded_trunc_seqs, \
                initial_h_seqs, attention_masks = self._collate_and_pad_sequences(all_trajectory_data_for_update)

                if not padded_obs_seqs or (isinstance(padded_obs_seqs, dict) and not any(v.numel() > 0 for v in padded_obs_seqs.values())) : 
                    print(f"RLRunner Update {self.currentUpdate}: Collation of segments resulted in no data. Skipping agent update.")
                    continue

                # 6. Perform agent update.
                logs = self.agent.update(
                    padded_states_dict_seq=padded_obs_seqs,
                    padded_next_states_dict_seq=padded_next_obs_seqs,
                    padded_actions_seq=padded_act_seqs,
                    padded_rewards_seq=padded_rew_seqs,
                    padded_dones_seq=padded_done_seqs,    
                    padded_truncs_seq=padded_trunc_seqs,  
                    initial_hidden_states_batch=initial_h_seqs if self.enable_memory else None,
                    attention_mask_batch=attention_masks # Mask is always passed
                )
            
                self.log_step(logs) # Log training metrics.
                if self.currentUpdate % self.saveFrequency == 0:
                    torch.save(self.agent.state_dict(), f'model_update_{self.currentUpdate}.pth')
                    print(f"RLRunner: Saved model at update {self.currentUpdate}")


    def _collate_and_pad_sequences(self, trajectory_segments: List[TrajectorySegment]) \
        -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], 
                 torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # Collates a list of TrajectorySegment objects into batched tensors.
        # Pads sequences to target_len_for_batch if self.pad_trajectories is True.
        
        if not trajectory_segments: 
            # Return empty structures if no segments are provided.
            empty_state_dict = {} 
            return empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   None, torch.empty(0, device=self.agentComm.device)

        # Determine target length for batch.
        if self.pad_trajectories:
            target_len_for_batch = self.sequence_length
        else: 
            # If not padding, use the max true length in the current batch of segments.
            # The agent (MAPOCA) must be able to handle variable length sequences if this path is taken.
            # Typically, for RNNs in PPO, fixed-length (padded) sequences are preferred.
            valid_lengths = [seg.true_sequence_length for seg in trajectory_segments if seg.true_sequence_length > 0]
            target_len_for_batch = max(valid_lengths) if valid_lengths else 0
        
        if target_len_for_batch == 0 and trajectory_segments: 
             # This case implies all segments were empty, or became empty after filtering.
             empty_state_dict = {} 
             return empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   None, torch.empty(0, device=self.agentComm.device)

        # Determine observation keys from the first non-empty segment to structure the output dicts.
        first_valid_segment = next((seg for seg in trajectory_segments if seg.observations), None)
        if not first_valid_segment: # All segments have empty observation lists.
             empty_state_dict = {} 
             return empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   None, torch.empty(0, device=self.agentComm.device)

        obs_keys = first_valid_segment.observations[0].keys()
        collated_obs_seqs_dict = {key: [] for key in obs_keys}
        collated_next_obs_seqs_dict = {key: [] for key in obs_keys}
        collated_act_seqs, collated_rew_seqs, collated_done_seqs, collated_trunc_seqs = [], [], [], []
        collated_initial_h_seqs, collated_true_lengths = [], []

        for segment in trajectory_segments:
            if segment.true_sequence_length == 0: continue # Skip genuinely empty segments.

            obs_list, act_list, rew_list, next_obs_list, done_list, trunc_list, initial_h, true_len = segment.get_tensors_for_collation()
            
            collated_true_lengths.append(true_len)
            if self.enable_memory:
                if initial_h is not None:
                    collated_initial_h_seqs.append(initial_h)
                else: 
                    # This fallback should ideally not be needed if Runner logic is correct.
                    print("RLRunner Warning (_collate_and_pad): initial_hidden_state is None for a non-empty segment with memory enabled. Using zeros.")
                    collated_initial_h_seqs.append(torch.zeros(self.num_agents_max_config, self.memory_num_layers, self.memory_hidden_size, device=self.device))
            
            # Determine the length to pad/truncate this specific segment to.
            current_segment_target_len = target_len_for_batch if self.pad_trajectories else true_len

            # Helper function for padding or truncating individual tensor sequences within a segment.
            def _process_tensor_list(tensor_list, item_dtype, item_device, final_len, pad_val=0.0):
                if not tensor_list: # If the list for this item type is empty for the segment
                    # This needs a defined way to get the trailing dimensions for an empty item.
                    # For now, we'll assume if list is empty, we can't form a tensor. This should be rare for core items.
                    print(f"RLRunner Warning: _process_tensor_list received empty tensor_list for final_len {final_len}. Returning None or empty.")
                    return None # Or handle by creating a default shaped tensor if possible

                stacked_tensor = torch.stack(tensor_list, dim=0) # (true_len, ...)
                
                if stacked_tensor.shape[0] == final_len: return stacked_tensor
                if stacked_tensor.shape[0] > final_len: return stacked_tensor[:final_len] # Truncate

                # Else, padding is needed
                _padding_needed_for_item = final_len - stacked_tensor.shape[0]
                if _padding_needed_for_item > 0:
                    pad_shape = list(stacked_tensor.shape)
                    pad_shape[0] = _padding_needed_for_item
                    padding = torch.full(pad_shape, pad_val, device=item_device, dtype=item_dtype)
                    return torch.cat([stacked_tensor, padding], dim=0)
                return stacked_tensor # Should be covered by above conditions

            # Process observations
            for key in obs_keys:
                obs_for_key_list = [o[key] for o in obs_list if key in o and o[key] is not None]
                if obs_for_key_list:
                    processed_tensor = _process_tensor_list(obs_for_key_list, obs_for_key_list[0].dtype, self.device, current_segment_target_len)
                    if processed_tensor is not None: collated_obs_seqs_dict[key].append(processed_tensor)
            
            # Process next_observations
            for key in obs_keys: 
                 next_obs_for_key_list = [no[key] for no in next_obs_list if key in no and no[key] is not None]
                 if next_obs_for_key_list:
                    processed_tensor = _process_tensor_list(next_obs_for_key_list, next_obs_for_key_list[0].dtype, self.device, current_segment_target_len)
                    if processed_tensor is not None: collated_next_obs_seqs_dict[key].append(processed_tensor)

            # Process actions, rewards, dones, truncs
            if act_list: collated_act_seqs.append(_process_tensor_list(act_list, act_list[0].dtype, self.device, current_segment_target_len))
            if rew_list: collated_rew_seqs.append(_process_tensor_list(rew_list, rew_list[0].dtype, self.device, current_segment_target_len))
            if done_list: collated_done_seqs.append(_process_tensor_list(done_list, done_list[0].dtype, self.device, current_segment_target_len, pad_val=1.0))
            if trunc_list: collated_trunc_seqs.append(_process_tensor_list(trunc_list, trunc_list[0].dtype, self.device, current_segment_target_len, pad_val=1.0))
        
        # If all segments were empty after filtering, return empty structures
        if not collated_true_lengths: 
             empty_state_dict = {} 
             return empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   empty_state_dict, torch.empty(0, device=self.agentComm.device), torch.empty(0, device=self.agentComm.device), \
                   None, torch.empty(0, device=self.agentComm.device)

        # Stack all processed sequences in the batch
        # Ensure lists are not empty before stacking to avoid errors
        final_padded_obs_dict = {key: torch.stack(val_list, dim=0) for key, val_list in collated_obs_seqs_dict.items() if val_list}
        final_padded_next_obs_dict = {key: torch.stack(val_list, dim=0) for key, val_list in collated_next_obs_seqs_dict.items() if val_list}
        
        # For non-dict items, provide a default empty tensor with expected rank if the list is empty
        # This helps prevent errors in agent.update if a batch ends up with no valid data for some items.
        # The agent.update method should also be robust to potentially empty batches for certain items.
        batch_dim_size = len(collated_true_lengths)
        
        final_padded_act = torch.stack(collated_act_seqs, dim=0) if collated_act_seqs else torch.empty(batch_dim_size, target_len_for_batch, self.num_agents_max_config, 0, device=self.agentComm.device)    
        final_padded_rew = torch.stack(collated_rew_seqs, dim=0) if collated_rew_seqs else torch.empty(batch_dim_size, target_len_for_batch, 1, device=self.agentComm.device)    
        final_padded_done = torch.stack(collated_done_seqs, dim=0) if collated_done_seqs else torch.empty(batch_dim_size, target_len_for_batch, 1, device=self.agentComm.device)  
        final_padded_trunc = torch.stack(collated_trunc_seqs, dim=0) if collated_trunc_seqs else torch.empty(batch_dim_size, target_len_for_batch, 1, device=self.agentComm.device)
        
        final_initial_h = torch.stack(collated_initial_h_seqs, dim=0) if self.enable_memory and collated_initial_h_seqs else None 
        
        # Create attention mask (B, TargetLenForBatch)
        attention_mask = torch.zeros(num_batch_segments, target_len_for_batch, device=self.agentComm.device, dtype=torch.float32) 
        for i, length in enumerate(collated_true_lengths):
            # Mask only up to the true length of the sequence, or target_len_for_batch if shorter
            mask_len = min(length, target_len_for_batch)
            attention_mask[i, :mask_len] = 1.0
            
        return final_padded_obs_dict, final_padded_act, final_padded_rew, \
               final_padded_next_obs_dict, final_padded_done, final_padded_trunc, \
               final_initial_h, attention_mask

    def log_step(self, train_results: dict[str, Any])->None: 
        # Logs training metrics to TensorBoard.
        for metric, value in train_results.items():
            scalar_value = value.item() if isinstance(value, torch.Tensor) else float(value) # Ensure scalar
            self.writer.add_scalar(tag=metric, scalar_value=scalar_value, global_step=self.currentUpdate)

    def end(self) -> None:
        # Cleans up resources like the agent communication interface and TensorBoard writer.
        self.agentComm.cleanup()
        self.writer.close()
        print("RLRunner finished and cleaned up.")

