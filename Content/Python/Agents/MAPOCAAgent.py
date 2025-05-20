# NOTICE: This file includes modifications generated with the assistance of generative AI.
# Original code structure and logic by the project author.
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from Source.StateRecorder import StateRecorder
from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer, LinearValueScheduler
from Source.Networks import (
    MultiAgentEmbeddingNetwork,
    SharedCritic,
    DiscretePolicyNetwork,
    BetaPolicyNetwork,
    RNDTargetNetwork, # NEW RND: Import RND networks
    RNDPredictorNetwork # NEW RND: Import RND networks
)
from Source.Memory import RecurrentMemoryNetwork, GRUSequenceMemory

class MAPOCAAgent(Agent):
    """
    Multi-agent PPO (MAPOCA) agent using an Attention-based SharedCritic.
    Key features:
    - Modular recurrent memory (e.g., GRU, LSTM) for temporal processing.
    - Support for Beta (continuous) or Discrete policy networks.
    - PPO algorithm with clipped surrogate objective.
    - Optional clipped value and baseline loss functions.
    - Generalized Advantage Estimation (GAE) for advantage calculation.
    - Multi-epoch mini-batch updates over sequences of experiences.
    - Counterfactual baseline for multi-agent credit assignment.
    - Schedulers for learning rate, entropy, and clipping parameters.
    - Optional state and reward normalization.
    - Optional Random Network Distillation (RND) for exploration.
    """

    def __init__(self, config: Dict, device: torch.device):
        super().__init__(config, device)

        # Parse configuration dictionaries
        agent_cfg = config["agent"]["params"]
        train_cfg = config["train"]
        shape_cfg = config["environment"]["shape"]
        action_cfg = shape_cfg["action"]
        rec_cfg = config.get("StateRecorder", None) # Optional state recorder config
        env_params_cfg = config["environment"]["params"]

        # --- Core PPO & GAE Hyperparameters ---
        self.gamma = agent_cfg.get("gamma", 0.99)  # Discount factor for future rewards
        self.lmbda = agent_cfg.get("lambda", 0.95) # Lambda for GAE
        self.lr = agent_cfg.get("learning_rate", 3e-4) # Learning rate
        self.value_loss_coeff = agent_cfg.get("value_loss_coeff", 0.5) # Coefficient for value loss
        self.baseline_loss_coeff = agent_cfg.get("baseline_loss_coeff", 0.5) # Coefficient for baseline loss
        self.entropy_coeff = agent_cfg.get("entropy_coeff", 0.01) # Coefficient for entropy bonus
        self.max_grad_norm = agent_cfg.get("max_grad_norm", 0.5) # Max norm for gradient clipping
        self.normalize_adv = agent_cfg.get("normalize_advantages", True) # Whether to normalize advantages
        self.ppo_clip_range = agent_cfg.get("ppo_clip_range", 0.2) # PPO clipping parameter
        self.clipped_value_loss = agent_cfg.get("clipped_value_loss", True) # Whether to use clipped value loss
        self.value_clip_range = agent_cfg.get("value_clip_range", 0.2) # Clipping range for value loss
        # Batch size for forward passes in no_grad block (number of sequences) during agent update's data gathering phase
        self.no_grad_forward_batch_size = agent_cfg.get("no_grad_forward_batch_size", train_cfg.get("mini_batch_size", 64))

        # --- Training Loop Parameters ---
        self.epochs = train_cfg.get("epochs", 4) # Number of PPO epochs over collected data
        # Number of sequences in each mini-batch during PPO optimization epochs
        self.mini_batch_size = train_cfg.get("mini_batch_size", 64)

        # --- Optional State Recorder ---
        self.state_recorder = StateRecorder(rec_cfg) if rec_cfg else None

        # --- Action Space Configuration ---
        self._determine_action_space(action_cfg) # Sets self.action_space_type, self.action_dim, self.total_action_dim_per_agent
        
        # Max number of agents the network architecture is built for (from environment parameters)
        self.num_agents = env_params_cfg.get('MaxAgents', 1)
        if "agent" in shape_cfg["state"]: # Override if specified in environment's state shape config
            self.num_agents = shape_cfg["state"]["agent"].get("max", self.num_agents)

        # --- Modular Memory Initialization ---
        self.enable_memory = agent_cfg.get('enable_memory', False) # Flag to enable/disable recurrent memory
        self.memory_module: Optional[RecurrentMemoryNetwork] = None # Placeholder for the memory module instance
        self.memory_hidden_size = 0 # Hidden size of the memory module
        self.memory_num_layers = 0  # Number of layers in the memory module
        # Dimension of features output by the embedding network, which becomes input to memory or heads
        embedding_output_dim = agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]["cross_attention_feature_extractor"]["embed_dim"]

        if self.enable_memory:
            self.memory_type = agent_cfg.get('memory_type', 'gru') # Type of memory module (e.g., 'gru')
            memory_module_config = agent_cfg.get(self.memory_type, {}) # Get config for the specified memory type

            # Ensure input_size for memory module is defined, default to embedding_output_dim
            if 'input_size' not in memory_module_config:
                print(f"MAPOCAAgent Warning: 'input_size' not found in config for memory_type '{self.memory_type}'. "
                      f"Defaulting memory input_size to embedding_output_dim: {embedding_output_dim}.")
                memory_module_config['input_size'] = embedding_output_dim
            
            # Check for potential mismatch between embedding output and memory input size
            if memory_module_config['input_size'] != embedding_output_dim:
                 print(f"MAPOCAAgent Warning: Memory module input_size ({memory_module_config['input_size']}) "
                       f"from config ('{self.memory_type}.input_size') differs from embedding_network's output_dim ({embedding_output_dim}). "
                       f"Ensure this is intended, as embeddings are input to the memory module.")

            self.memory_hidden_size = memory_module_config.get('hidden_size', 128)
            self.memory_num_layers = memory_module_config.get('num_layers', 1)

            # Instantiate the specified memory module
            if self.memory_type == 'gru':
                self.memory_module = GRUSequenceMemory(**memory_module_config).to(device)
                # print(f"MAPOCAAgent: GRU Memory Module Initialized with config: {memory_module_config}") # Already printed by GRUSequenceMemory
            # elif self.memory_type == 'lstm': # Example for future LSTM use
            #    self.memory_module = LSTMSequenceMemory(**memory_module_config).to(device)
            else:
                raise ValueError(f"MAPOCAAgent Error: Unsupported memory_type '{self.memory_type}' specified in config.")
        
        # Determine the feature dimension for policy and critic heads.
        # If memory is enabled, this is the memory's hidden size; otherwise, it's the embedding network's output dimension.
        self.feature_dim_for_heads = self.memory_hidden_size if self.enable_memory and self.memory_module else embedding_output_dim

        # --- Build Core Networks ---
        self.embedding_network = MultiAgentEmbeddingNetwork(
            agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        ).to(device)
        self.shared_critic = SharedCritic(agent_cfg["networks"]["critic_network"]).to(device)
        
        policy_network_config = agent_cfg["networks"]["policy_network"].copy() # Use .copy() to avoid modifying original dict
        policy_network_config["in_features"] = self.feature_dim_for_heads # Set input features for policy

        # Instantiate policy network based on action space type
        if self.action_space_type == "continuous":
            policy_network_config['out_features'] = self.action_dim # For BetaPolicy, action_dim is the number of continuous actions
            self.policy_net = BetaPolicyNetwork(**policy_network_config).to(device)
        elif self.action_space_type == "discrete":
            # For DiscretePolicy, action_dim is a list of choices per branch or a single int
            policy_network_config['out_features'] = self.action_dim 
            self.policy_net = DiscretePolicyNetwork(**policy_network_config).to(device)
        else:
            raise ValueError(f"MAPOCAAgent Error: Unsupported action_space_type: {self.action_space_type}")
        # print(f"MAPOCAAgent: Policy Network ({type(self.policy_net).__name__}) Initialized with in_features={self.feature_dim_for_heads}.")

        # --- Schedulers ---
        # Store LR scheduler config to initialize optimizer and scheduler together later
        self.lr_scheduler_cfg = agent_cfg.get("schedulers", {}).get("lr", None)

        ent_sched_cfg = agent_cfg.get("schedulers", {}).get("entropy_coeff", None)
        self.entropy_scheduler = LinearValueScheduler(**ent_sched_cfg) if ent_sched_cfg else None
        if self.entropy_scheduler: self.entropy_coeff = self.entropy_scheduler.start_value # Initialize with scheduler's start value
        
        policy_clip_sched_cfg = agent_cfg.get("schedulers", {}).get("policy_clip", None)
        self.policy_clip_scheduler = LinearValueScheduler(**policy_clip_sched_cfg) if policy_clip_sched_cfg else None
        if self.policy_clip_scheduler: self.ppo_clip_range = self.policy_clip_scheduler.start_value
        
        value_clip_sched_cfg = agent_cfg.get("schedulers", {}).get("value_clip", None)
        self.value_clip_scheduler = LinearValueScheduler(**value_clip_sched_cfg) if value_clip_sched_cfg else None
        if self.value_clip_scheduler: self.value_clip_range = self.value_clip_scheduler.start_value
        
        grad_norm_sched_cfg = agent_cfg.get("schedulers", {}).get("max_grad_norm", None)
        self.max_grad_norm_scheduler = LinearValueScheduler(**grad_norm_sched_cfg) if grad_norm_sched_cfg else None
        if self.max_grad_norm_scheduler: self.max_grad_norm = self.max_grad_norm_scheduler.start_value

        # --- Normalizers ---
        rewards_normalization_cfg = agent_cfg.get('rewards_normalizer', None)
        self.rewards_normalizer = RunningMeanStdNormalizer(**rewards_normalization_cfg, device=self.device) if rewards_normalization_cfg else None

        # --- RND Initialization ---
        self.enable_rnd = agent_cfg.get("enable_rnd", False)
        self.rnd_target_network: Optional[RNDTargetNetwork] = None
        self.rnd_predictor_network: Optional[RNDPredictorNetwork] = None
        self.intrinsic_reward_normalizer: Optional[RunningMeanStdNormalizer] = None
        self.intrinsic_reward_coeff = 0.0
        self.rnd_update_proportion = 0.25 # Default proportion of samples for RND predictor update

        if self.enable_rnd:
            rnd_cfg = agent_cfg.get("rnd_params", {})
            # RND networks operate on the features that are input to the policy/critic heads
            rnd_input_dim = self.feature_dim_for_heads 
            rnd_output_dim = rnd_cfg.get("output_size", 128) # Size of the random embedding
            rnd_hidden_size = rnd_cfg.get("hidden_size", 256) # Hidden size for RND MLPs

            self.rnd_target_network = RNDTargetNetwork(rnd_input_dim, rnd_output_dim, rnd_hidden_size).to(self.device)
            self.rnd_predictor_network = RNDPredictorNetwork(rnd_input_dim, rnd_output_dim, rnd_hidden_size).to(self.device)

            self.intrinsic_reward_coeff = rnd_cfg.get("intrinsic_reward_coeff", 0.01)
            self.rnd_update_proportion = rnd_cfg.get("rnd_update_proportion", 0.25)
            # Normalizer for the intrinsic rewards themselves
            self.intrinsic_reward_normalizer = RunningMeanStdNormalizer(epsilon=1e-8, device=self.device) 
            print(f"MAPOCAAgent: RND Module Initialized. Input Dim: {rnd_input_dim}, Output Dim: {rnd_output_dim}")

        # --- Initialize Optimizer and LR Scheduler (after all network parameters are known) ---
        all_params = self.parameters() # self.parameters() will now include RND predictor params if enabled
        self.optimizer = optim.AdamW(all_params, lr=self.lr, eps=1e-7) # Using AdamW
        if self.lr_scheduler_cfg: # Initialize LR scheduler if config was provided
            self.lr_scheduler = LinearLR(self.optimizer, **self.lr_scheduler_cfg)
        else:
            self.lr_scheduler = None
        # --- End of __init__ ---

    def parameters(self):
        # Collects parameters from all networks for the optimizer.
        params_list = list(self.embedding_network.parameters()) + \
                      list(self.shared_critic.parameters()) + \
                      list(self.policy_net.parameters())
        if self.enable_memory and self.memory_module is not None:
            params_list += list(self.memory_module.parameters())
        # NEW RND: Add RND predictor network parameters to the optimizer list
        if self.enable_rnd and self.rnd_predictor_network is not None:
            params_list += list(self.rnd_predictor_network.parameters()) # Target network params are fixed
        return params_list

    def _determine_action_space(self, action_cfg: Dict):
        # Determines action space type (discrete/continuous) and dimensions from config.
        # self.action_dim: For continuous, it's an int (number of continuous actions per agent).
        #                   For discrete, it's a list of ints (number of choices for each discrete branch per agent).
        # self.total_action_dim_per_agent: Total scalar action values output by policy per agent.
        if "agent" in action_cfg:
            agent_action_cfg = action_cfg["agent"]
            if "discrete" in agent_action_cfg:
                self.action_space_type = "discrete"
                d_list = agent_action_cfg["discrete"] # e.g., [{"num_choices": N1}, {"num_choices": N2}]
                self.action_dim = [d["num_choices"] for d in d_list] # List of choices for each branch
                self.total_action_dim_per_agent = len(self.action_dim) # Number of discrete branches
            elif "continuous" in agent_action_cfg:
                self.action_space_type = "continuous"
                # For continuous, action_dim is the number of continuous action values
                self.action_dim = len(agent_action_cfg["continuous"]) 
                self.total_action_dim_per_agent = self.action_dim # Same as action_dim for continuous
            else:
                raise ValueError("MAPOCAAgent Error: Missing 'discrete' or 'continuous' under 'action.agent' in config.")
        else:
            raise ValueError("MAPOCAAgent Error: No 'agent' block found in 'action' config.")
        # print(f"MAPOCAAgent: Action Space: Type='{self.action_space_type}', Per-Agent Dim/Branches={self.action_dim}, Total Per-Agent Output Dim (flat)={self.total_action_dim_per_agent}")


    def _get_batch_shape_info(self, states_input: Dict[str, torch.Tensor], context: str = "get_actions") -> Tuple[int, int, int, int]:
        # Helper to deduce batch dimensions (S, E, NA, Obs) from state dictionary.
        # S: Sequence/Time dimension (or outer batch for get_actions from Runner)
        # E: Environment dimension (or inner batch for get_actions from Runner)
        # NA: Number of Agents in the tensor
        # Obs: Observation dimension
        dim0, dim1, num_actual_agents_in_tensor, obs_d = -1, -1, -1, -1
        primary_key = 'agent' if 'agent' in states_input and states_input['agent'] is not None else 'central'
        
        if primary_key not in states_input or states_input[primary_key] is None:
            raise ValueError(f"MAPOCAAgent Error: Cannot determine batch dimensions: '{primary_key}' key missing or None in states_input for {context}.")
        
        tensor_shape = states_input[primary_key].shape

        if primary_key == 'agent': # Agent-specific observations
            if len(tensor_shape) < 3: 
                raise ValueError(f"MAPOCAAgent Error: Agent state tensor in {context} has {len(tensor_shape)} dims, expected at least 3 (e.g., E,A,O or S,E,A,O). Shape: {tensor_shape}")
            if len(tensor_shape) == 3 and context == "get_actions": # (NumEnvs, NumAgents, ObsDim)
                dim0 = 1 # Represents the single "step" or "outer batch" for get_actions
                dim1 = tensor_shape[0] # NumEnvs
                num_actual_agents_in_tensor = tensor_shape[1] # NumAgents in this tensor
            elif len(tensor_shape) == 4 : # (BatchOfSequences, SeqLen, NumAgents, ObsDim)
                dim0 = tensor_shape[0] # BatchOfSequences
                dim1 = tensor_shape[1] # SeqLen
                num_actual_agents_in_tensor = tensor_shape[2] # NumAgents in this tensor
            else:
                raise ValueError(f"MAPOCAAgent Error: Unexpected agent state tensor shape in {context}: {tensor_shape}")
            obs_d = tensor_shape[-1]
        elif primary_key == 'central': # Centralized observation
            if len(tensor_shape) < 2:
                raise ValueError(f"MAPOCAAgent Error: Central state tensor in {context} has {len(tensor_shape)} dims, expected at least 2. Shape: {tensor_shape}")
            if len(tensor_shape) == 2 and context == "get_actions": # (NumEnvs, CentralObsDim)
                dim0 = 1
                dim1 = tensor_shape[0]
            elif len(tensor_shape) == 3: # (BatchOfSequences, SeqLen, CentralObsDim)
                dim0 = tensor_shape[0]
                dim1 = tensor_shape[1]
            else:
                 raise ValueError(f"MAPOCAAgent Error: Unexpected central state tensor shape in {context}: {tensor_shape}")
            num_actual_agents_in_tensor = 1 # Central state is shared, conceptually 1 "agent" for this obs type
            obs_d = tensor_shape[-1]
        return dim0, dim1, num_actual_agents_in_tensor, obs_d


    @torch.no_grad() # Actions are sampled without gradient calculation
    def get_actions(self, states: dict, dones: torch.Tensor, truncs: torch.Tensor,
                    h_prev_batch: Optional[torch.Tensor] = None, eval: bool=False, record: bool=True) \
                    -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        # --- Input Processing & Embedding ---
        # s_dim: Typically 1 for get_actions from Runner (batch of environments at current step).
        # num_envs_from_state: Number of parallel environments.
        s_dim, num_envs_from_state, _, _ = self._get_batch_shape_info(states, context="get_actions")

        # Get base embeddings from observations.
        # Output shape: (s_dim, num_envs_from_state, self.num_agents (MaxConfiguredAgents), embedding_dim)
        base_embeddings, _ = self.embedding_network.get_base_embedding(states)
        
        # Squeeze s_dim if it's 1 (standard for get_actions from Runner).
        # features_for_memory_input becomes (NumEnvs, MaxConfiguredAgents, EmbDim)
        if base_embeddings.shape[0] == 1 and s_dim == 1:
            features_for_memory_input = base_embeddings.squeeze(0) 
        else: # Fallback if s_dim > 1 or other unexpected shapes (e.g. if called directly with a batch of steps)
            features_for_memory_input = base_embeddings.reshape(-1, self.num_agents, base_embeddings.shape[-1])
            num_envs_from_state = features_for_memory_input.shape[0] # Update num_envs if reshaped

        embedding_dim = features_for_memory_input.shape[-1] # Dimension of embeddings from embedding_network
        features_for_policy = features_for_memory_input     # Input to policy heads (initially embeddings)
        next_hidden_state_to_return = h_prev_batch          # Default if memory is not enabled

        # --- Memory Module Processing (Single Step) ---
        if self.enable_memory and self.memory_module is not None:
            # Initialize hidden state if Runner didn't provide (e.g., first step)
            if h_prev_batch is None: 
                h_prev_batch = torch.zeros(num_envs_from_state, self.num_agents, self.memory_num_layers, self.memory_hidden_size, device=self.device)
            
            # Ensure h_prev_batch matches current batch dimensions, adapting if necessary.
            # This handles cases where num_envs_from_state might change or num_agents in tensor differs from config.
            if h_prev_batch.shape[0] != num_envs_from_state or h_prev_batch.shape[1] != self.num_agents:
                 print(f"MAPOCAAgent Warning in get_actions: h_prev_batch shape {h_prev_batch.shape} mismatch. Expected num_envs={num_envs_from_state}, num_agents={self.num_agents}. Adapting h_prev_batch.")
                 h_prev_batch_adapted = torch.zeros(num_envs_from_state, self.num_agents, self.memory_num_layers, self.memory_hidden_size, device=self.device)
                 min_envs = min(h_prev_batch.shape[0], num_envs_from_state)
                 min_agents_dim = min(h_prev_batch.shape[1], self.num_agents) # Use self.num_agents (max_config)
                 h_prev_batch_adapted[:min_envs, :min_agents_dim, :, :] = h_prev_batch[:min_envs, :min_agents_dim, :, :]
                 h_prev_batch = h_prev_batch_adapted

            # Prepare inputs for memory_module.forward_step:
            # Input features: (N, H_in) where N = NumEnvs * MaxConfiguredAgents
            # Hidden state: (D, N, H_out) where D = NumMemoryLayers
            memory_input_features_flat = features_for_memory_input.reshape(num_envs_from_state * self.num_agents, embedding_dim)
            # Reshape h_prev: (NumEnvs, MaxConfigAgents, NumLayers, Hidden) -> (NumEnvs*MaxConfigAgents, NumLayers, Hidden) -> (NumLayers, NumEnvs*MaxConfigAgents, Hidden)
            h_prev_for_memory_module = h_prev_batch.reshape(num_envs_from_state * self.num_agents, self.memory_num_layers, self.memory_hidden_size).permute(1, 0, 2).contiguous()

            # Process through the memory module
            features_for_policy_flat, h_next_flat = self.memory_module.forward_step(memory_input_features_flat, h_prev_for_memory_module)
            
            # Reshape memory module output features to (NumEnvs, MaxConfiguredAgents, MemoryHiddenDim)
            features_for_policy = features_for_policy_flat.reshape(num_envs_from_state, self.num_agents, self.memory_hidden_size)
            # Reshape next hidden state: (NumLayers, NumEnvs*MaxConfigAgents, Hidden) -> (NumEnvs*MaxConfigAgents, NumLayers, Hidden) -> (NumEnvs, MaxConfigAgents, NumLayers, Hidden)
            next_hidden_intermediate = h_next_flat.permute(1, 0, 2).reshape(num_envs_from_state, self.num_agents, self.memory_num_layers, self.memory_hidden_size)
            
            # --- Hidden State Reset for Done/Truncated Environments ---
            # dones/truncs from Runner are (NumEnvs,)
            if dones.ndim == 1: dones = dones.unsqueeze(1).expand(-1, self.num_agents) # (NumEnvs, MaxConfigAgents)
            if truncs.ndim == 1: truncs = truncs.unsqueeze(1).expand(-1, self.num_agents) # (NumEnvs, MaxConfigAgents)

            # Ensure shapes match for reset_mask, slicing if necessary (though Runner should provide correct NumEnvs)
            if dones.shape[0] != num_envs_from_state: dones = dones[:num_envs_from_state, :] 
            if truncs.shape[0] != num_envs_from_state: truncs = truncs[:num_envs_from_state, :]

            reset_mask_bool = dones.bool() | truncs.bool() # True if done OR truncated for any agent in an env group
            # Expand reset_mask to match shape of next_hidden_intermediate for broadcasting
            reset_mask = reset_mask_bool.unsqueeze(-1).unsqueeze(-1).expand_as(next_hidden_intermediate).float()
            next_hidden_state_to_return = next_hidden_intermediate * (1.0 - reset_mask) # Zero out hidden states for terminated envs
        
        # --- Action Generation from Policy Network ---
        # Flatten features for policy network input: (NumEnvs*MaxConfiguredAgents, FeatureDimForHeads)
        flat_features_for_policy = features_for_policy.reshape(num_envs_from_state * self.num_agents, features_for_policy.shape[-1])
        actions_flat, lp_flat, ent_flat = self.policy_net.get_actions(flat_features_for_policy, eval=eval)

        # Reshape policy outputs back to (NumEnvs, MaxConfiguredAgents, ...)
        log_probs = lp_flat.reshape(num_envs_from_state, self.num_agents)
        entropies = ent_flat.reshape(num_envs_from_state, self.num_agents)

        # Reshape actions to (NumEnvs, MaxConfiguredAgents, PerAgentActionDim)
        if self.action_space_type == "continuous":
            actions_per_agent_reshaped = actions_flat.reshape(num_envs_from_state, self.num_agents, self.total_action_dim_per_agent)
        else: # Discrete (multi-discrete or single discrete)
            # If single discrete and total_action_dim_per_agent is 1, actions_flat might be (N,)
            if actions_flat.dim() == 1 and self.total_action_dim_per_agent == 1: 
                actions_per_agent_reshaped = actions_flat.reshape(num_envs_from_state, self.num_agents, 1)
            else: # Handles multi-discrete where actions_flat is (N, num_branches)
                actions_per_agent_reshaped = actions_flat.reshape(num_envs_from_state, self.num_agents, self.total_action_dim_per_agent)
        
        # Flatten the agent and per-agent action dimensions for SharedMemoryInterface compatibility
        # Resulting shape: (NumEnvs, MaxConfiguredAgents * TotalPerAgentActionDim)
        actions_out_final_shape = actions_per_agent_reshaped.reshape(num_envs_from_state, -1)

        # --- Optional State Recording ---
        if record and self.state_recorder is not None and 'central' in states:
            # Record central state of the first environment in the batch
            if states['central'].ndim >= 2 and states['central'].shape[0] == 1 and states['central'].shape[1] > 0 : 
                c_state_to_record = states["central"][0,0].cpu().numpy() # Assuming s_dim is 1, take first env
                self.state_recorder.record_frame(c_state_to_record.flatten())
        
        return actions_out_final_shape, (log_probs, entropies), next_hidden_state_to_return


    def update(self,
               padded_states_dict_seq: Dict[str, torch.Tensor],       # (B, MaxSeqLen, NA, ObsDim) or (B, MaxSeqLen, CentralObsDim)
               padded_next_states_dict_seq: Dict[str, torch.Tensor], # (B, MaxSeqLen, NA, ObsDim) or (B, MaxSeqLen, CentralObsDim)
               padded_actions_seq: torch.Tensor,                     # (B, MaxSeqLen, NA, ActionDimFlat)
               padded_rewards_seq: torch.Tensor,                     # (B, MaxSeqLen, 1)
               padded_dones_seq: torch.Tensor,                       # (B, MaxSeqLen, 1)
               padded_truncs_seq: torch.Tensor,                      # (B, MaxSeqLen, 1)
               initial_hidden_states_batch: Optional[torch.Tensor],  # (B, NA, NumMemoryLayers, MemoryHiddenSize)
               attention_mask_batch: Optional[torch.Tensor]):        # (B, MaxSeqLen)
        
        # B: num_sequences (segments), MaxSeqLen: sequence length, NA: self.num_agents (max configured)
        B, MaxSeqLen, _, _ = self._get_batch_shape_info(padded_states_dict_seq, context="update")
        NA = self.num_agents # Use max_agents (self.num_agents) for network architecture consistency
        
        # EmbDim: Output dimension of the embedding network (input to memory module or heads)
        EmbDim = self.embedding_network.base_encoder.embed_dim 
       
        # --- Reward Normalization (Optional) ---
        # Normalize rewards using only valid steps indicated by attention_mask_batch
        valid_rewards_mask = attention_mask_batch.unsqueeze(-1).expand_as(padded_rewards_seq).bool()

        rewards_raw_mean = 0.0 # Initialize to a default
        if valid_rewards_mask.sum() > 0: # Ensure there are valid rewards before calculating mean
            rewards_raw_mean = padded_rewards_seq[valid_rewards_mask].mean().item()

        if self.rewards_normalizer:
            # Extract valid rewards; this creates a 1D tensor of shape (M,)
            # where M is the number of True elements in valid_rewards_mask.
            valid_rewards_flat = padded_rewards_seq[valid_rewards_mask]

            if valid_rewards_flat.numel() > 0: 
                # Reshape valid_rewards_flat to (M, 1) to treat rewards as a single feature
                # before passing to the normalizer.
                valid_rewards_for_norm_update = valid_rewards_flat.unsqueeze(-1)
                
                # Update the normalizer's statistics with the (M, 1) shaped tensor
                self.rewards_normalizer.update(valid_rewards_for_norm_update)
                
                # Normalize these valid rewards. Input is (M, 1), output will also be (M, 1).
                normalized_valid_rewards = self.rewards_normalizer.normalize(valid_rewards_for_norm_update)

                # Create a temporary tensor to place normalized rewards back into the original sequence structure.
                norm_rewards_seq = torch.zeros_like(padded_rewards_seq)
                
                # Assign the normalized rewards back.
                # normalized_valid_rewards is (M, 1), squeeze it to (M,) for assignment
                # to the locations indicated by valid_rewards_mask (which effectively flattens those locations).
                norm_rewards_seq[valid_rewards_mask] = normalized_valid_rewards.squeeze(-1).to(norm_rewards_seq.dtype)
                
                # Update padded_rewards_seq to use the normalized rewards for GAE calculation
                padded_rewards_seq = norm_rewards_seq 

        # Calculate the mean of normalized rewards for logging
        rewards_norm_mean = 0.0 # Initialize to a default
        if valid_rewards_mask.sum() > 0: # Ensure there are valid rewards
            rewards_norm_mean = padded_rewards_seq[valid_rewards_mask].mean().item()

        # --- Intrinsic Rewards (RND) Section (if enabled) ---
        all_intrinsic_rewards_seq = torch.zeros_like(padded_rewards_seq) # Initialize intrinsic rewards

        # ... (rest of the 'with torch.no_grad():' block for gathering old values, including RND calculation) ...
        # The combined_rewards_seq will use the potentially normalized padded_rewards_seq


        # --- NEW RND: Prepare placeholder for intrinsic rewards ---
        # This tensor will be populated during the no_grad forward pass if RND is enabled.
        all_intrinsic_rewards_seq = torch.zeros_like(padded_rewards_seq) # Shape: (B, MaxSeqLen, 1)

        # --- Data Gathering (old policy/values) with no_grad, using sub-batches of sequences for memory efficiency ---
        with torch.no_grad():
            all_old_log_probs_seq_parts, all_old_values_seq_parts = [], []
            all_old_baselines_seq_parts, all_next_values_seq_parts = [], []
            # NEW RND: List to store intrinsic rewards calculated for each sub-batch
            all_intrinsic_rewards_sub_batch_parts = [] 
            num_sequences_total = B 

            # Loop over the batch of B sequences in sub-batches of size self.no_grad_forward_batch_size
            for i in range(0, num_sequences_total, self.no_grad_forward_batch_size):
                start_idx, end_idx = i, min(i + self.no_grad_forward_batch_size, num_sequences_total)
                current_sub_batch_size = end_idx - start_idx
                if current_sub_batch_size == 0: continue

                # Slice all sequence data for the current sub-batch
                sub_batch_states_dict_seq = {k: v[start_idx:end_idx] for k, v in padded_states_dict_seq.items() if v is not None}
                sub_batch_next_states_dict_seq = {k: v[start_idx:end_idx] for k, v in padded_next_states_dict_seq.items() if v is not None}
                sub_batch_actions_seq = padded_actions_seq[start_idx:end_idx]
                sub_batch_initial_hidden = initial_hidden_states_batch[start_idx:end_idx] if self.enable_memory and initial_hidden_states_batch is not None else None

                # Get base embeddings for current and next states for the sub-batch
                sub_base_embeddings_seq, _ = self.embedding_network.get_base_embedding(sub_batch_states_dict_seq)
                sub_next_base_embeddings_seq, _ = self.embedding_network.get_base_embedding(sub_batch_next_states_dict_seq)
                
                # Initialize features (will be memory module output if enabled)
                sub_processed_features_seq = sub_base_embeddings_seq
                sub_next_processed_features_seq = sub_next_base_embeddings_seq
                
                # Process sub-batch through memory module if enabled
                if self.enable_memory and self.memory_module is not None:
                    if sub_batch_initial_hidden is None: # Should ideally not happen if Runner provides it
                        sub_batch_initial_hidden = torch.zeros(current_sub_batch_size, NA, self.memory_num_layers, self.memory_hidden_size, device=self.device)
                    
                    # Reshape for memory_module.forward_sequence
                    memory_input_sub_seq = sub_base_embeddings_seq.reshape(current_sub_batch_size * NA, MaxSeqLen, EmbDim)
                    initial_h_for_memory_sub = sub_batch_initial_hidden.reshape(current_sub_batch_size * NA, self.memory_num_layers, self.memory_hidden_size).permute(1, 0, 2).contiguous()
                    sub_processed_features_flat_seq, _ = self.memory_module.forward_sequence(memory_input_sub_seq, initial_h_for_memory_sub)
                    sub_processed_features_seq = sub_processed_features_flat_seq.reshape(current_sub_batch_size, MaxSeqLen, NA, self.memory_hidden_size)

                    memory_input_next_sub_seq = sub_next_base_embeddings_seq.reshape(current_sub_batch_size * NA, MaxSeqLen, EmbDim)
                    # Re-using initial_h_for_memory_sub for next_states is an approximation if next_states should use h_final from current states.
                    # However, for PPO, next_value is often calculated from s_next independently.
                    sub_next_processed_features_flat_seq, _ = self.memory_module.forward_sequence(memory_input_next_sub_seq, initial_h_for_memory_sub) 
                    sub_next_processed_features_seq = sub_next_processed_features_flat_seq.reshape(current_sub_batch_size, MaxSeqLen, NA, self.memory_hidden_size)

                # --- NEW RND: Calculate intrinsic reward for this sub-batch ---
                if self.enable_rnd and self.rnd_target_network and self.rnd_predictor_network:
                    # RND input is sub_processed_features_seq (output of GRU or embedding if GRU disabled)
                    # Shape: (current_sub_batch_size, MaxSeqLen, NA, self.feature_dim_for_heads)
                    flat_features_for_rnd = sub_processed_features_seq.reshape(-1, self.feature_dim_for_heads)
                    
                    target_rnd_feats = self.rnd_target_network(flat_features_for_rnd)
                    # This predictor pass is only for generating the intrinsic reward, not for training the predictor yet.
                    predicted_rnd_feats_for_reward = self.rnd_predictor_network(flat_features_for_rnd) 
                    
                    # MSE loss terms, mean over the RND feature dimension
                    intrinsic_r_terms = F.mse_loss(predicted_rnd_feats_for_reward, target_rnd_feats, reduction='none').mean(dim=-1)
                    # Reshape to (current_sub_batch_size, MaxSeqLen, NA)
                    intrinsic_r_sub_batch_per_agent = intrinsic_r_terms.reshape(
                        current_sub_batch_size, MaxSeqLen, NA
                    )
                    # Average intrinsic reward across agents for a per-step, per-sequence-in-batch reward
                    # Result shape: (current_sub_batch_size, MaxSeqLen, 1)
                    all_intrinsic_rewards_sub_batch_parts.append(intrinsic_r_sub_batch_per_agent.mean(dim=-1, keepdim=True))
                else:
                    # If RND disabled, append zeros of the correct shape to maintain list consistency
                    zeros_for_intrinsic_r = torch.zeros(current_sub_batch_size, MaxSeqLen, 1, device=self.device)
                    all_intrinsic_rewards_sub_batch_parts.append(zeros_for_intrinsic_r)
                # --- End NEW RND intrinsic reward calculation for sub-batch ---

                # Calculate old log_probs, values, baselines for the sub-batch using processed features
                sub_old_log_probs_seq, _ = self._recompute_log_probs_from_features(sub_processed_features_seq, sub_batch_actions_seq)
                sub_old_values_seq = self.shared_critic.values(sub_processed_features_seq) # (SubBatch, MaxSeqLen, 1)
                sub_baseline_input_seq = self.embedding_network.get_baseline_embeddings(sub_processed_features_seq, sub_batch_actions_seq)
                sub_old_baselines_seq = self.shared_critic.baselines(sub_baseline_input_seq) # (SubBatch, MaxSeqLen, NA, 1)
                sub_next_values_seq = self.shared_critic.values(sub_next_processed_features_seq) # (SubBatch, MaxSeqLen, 1)

                all_old_log_probs_seq_parts.append(sub_old_log_probs_seq)
                all_old_values_seq_parts.append(sub_old_values_seq)
                all_old_baselines_seq_parts.append(sub_old_baselines_seq)
                all_next_values_seq_parts.append(sub_next_values_seq)
            
            # Concatenate results from all sub-batches to form full tensors
            old_log_probs_seq = torch.cat(all_old_log_probs_seq_parts, dim=0) # (B, MaxSeqLen, NA)
            old_values_seq = torch.cat(all_old_values_seq_parts, dim=0)       # (B, MaxSeqLen, 1)
            old_baselines_seq = torch.cat(all_old_baselines_seq_parts, dim=0)  # (B, MaxSeqLen, NA, 1)
            next_values_seq = torch.cat(all_next_values_seq_parts, dim=0)      # (B, MaxSeqLen, 1)
            
            # Concatenate all intrinsic rewards and normalize them
            all_intrinsic_rewards_seq = torch.cat(all_intrinsic_rewards_sub_batch_parts, dim=0) # (B, MaxSeqLen, 1)
            if self.enable_rnd and self.intrinsic_reward_normalizer:
                valid_intrinsic_mask = attention_mask_batch.unsqueeze(-1).expand_as(all_intrinsic_rewards_seq).bool()
                if valid_intrinsic_mask.sum() > 0:
                    valid_intrinsic_r_flat = all_intrinsic_rewards_seq[valid_intrinsic_mask] # Shape (M,)

                    if valid_intrinsic_r_flat.numel() > 0:
                        valid_intrinsic_r = valid_intrinsic_r_flat.unsqueeze(-1) # Shape (M, 1)

                        self.intrinsic_reward_normalizer.update(valid_intrinsic_r)
                        normalized_intrinsic_r_values = self.intrinsic_reward_normalizer.normalize(valid_intrinsic_r) # Shape (M,1)
                        
                        temp_norm_intrinsic_r = torch.zeros_like(all_intrinsic_rewards_seq) # Shape (B, MaxSeqLen, 1)
                        # Ensure dtypes match for masked assignment
                        temp_norm_intrinsic_r[valid_intrinsic_mask] = normalized_intrinsic_r_values.squeeze(-1).to(temp_norm_intrinsic_r.dtype) # <<< MODIFIED LINE
                        all_intrinsic_rewards_seq = temp_norm_intrinsic_r 

            # Add normalized intrinsic reward to extrinsic rewards before GAE
            # Ensure detach() is used for intrinsic rewards if they are not meant to propagate gradients back to RND predictor during GAE calculation
            combined_rewards_seq = padded_rewards_seq + self.intrinsic_reward_coeff * all_intrinsic_rewards_seq.detach()

            # Compute GAE returns using the (potentially augmented) combined_rewards_seq
            returns_seq = self._compute_gae_with_padding(
                combined_rewards_seq, 
                old_values_seq, next_values_seq, 
                padded_dones_seq, padded_truncs_seq, attention_mask_batch
            ) # Shape: (B, MaxSeqLen, 1)

        # --- PPO Update Loop (Mini-batches of sequences) ---
        num_sequences_in_batch = B 
        idxes = np.arange(num_sequences_in_batch) # Indices for shuffling sequences
        log_keys = ["Policy Loss", "Value Loss", "Baseline Loss", "Entropy", "Grad Norm",
                    "Advantage Raw Mean", "Advantage Raw Std", "Advantage Final Mean",
                    "Returns Mean", "Logprob Mean", "Logprob Min", "Logprob Max"]
        if self.enable_rnd: # NEW RND: Add RND related log keys
            log_keys.append("RND Predictor Loss")
            log_keys.append("Intrinsic Reward Mean (Normalized)") # Log the normalized intrinsic reward

        batch_logs = {k: [] for k in log_keys} # Initialize log dictionary

        for epoch in range(self.epochs):
            np.random.shuffle(idxes) # Shuffle sequences for each epoch
            for mb_start_idx in range(0, num_sequences_in_batch, self.mini_batch_size):
                mb_end_idx = min(mb_start_idx + self.mini_batch_size, num_sequences_in_batch)
                mb_indices = idxes[mb_start_idx:mb_end_idx] 
                MB_NUM_SEQ = len(mb_indices) 

                if MB_NUM_SEQ == 0: continue 

                # Slice data for the current mini-batch of sequences
                st_mb_seq_dict = self._slice_state_dict_batch(padded_states_dict_seq, mb_indices)
                act_mb_seq = padded_actions_seq[mb_indices].contiguous()
                init_h_mb = initial_hidden_states_batch[mb_indices].contiguous() if self.enable_memory and initial_hidden_states_batch is not None else None
                mask_mb = attention_mask_batch[mb_indices].contiguous() # Attention mask for this mini-batch
                
                olp_mb_seq = old_log_probs_seq[mb_indices].contiguous()
                oval_mb_seq = old_values_seq[mb_indices].contiguous()
                obase_mb_seq = old_baselines_seq[mb_indices].contiguous()
                ret_mb_seq = returns_seq[mb_indices].contiguous()

                # --- Forward Pass with Current Policy/Value Parameters for the Mini-batch ---
                base_emb_mb_seq, _ = self.embedding_network.get_base_embedding(st_mb_seq_dict)
                processed_features_mb_seq = base_emb_mb_seq # Default if no memory
                
                if self.enable_memory and self.memory_module is not None and init_h_mb is not None:
                    memory_input_mb_seq = base_emb_mb_seq.reshape(MB_NUM_SEQ * NA, MaxSeqLen, EmbDim)
                    initial_h_for_memory_mb = init_h_mb.reshape(MB_NUM_SEQ * NA, self.memory_num_layers, self.memory_hidden_size).permute(1, 0, 2).contiguous()
                    processed_features_mb_flat_seq, _ = self.memory_module.forward_sequence(memory_input_mb_seq, initial_h_for_memory_mb)
                    processed_features_mb_seq = processed_features_mb_flat_seq.reshape(MB_NUM_SEQ, MaxSeqLen, NA, self.memory_hidden_size)

                # Recompute log_probs and entropy with current policy using processed features
                new_lp_mb_seq, ent_mb_seq = self._recompute_log_probs_from_features(processed_features_mb_seq, act_mb_seq)
                
                # Compute new value and baseline estimates using processed features
                new_vals_mb_seq = self.shared_critic.values(processed_features_mb_seq)
                baseline_input_mb_seq = self.embedding_network.get_baseline_embeddings(processed_features_mb_seq, act_mb_seq)
                new_base_mb_seq = self.shared_critic.baselines(baseline_input_mb_seq)
                
                # Log stats for new log_probs (only for valid steps in this mini-batch)
                lp_detached = new_lp_mb_seq.detach()
                valid_lp_mask = mask_mb.unsqueeze(-1).expand_as(lp_detached).bool() # Mask for valid log_probs/entropy
                if valid_lp_mask.sum() > 0: 
                    batch_logs["Logprob Mean"].append(lp_detached[valid_lp_mask].mean().item())
                    batch_logs["Logprob Min"].append(lp_detached[valid_lp_mask].min().item())
                    batch_logs["Logprob Max"].append(lp_detached[valid_lp_mask].max().item())

                # --- Advantage Calculation (Per Agent, Per Step for the mini-batch) ---
                adv_mb_seq = ret_mb_seq.unsqueeze(2).expand_as(new_base_mb_seq) - new_base_mb_seq 
                adv_mb_squeezed_seq = adv_mb_seq.squeeze(-1) # Shape: (MB_NUM_SEQ, MaxSeqLen, NA)

                # Normalize advantages using only valid steps within this mini-batch
                valid_adv_mask = mask_mb.unsqueeze(-1).expand_as(adv_mb_squeezed_seq).bool()
                if valid_adv_mask.sum() > 0:
                    valid_adv_elements = adv_mb_squeezed_seq[valid_adv_mask]
                    batch_logs["Advantage Raw Mean"].append(valid_adv_elements.mean().item())
                    batch_logs["Advantage Raw Std"].append(valid_adv_elements.std().item())
                    if self.normalize_adv:
                        normalized_valid_adv = (valid_adv_elements - valid_adv_elements.mean()) / (valid_adv_elements.std() + 1e-8)
                        temp_adv_norm = torch.zeros_like(adv_mb_squeezed_seq) # Create temp tensor for masked assignment
                        temp_adv_norm[valid_adv_mask] = normalized_valid_adv
                        adv_mb_squeezed_seq = temp_adv_norm # Use the tensor with normalized values only at valid spots
                    batch_logs["Advantage Final Mean"].append(adv_mb_squeezed_seq[valid_adv_mask].mean().item())
                else: # Handle case where no valid advantages exist in the mini-batch
                    batch_logs["Advantage Raw Mean"].append(0.0); batch_logs["Advantage Raw Std"].append(0.0); batch_logs["Advantage Final Mean"].append(0.0)
                detached_adv_seq = adv_mb_squeezed_seq.detach() # Detach advantages for policy loss

                # --- Loss Calculation with Masking for the Mini-batch ---
                mask_mb_policy_entropy = mask_mb.unsqueeze(-1).expand_as(new_lp_mb_seq)
                mask_mb_value = mask_mb.unsqueeze(-1).expand_as(new_vals_mb_seq)
                mask_mb_baseline = mask_mb.unsqueeze(-1).unsqueeze(-1).expand_as(new_base_mb_seq)
                num_valid_policy_entropy = mask_mb_policy_entropy.sum().clamp(min=1)
                num_valid_value = mask_mb_value.sum().clamp(min=1)
                num_valid_baseline = mask_mb_baseline.sum().clamp(min=1)

                pol_loss_terms = self._ppo_clip_loss(new_lp_mb_seq, olp_mb_seq, detached_adv_seq, self.ppo_clip_range, reduction='none')
                pol_loss = (pol_loss_terms * mask_mb_policy_entropy).sum() / num_valid_policy_entropy
                
                ret_mb_for_value = ret_mb_seq # Target for V(s) is GAE return
                if self.clipped_value_loss:
                    value_loss_terms = self._clipped_value_loss(oval_mb_seq, new_vals_mb_seq, ret_mb_for_value, self.value_clip_range, reduction='none')
                else:
                    value_loss_terms = F.mse_loss(new_vals_mb_seq, ret_mb_for_value, reduction='none')
                val_loss = (value_loss_terms * mask_mb_value).sum() / num_valid_value

                ret_mb_exp_baseline = ret_mb_seq.unsqueeze(2).expand_as(new_base_mb_seq) # Target for B(s,a)
                if self.clipped_value_loss:
                    baseline_loss_terms = self._clipped_value_loss(obase_mb_seq, new_base_mb_seq, ret_mb_exp_baseline, self.value_clip_range, reduction='none')
                else:
                    baseline_loss_terms = F.mse_loss(new_base_mb_seq, ret_mb_exp_baseline, reduction='none')
                base_loss = (baseline_loss_terms * mask_mb_baseline).sum() / num_valid_baseline
                
                entropy_terms = -ent_mb_seq # Maximize entropy = Minimize negative entropy
                ent_loss = (entropy_terms * mask_mb_policy_entropy).sum() / num_valid_policy_entropy
                
                total_loss = (pol_loss +
                              self.value_loss_coeff * val_loss +
                              self.baseline_loss_coeff * base_loss +
                              self.entropy_coeff * ent_loss)

                # --- NEW RND: Predictor Loss Calculation for this mini-batch ---
                rnd_predictor_loss = torch.tensor(0.0, device=self.device)
                if self.enable_rnd and self.rnd_target_network and self.rnd_predictor_network:
                    # RND input is processed_features_mb_seq
                    # Shape: (MB_NUM_SEQ, MaxSeqLen, NA, self.feature_dim_for_heads)
                    flat_features_for_rnd_update_mb = processed_features_mb_seq.reshape(-1, self.feature_dim_for_heads)
                    # Create a mask for valid steps in the flattened features
                    # Mask_mb is (MB_NUM_SEQ, MaxSeqLen). Expand for NA dimension then flatten.
                    expanded_mask_for_na = mask_mb.unsqueeze(-1).expand(-1, -1, NA) # (MB_NUM_SEQ, MaxSeqLen, NA)
                    flat_mask_mb_rnd = expanded_mask_for_na.reshape(-1) # (MB_NUM_SEQ * MaxSeqLen * NA)
                    
                    valid_indices_rnd = flat_mask_mb_rnd.nonzero(as_tuple=False).squeeze(-1) # Get indices of valid steps
                    
                    if valid_indices_rnd.numel() > 0: # If there are any valid steps
                        num_rnd_update_samples = int(valid_indices_rnd.numel() * self.rnd_update_proportion)
                        if num_rnd_update_samples > 0:
                            # Randomly select a proportion of the valid steps for RND update
                            perm_rnd = torch.randperm(valid_indices_rnd.numel(), device=self.device)[:num_rnd_update_samples]
                            rnd_mb_update_indices = valid_indices_rnd[perm_rnd]
                            
                            features_for_rnd_predictor_update = flat_features_for_rnd_update_mb[rnd_mb_update_indices]

                            if features_for_rnd_predictor_update.numel() > 0: # Ensure some samples were selected
                                with torch.no_grad(): # Target network is fixed, no gradients needed
                                    target_rnd_feats_update = self.rnd_target_network(features_for_rnd_predictor_update)
                                predicted_rnd_feats_update = self.rnd_predictor_network(features_for_rnd_predictor_update)
                                rnd_predictor_loss = F.mse_loss(predicted_rnd_feats_update, target_rnd_feats_update)
                                total_loss = total_loss + rnd_predictor_loss # Add RND predictor loss to the main loss
                # --- End NEW RND Predictor Loss ---

                self.optimizer.zero_grad()
                total_loss.backward()
                gn = clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logging for this mini-batch
                batch_logs["Policy Loss"].append(pol_loss.item())
                batch_logs["Value Loss"].append(val_loss.item())
                batch_logs["Baseline Loss"].append(base_loss.item())
                if valid_lp_mask.sum() > 0: batch_logs["Entropy"].append(ent_mb_seq[valid_lp_mask].mean().item())
                else: batch_logs["Entropy"].append(0.0)
                batch_logs["Grad Norm"].append(gn.item() if torch.is_tensor(gn) else gn) # Handle if gn is not tensor (e.g. float)
                
                valid_returns_mask_mb = mask_mb.unsqueeze(-1).expand_as(ret_mb_seq).bool()
                if valid_returns_mask_mb.sum() > 0 : batch_logs["Returns Mean"].append(ret_mb_seq[valid_returns_mask_mb].mean().item())
                else: batch_logs["Returns Mean"].append(0.0)
                
                if self.enable_rnd: # NEW RND Logging
                    batch_logs["RND Predictor Loss"].append(rnd_predictor_loss.item())
                    # For intrinsic reward mean, use the normalized values from the start of the update cycle,
                    # masked for this mini-batch's valid steps.
                    intrinsic_rewards_for_mb_log = all_intrinsic_rewards_seq[mb_indices] # (MB_NUM_SEQ, MaxSeqLen, 1)
                    valid_intrinsic_mb_mask = mask_mb.unsqueeze(-1).expand_as(intrinsic_rewards_for_mb_log).bool()
                    if valid_intrinsic_mb_mask.sum() > 0:
                         batch_logs["Intrinsic Reward Mean (Normalized)"].append(intrinsic_rewards_for_mb_log[valid_intrinsic_mb_mask].mean().item())
                    else:
                         batch_logs["Intrinsic Reward Mean (Normalized)"].append(0.0)

            # --- End of Mini-batch Loop for one epoch ---

            # --- Scheduler Steps (End of Epoch) ---
            current_lr = self._get_avg_lr(self.optimizer)
            if self.lr_scheduler: self.lr_scheduler.step()
            if self.entropy_scheduler: self.entropy_coeff = self.entropy_scheduler.step()
            if self.policy_clip_scheduler: self.ppo_clip_range = self.policy_clip_scheduler.step()
            if self.value_clip_scheduler: self.value_clip_range = self.value_clip_scheduler.step()
            if self.max_grad_norm_scheduler: self.max_grad_norm = self.max_grad_norm_scheduler.step()
        # --- End of Epoch Loop ---

        # --- Final Logging (Averaged over all mini-batches for this update cycle) ---
        final_logs = {k: np.mean(v) if v else 0.0 for k, v in batch_logs.items()} # Calculate mean of collected log values
        final_logs["Entropy coeff"] = self.entropy_coeff
        final_logs["Learning Rate"] = current_lr
        final_logs["PPO Clip Range"] = self.ppo_clip_range
        final_logs["Max Grad Norm"] = self.max_grad_norm
        final_logs["Raw Reward Mean"] = rewards_raw_mean # Extrinsic reward mean before normalization
        final_logs["Norm Reward Mean"] = rewards_norm_mean # Extrinsic reward mean after normalization
        if self.enable_rnd and "Intrinsic Reward Mean (Normalized)" not in final_logs: # Ensure key exists if RND was enabled but no updates happened
             final_logs["Intrinsic Reward Mean (Normalized)"] = 0.0 # Default if no valid intrinsic rewards were logged

        return final_logs

    def _compute_gae_with_padding(self, rewards_seq, values_seq, next_values_seq,
                                  dones_seq, truncs_seq, attention_mask):
        # Computes Generalized Advantage Estimation (GAE) respecting padded sequences.
        # All input sequences (rewards, values, next_values, dones, truncs) are expected
        # to have shape (B, MaxSeqLen, 1), where B is the number of sequences (segments)
        # in the batch, and MaxSeqLen is the padded sequence length.
        # attention_mask has shape (B, MaxSeqLen), with 1.0 for valid steps and 0.0 for padded steps.

        B, MaxSeqLen, _ = rewards_seq.shape
        returns_seq = torch.zeros_like(rewards_seq)  # Will store target returns (GAE + V(s_t))
        gae = torch.zeros(B, 1, device=rewards_seq.device)  # GAE accumulator, reset for each sequence in batch implicitly by loop start

        # Convert done flags to float for masking operations.
        # The 'dones_seq' indicates true terminal states from the environment.
        dones_float_seq = dones_seq.float()
        # 'truncs_seq' indicates episode termination due to timeout/MaxSteps.
        # It's not directly used in the modified bootstrap logic here but is an input.
        # Its primary role is handled by the attention_mask and by the runner in resetting envs.

        # Iterate backwards through the sequence timesteps
        for t in reversed(range(MaxSeqLen)):
            # Mask for the current timestep t: 1.0 if valid, 0.0 if padded.
            # Sliced as [:, t:t+1] to keep the dimension for broadcasting: (B, 1)
            current_step_is_valid_mask_t = attention_mask[:, t:t+1]

            # Mask for the next timestep t+1: 1.0 if valid, 0.0 if padded or beyond MaxSeqLen.
            # If t is the last step (MaxSeqLen - 1), there's no valid next step in the sequence.
            if t < MaxSeqLen - 1:
                next_step_is_valid_in_sequence_mask_t_plus_1 = attention_mask[:, t+1:t+2]
            else:
                next_step_is_valid_in_sequence_mask_t_plus_1 = torch.zeros_like(current_step_is_valid_mask_t)

            # Determine if we should bootstrap the value of the next state V(s_{t+1}).
            # We bootstrap if:
            #   1. The current state s_t is NOT a true terminal state (dones_float_seq[:, t] is 0).
            #   2. The next state s_{t+1} is a valid (non-padded) step within the current sequence.
            # This correctly handles pure truncations (dones=0, truncs=1) by allowing bootstrapping.
            # It also correctly handles true terminal states (dones=1) by not bootstrapping.
            can_bootstrap_from_next_value = (1.0 - dones_float_seq[:, t]) * next_step_is_valid_in_sequence_mask_t_plus_1

            # Get V(s_{t+1}), scaled by can_bootstrap_from_next_value.
            # If cannot bootstrap (e.g., s_t is terminal or s_{t+1} is padded), v_next_bootstrapped becomes 0.
            # values_seq[:, t] is V(s_t) and next_values_seq[:, t] is V(s_{t+1}) for the current step t.
            v_next_bootstrapped = next_values_seq[:, t] * can_bootstrap_from_next_value

            # Calculate TD error (delta_t) for the current step t.
            # delta_t = r_t + gamma * V(s_{t+1})_bootstrapped - V(s_t)
            # Multiply by current_step_is_valid_mask_t so delta is 0 for padded steps.
            delta = (rewards_seq[:, t] + self.gamma * v_next_bootstrapped - values_seq[:, t]) * current_step_is_valid_mask_t

            # Calculate GAE (A_t).
            # A_t = delta_t + gamma * lambda * A_{t+1} * (mask_for_gae_propagation)
            # GAE should propagate from A_{t+1} if s_t was not a terminal state and s_{t+1} was valid.
            # This uses the same condition as can_bootstrap_from_next_value.
            gae_prop_mask = (1.0 - dones_float_seq[:, t]) * next_step_is_valid_in_sequence_mask_t_plus_1
            gae = delta + self.gamma * self.lmbda * gae_prop_mask * gae

            # Calculate the target return R_t = A_t + V(s_t) for the value function.
            current_returns_t = gae + values_seq[:, t]

            # Store the calculated return for the current step t.
            # If current_step_is_valid_mask_t is 0 (padded step), store 0.
            returns_seq[:, t] = torch.where(current_step_is_valid_mask_t.bool(),
                                            current_returns_t,
                                            torch.zeros_like(current_returns_t))

            # If the current step itself was padded, reset gae to 0.
            # This prevents gae from a padded step (which would be 0 due to delta being 0)
            # from being incorrectly used by a previous valid step in the next iteration of the loop.
            # (Essentially, gae should be 0 if it's calculated based on a padded 'current_step_is_valid_mask_t').
            gae = gae * current_step_is_valid_mask_t

        return returns_seq
    
    def _ppo_clip_loss(self, new_lp, old_lp, advantages, clip_range, reduction='mean'):
        # new_lp, old_lp, advantages: (MiniBatchNumSeq, MaxSeqLen, NumAgents)
        log_ratio = new_lp - old_lp
        ratio = torch.exp(log_ratio.clamp(-10,10)) # Clamp log_ratio for numerical stability
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        # Per-element policy loss (negative because we want to maximize the PPO objective)
        loss_terms = -torch.min(surr1, surr2) # Shape: (MB, MaxSeqLen, NA)
        
        if reduction == 'mean': 
            return loss_terms.mean() # Note: Masking should be applied *before* this mean if called directly.
        elif reduction == 'none':
            return loss_terms # Return per-element losses for manual masking and averaging
        else:
            raise ValueError(f"MAPOCAAgent Error: Unsupported reduction type in _ppo_clip_loss: {reduction}")

    def _clipped_value_loss(self, old_values, new_values, targets, clip_range, reduction='mean'):
        # old_values, new_values, targets could be (MB, MaxSeqLen, 1) for V(s) or (MB, MaxSeqLen, NA, 1) for B(s,a)
        # Clip new_values around old_values
        values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
        # MSE for unclipped and clipped values
        mse_unclipped = F.mse_loss(new_values, targets, reduction='none')
        mse_clipped = F.mse_loss(values_clipped, targets, reduction='none')
        # Per-element clipped value loss is the max of the two MSEs
        loss_terms = torch.max(mse_unclipped, mse_clipped)
        
        if reduction == 'mean':
            return loss_terms.mean() # Note: Masking should be applied *before* this mean.
        elif reduction == 'none':
            return loss_terms # Return per-element losses for manual masking and averaging
        else:
            raise ValueError(f"MAPOCAAgent Error: Unsupported reduction type in _clipped_value_loss: {reduction}")

    def _recompute_log_probs_from_features(self, processed_features: torch.Tensor, actions: torch.Tensor):
        # processed_features shape: (B_or_MB, MaxSeqLen, NA, FeatureDimForHeads)
        # actions shape: (B_or_MB, MaxSeqLen, NA, ActionDimFlatForAgent)
        input_shape = processed_features.shape
        FeatureDim = input_shape[-1]
        NA = input_shape[-2] # Should be self.num_agents (MaxConfiguredAgents)
        leading_dims = input_shape[:-2] # (B_or_MB, MaxSeqLen)
        
        # Flatten features for policy network input: (B_or_MB * MaxSeqLen * NA, FeatureDimForHeads)
        flat_features = processed_features.reshape(-1, FeatureDim)

        # Flatten actions to match flat_features: (B_or_MB * MaxSeqLen * NA, ActionDimPerAgent)
        # The 'actions' tensor from Runner is (B, MaxSeqLen, NA, ActionDimFlatForAgent)
        act_flat = actions.reshape(-1, actions.shape[-1]) # Assumes last dim is ActionDimFlatForAgent
        
        # Get log_probs and entropy from the policy network using the current parameters
        lp_flat, ent_flat = self.policy_net.recompute_log_probs(flat_features, act_flat)

        # Reshape results back to (B_or_MB, MaxSeqLen, NA)
        log_probs = lp_flat.reshape(*leading_dims, NA)
        entropies = ent_flat.reshape(*leading_dims, NA)
        return log_probs, entropies

    def _slice_state_dict_batch(self, states_dict_seq: dict, mb_indices: np.ndarray):
        # Slices the first dimension (batch of sequences) of each tensor in the state dictionary.
        new_dict = {}
        for k, v_seq in states_dict_seq.items(): # v_seq is (B, MaxSeqLen, ...)
            if v_seq is not None:
                new_dict[k] = v_seq[mb_indices].contiguous()
            else:
                new_dict[k] = None
        return new_dict

    def _get_avg_lr(self, optimizer) -> float:
        # Utility to get current average learning rate from optimizer param groups.
        lr_sum, count = 0.0, 0
        for pg in optimizer.param_groups:
            lr_sum += pg.get('lr', 0.0) 
            count += 1
        return lr_sum / max(count, 1) # Avoid division by zero if no param_groups

    def _accumulate_per_layer_grad_norms(self, grad_norms_sum_per_layer: dict):
        # Accumulates L2 norm of gradients for each named parameter (for logging/debugging).
        if not hasattr(self, '_param_names_with_grads'):
            # Cache the names of parameters that require gradients to avoid repeated checks.
            self._param_names_with_grads = {name for name, param in self.named_parameters() if param.requires_grad}
        
        for name, param in self.named_parameters():
            # Check if the parameter requires grad and has a computed gradient.
            if name in self._param_names_with_grads and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item() # Calculate L2 norm of the gradient.
                # Accumulate the norm, initializing if key doesn't exist in the tracking dict.
                grad_norms_sum_per_layer[name] = grad_norms_sum_per_layer.get(name, 0.0) + grad_norm