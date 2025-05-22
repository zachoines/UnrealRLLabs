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
    RNDTargetNetwork, 
    RNDPredictorNetwork
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
    MODIFIED: Now uses separate base embedding and memory networks for policy, value, and baseline.
    MODIFIED: Now uses separate optimizers for policy, value, baseline, and RND predictor.
    """

    def __init__(self, config: Dict, device: torch.device):
        super().__init__(config, device)

        # Parse configuration dictionaries
        agent_cfg = config["agent"]["params"]
        train_cfg = config["train"]
        shape_cfg = config["environment"]["shape"]
        action_cfg = shape_cfg["action"]
        rec_cfg = config.get("StateRecorder", None) 
        env_params_cfg = config["environment"]["params"]

        # --- Core PPO & GAE Hyperparameters ---
        self.gamma = agent_cfg.get("gamma", 0.99)
        self.lmbda = agent_cfg.get("lambda", 0.95)
        # General learning rate, used as fallback if specific LRs are not provided
        self.lr = agent_cfg.get("learning_rate", 3e-4) 
        self.value_loss_coeff = agent_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff = agent_cfg.get("baseline_loss_coeff", 0.5)
        self.entropy_coeff = agent_cfg.get("entropy_coeff", 0.01)
        self.max_grad_norm = agent_cfg.get("max_grad_norm", 0.5)
        self.normalize_adv = agent_cfg.get("normalize_advantages", True)
        self.ppo_clip_range = agent_cfg.get("ppo_clip_range", 0.2)
        self.clipped_value_loss = agent_cfg.get("clipped_value_loss", True)
        self.value_clip_range = agent_cfg.get("value_clip_range", 0.2)
        self.no_grad_forward_batch_size = agent_cfg.get("no_grad_forward_batch_size", train_cfg.get("mini_batch_size", 64))

        self.epochs = train_cfg.get("epochs", 4)
        self.mini_batch_size = train_cfg.get("mini_batch_size", 64)
        self.state_recorder = StateRecorder(rec_cfg) if rec_cfg else None
        self._determine_action_space(action_cfg)
        
        self.num_agents = env_params_cfg.get('MaxAgents', 1)
        if "agent" in shape_cfg["state"]:
            self.num_agents = shape_cfg["state"]["agent"].get("max", self.num_agents)

        self.enable_memory = agent_cfg.get('enable_memory', False)
        embedding_output_dim = agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]["cross_attention_feature_extractor"]["embed_dim"]
        
        embedding_network_config = agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        self.policy_embedding_network = MultiAgentEmbeddingNetwork(embedding_network_config).to(device)
        self.value_embedding_network = MultiAgentEmbeddingNetwork(embedding_network_config).to(device)
        self.baseline_embedding_network = MultiAgentEmbeddingNetwork(embedding_network_config).to(device)

        self.policy_memory_module: Optional[RecurrentMemoryNetwork] = None
        self.value_memory_module: Optional[RecurrentMemoryNetwork] = None
        self.baseline_memory_module: Optional[RecurrentMemoryNetwork] = None
        self.memory_hidden_size = 0
        self.memory_num_layers = 0

        if self.enable_memory:
            self.memory_type = agent_cfg.get('memory_type', 'gru')
            memory_module_config = agent_cfg.get(self.memory_type, {})
            if 'input_size' not in memory_module_config:
                memory_module_config['input_size'] = embedding_output_dim
            if memory_module_config['input_size'] != embedding_output_dim:
                 print(f"MAPOCAAgent Warning: Memory module input_size ({memory_module_config['input_size']}) "
                       f"differs from embedding_network's output_dim ({embedding_output_dim}).")
            self.memory_hidden_size = memory_module_config.get('hidden_size', 128)
            self.memory_num_layers = memory_module_config.get('num_layers', 1)
            if self.memory_type == 'gru':
                self.policy_memory_module = GRUSequenceMemory(**memory_module_config).to(device)
                self.value_memory_module = GRUSequenceMemory(**memory_module_config).to(device)
                self.baseline_memory_module = GRUSequenceMemory(**memory_module_config).to(device)
            else:
                raise ValueError(f"MAPOCAAgent Error: Unsupported memory_type '{self.memory_type}' specified in config.")
        
        self.feature_dim_for_heads = self.memory_hidden_size if self.enable_memory else embedding_output_dim
        self.shared_critic = SharedCritic(agent_cfg["networks"]["critic_network"]).to(device)
        
        policy_network_config = agent_cfg["networks"]["policy_network"].copy()
        policy_network_config["in_features"] = self.feature_dim_for_heads
        if self.action_space_type == "continuous":
            policy_network_config['out_features'] = self.action_dim
            self.policy_net = BetaPolicyNetwork(**policy_network_config).to(device)
        elif self.action_space_type == "discrete":
            policy_network_config['out_features'] = self.action_dim 
            self.policy_net = DiscretePolicyNetwork(**policy_network_config).to(device)
        else:
            raise ValueError(f"MAPOCAAgent Error: Unsupported action_space_type: {self.action_space_type}")

        self.lr_scheduler_cfg = agent_cfg.get("schedulers", {}).get("lr", None) # Main LR scheduler config
        ent_sched_cfg = agent_cfg.get("schedulers", {}).get("entropy_coeff", None)
        self.entropy_scheduler = LinearValueScheduler(**ent_sched_cfg) if ent_sched_cfg else None
        if self.entropy_scheduler: self.entropy_coeff = self.entropy_scheduler.start_value
        
        policy_clip_sched_cfg = agent_cfg.get("schedulers", {}).get("policy_clip", None)
        self.policy_clip_scheduler = LinearValueScheduler(**policy_clip_sched_cfg) if policy_clip_sched_cfg else None
        if self.policy_clip_scheduler: self.ppo_clip_range = self.policy_clip_scheduler.start_value
        
        value_clip_sched_cfg = agent_cfg.get("schedulers", {}).get("value_clip", None)
        self.value_clip_scheduler = LinearValueScheduler(**value_clip_sched_cfg) if value_clip_sched_cfg else None
        if self.value_clip_scheduler: self.value_clip_range = self.value_clip_scheduler.start_value
        
        grad_norm_sched_cfg = agent_cfg.get("schedulers", {}).get("max_grad_norm", None)
        self.max_grad_norm_scheduler = LinearValueScheduler(**grad_norm_sched_cfg) if grad_norm_sched_cfg else None
        if self.max_grad_norm_scheduler: self.max_grad_norm = self.max_grad_norm_scheduler.start_value

        rewards_normalization_cfg = agent_cfg.get('rewards_normalizer', None)
        self.rewards_normalizer = RunningMeanStdNormalizer(**rewards_normalization_cfg, device=self.device) if rewards_normalization_cfg else None

        self.enable_rnd = agent_cfg.get("enable_rnd", False)
        self.rnd_target_network: Optional[RNDTargetNetwork] = None
        self.rnd_predictor_network: Optional[RNDPredictorNetwork] = None
        self.intrinsic_reward_normalizer: Optional[RunningMeanStdNormalizer] = None
        self.intrinsic_reward_coeff = 0.0
        self.rnd_update_proportion = 0.25

        if self.enable_rnd:
            rnd_cfg = agent_cfg.get("rnd_params", {})
            rnd_input_dim = self.feature_dim_for_heads 
            rnd_output_dim = rnd_cfg.get("output_size", 128)
            rnd_hidden_size = rnd_cfg.get("hidden_size", 256)
            self.rnd_target_network = RNDTargetNetwork(rnd_input_dim, rnd_output_dim, rnd_hidden_size).to(self.device)
            self.rnd_predictor_network = RNDPredictorNetwork(rnd_input_dim, rnd_output_dim, rnd_hidden_size).to(self.device)
            self.intrinsic_reward_coeff = rnd_cfg.get("intrinsic_reward_coeff", 0.01)
            self.rnd_update_proportion = rnd_cfg.get("rnd_update_proportion", 0.25)
            self.intrinsic_reward_normalizer = RunningMeanStdNormalizer(epsilon=1e-8, device=self.device) 
            print(f"MAPOCAAgent: RND Module Initialized. Input Dim: {rnd_input_dim}, Output Dim: {rnd_output_dim}")

        # --- Initialize Separate Optimizers ---
        # Learning rates for each component (can be specified in config, otherwise defaults to self.lr)
        lr_policy = agent_cfg.get("lr_policy", self.lr)
        lr_value = agent_cfg.get("lr_value", self.lr)
        lr_baseline = agent_cfg.get("lr_baseline", self.lr)
        lr_rnd_predictor = agent_cfg.get("lr_rnd_predictor", self.lr) # For RND predictor

        # Policy pathway parameters
        policy_params = list(self.policy_embedding_network.parameters()) + \
                        list(self.policy_net.parameters())
        if self.enable_memory and self.policy_memory_module:
            policy_params += list(self.policy_memory_module.parameters())
        self.policy_optimizer = optim.Adam(policy_params, lr=lr_policy, eps=1e-7)

        # Value pathway parameters (Value Embedding, Value Memory, Critic's Value Head & Attention)
        value_params = list(self.value_embedding_network.parameters()) + \
                       list(self.shared_critic.value_head.parameters()) + \
                       list(self.shared_critic.value_attention.parameters())
        if self.enable_memory and self.value_memory_module:
            value_params += list(self.value_memory_module.parameters())
        self.value_optimizer = optim.Adam(value_params, lr=lr_value, eps=1e-7)

        # Baseline pathway parameters (Baseline Embedding, Baseline Memory, Critic's Baseline Head & Attention)
        baseline_params = list(self.baseline_embedding_network.parameters()) + \
                          list(self.shared_critic.baseline_head.parameters()) + \
                          list(self.shared_critic.baseline_attention.parameters())
        if self.enable_memory and self.baseline_memory_module:
            baseline_params += list(self.baseline_memory_module.parameters())
        self.baseline_optimizer = optim.Adam(baseline_params, lr=lr_baseline, eps=1e-7)
        
        # RND Predictor Optimizer (if RND is enabled)
        self.rnd_predictor_optimizer: Optional[optim.Adam] = None
        if self.enable_rnd and self.rnd_predictor_network:
            self.rnd_predictor_optimizer = optim.Adam(self.rnd_predictor_network.parameters(), lr=lr_rnd_predictor, eps=1e-7)

        # Main LR scheduler is now applied to the policy_optimizer
        self.policy_lr_scheduler: Optional[LinearLR] = None
        if self.lr_scheduler_cfg: 
            self.policy_lr_scheduler = LinearLR(self.policy_optimizer, **self.lr_scheduler_cfg)
        
        self.current_policy_memory_hidden_states: Optional[torch.Tensor] = None


    def parameters(self):
        # Collects ALL trainable parameters from all networks.
        # This is used for global gradient clipping if applied to all params at once.
        params_list = list(self.policy_embedding_network.parameters()) + \
                      list(self.value_embedding_network.parameters()) + \
                      list(self.baseline_embedding_network.parameters()) + \
                      list(self.shared_critic.parameters()) + \
                      list(self.policy_net.parameters())
        if self.enable_memory:
            if self.policy_memory_module: params_list += list(self.policy_memory_module.parameters())
            if self.value_memory_module: params_list += list(self.value_memory_module.parameters())
            if self.baseline_memory_module: params_list += list(self.baseline_memory_module.parameters())
        if self.enable_rnd and self.rnd_predictor_network is not None:
            params_list += list(self.rnd_predictor_network.parameters())
        return params_list

    def _determine_action_space(self, action_cfg: Dict):
        if "agent" in action_cfg:
            agent_action_cfg = action_cfg["agent"]
            if "discrete" in agent_action_cfg:
                self.action_space_type = "discrete"
                d_list = agent_action_cfg["discrete"]
                self.action_dim = [d["num_choices"] for d in d_list]
                self.total_action_dim_per_agent = len(self.action_dim)
            elif "continuous" in agent_action_cfg:
                self.action_space_type = "continuous"
                self.action_dim = len(agent_action_cfg["continuous"]) 
                self.total_action_dim_per_agent = self.action_dim
            else:
                raise ValueError("MAPOCAAgent Error: Missing 'discrete' or 'continuous' under 'action.agent' in config.")
        else:
            raise ValueError("MAPOCAAgent Error: No 'agent' block found in 'action' config.")

    def _get_batch_shape_info(self, states_input: Dict[str, torch.Tensor], context: str = "get_actions") -> Tuple[int, int, int, int]:
        dim0, dim1, num_actual_agents_in_tensor, obs_d = -1, -1, -1, -1
        primary_key = 'agent' if 'agent' in states_input and states_input['agent'] is not None else 'central'
        if primary_key not in states_input or states_input[primary_key] is None:
            raise ValueError(f"MAPOCAAgent Error: Cannot determine batch dimensions: '{primary_key}' key missing or None in states_input for {context}.")
        tensor_shape = states_input[primary_key].shape
        if primary_key == 'agent':
            if len(tensor_shape) < 3: 
                raise ValueError(f"MAPOCAAgent Error: Agent state tensor in {context} has {len(tensor_shape)} dims, expected at least 3. Shape: {tensor_shape}")
            if len(tensor_shape) == 3 and context == "get_actions": 
                dim0 = 1; dim1 = tensor_shape[0]; num_actual_agents_in_tensor = tensor_shape[1]
            elif len(tensor_shape) == 4 : 
                dim0 = tensor_shape[0]; dim1 = tensor_shape[1]; num_actual_agents_in_tensor = tensor_shape[2]
            else: raise ValueError(f"MAPOCAAgent Error: Unexpected agent state tensor shape in {context}: {tensor_shape}")
            obs_d = tensor_shape[-1]
        elif primary_key == 'central':
            if len(tensor_shape) < 2: raise ValueError(f"MAPOCAAgent Error: Central state tensor in {context} has {len(tensor_shape)} dims, expected at least 2. Shape: {tensor_shape}")
            if len(tensor_shape) == 2 and context == "get_actions": dim0 = 1; dim1 = tensor_shape[0]
            elif len(tensor_shape) == 3: dim0 = tensor_shape[0]; dim1 = tensor_shape[1]
            else: raise ValueError(f"MAPOCAAgent Error: Unexpected central state tensor shape in {context}: {tensor_shape}")
            num_actual_agents_in_tensor = 1; obs_d = tensor_shape[-1]
        return dim0, dim1, num_actual_agents_in_tensor, obs_d


    @torch.no_grad()
    def get_actions(self, states: dict, dones: torch.Tensor, truncs: torch.Tensor,
                    h_prev_batch: Optional[torch.Tensor] = None, eval: bool=False, record: bool=True) \
                    -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        s_dim, num_envs_from_state, _, _ = self._get_batch_shape_info(states, context="get_actions")
        
        base_embeddings, _ = self.policy_embedding_network.get_base_embedding(states)
        
        if base_embeddings.shape[0] == 1 and s_dim == 1:
            features_for_memory_input = base_embeddings.squeeze(0) 
        else: 
            features_for_memory_input = base_embeddings.reshape(-1, self.num_agents, base_embeddings.shape[-1])
            num_envs_from_state = features_for_memory_input.shape[0]

        embedding_dim = features_for_memory_input.shape[-1]
        features_for_policy_head = features_for_memory_input
        next_policy_hidden_state_to_return = h_prev_batch

        if self.enable_memory and self.policy_memory_module is not None:
            if h_prev_batch is None: 
                h_prev_batch = torch.zeros(num_envs_from_state, self.num_agents, self.memory_num_layers, self.memory_hidden_size, device=self.device)
            
            if h_prev_batch.shape[0] != num_envs_from_state or h_prev_batch.shape[1] != self.num_agents:
                 print(f"MAPOCAAgent Warning in get_actions: Policy h_prev_batch shape {h_prev_batch.shape} mismatch. Expected num_envs={num_envs_from_state}, num_agents={self.num_agents}. Adapting.")
                 h_prev_batch_adapted = torch.zeros(num_envs_from_state, self.num_agents, self.memory_num_layers, self.memory_hidden_size, device=self.device)
                 min_envs = min(h_prev_batch.shape[0], num_envs_from_state)
                 min_agents_dim = min(h_prev_batch.shape[1], self.num_agents)
                 h_prev_batch_adapted[:min_envs, :min_agents_dim, :, :] = h_prev_batch[:min_envs, :min_agents_dim, :, :]
                 h_prev_batch = h_prev_batch_adapted

            memory_input_features_flat = features_for_memory_input.reshape(num_envs_from_state * self.num_agents, embedding_dim)
            h_prev_for_policy_memory = h_prev_batch.reshape(num_envs_from_state * self.num_agents, self.memory_num_layers, self.memory_hidden_size).permute(1, 0, 2).contiguous()
            features_for_policy_head_flat, h_next_policy_flat = self.policy_memory_module.forward_step(memory_input_features_flat, h_prev_for_policy_memory)
            features_for_policy_head = features_for_policy_head_flat.reshape(num_envs_from_state, self.num_agents, self.memory_hidden_size)
            next_hidden_intermediate_policy = h_next_policy_flat.permute(1, 0, 2).reshape(num_envs_from_state, self.num_agents, self.memory_num_layers, self.memory_hidden_size)
            
            if dones.ndim == 1: dones = dones.unsqueeze(1).expand(-1, self.num_agents)
            if truncs.ndim == 1: truncs = truncs.unsqueeze(1).expand(-1, self.num_agents)
            if dones.shape[0] != num_envs_from_state: dones = dones[:num_envs_from_state, :] 
            if truncs.shape[0] != num_envs_from_state: truncs = truncs[:num_envs_from_state, :]
            reset_mask_bool = dones.bool() | truncs.bool()
            reset_mask = reset_mask_bool.unsqueeze(-1).unsqueeze(-1).expand_as(next_hidden_intermediate_policy).float()
            next_policy_hidden_state_to_return = next_hidden_intermediate_policy * (1.0 - reset_mask)
        
        flat_features_for_policy_head = features_for_policy_head.reshape(num_envs_from_state * self.num_agents, features_for_policy_head.shape[-1])
        actions_flat, lp_flat, ent_flat = self.policy_net.get_actions(flat_features_for_policy_head, eval=eval)

        log_probs = lp_flat.reshape(num_envs_from_state, self.num_agents)
        entropies = ent_flat.reshape(num_envs_from_state, self.num_agents)

        if self.action_space_type == "continuous":
            actions_per_agent_reshaped = actions_flat.reshape(num_envs_from_state, self.num_agents, self.total_action_dim_per_agent)
        else: 
            if actions_flat.dim() == 1 and self.total_action_dim_per_agent == 1: 
                actions_per_agent_reshaped = actions_flat.reshape(num_envs_from_state, self.num_agents, 1)
            else: 
                actions_per_agent_reshaped = actions_flat.reshape(num_envs_from_state, self.num_agents, self.total_action_dim_per_agent)
        
        actions_out_final_shape = actions_per_agent_reshaped.reshape(num_envs_from_state, -1)

        if record and self.state_recorder is not None and 'central' in states:
            if states['central'].ndim >= 2 and states['central'].shape[0] == 1 and states['central'].shape[1] > 0 : 
                c_state_to_record = states["central"][0,0].cpu().numpy()
                self.state_recorder.record_frame(c_state_to_record.flatten())
        
        return actions_out_final_shape, (log_probs, entropies), next_policy_hidden_state_to_return

    def update(self,
               padded_states_dict_seq: Dict[str, torch.Tensor],
               padded_next_states_dict_seq: Dict[str, torch.Tensor],
               padded_actions_seq: torch.Tensor,
               padded_rewards_seq: torch.Tensor,
               padded_dones_seq: torch.Tensor,
               padded_truncs_seq: torch.Tensor,
               initial_hidden_states_batch: Optional[torch.Tensor], 
               attention_mask_batch: Optional[torch.Tensor]):
        
        B, MaxSeqLen, _, _ = self._get_batch_shape_info(padded_states_dict_seq, context="update")
        NA_config = self.num_agents 
        
        EmbDimPolicy = self.policy_embedding_network.base_encoder.embed_dim
        EmbDimValue = self.value_embedding_network.base_encoder.embed_dim
        EmbDimBaseline = self.baseline_embedding_network.base_encoder.embed_dim

        valid_rewards_mask = attention_mask_batch.unsqueeze(-1).expand_as(padded_rewards_seq).bool()
        rewards_raw_mean = 0.0
        if valid_rewards_mask.sum() > 0: 
            rewards_raw_mean = padded_rewards_seq[valid_rewards_mask].mean().item()
        if self.rewards_normalizer:
            valid_rewards_flat = padded_rewards_seq[valid_rewards_mask]
            if valid_rewards_flat.numel() > 0: 
                valid_rewards_for_norm_update = valid_rewards_flat.unsqueeze(-1)
                self.rewards_normalizer.update(valid_rewards_for_norm_update)
                normalized_valid_rewards = self.rewards_normalizer.normalize(valid_rewards_for_norm_update)
                norm_rewards_seq = torch.zeros_like(padded_rewards_seq)
                norm_rewards_seq[valid_rewards_mask] = normalized_valid_rewards.squeeze(-1).to(norm_rewards_seq.dtype)
                padded_rewards_seq = norm_rewards_seq 
        rewards_norm_mean = 0.0
        if valid_rewards_mask.sum() > 0:
            rewards_norm_mean = padded_rewards_seq[valid_rewards_mask].mean().item()

        all_intrinsic_rewards_seq = torch.zeros_like(padded_rewards_seq)

        with torch.no_grad():
            all_old_log_probs_seq_parts, all_old_values_seq_parts = [], []
            all_old_baselines_seq_parts, all_next_values_seq_parts = [], []
            all_intrinsic_rewards_sub_batch_parts = [] 
            num_sequences_total = B 

            for i in range(0, num_sequences_total, self.no_grad_forward_batch_size):
                start_idx, end_idx = i, min(i + self.no_grad_forward_batch_size, num_sequences_total)
                current_sub_batch_size = end_idx - start_idx
                if current_sub_batch_size == 0: continue

                sub_batch_states_dict_seq = {k: v[start_idx:end_idx] for k, v in padded_states_dict_seq.items() if v is not None}
                sub_batch_next_states_dict_seq = {k: v[start_idx:end_idx] for k, v in padded_next_states_dict_seq.items() if v is not None}
                sub_batch_actions_seq = padded_actions_seq[start_idx:end_idx]
                
                sub_base_emb_policy, _ = self.policy_embedding_network.get_base_embedding(sub_batch_states_dict_seq)
                sub_processed_feat_policy = sub_base_emb_policy
                if self.enable_memory and self.policy_memory_module:
                    sub_batch_initial_hidden_policy = initial_hidden_states_batch[start_idx:end_idx] if initial_hidden_states_batch is not None else None
                    if sub_batch_initial_hidden_policy is None:
                        sub_batch_initial_hidden_policy = torch.zeros(current_sub_batch_size, NA_config, self.memory_num_layers, self.memory_hidden_size, device=self.device)
                    mem_in_policy = sub_base_emb_policy.reshape(current_sub_batch_size * NA_config, MaxSeqLen, EmbDimPolicy)
                    init_h_policy = sub_batch_initial_hidden_policy.reshape(current_sub_batch_size * NA_config, self.memory_num_layers, self.memory_hidden_size).permute(1,0,2).contiguous()
                    proc_feat_flat_policy, _ = self.policy_memory_module.forward_sequence(mem_in_policy, init_h_policy)
                    sub_processed_feat_policy = proc_feat_flat_policy.reshape(current_sub_batch_size, MaxSeqLen, NA_config, self.memory_hidden_size)

                sub_base_emb_value, _ = self.value_embedding_network.get_base_embedding(sub_batch_states_dict_seq)
                sub_next_base_emb_value, _ = self.value_embedding_network.get_base_embedding(sub_batch_next_states_dict_seq)
                sub_processed_feat_value = sub_base_emb_value
                sub_next_processed_feat_value = sub_next_base_emb_value
                if self.enable_memory and self.value_memory_module:
                    mem_in_value = sub_base_emb_value.reshape(current_sub_batch_size * NA_config, MaxSeqLen, EmbDimValue)
                    proc_feat_flat_value, _ = self.value_memory_module.forward_sequence(mem_in_value, None) 
                    sub_processed_feat_value = proc_feat_flat_value.reshape(current_sub_batch_size, MaxSeqLen, NA_config, self.memory_hidden_size)
                    mem_in_next_value = sub_next_base_emb_value.reshape(current_sub_batch_size*NA_config, MaxSeqLen, EmbDimValue)
                    proc_feat_flat_next_value, _ = self.value_memory_module.forward_sequence(mem_in_next_value, None)
                    sub_next_processed_feat_value = proc_feat_flat_next_value.reshape(current_sub_batch_size, MaxSeqLen, NA_config, self.memory_hidden_size)

                sub_base_emb_baseline, _ = self.baseline_embedding_network.get_base_embedding(sub_batch_states_dict_seq)
                sub_processed_feat_baseline = sub_base_emb_baseline
                if self.enable_memory and self.baseline_memory_module:
                    mem_in_baseline = sub_base_emb_baseline.reshape(current_sub_batch_size*NA_config, MaxSeqLen, EmbDimBaseline)
                    proc_feat_flat_baseline, _ = self.baseline_memory_module.forward_sequence(mem_in_baseline, None)
                    sub_processed_feat_baseline = proc_feat_flat_baseline.reshape(current_sub_batch_size, MaxSeqLen, NA_config, self.memory_hidden_size)

                if self.enable_rnd and self.rnd_target_network and self.rnd_predictor_network:
                    flat_features_for_rnd = sub_processed_feat_value.reshape(-1, self.feature_dim_for_heads)
                    target_rnd_feats = self.rnd_target_network(flat_features_for_rnd)
                    predicted_rnd_feats_for_reward = self.rnd_predictor_network(flat_features_for_rnd) 
                    intrinsic_r_terms = F.mse_loss(predicted_rnd_feats_for_reward, target_rnd_feats, reduction='none').mean(dim=-1)
                    intrinsic_r_sub_batch_per_agent = intrinsic_r_terms.reshape(current_sub_batch_size, MaxSeqLen, NA_config)
                    all_intrinsic_rewards_sub_batch_parts.append(intrinsic_r_sub_batch_per_agent.mean(dim=-1, keepdim=True))
                else:
                    all_intrinsic_rewards_sub_batch_parts.append(torch.zeros(current_sub_batch_size, MaxSeqLen, 1, device=self.device))

                sub_old_log_probs_seq, _ = self._recompute_log_probs_from_features(sub_processed_feat_policy, sub_batch_actions_seq)
                sub_old_values_seq = self.shared_critic.values(sub_processed_feat_value)
                sub_baseline_input_seq = self.baseline_embedding_network.get_baseline_embeddings(sub_processed_feat_baseline, sub_batch_actions_seq)
                sub_old_baselines_seq = self.shared_critic.baselines(sub_baseline_input_seq)
                sub_next_values_seq = self.shared_critic.values(sub_next_processed_feat_value)

                all_old_log_probs_seq_parts.append(sub_old_log_probs_seq)
                all_old_values_seq_parts.append(sub_old_values_seq)
                all_old_baselines_seq_parts.append(sub_old_baselines_seq)
                all_next_values_seq_parts.append(sub_next_values_seq)
            
            old_log_probs_seq = torch.cat(all_old_log_probs_seq_parts, dim=0)
            old_values_seq = torch.cat(all_old_values_seq_parts, dim=0)
            old_baselines_seq = torch.cat(all_old_baselines_seq_parts, dim=0)
            next_values_seq = torch.cat(all_next_values_seq_parts, dim=0)
            
            all_intrinsic_rewards_seq = torch.cat(all_intrinsic_rewards_sub_batch_parts, dim=0)
            if self.enable_rnd and self.intrinsic_reward_normalizer:
                valid_intrinsic_mask = attention_mask_batch.unsqueeze(-1).expand_as(all_intrinsic_rewards_seq).bool()
                if valid_intrinsic_mask.sum() > 0:
                    valid_intrinsic_r_flat = all_intrinsic_rewards_seq[valid_intrinsic_mask]
                    if valid_intrinsic_r_flat.numel() > 0:
                        valid_intrinsic_r = valid_intrinsic_r_flat.unsqueeze(-1)
                        self.intrinsic_reward_normalizer.update(valid_intrinsic_r)
                        normalized_intrinsic_r_values = self.intrinsic_reward_normalizer.normalize(valid_intrinsic_r)
                        temp_norm_intrinsic_r = torch.zeros_like(all_intrinsic_rewards_seq)
                        temp_norm_intrinsic_r[valid_intrinsic_mask] = normalized_intrinsic_r_values.squeeze(-1).to(temp_norm_intrinsic_r.dtype)
                        all_intrinsic_rewards_seq = temp_norm_intrinsic_r 

            combined_rewards_seq = padded_rewards_seq + self.intrinsic_reward_coeff * all_intrinsic_rewards_seq.detach()
            returns_seq = self._compute_gae_with_padding(combined_rewards_seq, old_values_seq, next_values_seq, padded_dones_seq, padded_truncs_seq, attention_mask_batch)

        num_sequences_in_batch = B 
        idxes = np.arange(num_sequences_in_batch)
        log_keys = ["Policy Loss", "Value Loss", "Baseline Loss", "Entropy", "Grad Norm", "Advantage Raw Mean", "Advantage Raw Std", "Advantage Final Mean", "Returns Mean", "Logprob Mean", "Logprob Min", "Logprob Max"]
        if self.enable_rnd: log_keys.extend(["RND Predictor Loss", "Intrinsic Reward Mean (Normalized)"])
        batch_logs = {k: [] for k in log_keys}

        for epoch in range(self.epochs):
            np.random.shuffle(idxes)
            for mb_start_idx in range(0, num_sequences_in_batch, self.mini_batch_size):
                mb_end_idx = min(mb_start_idx + self.mini_batch_size, num_sequences_in_batch)
                mb_indices = idxes[mb_start_idx:mb_end_idx] 
                MB_NUM_SEQ = len(mb_indices)
                if MB_NUM_SEQ == 0: continue 

                st_mb_seq_dict = self._slice_state_dict_batch(padded_states_dict_seq, mb_indices)
                act_mb_seq = padded_actions_seq[mb_indices].contiguous()
                mask_mb = attention_mask_batch[mb_indices].contiguous()
                
                olp_mb_seq = old_log_probs_seq[mb_indices].contiguous()
                oval_mb_seq = old_values_seq[mb_indices].contiguous()
                obase_mb_seq = old_baselines_seq[mb_indices].contiguous()
                ret_mb_seq = returns_seq[mb_indices].contiguous()
                
                init_h_mb_policy = initial_hidden_states_batch[mb_indices].contiguous() if self.enable_memory and initial_hidden_states_batch is not None else None
                base_emb_mb_seq_policy, _ = self.policy_embedding_network.get_base_embedding(st_mb_seq_dict)
                processed_features_mb_seq_policy = base_emb_mb_seq_policy
                if self.enable_memory and self.policy_memory_module and init_h_mb_policy is not None:
                    mem_in_policy = base_emb_mb_seq_policy.reshape(MB_NUM_SEQ * NA_config, MaxSeqLen, EmbDimPolicy)
                    init_h_policy = init_h_mb_policy.reshape(MB_NUM_SEQ * NA_config, self.memory_num_layers, self.memory_hidden_size).permute(1,0,2).contiguous()
                    proc_feat_flat_policy, _ = self.policy_memory_module.forward_sequence(mem_in_policy, init_h_policy)
                    processed_features_mb_seq_policy = proc_feat_flat_policy.reshape(MB_NUM_SEQ, MaxSeqLen, NA_config, self.memory_hidden_size)
                new_lp_mb_seq, ent_mb_seq = self._recompute_log_probs_from_features(processed_features_mb_seq_policy, act_mb_seq)
                
                base_emb_mb_seq_value, _ = self.value_embedding_network.get_base_embedding(st_mb_seq_dict)
                processed_features_mb_seq_value = base_emb_mb_seq_value
                if self.enable_memory and self.value_memory_module:
                    mem_in_value = base_emb_mb_seq_value.reshape(MB_NUM_SEQ*NA_config, MaxSeqLen, EmbDimValue)
                    proc_feat_flat_value, _ = self.value_memory_module.forward_sequence(mem_in_value, None) 
                    processed_features_mb_seq_value = proc_feat_flat_value.reshape(MB_NUM_SEQ, MaxSeqLen, NA_config, self.memory_hidden_size)
                new_vals_mb_seq = self.shared_critic.values(processed_features_mb_seq_value)

                base_emb_mb_seq_baseline, _ = self.baseline_embedding_network.get_base_embedding(st_mb_seq_dict)
                processed_features_mb_seq_baseline = base_emb_mb_seq_baseline
                if self.enable_memory and self.baseline_memory_module:
                    mem_in_baseline = base_emb_mb_seq_baseline.reshape(MB_NUM_SEQ*NA_config, MaxSeqLen, EmbDimBaseline)
                    proc_feat_flat_baseline, _ = self.baseline_memory_module.forward_sequence(mem_in_baseline, None)
                    processed_features_mb_seq_baseline = proc_feat_flat_baseline.reshape(MB_NUM_SEQ, MaxSeqLen, NA_config, self.memory_hidden_size)
                baseline_input_mb_seq_for_loss = self.baseline_embedding_network.get_baseline_embeddings(processed_features_mb_seq_baseline, act_mb_seq)
                new_base_mb_seq = self.shared_critic.baselines(baseline_input_mb_seq_for_loss)
                
                lp_detached = new_lp_mb_seq.detach()
                valid_lp_mask = mask_mb.unsqueeze(-1).expand_as(lp_detached).bool()
                if valid_lp_mask.sum() > 0: 
                    batch_logs["Logprob Mean"].append(lp_detached[valid_lp_mask].mean().item())
                    batch_logs["Logprob Min"].append(lp_detached[valid_lp_mask].min().item())
                    batch_logs["Logprob Max"].append(lp_detached[valid_lp_mask].max().item())

                adv_mb_seq = ret_mb_seq.unsqueeze(2).expand_as(new_base_mb_seq) - new_base_mb_seq 
                adv_mb_squeezed_seq = adv_mb_seq.squeeze(-1)
                valid_adv_mask = mask_mb.unsqueeze(-1).expand_as(adv_mb_squeezed_seq).bool()
                if valid_adv_mask.sum() > 0:
                    valid_adv_elements = adv_mb_squeezed_seq[valid_adv_mask]
                    batch_logs["Advantage Raw Mean"].append(valid_adv_elements.mean().item())
                    batch_logs["Advantage Raw Std"].append(valid_adv_elements.std().item())
                    if self.normalize_adv:
                        normalized_valid_adv = (valid_adv_elements - valid_adv_elements.mean()) / (valid_adv_elements.std() + 1e-8)
                        temp_adv_norm = torch.zeros_like(adv_mb_squeezed_seq)
                        temp_adv_norm[valid_adv_mask] = normalized_valid_adv
                        adv_mb_squeezed_seq = temp_adv_norm
                    batch_logs["Advantage Final Mean"].append(adv_mb_squeezed_seq[valid_adv_mask].mean().item())
                else: 
                    batch_logs["Advantage Raw Mean"].append(0.0); batch_logs["Advantage Raw Std"].append(0.0); batch_logs["Advantage Final Mean"].append(0.0)
                detached_adv_seq = adv_mb_squeezed_seq.detach()

                mask_mb_policy_entropy = mask_mb.unsqueeze(-1).expand_as(new_lp_mb_seq)
                mask_mb_value = mask_mb.unsqueeze(-1).expand_as(new_vals_mb_seq)
                mask_mb_baseline = mask_mb.unsqueeze(-1).unsqueeze(-1).expand_as(new_base_mb_seq)
                num_valid_policy_entropy = mask_mb_policy_entropy.sum().clamp(min=1)
                num_valid_value = mask_mb_value.sum().clamp(min=1)
                num_valid_baseline = mask_mb_baseline.sum().clamp(min=1)

                pol_loss_terms = self._ppo_clip_loss(new_lp_mb_seq, olp_mb_seq, detached_adv_seq, self.ppo_clip_range, reduction='none')
                pol_loss = (pol_loss_terms * mask_mb_policy_entropy).sum() / num_valid_policy_entropy
                
                ret_mb_for_value = ret_mb_seq
                if self.clipped_value_loss:
                    value_loss_terms = self._clipped_value_loss(oval_mb_seq, new_vals_mb_seq, ret_mb_for_value, self.value_clip_range, reduction='none')
                else:
                    value_loss_terms = F.mse_loss(new_vals_mb_seq, ret_mb_for_value, reduction='none')
                val_loss = (value_loss_terms * mask_mb_value).sum() / num_valid_value

                ret_mb_exp_baseline = ret_mb_seq.unsqueeze(2).expand_as(new_base_mb_seq)
                if self.clipped_value_loss:
                    baseline_loss_terms = self._clipped_value_loss(obase_mb_seq, new_base_mb_seq, ret_mb_exp_baseline, self.value_clip_range, reduction='none')
                else:
                    baseline_loss_terms = F.mse_loss(new_base_mb_seq, ret_mb_exp_baseline, reduction='none')
                base_loss = (baseline_loss_terms * mask_mb_baseline).sum() / num_valid_baseline
                
                entropy_terms = -ent_mb_seq
                ent_loss = (entropy_terms * mask_mb_policy_entropy).sum() / num_valid_policy_entropy
                
                total_loss_rl = (pol_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss + self.entropy_coeff * ent_loss)
                total_loss = total_loss_rl # Initialize total loss with RL components

                rnd_predictor_loss = torch.tensor(0.0, device=self.device)
                if self.enable_rnd and self.rnd_target_network and self.rnd_predictor_network:
                    flat_features_for_rnd_update_mb = processed_features_mb_seq_value.reshape(-1, self.feature_dim_for_heads)
                    expanded_mask_for_na = mask_mb.unsqueeze(-1).expand(-1, -1, NA_config)
                    flat_mask_mb_rnd = expanded_mask_for_na.reshape(-1)
                    valid_indices_rnd = flat_mask_mb_rnd.nonzero(as_tuple=False).squeeze(-1)
                    
                    if valid_indices_rnd.numel() > 0:
                        num_rnd_update_samples = int(valid_indices_rnd.numel() * self.rnd_update_proportion)
                        if num_rnd_update_samples > 0:
                            perm_rnd = torch.randperm(valid_indices_rnd.numel(), device=self.device)[:num_rnd_update_samples]
                            rnd_mb_update_indices = valid_indices_rnd[perm_rnd]
                            features_for_rnd_predictor_update = flat_features_for_rnd_update_mb[rnd_mb_update_indices]
                            if features_for_rnd_predictor_update.numel() > 0:
                                with torch.no_grad():
                                    target_rnd_feats_update = self.rnd_target_network(features_for_rnd_predictor_update)
                                predicted_rnd_feats_update = self.rnd_predictor_network(features_for_rnd_predictor_update)
                                rnd_predictor_loss = F.mse_loss(predicted_rnd_feats_update, target_rnd_feats_update)
                                # Add RND predictor loss to the total loss that will be backpropagated
                                # This means RND predictor loss will affect the value pathway's embedding and memory if they exist
                                total_loss = total_loss + rnd_predictor_loss 
                
                # Zero gradients for all optimizers before backward pass
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                self.baseline_optimizer.zero_grad()
                if self.rnd_predictor_optimizer:
                    self.rnd_predictor_optimizer.zero_grad()

                total_loss.backward() # Compute gradients for all parts contributing to total_loss
                
                # Clip gradients globally across all parameters managed by the agent
                # This is a common approach even with separate optimizers if a single clipping threshold is desired.
                # Alternatively, one could clip per parameter group before each optimizer step.
                all_trainable_params = self.parameters() # Gets all parameters from all optimizers' groups
                gn = clip_grad_norm_(all_trainable_params, self.max_grad_norm)
                
                # Step each optimizer
                self.policy_optimizer.step()
                self.value_optimizer.step()
                self.baseline_optimizer.step()
                if self.enable_rnd and self.rnd_predictor_optimizer:
                    # If RND predictor loss was part of total_loss, its gradients are already computed.
                    # If RND predictor had a separate loss.backward(), it would be stepped here.
                    # Since rnd_predictor_loss is added to total_loss, its gradients are included.
                    # The RND predictor optimizer will only update rnd_predictor_network.parameters().
                    self.rnd_predictor_optimizer.step()


                batch_logs["Policy Loss"].append(pol_loss.item())
                batch_logs["Value Loss"].append(val_loss.item())
                batch_logs["Baseline Loss"].append(base_loss.item())
                if valid_lp_mask.sum() > 0: batch_logs["Entropy"].append(ent_mb_seq[valid_lp_mask].mean().item())
                else: batch_logs["Entropy"].append(0.0)
                batch_logs["Grad Norm"].append(gn.item() if torch.is_tensor(gn) else gn)
                
                valid_returns_mask_mb = mask_mb.unsqueeze(-1).expand_as(ret_mb_seq).bool()
                if valid_returns_mask_mb.sum() > 0 : batch_logs["Returns Mean"].append(ret_mb_seq[valid_returns_mask_mb].mean().item())
                else: batch_logs["Returns Mean"].append(0.0)
                
                if self.enable_rnd:
                    batch_logs["RND Predictor Loss"].append(rnd_predictor_loss.item())
                    intrinsic_rewards_for_mb_log = all_intrinsic_rewards_seq[mb_indices]
                    valid_intrinsic_mb_mask = mask_mb.unsqueeze(-1).expand_as(intrinsic_rewards_for_mb_log).bool()
                    if valid_intrinsic_mb_mask.sum() > 0:
                         batch_logs["Intrinsic Reward Mean (Normalized)"].append(intrinsic_rewards_for_mb_log[valid_intrinsic_mb_mask].mean().item())
                    else:
                         batch_logs["Intrinsic Reward Mean (Normalized)"].append(0.0)

            current_lr = self._get_policy_lr() # Get policy LR for logging
            if self.policy_lr_scheduler: self.policy_lr_scheduler.step() # Step only policy LR scheduler
            if self.entropy_scheduler: self.entropy_coeff = self.entropy_scheduler.step()
            if self.policy_clip_scheduler: self.ppo_clip_range = self.policy_clip_scheduler.step()
            if self.value_clip_scheduler: self.value_clip_range = self.value_clip_scheduler.step()
            if self.max_grad_norm_scheduler: self.max_grad_norm = self.max_grad_norm_scheduler.step()

        final_logs = {k: np.mean(v) if v else 0.0 for k, v in batch_logs.items()}
        final_logs["Entropy coeff"] = self.entropy_coeff
        final_logs["Learning Rate (Policy)"] = current_lr # Log policy LR
        final_logs["PPO Clip Range"] = self.ppo_clip_range
        final_logs["Max Grad Norm"] = self.max_grad_norm
        final_logs["Raw Reward Mean"] = rewards_raw_mean
        final_logs["Norm Reward Mean"] = rewards_norm_mean
        if self.enable_rnd and "Intrinsic Reward Mean (Normalized)" not in final_logs:
             final_logs["Intrinsic Reward Mean (Normalized)"] = 0.0

        return final_logs

    def _compute_gae_with_padding(self, rewards_seq, values_seq, next_values_seq,
                                  dones_seq, truncs_seq, attention_mask):
        B, MaxSeqLen, _ = rewards_seq.shape
        returns_seq = torch.zeros_like(rewards_seq)
        gae = torch.zeros(B, 1, device=rewards_seq.device)
        dones_float_seq = dones_seq.float()

        for t in reversed(range(MaxSeqLen)):
            current_step_is_valid_mask_t = attention_mask[:, t:t+1] 
            next_step_is_valid_in_sequence_mask_t_plus_1 = attention_mask[:, t+1:t+2] if t < MaxSeqLen - 1 else torch.zeros_like(current_step_is_valid_mask_t)
            can_bootstrap_from_next_value = (1.0 - dones_float_seq[:, t]) * next_step_is_valid_in_sequence_mask_t_plus_1
            v_next_bootstrapped = next_values_seq[:, t] * can_bootstrap_from_next_value
            delta = (rewards_seq[:, t] + self.gamma * v_next_bootstrapped - values_seq[:, t]) * current_step_is_valid_mask_t
            gae_prop_mask = can_bootstrap_from_next_value
            gae = delta + self.gamma * self.lmbda * gae_prop_mask * gae
            current_returns_t = gae + values_seq[:, t]
            returns_seq[:, t] = torch.where(current_step_is_valid_mask_t.bool(), current_returns_t, torch.zeros_like(current_returns_t))
            gae = gae * current_step_is_valid_mask_t
        return returns_seq
    
    def _ppo_clip_loss(self, new_lp, old_lp, advantages, clip_range, reduction='mean'):
        log_ratio = new_lp - old_lp
        ratio = torch.exp(log_ratio.clamp(-10,10))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        loss_terms = -torch.min(surr1, surr2)
        if reduction == 'mean': return loss_terms.mean()
        elif reduction == 'none': return loss_terms
        else: raise ValueError(f"MAPOCAAgent Error: Unsupported reduction type in _ppo_clip_loss: {reduction}")

    def _clipped_value_loss(self, old_values, new_values, targets, clip_range, reduction='mean'):
        values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
        mse_unclipped = F.mse_loss(new_values, targets, reduction='none')
        mse_clipped = F.mse_loss(values_clipped, targets, reduction='none')
        loss_terms = torch.max(mse_unclipped, mse_clipped)
        if reduction == 'mean': return loss_terms.mean()
        elif reduction == 'none': return loss_terms
        else: raise ValueError(f"MAPOCAAgent Error: Unsupported reduction type in _clipped_value_loss: {reduction}")

    def _recompute_log_probs_from_features(self, processed_features: torch.Tensor, actions: torch.Tensor):
        input_shape = processed_features.shape
        FeatureDim = input_shape[-1]
        NA = input_shape[-2]
        leading_dims = input_shape[:-2]
        flat_features = processed_features.reshape(-1, FeatureDim)
        act_flat = actions.reshape(-1, actions.shape[-1]) 
        lp_flat, ent_flat = self.policy_net.recompute_log_probs(flat_features, act_flat)
        log_probs = lp_flat.reshape(*leading_dims, NA)
        entropies = ent_flat.reshape(*leading_dims, NA)
        return log_probs, entropies

    def _slice_state_dict_batch(self, states_dict_seq: dict, mb_indices: np.ndarray):
        new_dict = {}
        for k, v_seq in states_dict_seq.items():
            if v_seq is not None: new_dict[k] = v_seq[mb_indices].contiguous()
            else: new_dict[k] = None
        return new_dict

    def _get_policy_lr(self) -> float: # Renamed from _get_avg_lr
        # Utility to get current learning rate from the policy optimizer.
        lr_sum, count = 0.0, 0
        for pg in self.policy_optimizer.param_groups: # Use policy_optimizer
            lr_sum += pg.get('lr', 0.0) 
            count += 1
        return lr_sum / max(count, 1)

    def _accumulate_per_layer_grad_norms(self, grad_norms_sum_per_layer: dict):
        if not hasattr(self, '_param_names_with_grads'):
            self._param_names_with_grads = {name for name, param in self.named_parameters() if param.requires_grad}
        for name, param in self.named_parameters():
            if name in self._param_names_with_grads and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms_sum_per_layer[name] = grad_norms_sum_per_layer.get(name, 0.0) + grad_norm
