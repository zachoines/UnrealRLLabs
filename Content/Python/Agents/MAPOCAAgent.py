# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
import copy # For deepcopying target networks
import random # For replay buffer sampling

from Source.Agent import Agent
from Source.StateRecorder import StateRecorder
from Source.Utility import RunningMeanStdNormalizer, LinearValueScheduler, PopArtNormalizer
from Source.Networks import (
    MultiAgentEmbeddingNetwork,
    SharedCritic,
    BetaPolicyNetwork,
    DiscretePolicyNetwork,
    RNDTargetNetwork,
    RNDPredictorNetwork,
    # ImplicitQuantileNetwork and DistillationNetwork are used by SharedCritic
)
from Source.Memory import GRUSequenceMemory

# -----------------------------------------------------------------------------
# IQN Replay Buffer - Stores individual sequences
# -----------------------------------------------------------------------------
class IQNReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device # Device to move tensors to when sampling
        # Stored: (feat_single_seq_cpu, next_feat_single_seq_cpu, reward_single_seq_cpu, 
        #          done_single_seq_cpu, attention_mask_single_seq_cpu, 
        #          initial_h_single_seq_cpu, next_initial_h_single_seq_cpu)
        # Each element in the buffer is a tuple representing one sequence.
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        self.position = 0

    def push_batch(self, 
                   batch_feats_seq: torch.Tensor,      # (B_ppo, T, NA, F)
                   batch_next_feats_seq: torch.Tensor, # (B_ppo, T, NA, F)
                   batch_rewards_seq: torch.Tensor,    # (B_ppo, T, 1)
                   batch_dones_seq: torch.Tensor,      # (B_ppo, T, 1)
                   batch_attention_mask: torch.Tensor, # (B_ppo, T)
                   batch_initial_h: Optional[torch.Tensor] = None, # (B_ppo, NA, Layers, Hidden)
                   batch_next_initial_h: Optional[torch.Tensor] = None): # (B_ppo, NA, Layers, Hidden)
        
        if self.capacity == 0: return # Do not store if capacity is zero

        num_sequences_in_batch = batch_feats_seq.shape[0]

        for i in range(num_sequences_in_batch):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None) # type: ignore
            
            # Ensure tensors are detached and moved to CPU before storing
            initial_h_single = batch_initial_h[i].detach().cpu() if batch_initial_h is not None and batch_initial_h.numel() > 0 else None
            next_initial_h_single = batch_next_initial_h[i].detach().cpu() if batch_next_initial_h is not None and batch_next_initial_h.numel() > 0 else None


            self.buffer[self.position] = (
                batch_feats_seq[i].detach().cpu(),      # (T, NA, F)
                batch_next_feats_seq[i].detach().cpu(), # (T, NA, F)
                batch_rewards_seq[i].detach().cpu(),    # (T, 1)
                batch_dones_seq[i].detach().cpu(),      # (T, 1)
                batch_attention_mask[i].detach().cpu(), # (T,)
                initial_h_single,
                next_initial_h_single
            )
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        if len(self.buffer) < batch_size:
            return None
        
        batch_tuples = random.sample(self.buffer, batch_size)
        
        feats_seq_list, next_feats_seq_list, rewards_seq_list, \
        dones_seq_list, attention_mask_list, initial_h_list, next_initial_h_list = zip(*batch_tuples)

        # Stack along the new batch dimension (dim 0) and move to target device
        feats_seq_dev = torch.stack(feats_seq_list).to(self.device)
        next_feats_seq_dev = torch.stack(next_feats_seq_list).to(self.device)
        rewards_seq_dev = torch.stack(rewards_seq_list).to(self.device)
        dones_seq_dev = torch.stack(dones_seq_list).to(self.device)
        attention_mask_dev = torch.stack(attention_mask_list).to(self.device)

        initial_h_filtered = [h for h in initial_h_list if h is not None]
        initial_h_dev = None
        if initial_h_filtered:
            # Ensure all non-None tensors have the same shape before stacking
            if all(h.shape == initial_h_filtered[0].shape for h in initial_h_filtered):
                 initial_h_dev = torch.stack(initial_h_filtered).to(self.device) if len(initial_h_filtered) == batch_size else None # Only stack if all sequences had initial_h
            # else: handle potentially mixed (None/Tensor) or differently shaped h-states if necessary

        next_initial_h_filtered = [h for h in next_initial_h_list if h is not None]
        next_initial_h_dev = None
        if next_initial_h_filtered:
            if all(h.shape == next_initial_h_filtered[0].shape for h in next_initial_h_filtered):
                next_initial_h_dev = torch.stack(next_initial_h_filtered).to(self.device) if len(next_initial_h_filtered) == batch_size else None
            
        return feats_seq_dev, next_feats_seq_dev, rewards_seq_dev, dones_seq_dev, attention_mask_dev, initial_h_dev, next_initial_h_dev

    def __len__(self) -> int:
        return len(self.buffer)

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _bool_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise OR that works on float or bool tensors."""
    return (a > 0.5) | (b > 0.5)

# -----------------------------------------------------------------------------
class MAPOCAAgent(Agent):
    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__(cfg, device)
        self.device = device
        a_cfg = cfg["agent"]["params"]
        t_cfg = cfg["train"]
        env_shape = cfg["environment"]["shape"]
        env_params = cfg["environment"]["params"]

        # --- Hyperparameters ---
        self.gamma = a_cfg.get("gamma", 0.99)
        self.lmbda = a_cfg.get("lambda", 0.95)
        self.entropy_coeff = a_cfg.get("entropy_coeff", 0.01)
        self.value_loss_coeff = a_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff = a_cfg.get("baseline_loss_coeff", 0.5)
        self.max_grad_norm = a_cfg.get("max_grad_norm", 0.5) 
        self.normalize_adv = a_cfg.get("normalize_advantages", True)
        self.ppo_clip_range = a_cfg.get("ppo_clip_range", 0.2)
        self.value_clip_range = a_cfg.get("value_clip_range", 0.2) 
        self.lr_base = a_cfg.get("learning_rate", 3e-4)
        self.epochs = t_cfg.get("epochs", 4) 
        self.mini_batch_size = t_cfg.get("mini_batch_size", 64)
        self.no_grad_bs = a_cfg.get("no_grad_forward_batch_size", self.mini_batch_size)

        # --- IQN & Distillation Parameters ---
        self.enable_iqn_distillation = a_cfg.get("enable_iqn_distillation", False)
        critic_full_config = a_cfg["networks"]["critic_network"] 
        
        if self.enable_iqn_distillation:
            iqn_params_from_critic_config = critic_full_config.get("iqn_params", {})
            self.num_quantiles = iqn_params_from_critic_config.get("num_quantiles", 32)
            self.num_quantiles_prime = iqn_params_from_critic_config.get("num_quantiles_prime", self.num_quantiles) 
            self.iqn_kappa = iqn_params_from_critic_config.get("kappa", 1.0) 
            self.lr_iqn = a_cfg.get("lr_iqn", 0.0001)
            self.iqn_loss_coeff = a_cfg.get("iqn_loss_coeff", 0.5)
            self.iqn_target_update_freq = t_cfg.get("iqn_target_update_freq", 1000) 
            self.iqn_polyak = iqn_params_from_critic_config.get("polyak_tau", 0.005) 
            self.iqn_epochs = t_cfg.get("iqn_epochs", t_cfg.get("epochs",1)) 
            self.iqn_batch_size = t_cfg.get("iqn_batch_size", self.mini_batch_size)
            self.iqn_max_grad_norm = a_cfg.get("iqn_max_grad_norm", self.max_grad_norm)
            self._iqn_total_updates_counter = 0 
            replay_buffer_size = t_cfg.get("iqn_replay_buffer_size", 10000)
            # Pass self.device to replay buffer if samples are moved to device upon sampling
            self.iqn_replay_buffer = IQNReplayBuffer(replay_buffer_size, self.device) 
        else:
            self.num_quantiles = 0 
            self.num_quantiles_prime = 0

        # --- Environment & Agent Setup ---
        self.num_agents = env_params.get("MaxAgents", 1)
        if "agent" in env_shape["state"]: self.num_agents = env_shape["state"]["agent"].get("max", self.num_agents)
        self.state_recorder = StateRecorder(cfg.get("StateRecorder")) if "StateRecorder" in cfg and cfg.get("StateRecorder") else None
        self._determine_action_space(env_shape["action"])

        # --- Network Definitions ---
        emb_cfg = a_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        self.embedding_net = MultiAgentEmbeddingNetwork(emb_cfg).to(device)
        emb_out = emb_cfg["cross_attention_feature_extractor"]["embed_dim"]

        self.enable_memory = a_cfg.get("enable_memory", False)
        if self.enable_memory:
            mem_cfg = a_cfg.get(a_cfg.get("memory_type", "gru"), {}); mem_cfg.setdefault("input_size", emb_out)
            self.memory_hidden = mem_cfg.get("hidden_size", 128); self.memory_layers = mem_cfg.get("num_layers", 1)
            self.memory_module = GRUSequenceMemory(**mem_cfg).to(device); trunk_dim = self.memory_hidden
        else:
            self.memory_module = None; self.memory_hidden = self.memory_layers = 0; trunk_dim = emb_out

        pol_cfg = a_cfg["networks"]["policy_network"].copy(); pol_cfg.update({"in_features": trunk_dim, "out_features": self.action_dim})
        self.policy_net = BetaPolicyNetwork(**pol_cfg).to(device) if self.action_space_type == "continuous" else DiscretePolicyNetwork(**pol_cfg).to(device)
        
        self.shared_critic = SharedCritic(net_cfg=critic_full_config).to(device)

        if self.enable_iqn_distillation:
            self.target_value_iqn_net = copy.deepcopy(self.shared_critic.value_iqn_net).to(device)
            self.target_baseline_iqn_net = copy.deepcopy(self.shared_critic.baseline_iqn_net).to(device)
            for p in self.target_value_iqn_net.parameters(): p.requires_grad = False
            for p in self.target_baseline_iqn_net.parameters(): p.requires_grad = False

        # --- PopArt / Reward Normalization ---
        self.enable_popart = a_cfg.get("enable_popart", False)
        if self.enable_popart:
            beta, eps = a_cfg.get("popart_beta", 0.999), a_cfg.get("popart_epsilon", 1e-5)
            if self.enable_iqn_distillation:
                self.value_popart = PopArtNormalizer(self.shared_critic.value_distill_net.mlp[-1], beta, eps, device).to(device)
                self.baseline_popart = PopArtNormalizer(self.shared_critic.baseline_distill_net.mlp[-1], beta, eps, device).to(device)
            else: 
                self.value_popart = PopArtNormalizer(self.shared_critic.value_head_ppo.output_layer, beta, eps, device).to(device)
                self.baseline_popart = PopArtNormalizer(self.shared_critic.baseline_head_ppo.output_layer, beta, eps, device).to(device)
            self.rewards_normalizer = None
        else:
            self.value_popart = self.baseline_popart = None
            rnorm_cfg = a_cfg.get("rewards_normalizer_disabled", None); 
            self.rewards_normalizer = RunningMeanStdNormalizer(**rnorm_cfg, device=device) if rnorm_cfg else None
        
        # --- RND Initialization ---
        self.enable_rnd = a_cfg.get("enable_rnd", False) 
        if self.enable_rnd:
            rnd_cfg = a_cfg.get("rnd_params", {}); rnd_in, rnd_out = trunk_dim, rnd_cfg.get("output_size", 128)
            rnd_hid = rnd_cfg.get("hidden_size", 256); self.intrinsic_reward_normalizer = RunningMeanStdNormalizer(epsilon=1e-8, device=device)
            self.rnd_target_network = RNDTargetNetwork(rnd_in, rnd_out, rnd_hid).to(device); self.rnd_target_network.eval()
            for p in self.rnd_target_network.parameters(): p.requires_grad_(False)
            self.rnd_predictor_network = RNDPredictorNetwork(rnd_in, rnd_out, rnd_hid).to(device)
            self.intrinsic_reward_coeff = rnd_cfg.get("intrinsic_reward_coeff", 0.01); self.rnd_update_prop = rnd_cfg.get("rnd_update_proportion", 0.25)
        else: self.rnd_target_network = self.rnd_predictor_network = self.intrinsic_reward_normalizer = None; self.intrinsic_reward_coeff = self.rnd_update_prop = 0.0

        # --- Optimizers ---
        lr_conf = { "lr_trunk": a_cfg.get("lr_trunk", a_cfg.get("lr_policy", self.lr_base)), 
                    "lr_policy": a_cfg.get("lr_policy", self.lr_base),
                    "lr_value_distill": a_cfg.get("lr_value", self.lr_base), 
                    "lr_baseline_distill": a_cfg.get("lr_baseline", self.lr_base),
                    "lr_iqn": a_cfg.get("lr_iqn", self.lr_base), 
                    "lr_rnd": a_cfg.get("lr_rnd_predictor", self.lr_base)}
        beta_conf = a_cfg.get("optimizer_betas", {}); wd_conf = a_cfg.get("weight_decay", {})

        trunk_params = list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])
        self.trunk_opt = optim.Adam(trunk_params, lr=lr_conf["lr_trunk"], betas=tuple(beta_conf.get("trunk", (0.9,0.98))), weight_decay=wd_conf.get("trunk",1e-5))
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr_conf["lr_policy"], betas=tuple(beta_conf.get("policy",(0.9,0.98))), weight_decay=wd_conf.get("policy",1e-5))
        
        val_dist_params = list(self.shared_critic.value_head_ppo.parameters()) + list(self.shared_critic.value_attention.parameters())
        if self.enable_iqn_distillation: val_dist_params += list(self.shared_critic.value_distill_net.parameters())
        self.value_distill_opt = optim.Adam(val_dist_params, lr=lr_conf["lr_value_distill"], betas=tuple(beta_conf.get("value",(0.9,0.95))), weight_decay=wd_conf.get("value",1e-4))
        
        base_dist_params = list(self.shared_critic.baseline_head_ppo.parameters()) + list(self.shared_critic.baseline_attention.parameters())
        if self.enable_iqn_distillation: base_dist_params += list(self.shared_critic.baseline_distill_net.parameters())
        self.baseline_distill_opt = optim.Adam(base_dist_params, lr=lr_conf["lr_baseline_distill"], betas=tuple(beta_conf.get("baseline",(0.9,0.95))), weight_decay=wd_conf.get("baseline",1e-4))

        if self.enable_iqn_distillation:
            self.value_iqn_opt = optim.Adam(self.shared_critic.value_iqn_net.parameters(), lr=lr_conf["lr_iqn"], betas=tuple(beta_conf.get("iqn",(0.9,0.98))), weight_decay=wd_conf.get("iqn",0.0))
            self.baseline_iqn_opt = optim.Adam(self.shared_critic.baseline_iqn_net.parameters(), lr=lr_conf["lr_iqn"], betas=tuple(beta_conf.get("iqn",(0.9,0.98))), weight_decay=wd_conf.get("iqn",0.0))
        else: self.value_iqn_opt = self.baseline_iqn_opt = None
        
        self.rnd_opt = optim.Adam(self.rnd_predictor_network.parameters(), lr=lr_conf["lr_rnd"], betas=tuple(beta_conf.get("rnd",(0.9,0.98))), weight_decay=wd_conf.get("rnd",0.0)) if self.enable_rnd else None
        
        # --- Schedulers ---
        sched_cfg = a_cfg.get("schedulers", {})
        def_lin_sched = {"start_factor":1.0, "end_factor":1.0, "total_iters":1} 
        def_val_sched = {"start_value":0.0, "end_value":0.0, "total_iters":1} 

        self.schedulers = {
            "lr_trunk": LinearLR(self.trunk_opt, **sched_cfg.get("lr_trunk", def_lin_sched)),
            "lr_policy": LinearLR(self.policy_opt, **sched_cfg.get("lr_policy", def_lin_sched)), 
            "lr_value_distill": LinearLR(self.value_distill_opt, **sched_cfg.get("lr_value", def_lin_sched)), 
            "lr_baseline_distill": LinearLR(self.baseline_distill_opt, **sched_cfg.get("lr_baseline", def_lin_sched)), 
            "entropy_coeff": LinearValueScheduler(**sched_cfg.get("entropy_coeff", {"start_value":self.entropy_coeff, "end_value":self.entropy_coeff, "total_iters":1})),
            "policy_clip": LinearValueScheduler(**sched_cfg.get("policy_clip", {"start_value":self.ppo_clip_range, "end_value":self.ppo_clip_range, "total_iters":1})),
            "value_clip": LinearValueScheduler(**sched_cfg.get("value_clip", {"start_value":self.value_clip_range, "end_value":self.value_clip_range, "total_iters":1})), 
            "max_grad_norm": LinearValueScheduler(**sched_cfg.get("max_grad_norm", {"start_value":self.max_grad_norm, "end_value":self.max_grad_norm, "total_iters":1}))
        }
        # Initialize scheduled values
        if self.schedulers["entropy_coeff"]: self.entropy_coeff = self.schedulers["entropy_coeff"].current_value()
        if self.schedulers["policy_clip"]: self.ppo_clip_range = self.schedulers["policy_clip"].current_value()
        if self.schedulers["value_clip"]: self.value_clip_range = self.schedulers["value_clip"].current_value()
        if self.schedulers["max_grad_norm"]: self.max_grad_norm = self.schedulers["max_grad_norm"].current_value()

        if self.enable_iqn_distillation and self.value_iqn_opt and self.baseline_iqn_opt:
            self.schedulers["lr_value_iqn"] = LinearLR(self.value_iqn_opt, **sched_cfg.get("lr_iqn", def_lin_sched))
            self.schedulers["lr_baseline_iqn"] = LinearLR(self.baseline_iqn_opt, **sched_cfg.get("lr_iqn", def_lin_sched))
        if self.enable_rnd and self.rnd_opt:
            self.schedulers["lr_rnd_predictor"] = LinearLR(self.rnd_opt, **sched_cfg.get("lr_rnd_predictor", def_lin_sched))

    def _determine_action_space(self, cfg: Dict[str, Any]): # Unchanged
        if "agent" not in cfg: raise ValueError("Missing 'action.agent' block")
        a_cfg = cfg["agent"]
        if "continuous" in a_cfg:
            self.action_space_type = "continuous"; self.action_dim = len(a_cfg["continuous"])
            self.total_action_dim_per_agent = self.action_dim
        elif "discrete" in a_cfg:
            self.action_space_type = "discrete"; self.action_dim = [d["num_choices"] for d in a_cfg["discrete"]]
            self.total_action_dim_per_agent = len(self.action_dim)
        else: raise ValueError("Specify discrete or continuous for action space")

    def _apply_memory_seq(self, seq_feats: torch.Tensor, init_h: Optional[torch.Tensor], return_hidden: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: # Unchanged
        if not self.memory_module: return (seq_feats, None) if return_hidden else seq_feats
        B, T, NA, F = seq_feats.shape; flat = seq_feats.reshape(B * NA, T, F)
        h0 = init_h.reshape(B * NA, self.memory_layers, self.memory_hidden).permute(1, 0, 2).contiguous() if init_h is not None else None
        flat_out, h_n = self.memory_module.forward_sequence(flat, h0)
        out = flat_out.reshape(B, T, NA, self.memory_hidden)
        if return_hidden:
            h_n_reshaped = h_n.permute(1, 0, 2).reshape(B, NA, self.memory_layers, self.memory_hidden)
            return out, h_n_reshaped
        return out

    def _sample_taus(self, batch_dim0_size: int, num_quantiles: int) -> torch.Tensor: # Unchanged
        return torch.rand(batch_dim0_size, num_quantiles, 1, device=self.device)

    @torch.no_grad()
    def get_actions(self, states: Dict[str, torch.Tensor], dones: torch.Tensor, truncs: torch.Tensor, h_prev_batch: Optional[torch.Tensor] = None, eval: bool = False, record: bool = True) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]: # Unchanged from previous full regen
        S_batch, B_env, NA_runtime = 1, states["agent"].shape[1] if "agent" in states else (states["central"].shape[1] if "central" in states else 1), self.num_agents
        
        emb, _ = self.embedding_net.get_base_embedding(states); emb = emb.squeeze(0) 

        if self.memory_module:
            if h_prev_batch is None: h_prev_batch = torch.zeros(B_env, NA_runtime, self.memory_layers, self.memory_hidden, device=self.device)
            flat_emb = emb.reshape(B_env * NA_runtime, -1)
            h_prev_flat = h_prev_batch.reshape(B_env * NA_runtime, self.memory_layers, self.memory_hidden).permute(1,0,2).contiguous()
            flat_out, h_next_flat = self.memory_module.forward_step(flat_emb, h_prev_flat)
            feats = flat_out.reshape(B_env, NA_runtime, -1)
            h_next = h_next_flat.permute(1,0,2).reshape(B_env, NA_runtime, self.memory_layers, self.memory_hidden)
            reset_mask = _bool_or(dones.squeeze(0), truncs.squeeze(0)).view(B_env, 1, 1, 1).float()
            h_next = h_next * (1.0 - reset_mask)
        else: feats, h_next = emb, None

        pol_in = feats.reshape(B_env * NA_runtime, -1)
        act_flat, lp_flat, ent_flat = self.policy_net.get_actions(pol_in, eval)
        
        if self.action_space_type == "discrete" and isinstance(self.action_dim, list) and len(self.action_dim) > 1:
             actions_shaped = act_flat.reshape(B_env, NA_runtime, len(self.action_dim))
        else:
             actions_shaped = act_flat.reshape(B_env, NA_runtime, self.action_dim if isinstance(self.action_dim, int) else self.action_dim[0])
        actions_ue = actions_shaped.reshape(B_env, NA_runtime * self.total_action_dim_per_agent)
        
        logp = lp_flat.reshape(B_env, NA_runtime); ent = ent_flat.reshape(B_env, NA_runtime)
        if record and self.state_recorder and "central" in states and states["central"].ndim >=2 :
            self.state_recorder.record_frame(states["central"][0,0].cpu().numpy().flatten())
        return actions_ue, (logp,ent), h_next

    def _compute_huber_quantile_loss(self, current_quantiles, target_quantiles_detached, taus_for_current, kappa): # Unchanged
        td_errors = target_quantiles_detached.unsqueeze(1) - current_quantiles.unsqueeze(2)  
        abs_td_errors = torch.abs(td_errors)
        huber_loss_matrix = torch.where(abs_td_errors <= kappa, 0.5 * td_errors.pow(2), kappa * (abs_td_errors - 0.5 * kappa))
        indicator = (td_errors < 0).float() 
        delta_weight = torch.abs(taus_for_current - indicator) 
        loss = (delta_weight * huber_loss_matrix).mean() 
        return loss

    def _update_target_iqn_networks(self): # Unchanged
        if self.enable_iqn_distillation:
            for target_param, online_param in zip(self.target_value_iqn_net.parameters(), self.shared_critic.value_iqn_net.parameters()):
                target_param.data.copy_(self.iqn_polyak * online_param.data + (1.0 - self.iqn_polyak) * target_param.data)
            for target_param, online_param in zip(self.target_baseline_iqn_net.parameters(), self.shared_critic.baseline_iqn_net.parameters()):
                target_param.data.copy_(self.iqn_polyak * online_param.data + (1.0 - self.iqn_polyak) * target_param.data)

    def _compute_gae_with_padding(self, r: torch.Tensor, v: torch.Tensor, v_next: torch.Tensor, d: torch.Tensor, tr: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor: # Unchanged
        B, T, _ = r.shape; returns = torch.zeros_like(r) 
        gae = torch.zeros(B, 1, device=self.device); mask_bt_expanded = mask_bt.unsqueeze(-1) 
        for t in reversed(range(T)):
            valid_mask_step = mask_bt_expanded[:, t] 
            valid_mask_next_step = mask_bt_expanded[:, t + 1] if t < T - 1 else torch.zeros_like(valid_mask_step)
            non_terminal_and_valid_next = (1.0 - d[:, t]) * (1.0 - tr[:, t]) * valid_mask_next_step
            delta = r[:, t] + self.gamma * v_next[:, t] * non_terminal_and_valid_next - v[:, t]
            gae = delta + self.gamma * self.lmbda * non_terminal_and_valid_next * gae
            current_return = gae + v[:, t] 
            returns[:, t] = current_return * valid_mask_step 
            gae = gae * valid_mask_step 
        return returns

    def update(self, padded_states_dict_seq: Dict[str, torch.Tensor], padded_actions_seq: torch.Tensor, padded_rewards_seq: torch.Tensor, padded_next_states_dict_seq: Dict[str, torch.Tensor], padded_dones_seq: torch.Tensor, padded_truncs_seq: torch.Tensor, initial_hidden_states_batch: Optional[torch.Tensor], attention_mask_batch: torch.Tensor) -> Dict[str, float]:
        B, T = attention_mask_batch.shape; NA = self.num_agents
        mask_bt = attention_mask_batch; mask_bt1 = mask_bt.unsqueeze(-1); mask_btna1 = mask_bt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, NA, 1)

        rewards_seq = padded_rewards_seq.clone()
        if self.rewards_normalizer:
            valid_r = rewards_seq[mask_bt1.bool()]; 
            if valid_r.numel(): self.rewards_normalizer.update(valid_r.unsqueeze(-1)); rewards_seq[mask_bt1.bool()] = self.rewards_normalizer.normalize(valid_r.unsqueeze(-1)).squeeze(-1).to(rewards_seq.dtype)
        
        # --- NO-GRAD PASS for PPO GAE and IQN Replay Buffer Storage ---
        with torch.no_grad():
            emb_seq, _ = self.embedding_net.get_base_embedding(padded_states_dict_seq)
            emb_next_seq, _ = self.embedding_net.get_base_embedding(padded_next_states_dict_seq)

            h_n_for_next_buffer = None
            if self.memory_module:
                feats_seq, h_n_for_next_buffer = self._apply_memory_seq(emb_seq, initial_hidden_states_batch, return_hidden=True)
                feats_next_seq = self._apply_memory_seq(emb_next_seq, h_n_for_next_buffer)
            else:
                feats_seq = self._apply_memory_seq(emb_seq, None); feats_next_seq = self._apply_memory_seq(emb_next_seq, None)
            
            # Push to IQN replay buffer before any reshaping for critic calls
            if self.enable_iqn_distillation and hasattr(self, 'iqn_replay_buffer') and self.iqn_replay_buffer.capacity > 0:
                 self.iqn_replay_buffer.push_batch(
                     feats_seq, feats_next_seq, rewards_seq, padded_dones_seq, 
                     attention_mask_batch, initial_hidden_states_batch, h_n_for_next_buffer
                 )

            B_T_flat_dim = B*T; B_T_NA_flat_dim = B*T*NA
            taus_no_grad_val = self._sample_taus(B_T_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None
            taus_no_grad_base = self._sample_taus(B_T_NA_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None

            old_logp_seq, _ = self._recompute_log_probs_from_features(feats_seq, padded_actions_seq)
            
            critic_args_val_no_grad = (feats_seq.view(B_T_flat_dim, NA, -1), taus_no_grad_val) if self.enable_iqn_distillation else (feats_seq.view(B_T_flat_dim, NA, -1),)
            old_val_output_no_grad = self.shared_critic.values(*critic_args_val_no_grad)
            old_val_distilled_norm = old_val_output_no_grad[0] if self.enable_iqn_distillation else old_val_output_no_grad
            old_val_distilled_norm = old_val_distilled_norm.view(B,T,1)

            critic_args_next_val_no_grad = (feats_next_seq.view(B_T_flat_dim, NA, -1), taus_no_grad_val) if self.enable_iqn_distillation else (feats_next_seq.view(B_T_flat_dim, NA, -1),)
            next_val_output_no_grad = self.shared_critic.values(*critic_args_next_val_no_grad)
            next_val_distilled_norm = next_val_output_no_grad[0] if self.enable_iqn_distillation else next_val_output_no_grad
            next_val_distilled_norm = next_val_distilled_norm.view(B,T,1)

            base_in_seq = self.embedding_net.get_baseline_embeddings(feats_seq, padded_actions_seq)
            base_in_seq_flat = base_in_seq.view(B_T_NA_flat_dim, base_in_seq.shape[-2], base_in_seq.shape[-1])
            critic_args_base_no_grad = (base_in_seq_flat, taus_no_grad_base) if self.enable_iqn_distillation else (base_in_seq_flat,)
            old_base_output_no_grad = self.shared_critic.baselines(*critic_args_base_no_grad)
            old_base_distilled_norm = old_base_output_no_grad[0] if self.enable_iqn_distillation else old_base_output_no_grad
            old_base_distilled_norm = old_base_distilled_norm.view(B,T,NA,1)
            
            intrinsic_na = torch.zeros(B,T,NA, device=self.device) 
            if self.enable_rnd: # RND Logic
                flat_feats_for_rnd = feats_seq.reshape(B*T*NA, -1)
                with torch.no_grad(): tgt_rnd = self.rnd_target_network(flat_feats_for_rnd)
                pred_rnd_no_grad = self.rnd_predictor_network(flat_feats_for_rnd).detach()
                mse_rnd = F.mse_loss(pred_rnd_no_grad, tgt_rnd, reduction="none").mean(-1)
                intrinsic_na = mse_rnd.reshape(B,T,NA)
                valid_intr_mask = mask_bt.unsqueeze(-1).expand_as(intrinsic_na).bool()
                valid_intr = intrinsic_na[valid_intr_mask]
                if valid_intr.numel() > 0:
                    self.intrinsic_reward_normalizer.update(valid_intr.unsqueeze(-1))
                    intrinsic_na[valid_intr_mask] = self.intrinsic_reward_normalizer.normalize(valid_intr.unsqueeze(-1)).squeeze(-1).to(intrinsic_na.dtype)
            intrinsic_seq = intrinsic_na.mean(-1, keepdim=True)

            rew_gae = rewards_seq + self.intrinsic_reward_coeff * intrinsic_seq
            old_val_distilled = self.value_popart.denormalize_outputs(old_val_distilled_norm) if self.enable_popart else old_val_distilled_norm
            next_val_distilled = self.value_popart.denormalize_outputs(next_val_distilled_norm) if self.enable_popart else next_val_distilled_norm
            returns_seq = self._compute_gae_with_padding(rew_gae, old_val_distilled, next_val_distilled, padded_dones_seq, padded_truncs_seq, mask_bt)

            raw_ext_mean = float(padded_rewards_seq[mask_bt1.bool()].mean()) if mask_bt1.any() else 0.0
            proc_rew_mean = float(rewards_seq[mask_bt1.bool()].mean()) if mask_bt1.any() else 0.0
            ret_vals = returns_seq[mask_bt1.bool()]; ret_mean = float(ret_vals.mean()) if ret_vals.numel() else 0.0; ret_std = float(ret_vals.std()) if ret_vals.numel() else 0.0
            
            if self.enable_popart:
                self.value_popart.update_stats(returns_seq[mask_bt1.bool()])
                self.baseline_popart.update_stats(returns_seq.unsqueeze(2).expand(-1,-1,NA,-1)[mask_btna1.bool()])
        
        # --- PPO Training Loop ---
        logs_acc: Dict[str, List[float]] = {k: [] for k in ["policy", "value_distill", "baseline_distill", "entropy", "grad_ppo", "adv_mean", "adv_std", "value_iqn", "baseline_iqn", "grad_iqn"]}
        if self.enable_rnd: logs_acc["rnd"] = []
        
        idx = np.arange(B)
        for _epoch_ppo in range(self.epochs):
            np.random.shuffle(idx)
            for mb_start in range(0, B, self.mini_batch_size):
                mb_idx = idx[mb_start : mb_start + self.mini_batch_size]; M = mb_idx.shape[0]
                if M == 0: continue
                mask_mb = mask_bt[mb_idx] # (M, T)
                
                M_T_flat_dim = M*T; M_T_NA_flat_dim = M*T*NA
                taus_for_ppo_mb_val = self._sample_taus(M_T_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None
                taus_for_ppo_mb_base = self._sample_taus(M_T_NA_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None

                emb_mb, _ = self.embedding_net.get_base_embedding({k: v[mb_idx] for k,v in padded_states_dict_seq.items()})
                init_h_mb = initial_hidden_states_batch[mb_idx] if initial_hidden_states_batch is not None and initial_hidden_states_batch.shape[0] == B else None 
                if init_h_mb is not None and init_h_mb.shape[0] != M : init_h_mb = init_h_mb[:M] 

                feats_mb = self._apply_memory_seq(emb_mb, init_h_mb) # (M, T, NA, feat_dim)
                
                new_lp_mb, ent_mb = self._recompute_log_probs_from_features(feats_mb, padded_actions_seq[mb_idx]) # (M,T,NA)
                
                critic_args_val_mb = (feats_mb.view(M_T_flat_dim, NA, -1), taus_for_ppo_mb_val) if self.enable_iqn_distillation else (feats_mb.view(M_T_flat_dim, NA, -1),)
                new_val_output_mb = self.shared_critic.values(*critic_args_val_mb)
                new_val_distilled_norm_mb = new_val_output_mb[0] if self.enable_iqn_distillation else new_val_output_mb
                new_val_distilled_norm_mb = new_val_distilled_norm_mb.view(M,T,1)

                base_in_mb = self.embedding_net.get_baseline_embeddings(feats_mb, padded_actions_seq[mb_idx]) # (M,T,NA,SeqLen,H)
                base_in_mb_flat = base_in_mb.view(M_T_NA_flat_dim, base_in_mb.shape[-2], base_in_mb.shape[-1])
                critic_args_base_mb = (base_in_mb_flat, taus_for_ppo_mb_base) if self.enable_iqn_distillation else (base_in_mb_flat,)
                new_base_output_mb = self.shared_critic.baselines(*critic_args_base_mb)
                new_base_distilled_norm_mb = new_base_output_mb[0] if self.enable_iqn_distillation else new_base_output_mb
                new_base_distilled_norm_mb = new_base_distilled_norm_mb.view(M,T,NA,1)

                old_lp_mb = old_logp_seq[mb_idx]
                old_val_distilled_mb_norm = old_val_distilled_norm[mb_idx]
                old_base_distilled_mb_norm = old_base_distilled_norm[mb_idx]
                returns_mb = returns_seq[mb_idx]

                new_base_denorm_distilled = self.baseline_popart.denormalize_outputs(new_base_distilled_norm_mb) if self.enable_popart else new_base_distilled_norm_mb
                adv = (returns_mb.unsqueeze(2).expand_as(new_base_distilled_norm_mb) - new_base_denorm_distilled).squeeze(-1)
                
                adv_mask_mb = mask_mb.unsqueeze(-1).expand_as(adv) 
                if self.normalize_adv:
                    valid_adv_data = adv[adv_mask_mb.bool()]
                    if valid_adv_data.numel() > 0:
                        m,s = valid_adv_data.mean(), valid_adv_data.std()+1e-8; adv_norm_data = (valid_adv_data-m)/s
                        tmp_adv = torch.zeros_like(adv); tmp_adv[adv_mask_mb.bool()] = adv_norm_data; adv = tmp_adv
                        if _epoch_ppo == 0 and mb_start == 0: logs_acc["adv_mean"].append(m.item()); logs_acc["adv_std"].append((s-1e-8).item())
                elif adv_mask_mb.any() and _epoch_ppo == 0 and mb_start == 0:
                    logs_acc["adv_mean"].append(float(adv[adv_mask_mb.bool()].mean())); logs_acc["adv_std"].append(float(adv[adv_mask_mb.bool()].std()))
                
                pol_loss = self._ppo_clip_loss(new_lp_mb, old_lp_mb, adv.detach(), self.ppo_clip_range, mask_mb.unsqueeze(-1).expand_as(new_lp_mb))
                
                tgt_val_distilled = returns_mb if not self.enable_popart else self.value_popart.normalize_targets(returns_mb)
                val_loss = self._clipped_value_loss(old_val_distilled_mb_norm.detach(), new_val_distilled_norm_mb, tgt_val_distilled, self.value_clip_range, mask=mask_mb.unsqueeze(-1))
                
                tgt_base_distilled = returns_mb.unsqueeze(2).expand_as(new_base_distilled_norm_mb)
                if self.enable_popart: tgt_base_distilled = self.baseline_popart.normalize_targets(tgt_base_distilled)
                base_loss = self._clipped_value_loss(old_base_distilled_mb_norm.detach(), new_base_distilled_norm_mb, tgt_base_distilled, self.value_clip_range, mask=mask_mb.unsqueeze(-1).unsqueeze(-1).expand_as(new_base_distilled_norm_mb))
                
                ent_mask_mb_expanded = mask_mb.unsqueeze(-1).expand_as(ent_mb)
                ent_terms_masked = -ent_mb * ent_mask_mb_expanded 
                ent_loss = ent_terms_masked.sum() / ent_mask_mb_expanded.sum().clamp(min=1e-8)


                ppo_total_loss = pol_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss + self.entropy_coeff * ent_loss

                rnd_loss = torch.tensor(0.0, device=self.device) 
                if self.enable_rnd and self.rnd_update_prop > 0.0:
                    # RND features mask: (M, T, NA)
                    valid_rnd_feats_mask_mb_bool = mask_mb.unsqueeze(-1).expand_as(feats_mb[...,0]).bool() 
                    # Flatten feats_mb to (M*T*NA, FeatDim) then apply mask
                    flat_feats_mb_for_rnd = feats_mb.reshape(M*T*NA, -1)
                    valid_feats_mb_for_rnd = flat_feats_mb_for_rnd[valid_rnd_feats_mask_mb_bool.reshape(-1)]

                    if valid_feats_mb_for_rnd.numel() > 0:
                        num_rnd_samples = valid_feats_mb_for_rnd.shape[0]
                        k_rnd = int(num_rnd_samples * self.rnd_update_prop)
                        if k_rnd > 0:
                            sel_rnd = torch.randperm(num_rnd_samples, device=self.device)[:k_rnd]
                            pred_rnd_val = self.rnd_predictor_network(valid_feats_mb_for_rnd[sel_rnd])
                            with torch.no_grad(): tgt_rnd_val = self.rnd_target_network(valid_feats_mb_for_rnd[sel_rnd])
                            rnd_loss = F.mse_loss(pred_rnd_val, tgt_rnd_val); ppo_total_loss += rnd_loss
                
                self.trunk_opt.zero_grad(); self.policy_opt.zero_grad()
                self.value_distill_opt.zero_grad(); self.baseline_distill_opt.zero_grad()
                if self.rnd_opt: self.rnd_opt.zero_grad()
                ppo_total_loss.backward()
                
                ppo_path_params = list(self.embedding_net.parameters()) + list(self.policy_net.parameters()) + \
                                  list(self.shared_critic.value_head_ppo.parameters()) + list(self.shared_critic.value_attention.parameters()) + \
                                  list(self.shared_critic.baseline_head_ppo.parameters()) + list(self.shared_critic.baseline_attention.parameters())
                if self.enable_memory: ppo_path_params += list(self.memory_module.parameters())
                if self.enable_iqn_distillation:
                    ppo_path_params += list(self.shared_critic.value_distill_net.parameters())
                    ppo_path_params += list(self.shared_critic.baseline_distill_net.parameters())
                if self.enable_rnd: ppo_path_params += list(self.rnd_predictor_network.parameters())
                gn_ppo = clip_grad_norm_(ppo_path_params, self.max_grad_norm)

                self.trunk_opt.step(); self.policy_opt.step()
                self.value_distill_opt.step(); self.baseline_distill_opt.step()
                if self.rnd_opt: self.rnd_opt.step()

                logs_acc["policy"].append(pol_loss.item()); logs_acc["value_distill"].append(val_loss.item())
                logs_acc["baseline_distill"].append(base_loss.item()); 
                valid_ent_mb_mask = mask_mb.unsqueeze(-1).expand_as(ent_mb).bool()
                logs_acc["entropy"].append(ent_mb[valid_ent_mb_mask].mean().item() if valid_ent_mb_mask.any() else 0.0)
                logs_acc["grad_ppo"].append(gn_ppo.item() if torch.is_tensor(gn_ppo) else gn_ppo)
                if self.enable_rnd: logs_acc["rnd"].append(rnd_loss.item())
            
            for sch_name_key in ["lr_trunk", "lr_policy", "lr_value_distill", "lr_baseline_distill"]:
                if self.schedulers.get(sch_name_key): self.schedulers[sch_name_key].step()
            if self.enable_rnd and self.schedulers.get("lr_rnd_predictor"): self.schedulers["lr_rnd_predictor"].step()
            if self.schedulers.get("entropy_coeff"): self.entropy_coeff = self.schedulers["entropy_coeff"].step()
            if self.schedulers.get("policy_clip"): self.ppo_clip_range = self.schedulers["policy_clip"].step()
            if self.schedulers.get("value_clip"): self.value_clip_range = self.schedulers["value_clip"].step()
            if self.schedulers.get("max_grad_norm"): self.max_grad_norm = self.schedulers["max_grad_norm"].step()

        # --- IQN Training Loop ---
        if self.enable_iqn_distillation and hasattr(self, 'iqn_replay_buffer') and len(self.iqn_replay_buffer) >= self.iqn_batch_size :
            self._iqn_total_updates_counter +=1
            current_iqn_grad_norm_sum = 0.0
            num_iqn_batches_processed = 0
            for _epoch_iqn in range(self.iqn_epochs):
                iqn_batch_tuple = self.iqn_replay_buffer.sample(self.iqn_batch_size)
                if iqn_batch_tuple is None: continue

                s_feats_iqn_mb, ns_feats_iqn_mb, r_iqn_mb, d_iqn_mb, mask_iqn_mb, _, _ = iqn_batch_tuple
                # Shapes from buffer: (iqn_B, T, NA, Feat), (iqn_B, T, 1), (iqn_B, T)
                
                iqn_mb_B, iqn_mb_T, iqn_mb_NA, iqn_mb_F = s_feats_iqn_mb.shape
                
                # Flatten sequences and select valid steps using mask_iqn_mb
                valid_mask_flat = mask_iqn_mb.reshape(-1).bool() # (iqn_B * T)
                
                # These are now features for valid (non-padded) timesteps across the sampled sequences
                s_feats_iqn_valid_steps = s_feats_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, iqn_mb_NA, iqn_mb_F)[valid_mask_flat]
                ns_feats_iqn_valid_steps = ns_feats_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, iqn_mb_NA, iqn_mb_F)[valid_mask_flat]
                r_iqn_valid_steps = r_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, 1)[valid_mask_flat]
                d_iqn_valid_steps = d_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, 1)[valid_mask_flat]

                if s_feats_iqn_valid_steps.shape[0] == 0: continue # No valid steps in this minibatch
                
                eff_iqn_b_steps = s_feats_iqn_valid_steps.shape[0] # Number of valid (state, next_state) items (each is a (NA,Feat) tensor)

                # For value_iqn, aggregate agent features: (eff_iqn_b_steps, Feat)
                s_val_feats_agg = s_feats_iqn_valid_steps.mean(dim=1) 
                ns_val_feats_agg = ns_feats_iqn_valid_steps.mean(dim=1) 
                
                # For baseline_iqn, features are per-agent: (eff_iqn_b_steps * NA_from_data, Feat)
                s_base_feats_per_agent = s_feats_iqn_valid_steps.reshape(-1, iqn_mb_F) 
                ns_base_feats_per_agent = ns_feats_iqn_valid_steps.reshape(-1, iqn_mb_F) 
                
                r_val_iqn = r_iqn_valid_steps # (eff_iqn_b_steps, 1)
                d_val_iqn = d_iqn_valid_steps # (eff_iqn_b_steps, 1)
                r_base_iqn = r_iqn_valid_steps.repeat_interleave(iqn_mb_NA, dim=0) if s_base_feats_per_agent.shape[0] > 0 else torch.empty(0,1,device=self.device)
                d_base_iqn = d_iqn_valid_steps.repeat_interleave(iqn_mb_NA, dim=0) if s_base_feats_per_agent.shape[0] > 0 else torch.empty(0,1,device=self.device)

                # Sample taus for current and target estimations
                taus_current_val = self._sample_taus(eff_iqn_b_steps, self.num_quantiles)
                taus_next_val    = self._sample_taus(eff_iqn_b_steps, self.num_quantiles_prime) # num_quantiles_prime for target
                
                value_iqn_loss = torch.tensor(0.0, device=self.device)
                if eff_iqn_b_steps > 0:
                    current_value_quantiles = self.shared_critic.value_iqn_net(s_val_feats_agg, taus_current_val) # (eff_iqn_b_steps, N)
                    with torch.no_grad(): next_value_quantiles_target = self.target_value_iqn_net(ns_val_feats_agg, taus_next_val) # (eff_iqn_b_steps, N')
                    # Bellman targets: (eff_iqn_b_steps, 1) + scalar * (eff_iqn_b_steps, 1) * (eff_iqn_b_steps, N') -> (eff_iqn_b_steps, N')
                    value_bellman_targets = r_val_iqn + self.gamma * (1.0 - d_val_iqn) * next_value_quantiles_target.detach()
                    value_iqn_loss = self._compute_huber_quantile_loss(current_value_quantiles, value_bellman_targets, taus_current_val, self.iqn_kappa)
                
                baseline_iqn_loss = torch.tensor(0.0, device=self.device)
                if s_base_feats_per_agent.shape[0] > 0: # If there are any per-agent features to process
                    eff_base_batch_size = s_base_feats_per_agent.shape[0]
                    taus_current_base = self._sample_taus(eff_base_batch_size, self.num_quantiles)
                    taus_next_base    = self._sample_taus(eff_base_batch_size, self.num_quantiles_prime)
                    current_baseline_quantiles = self.shared_critic.baseline_iqn_net(s_base_feats_per_agent, taus_current_base)
                    with torch.no_grad(): next_baseline_quantiles_target = self.target_baseline_iqn_net(ns_base_feats_per_agent, taus_next_base)
                    baseline_bellman_targets = r_base_iqn + self.gamma * (1.0 - d_base_iqn) * next_baseline_quantiles_target.detach()
                    baseline_iqn_loss = self._compute_huber_quantile_loss(current_baseline_quantiles, baseline_bellman_targets, taus_current_base, self.iqn_kappa)
                
                total_iqn_loss = self.iqn_loss_coeff * (value_iqn_loss + baseline_iqn_loss)

                self.value_iqn_opt.zero_grad(); self.baseline_iqn_opt.zero_grad()
                if total_iqn_loss.requires_grad and total_iqn_loss.abs().item() > 1e-9 : # Only backward if loss is not trivial
                    total_iqn_loss.backward()
                    gn_val_iqn = clip_grad_norm_(self.shared_critic.value_iqn_net.parameters(), self.iqn_max_grad_norm)
                    gn_base_iqn = clip_grad_norm_(self.shared_critic.baseline_iqn_net.parameters(), self.iqn_max_grad_norm)
                    self.value_iqn_opt.step(); self.baseline_iqn_opt.step()
                    current_iqn_grad_norm_sum += (( (gn_val_iqn.item() if torch.is_tensor(gn_val_iqn) else gn_val_iqn) + \
                                                   (gn_base_iqn.item() if torch.is_tensor(gn_base_iqn) else gn_base_iqn) ) / 2.0) if gn_val_iqn is not None and gn_base_iqn is not None else 0.0
                num_iqn_batches_processed +=1
                
                logs_acc["value_iqn"].append(value_iqn_loss.item()); logs_acc["baseline_iqn"].append(baseline_iqn_loss.item())
            
            if num_iqn_batches_processed > 0: 
                logs_acc["grad_iqn"].append(current_iqn_grad_norm_sum / num_iqn_batches_processed)
            elif "grad_iqn" not in logs_acc or not logs_acc["grad_iqn"]: # Add 0 if no batches processed in this update
                 logs_acc["grad_iqn"].append(0.0)


            if self.schedulers.get("lr_value_iqn"): self.schedulers["lr_value_iqn"].step()
            if self.schedulers.get("lr_baseline_iqn"): self.schedulers["lr_baseline_iqn"].step()
            if self._iqn_total_updates_counter % self.iqn_target_update_freq == 0: self._update_target_iqn_networks()
        
        # --- Logging ---
        logs = {k: (np.mean(v) if v else 0.0) for k,v in logs_acc.items()}
        logs.update({
            "raw_reward_mean": raw_ext_mean, "processed_reward_mean": proc_rew_mean, "return_mean": ret_mean, "return_std": ret_std,
            "entropy_coeff": self.entropy_coeff, "ppo_clip": self.ppo_clip_range, "value_clip": self.value_clip_range, "max_grad_norm_ppo": self.max_grad_norm,
            "lr_trunk": self.schedulers["lr_trunk"].get_last_lr()[0] if self.schedulers.get("lr_trunk") else lr_conf["lr_trunk"],
            "lr_policy": self.schedulers["lr_policy"].get_last_lr()[0] if self.schedulers.get("lr_policy") else lr_conf["lr_policy"],
            "lr_value_distill": self.schedulers["lr_value_distill"].get_last_lr()[0] if self.schedulers.get("lr_value_distill") else lr_conf["lr_value_distill"],
            "lr_baseline_distill": self.schedulers["lr_baseline_distill"].get_last_lr()[0] if self.schedulers.get("lr_baseline_distill") else lr_conf["lr_baseline_distill"],
        })
        if self.enable_iqn_distillation:
            logs["lr_value_iqn"] = self.schedulers["lr_value_iqn"].get_last_lr()[0] if self.schedulers.get("lr_value_iqn") else lr_conf["lr_iqn"]
            logs["lr_baseline_iqn"] = self.schedulers["lr_baseline_iqn"].get_last_lr()[0] if self.schedulers.get("lr_baseline_iqn") else lr_conf["lr_iqn"]
            logs["iqn_max_grad_norm"] = self.iqn_max_grad_norm
        if self.enable_rnd: logs["lr_rnd"] = self.schedulers["lr_rnd_predictor"].get_last_lr()[0] if self.schedulers.get("lr_rnd_predictor") else lr_conf["lr_rnd"]
        if self.enable_popart:
            logs.update({"popart_value_mu": self.value_popart.mu.item(), "popart_value_sigma": self.value_popart.sigma.item(),
                         "popart_baseline_mu": self.baseline_popart.mu.item(), "popart_baseline_sigma": self.baseline_popart.sigma.item()})
        return logs

    def _ppo_clip_loss(self, new_lp, old_lp, adv, clip_range, mask): 
        # Ensure mask has same number of dims as new_lp for broadcasting before sum/mean
        if mask.ndim < new_lp.ndim: mask = mask.unsqueeze(-1).expand_as(new_lp)
        ratio = torch.exp((new_lp - old_lp).clamp(-10,10))
        s1 = ratio * adv; s2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * adv
        loss_terms = -torch.min(s1,s2) * mask 
        return loss_terms.sum() / mask.sum().clamp(min=1e-8)

    def _clipped_value_loss(self, old, new, target, clip_range, mask, reduction="mean"): 
        if mask.ndim < new.ndim: mask = mask.unsqueeze(-1).expand_as(new)
        clipped_new = old + (new - old).clamp(-clip_range, clip_range)
        loss_terms = torch.max((new - target).pow(2), (clipped_new - target).pow(2))
        masked_loss_terms = loss_terms * mask 
        if reduction == "mean": return masked_loss_terms.sum() / mask.sum().clamp(min=1e-8) 
        return masked_loss_terms

    def _recompute_log_probs_from_features(self, feats_seq, actions_seq): # Unchanged
        B,T,NA,F = feats_seq.shape; flat_feats = feats_seq.reshape(B*T*NA, F)
        if self.action_space_type == "discrete" and isinstance(self.action_dim, list) and len(self.action_dim) > 1: 
            flat_actions = actions_seq.reshape(B*T*NA, self.total_action_dim_per_agent)
        else: 
             flat_actions = actions_seq.reshape(B*T*NA, self.action_dim if isinstance(self.action_dim, int) else self.action_dim[0] )
        lp_flat, ent_flat = self.policy_net.recompute_log_probs(flat_feats, flat_actions)
        return lp_flat.reshape(B,T,NA), ent_flat.reshape(B,T,NA)

    def parameters(self): # For PPO path global grad clipping
        ppo_path_params = list(self.embedding_net.parameters()) + list(self.policy_net.parameters()) + \
                          list(self.shared_critic.value_head_ppo.parameters()) + list(self.shared_critic.value_attention.parameters()) + \
                          list(self.shared_critic.baseline_head_ppo.parameters()) + list(self.shared_critic.baseline_attention.parameters())
        if self.enable_memory: ppo_path_params += list(self.memory_module.parameters())
        if self.enable_iqn_distillation:
            ppo_path_params += list(self.shared_critic.value_distill_net.parameters())
            ppo_path_params += list(self.shared_critic.baseline_distill_net.parameters())
        if self.enable_rnd: ppo_path_params += list(self.rnd_predictor_network.parameters())
        return ppo_path_params

    def _get_policy_lr(self): # Unchanged
        return float(np.mean([pg["lr"] for pg in self.policy_opt.param_groups]))