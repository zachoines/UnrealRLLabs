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
    TanhContinuousPolicyNetwork,
    GaussianPolicyNetwork,
    RNDTargetNetwork,
    RNDPredictorNetwork,
)
from Source.Memory import GRUSequenceMemory

# -----------------------------------------------------------------------------
# IQN Replay Buffer - Stores individual sequences
# -----------------------------------------------------------------------------
class IQNReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        # Stored:
        # (
        #   s_val_feats_agg_cpu,          # Aggregated features for centralized value IQN (T, F_agg_val)
        #   ns_val_feats_agg_cpu,         # Next aggregated features for centralized value IQN (T, F_agg_val)
        #   s_base_feats_comb_agg_cpu,    # Aggregated counterfactual features for baseline IQN (T, NA, F_agg_base)
        #   ns_base_feats_comb_agg_cpu,   # Next aggregated counterfactual features for baseline IQN (T, NA, F_agg_base)
        #   actions_seq_cpu,              # Actions taken in the sequence (T, NA, ActionDim)
        #   reward_single_seq_cpu,        # Rewards (T, 1) - centralized
        #   done_single_seq_cpu,          # Dones (T, 1) - centralized
        #   attention_mask_single_seq_cpu,# Attention mask (T,)
        #   initial_h_single_seq_cpu,     # Optional: Initial hidden state for memory (NA, Layers, Hidden)
        #   next_initial_h_single_seq_cpu # Optional: Hidden state for next sequence start (NA, Layers, Hidden)
        # )
        self.buffer: List[Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            Optional[torch.Tensor], Optional[torch.Tensor]
        ]] = []
        self.position = 0

    def push_batch(self,
                   # Features for centralized value IQN (already attention-processed and aggregated)
                   batch_s_val_feats_agg: torch.Tensor,      # (B_ppo, T, F_agg_val)
                   batch_ns_val_feats_agg: torch.Tensor,     # (B_ppo, T, F_agg_val)
                   # Features for baseline IQN (already counterfactual, attention-processed, and aggregated)
                   batch_s_base_feats_comb_agg: torch.Tensor, # (B_ppo, T, NA, F_agg_base)
                   batch_ns_base_feats_comb_agg: torch.Tensor,# (B_ppo, T, NA, F_agg_base)
                   # Other sequence data
                   batch_actions_seq: torch.Tensor,          # (B_ppo, T, NA, ActionDim)
                   batch_rewards_seq: torch.Tensor,          # (B_ppo, T, 1)
                   batch_dones_seq: torch.Tensor,            # (B_ppo, T, 1)
                   batch_attention_mask: torch.Tensor,       # (B_ppo, T)
                   batch_initial_h: Optional[torch.Tensor] = None, # (B_ppo, NA, Layers, Hidden)
                   batch_next_initial_h: Optional[torch.Tensor] = None): # (B_ppo, NA, Layers, Hidden)

        if self.capacity == 0: return

        num_sequences_in_batch = batch_s_val_feats_agg.shape[0]

        for i in range(num_sequences_in_batch):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None) # type: ignore

            initial_h_single = batch_initial_h[i].detach().cpu() if batch_initial_h is not None and batch_initial_h.numel() > 0 else None
            next_initial_h_single = batch_next_initial_h[i].detach().cpu() if batch_next_initial_h is not None and batch_next_initial_h.numel() > 0 else None

            self.buffer[self.position] = (
                batch_s_val_feats_agg[i].detach().cpu(),
                batch_ns_val_feats_agg[i].detach().cpu(),
                batch_s_base_feats_comb_agg[i].detach().cpu(),
                batch_ns_base_feats_comb_agg[i].detach().cpu(),
                batch_actions_seq[i].detach().cpu(),
                batch_rewards_seq[i].detach().cpu(),
                batch_dones_seq[i].detach().cpu(),
                batch_attention_mask[i].detach().cpu(),
                initial_h_single,
                next_initial_h_single
            )
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Optional[Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            Optional[torch.Tensor], Optional[torch.Tensor]]]:

        if len(self.buffer) < batch_size:
            return None

        batch_tuples = random.sample(self.buffer, batch_size)

        s_val_agg_list, ns_val_agg_list, s_base_agg_list, ns_base_agg_list, \
        actions_list, rewards_list, dones_list, attn_mask_list, \
        initial_h_list, next_initial_h_list = zip(*batch_tuples)

        s_val_feats_agg_dev = torch.stack(s_val_agg_list).to(self.device)
        ns_val_feats_agg_dev = torch.stack(ns_val_agg_list).to(self.device)
        s_base_feats_comb_agg_dev = torch.stack(s_base_agg_list).to(self.device)
        ns_base_feats_comb_agg_dev = torch.stack(ns_base_agg_list).to(self.device)
        actions_seq_dev = torch.stack(actions_list).to(self.device)
        rewards_seq_dev = torch.stack(rewards_list).to(self.device)
        dones_seq_dev = torch.stack(dones_list).to(self.device)
        attention_mask_dev = torch.stack(attn_mask_list).to(self.device)

        initial_h_filtered = [h for h in initial_h_list if h is not None]
        initial_h_dev = None
        if initial_h_filtered and all(h.shape == initial_h_filtered[0].shape for h in initial_h_filtered):
            if len(initial_h_filtered) == batch_size:
                initial_h_dev = torch.stack(initial_h_filtered).to(self.device)

        next_initial_h_filtered = [h for h in next_initial_h_list if h is not None]
        next_initial_h_dev = None
        if next_initial_h_filtered and all(h.shape == next_initial_h_filtered[0].shape for h in next_initial_h_filtered):
            if len(next_initial_h_filtered) == batch_size:
                next_initial_h_dev = torch.stack(next_initial_h_filtered).to(self.device)

        return (
            s_val_feats_agg_dev, ns_val_feats_agg_dev,
            s_base_feats_comb_agg_dev, ns_base_feats_comb_agg_dev,
            actions_seq_dev,
            rewards_seq_dev, dones_seq_dev, attention_mask_dev,
            initial_h_dev, next_initial_h_dev
        )

    def __len__(self) -> int:
        return len(self.buffer)

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _bool_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise OR that works on float or bool tensors."""
    return (a > 0.5) | (b > 0.5)

def _get_avg_grad_mag(params_list: List[torch.nn.Parameter]) -> float:
    """Computes the average magnitude of gradients for a list of parameters."""
    total_abs_grad = 0.0
    num_params_with_grad = 0
    if not params_list: # Handle empty list
        return 0.0

    for p in params_list:
        if p.grad is not None:
            total_abs_grad += p.grad.abs().mean().item()
            num_params_with_grad += 1
    return total_abs_grad / num_params_with_grad if num_params_with_grad > 0 else 0.0

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
            self.lr_iqn = a_cfg.get("lr_iqn", 0.0001) # General LR for IQN components

            self.value_iqn_loss_coeff = a_cfg.get("value_iqn_loss_coeff", a_cfg.get("iqn_loss_coeff", 0.25))
            self.baseline_iqn_loss_coeff = a_cfg.get("baseline_iqn_loss_coeff", a_cfg.get("iqn_loss_coeff", 0.25))

            self.iqn_target_update_freq = t_cfg.get("iqn_target_update_freq", 1000)
            self.iqn_polyak = iqn_params_from_critic_config.get("polyak_tau", 0.005)
            self.iqn_epochs = t_cfg.get("iqn_epochs", t_cfg.get("epochs",1))
            self.iqn_batch_size = t_cfg.get("iqn_batch_size", self.mini_batch_size)
            self.iqn_max_grad_norm = a_cfg.get("iqn_max_grad_norm", self.max_grad_norm)
            self._iqn_total_updates_counter = 0
            replay_buffer_size = t_cfg.get("iqn_replay_buffer_size", 10000)
            self.iqn_replay_buffer = IQNReplayBuffer(replay_buffer_size, self.device)
        else:
            self.num_quantiles = 0
            self.num_quantiles_prime = 0
            self.value_iqn_loss_coeff = 0.0
            self.baseline_iqn_loss_coeff = 0.0


        # --- Environment & Agent Setup ---
        self.num_agents = env_params.get("MaxAgents", 1)
        if "agent" in env_shape["state"]: self.num_agents = env_shape["state"]["agent"].get("max", self.num_agents)
        self.state_recorder = StateRecorder(cfg.get("StateRecorder")) if "StateRecorder" in cfg and cfg.get("StateRecorder") else None
        self._determine_action_space(env_shape["action"])

        # --- Network Definitions ---
        emb_cfg = a_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        self.embedding_net = MultiAgentEmbeddingNetwork(emb_cfg).to(device)
        emb_out_dim = emb_cfg["cross_attention_feature_extractor"]["embed_dim"]

        self.enable_memory = a_cfg.get("enable_memory", False)
        if self.enable_memory:
            mem_cfg = a_cfg.get(a_cfg.get("memory_type", "gru"), {}); mem_cfg.setdefault("input_size", emb_out_dim)
            self.memory_hidden = mem_cfg.get("hidden_size", 128); self.memory_layers = mem_cfg.get("num_layers", 1)
            self.memory_module = GRUSequenceMemory(**mem_cfg).to(device); trunk_dim = self.memory_hidden
        else:
            self.memory_module = None; self.memory_hidden = self.memory_layers = 0; trunk_dim = emb_out_dim

        pol_cfg = a_cfg["networks"]["policy_network"].copy(); pol_cfg.update({"in_features": trunk_dim, "out_features": self.action_dim})

        policy_type_str = pol_cfg.get("type", self.action_space_type)
        if policy_type_str == "continuous" or policy_type_str == "beta":
            self.policy_net = BetaPolicyNetwork(**pol_cfg).to(device)
            self.action_space_type = "beta"
        elif policy_type_str == "tanh_continuous":
            self.policy_net = TanhContinuousPolicyNetwork(**pol_cfg).to(device)
            self.action_space_type = "tanh_continuous"
        elif policy_type_str == "gaussian":
            self.policy_net = GaussianPolicyNetwork(**pol_cfg).to(device)
            self.action_space_type = "gaussian"
        elif policy_type_str == "discrete":
            self.policy_net = DiscretePolicyNetwork(**pol_cfg).to(device)
            self.action_space_type = "discrete"
        else:
            raise ValueError(f"Unsupported policy_network type: {policy_type_str}")

        critic_full_config["feature_dim_for_iqn"] = emb_out_dim
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
            rnorm_cfg = a_cfg.get("rewards_normalizer", None);
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
                    "lr_value_iqn": a_cfg.get("lr_value_iqn", a_cfg.get("lr_iqn", self.lr_base)),
                    "lr_baseline_iqn": a_cfg.get("lr_baseline_iqn", a_cfg.get("lr_iqn", self.lr_base)),
                    "lr_rnd": a_cfg.get("lr_rnd_predictor", self.lr_base)}
        beta_conf = a_cfg.get("optimizer_betas", {}); wd_conf = a_cfg.get("weight_decay", {})

        trunk_params = list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])
        self.trunk_opt = optim.Adam(trunk_params, lr=lr_conf["lr_trunk"], betas=tuple(beta_conf.get("trunk", (0.9,0.98))), weight_decay=wd_conf.get("trunk",1e-5))
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr_conf["lr_policy"], betas=tuple(beta_conf.get("policy",(0.9,0.98))), weight_decay=wd_conf.get("policy",1e-5))

        val_dist_params = list(self.shared_critic.value_head_ppo.parameters()) + \
                          list(self.shared_critic.value_attention.parameters())
        if self.enable_iqn_distillation: val_dist_params += list(self.shared_critic.value_distill_net.parameters())
        self.value_distill_opt = optim.Adam(val_dist_params, lr=lr_conf["lr_value_distill"], betas=tuple(beta_conf.get("value",(0.9,0.95))), weight_decay=wd_conf.get("value",1e-4))

        base_dist_params = list(self.shared_critic.baseline_head_ppo.parameters()) + \
                           list(self.shared_critic.baseline_attention.parameters())
        if self.enable_iqn_distillation: base_dist_params += list(self.shared_critic.baseline_distill_net.parameters())
        self.baseline_distill_opt = optim.Adam(base_dist_params, lr=lr_conf["lr_baseline_distill"], betas=tuple(beta_conf.get("baseline",(0.9,0.95))), weight_decay=wd_conf.get("baseline",1e-4))

        if self.enable_iqn_distillation:
            self.value_iqn_opt = optim.Adam(self.shared_critic.value_iqn_net.parameters(), lr=lr_conf["lr_value_iqn"], betas=tuple(beta_conf.get("iqn_value",(0.9,0.98))), weight_decay=wd_conf.get("iqn_value",0.0))
            self.baseline_iqn_opt = optim.Adam(self.shared_critic.baseline_iqn_net.parameters(), lr=lr_conf["lr_baseline_iqn"], betas=tuple(beta_conf.get("iqn_baseline",(0.9,0.98))), weight_decay=wd_conf.get("iqn_baseline",0.0))
        else:
            self.value_iqn_opt = self.baseline_iqn_opt = None

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
        if self.schedulers["entropy_coeff"]: self.entropy_coeff = self.schedulers["entropy_coeff"].current_value()
        if self.schedulers["policy_clip"]: self.ppo_clip_range = self.schedulers["policy_clip"].current_value()
        if self.schedulers["value_clip"]: self.value_clip_range = self.schedulers["value_clip"].current_value()
        if self.schedulers["max_grad_norm"]: self.max_grad_norm = self.schedulers["max_grad_norm"].current_value()

        if self.enable_iqn_distillation and self.value_iqn_opt and self.baseline_iqn_opt:
            self.schedulers["lr_value_iqn"] = LinearLR(self.value_iqn_opt, **sched_cfg.get("lr_value_iqn", sched_cfg.get("lr_iqn", def_lin_sched)))
            self.schedulers["lr_baseline_iqn"] = LinearLR(self.baseline_iqn_opt, **sched_cfg.get("lr_baseline_iqn", sched_cfg.get("lr_iqn", def_lin_sched)))
        if self.enable_rnd and self.rnd_opt:
            self.schedulers["lr_rnd_predictor"] = LinearLR(self.rnd_opt, **sched_cfg.get("lr_rnd_predictor", def_lin_sched))

    def _determine_action_space(self, cfg: Dict[str, Any]):
        if "agent" not in cfg: raise ValueError("Missing 'action.agent' block")
        a_cfg = cfg["agent"]
        policy_type_override = self.config["agent"]["params"]["networks"]["policy_network"].get("type", None)

        if policy_type_override:
            self.action_space_type = policy_type_override
            if policy_type_override == "beta" or policy_type_override == "tanh_continuous" or policy_type_override == "gaussian":
                self.action_dim = len(a_cfg["continuous"]) if "continuous" in a_cfg else 0
            elif policy_type_override == "discrete":
                 self.action_dim = [d["num_choices"] for d in a_cfg["discrete"]] if "discrete" in a_cfg else []
            else:
                raise ValueError(f"Unsupported policy_network type in config: {policy_type_override}")
        elif "continuous" in a_cfg:
            self.action_space_type = "beta" # Defaulting to beta if continuous and no override
            self.action_dim = len(a_cfg["continuous"])
        elif "discrete" in a_cfg:
            self.action_space_type = "discrete"
            self.action_dim = [d["num_choices"] for d in a_cfg["discrete"]]
        else:
            raise ValueError("Specify discrete or continuous for action space, or a policy type in config.")

        self.total_action_dim_per_agent = self.action_dim if isinstance(self.action_dim, int) else len(self.action_dim)


    def _apply_memory_seq(self, seq_feats: torch.Tensor, init_h: Optional[torch.Tensor], return_hidden: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.memory_module: return (seq_feats, None) if return_hidden else seq_feats
        B, T, NA, F = seq_feats.shape; flat = seq_feats.reshape(B * NA, T, F)
        h0_gru = None
        if init_h is not None:
            h0_gru = init_h.reshape(B * NA, self.memory_layers, self.memory_hidden).permute(1, 0, 2).contiguous()

        flat_out, h_n_gru = self.memory_module.forward_sequence(flat, h0_gru)
        out = flat_out.reshape(B, T, NA, self.memory_hidden)
        if return_hidden:
            h_n_reshaped = h_n_gru.permute(1, 0, 2).reshape(B, NA, self.memory_layers, self.memory_hidden)
            return out, h_n_reshaped
        return out

    def _sample_taus(self, batch_dim0_size: int, num_quantiles: int) -> torch.Tensor:
        return torch.rand(batch_dim0_size, num_quantiles, 1, device=self.device)

    @torch.no_grad()
    def get_actions(self, states: Dict[str, torch.Tensor], dones: torch.Tensor, truncs: torch.Tensor, h_prev_batch: Optional[torch.Tensor] = None, eval: bool = False, record: bool = True) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        S_batch, B_env, NA_runtime = 1, states["agent"].shape[1] if "agent" in states else (states["central"].shape[1] if "central" in states else 1), self.num_agents

        emb, _ = self.embedding_net.get_base_embedding(states); emb = emb.squeeze(0)

        h_next = None
        if self.memory_module:
            if h_prev_batch is None:
                h_prev_batch = torch.zeros(B_env, NA_runtime, self.memory_layers, self.memory_hidden, device=self.device)
            flat_emb = emb.reshape(B_env * NA_runtime, -1)
            h_prev_gru = h_prev_batch.reshape(B_env * NA_runtime, self.memory_layers, self.memory_hidden).permute(1,0,2).contiguous()
            flat_out, h_next_gru = self.memory_module.forward_step(flat_emb, h_prev_gru)
            feats = flat_out.reshape(B_env, NA_runtime, -1)
            h_next = h_next_gru.permute(1,0,2).reshape(B_env, NA_runtime, self.memory_layers, self.memory_hidden)
            reset_mask = _bool_or(dones.squeeze(0), truncs.squeeze(0)).view(B_env, 1, 1, 1).float()
            h_next = h_next * (1.0 - reset_mask)
        else:
            feats = emb

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

    def _compute_huber_quantile_loss(self, current_quantiles, target_quantiles_detached, taus_for_current, kappa):
        # current_quantiles: (Batch, N_taus_current)
        # target_quantiles_detached: (Batch, N_taus_target)
        # taus_for_current: (Batch, N_taus_current, 1)
        td_errors = target_quantiles_detached.unsqueeze(1) - current_quantiles.unsqueeze(2) # (B, N_curr, N_targ)
        abs_td_errors = torch.abs(td_errors)
        huber_loss_matrix = torch.where(abs_td_errors <= kappa, 0.5 * td_errors.pow(2), kappa * (abs_td_errors - 0.5 * kappa))
        indicator = (td_errors < 0).float() # (B, N_curr, N_targ)
        # **FIXED: Broadcasting for delta_weight**
        # taus_for_current is (B, N_curr, 1). It correctly broadcasts with indicator (B, N_curr, N_targ).
        delta_weight = torch.abs(taus_for_current - indicator) # (B, N_curr, N_targ)
        loss = (delta_weight * huber_loss_matrix).mean() # Mean over all three dimensions (B, N_curr, N_targ)
        return loss

    def _update_target_iqn_networks(self):
        if self.enable_iqn_distillation:
            for target_param, online_param in zip(self.target_value_iqn_net.parameters(), self.shared_critic.value_iqn_net.parameters()):
                target_param.data.copy_(self.iqn_polyak * online_param.data + (1.0 - self.iqn_polyak) * target_param.data)
            for target_param, online_param in zip(self.target_baseline_iqn_net.parameters(), self.shared_critic.baseline_iqn_net.parameters()):
                target_param.data.copy_(self.iqn_polyak * online_param.data + (1.0 - self.iqn_polyak) * target_param.data)

    def _compute_gae_with_padding(self, r: torch.Tensor, v: torch.Tensor, v_next: torch.Tensor, d: torch.Tensor, tr: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        B, T, _ = r.shape
        returns = torch.zeros_like(r)
        gae = torch.zeros(B, 1, device=self.device)
        mask_bt_expanded = mask_bt.unsqueeze(-1) # (B, T, 1)

        for t in reversed(range(T)):
            # valid_mask_step indicates if the current step 't' is part of the actual trajectory (not padding)
            valid_mask_step = mask_bt_expanded[:, t]

            # valid_mask_next_step indicates if the next step 't+1' is part of the actual trajectory
            # If t is the last step of the sequence (T-1), there is no valid next step for bootstrapping within this segment.
            valid_mask_next_step = mask_bt_expanded[:, t + 1] if t < T - 1 else torch.zeros_like(valid_mask_step)

            # Condition for bootstrapping:
            # - The current step 't' must not be a true terminal state (d[:, t] should be 0).
            # - The next step 't+1' must be a valid step in the sequence (not padding).
            # Truncation (tr[:, t]) does not prevent bootstrapping from V(s_t+1) if not also 'done'.
            bootstrap_condition = (1.0 - d[:, t]) * valid_mask_next_step

            # Calculate TD error (delta)
            # v_next[:, t] is V(s_{t+1})
            # v[:, t] is V(s_t)
            delta = r[:, t] + self.gamma * v_next[:, t] * bootstrap_condition - v[:, t]

            # Update GAE: A_t = delta_t + gamma * lambda * A_{t+1} (if s_t is not terminal and s_{t+1} is valid)
            # The 'gae' variable here is A_{t+1} from the previous (later) timestep.
            gae = delta + self.gamma * self.lmbda * bootstrap_condition * gae

            # The return for the current step is GAE + V(s_t)
            current_return = gae + v[:, t]

            # Store returns only for valid steps; zero out for padded steps
            returns[:, t] = current_return * valid_mask_step

            # If the current step was padding, reset GAE for the next iteration (which is t-1)
            # This ensures GAE doesn't propagate across padded sections.
            gae = gae * valid_mask_step
            
        return returns

    def update(self, padded_states_dict_seq: Dict[str, torch.Tensor], padded_actions_seq: torch.Tensor, padded_rewards_seq: torch.Tensor, padded_next_states_dict_seq: Dict[str, torch.Tensor], padded_dones_seq: torch.Tensor, padded_truncs_seq: torch.Tensor, initial_hidden_states_batch: Optional[torch.Tensor], attention_mask_batch: torch.Tensor) -> Dict[str, float]:
        B, T = attention_mask_batch.shape; NA = self.num_agents
        mask_bt = attention_mask_batch # (B, T)
        mask_bt1 = mask_bt.unsqueeze(-1) # (B, T, 1) for rewards, values
        mask_btna = mask_bt.unsqueeze(-1).expand(-1, -1, NA) # (B,T,NA) for per-agent logp, ent
        mask_btna1 = mask_bt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, NA, 1) # (B,T,NA,1) for per-agent baselines

        # --- Initialize logs accumulator for this update cycle ---
        logs_acc: Dict[str, List[float]] = {
            "policy_loss": [], "value_distill_loss": [], "baseline_distill_loss": [],
            "entropy_mean": [], "grad_norm_ppo": [], "adv_mean": [], "adv_std": [],
            "logp_old_mean": [], "logp_old_min": [], "logp_old_max": [],
            "logp_new_mean": [], "logp_new_min": [], "logp_new_max": [],
            "grad_norm_trunk": [], "grad_norm_policy": [],
            "grad_norm_value_distill": [], "grad_norm_baseline_distill": []
        }
        if self.enable_rnd:
            logs_acc.update({"rnd_loss": [], "rnd_reward_raw_mean": [], "rnd_reward_raw_std": [],
                             "rnd_reward_norm_mean": [], "rnd_reward_norm_std": [], "grad_norm_rnd": []})
        if self.enable_iqn_distillation:
            logs_acc.update({"value_iqn_loss": [], "baseline_iqn_loss": [],
                             "grad_norm_value_iqn": [], "grad_norm_baseline_iqn": [],
                             "distill_val_v_ppo_mean": [], "distill_val_v_ppo_std": [],
                             "distill_val_quantiles_mean": [], "distill_val_quantiles_std": [],
                             "distill_val_output_norm_mean": [], "distill_val_output_norm_std": [],
                             "distill_base_b_ppo_mean": [], "distill_base_b_ppo_std": [],
                             "distill_base_quantiles_mean": [], "distill_base_quantiles_std": [],
                             "distill_base_output_norm_mean": [], "distill_base_output_norm_std": []
                             })
        if self.action_space_type == "beta":
            logs_acc.update({"policy_alpha_mean": [], "policy_alpha_std": [],
                             "policy_beta_mean": [], "policy_beta_std": []})


        rewards_seq = padded_rewards_seq.clone()
        if self.rewards_normalizer:
            valid_r_mask = mask_bt1.bool()
            if valid_r_mask.any():
                valid_r = rewards_seq[valid_r_mask]
                self.rewards_normalizer.update(valid_r.unsqueeze(-1))
                rewards_seq[valid_r_mask] = self.rewards_normalizer.normalize(valid_r.unsqueeze(-1)).squeeze(-1).to(rewards_seq.dtype)

        # --- NO-GRAD PASS for PPO GAE and preparing data for IQN Replay Buffer ---
        with torch.no_grad():
            emb_s_seq, _ = self.embedding_net.get_base_embedding(padded_states_dict_seq)
            emb_ns_seq, _ = self.embedding_net.get_base_embedding(padded_next_states_dict_seq)

            h_n_for_next_buffer = None
            if self.memory_module:
                feats_s_seq, h_n_for_next_buffer = self._apply_memory_seq(emb_s_seq, initial_hidden_states_batch, return_hidden=True)
                feats_ns_seq = self._apply_memory_seq(emb_ns_seq, h_n_for_next_buffer)
            else:
                feats_s_seq = self._apply_memory_seq(emb_s_seq, None)
                feats_ns_seq = self._apply_memory_seq(emb_ns_seq, None)

            B_T_flat_dim = B * T
            B_T_NA_flat_dim = B * T * NA

            # 1. Features for Centralized Value IQN (s_val_feats_agg, ns_val_feats_agg)
            # These are the `aggregated_emb` from `SharedCritic.values` logic
            _val_input_s_flat_no_grad = feats_s_seq.reshape(B_T_flat_dim, NA, -1)
            _val_attn_out_s_no_grad, _ = self.shared_critic.value_attention(_val_input_s_flat_no_grad)
            s_val_feats_agg_for_buffer = _val_attn_out_s_no_grad.mean(dim=1).reshape(B, T, -1)

            _val_input_ns_flat_no_grad = feats_ns_seq.reshape(B_T_flat_dim, NA, -1)
            _val_attn_out_ns_no_grad, _ = self.shared_critic.value_attention(_val_input_ns_flat_no_grad)
            ns_val_feats_agg_for_buffer = _val_attn_out_ns_no_grad.mean(dim=1).reshape(B, T, -1)

            # 2. Features for Baseline IQN (s_base_feats_comb_agg, ns_base_feats_comb_agg)
            # These are `agent_emb_for_baseline` from `SharedCritic.baselines` logic
            _base_comb_inputs_s_no_grad = self.embedding_net.get_baseline_embeddings(feats_s_seq, padded_actions_seq)
            _base_attn_input_s_flat_no_grad = _base_comb_inputs_s_no_grad.reshape(B_T_NA_flat_dim, NA, -1) # SeqLen_A is NA
            _base_attn_out_s_no_grad, _ = self.shared_critic.baseline_attention(_base_attn_input_s_flat_no_grad)
            s_base_feats_comb_agg_for_buffer = _base_attn_out_s_no_grad.mean(dim=1).reshape(B, T, NA, -1)

            _base_comb_inputs_ns_no_grad = self.embedding_net.get_baseline_embeddings(feats_ns_seq, torch.zeros_like(padded_actions_seq))
            _base_attn_input_ns_flat_no_grad = _base_comb_inputs_ns_no_grad.reshape(B_T_NA_flat_dim, NA, -1) # SeqLen_A is NA
            _base_attn_out_ns_no_grad, _ = self.shared_critic.baseline_attention(_base_attn_input_ns_flat_no_grad)
            ns_base_feats_comb_agg_for_buffer = _base_attn_out_ns_no_grad.mean(dim=1).reshape(B, T, NA, -1)

            if self.enable_iqn_distillation and hasattr(self, 'iqn_replay_buffer') and self.iqn_replay_buffer.capacity > 0:
                 self.iqn_replay_buffer.push_batch(
                     s_val_feats_agg_for_buffer, ns_val_feats_agg_for_buffer,
                     s_base_feats_comb_agg_for_buffer, ns_base_feats_comb_agg_for_buffer,
                     padded_actions_seq,
                     rewards_seq, padded_dones_seq,
                     attention_mask_batch, initial_hidden_states_batch, h_n_for_next_buffer
                 )

            taus_no_grad_val = self._sample_taus(B_T_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None
            taus_no_grad_base = self._sample_taus(B_T_NA_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None

            old_logp_seq, _ = self._recompute_log_probs_from_features(feats_s_seq, padded_actions_seq)

            val_input_for_ppo_s_flat = feats_s_seq.reshape(B_T_flat_dim, NA, -1)
            val_critic_args_no_grad = (val_input_for_ppo_s_flat, taus_no_grad_val) if self.enable_iqn_distillation else (val_input_for_ppo_s_flat,)
            old_val_output_no_grad = self.shared_critic.values(*val_critic_args_no_grad)
            old_val_distilled_norm = old_val_output_no_grad[0] if self.enable_iqn_distillation else old_val_output_no_grad
            old_val_distilled_norm = old_val_distilled_norm.reshape(B,T,1)

            val_input_for_ppo_ns_flat = feats_ns_seq.reshape(B_T_flat_dim, NA, -1)
            next_val_critic_args_no_grad = (val_input_for_ppo_ns_flat, taus_no_grad_val) if self.enable_iqn_distillation else (val_input_for_ppo_ns_flat,)
            next_val_output_no_grad = self.shared_critic.values(*next_val_critic_args_no_grad)
            next_val_distilled_norm = next_val_output_no_grad[0] if self.enable_iqn_distillation else next_val_output_no_grad
            next_val_distilled_norm = next_val_distilled_norm.reshape(B,T,1)

            base_input_for_ppo_s_comb = self.embedding_net.get_baseline_embeddings(feats_s_seq, padded_actions_seq)
            base_input_for_ppo_s_flat = base_input_for_ppo_s_comb.reshape(B_T_NA_flat_dim, NA, -1)
            base_critic_args_no_grad = (base_input_for_ppo_s_flat, taus_no_grad_base) if self.enable_iqn_distillation else (base_input_for_ppo_s_flat,)
            old_base_output_no_grad = self.shared_critic.baselines(*base_critic_args_no_grad)
            old_base_distilled_norm = old_base_output_no_grad[0] if self.enable_iqn_distillation else old_base_output_no_grad
            old_base_distilled_norm = old_base_distilled_norm.reshape(B,T,NA,1)

            intrinsic_rewards_per_agent_seq = torch.zeros(B,T,NA, device=self.device)
            if self.enable_rnd:
                flat_feats_for_rnd = feats_s_seq.reshape(B_T_NA_flat_dim, -1)
                with torch.no_grad(): target_rnd_embeddings = self.rnd_target_network(flat_feats_for_rnd)
                predictor_rnd_embeddings_no_grad = self.rnd_predictor_network(flat_feats_for_rnd).detach()
                rnd_mse_errors = F.mse_loss(predictor_rnd_embeddings_no_grad, target_rnd_embeddings, reduction="none").mean(dim=-1)
                intrinsic_rewards_per_agent_seq = rnd_mse_errors.reshape(B,T,NA)
                valid_intrinsic_mask_btna = mask_btna.bool()
                if valid_intrinsic_mask_btna.any():
                    valid_intrinsic_rewards = intrinsic_rewards_per_agent_seq[valid_intrinsic_mask_btna]
                    logs_acc["rnd_reward_raw_mean"].append(valid_intrinsic_rewards.mean().item())
                    logs_acc["rnd_reward_raw_std"].append(valid_intrinsic_rewards.std().item())
                    self.intrinsic_reward_normalizer.update(valid_intrinsic_rewards.unsqueeze(-1))
                    normalized_intrinsic_rewards = self.intrinsic_reward_normalizer.normalize(valid_intrinsic_rewards.unsqueeze(-1)).squeeze(-1)
                    intrinsic_rewards_per_agent_seq[valid_intrinsic_mask_btna] = normalized_intrinsic_rewards.to(intrinsic_rewards_per_agent_seq.dtype)
                    logs_acc["rnd_reward_norm_mean"].append(normalized_intrinsic_rewards.mean().item())
                    logs_acc["rnd_reward_norm_std"].append(normalized_intrinsic_rewards.std().item())
                else:
                    logs_acc["rnd_reward_raw_mean"].append(0.0); logs_acc["rnd_reward_raw_std"].append(0.0)
                    logs_acc["rnd_reward_norm_mean"].append(0.0); logs_acc["rnd_reward_norm_std"].append(0.0)

            intrinsic_rewards_global_seq = intrinsic_rewards_per_agent_seq.mean(dim=-1, keepdim=True)
            rewards_for_gae = rewards_seq + self.intrinsic_reward_coeff * intrinsic_rewards_global_seq
            old_val_distilled_denorm = self.value_popart.denormalize_outputs(old_val_distilled_norm) if self.enable_popart else old_val_distilled_norm
            next_val_distilled_denorm = self.value_popart.denormalize_outputs(next_val_distilled_norm) if self.enable_popart else next_val_distilled_norm
            returns_seq = self._compute_gae_with_padding(rewards_for_gae, old_val_distilled_denorm, next_val_distilled_denorm, padded_dones_seq, padded_truncs_seq, mask_bt)

            if self.enable_popart:
                valid_returns_mask = mask_bt1.bool()
                if valid_returns_mask.any():
                    self.value_popart.update_stats(returns_seq[valid_returns_mask])
                    baseline_popart_targets = returns_seq.unsqueeze(2).expand(-1,-1,NA,-1)[mask_btna1.bool()]
                    if baseline_popart_targets.numel() > 0:
                         self.baseline_popart.update_stats(baseline_popart_targets)
                    logs_acc.setdefault("popart_val_mu", []).append(self.value_popart.mu.item())
                    logs_acc.setdefault("popart_val_sigma", []).append(self.value_popart.sigma.item())
                    logs_acc.setdefault("popart_base_mu", []).append(self.baseline_popart.mu.item())
                    logs_acc.setdefault("popart_base_sigma", []).append(self.baseline_popart.sigma.item())
                    sample_raw_ret = returns_seq[valid_returns_mask][:5].cpu().numpy()
                    sample_norm_targ_val = self.value_popart.normalize_targets(returns_seq[valid_returns_mask][:5]).cpu().numpy()
                    logs_acc.setdefault("popart_sample_raw_returns", []).append(sample_raw_ret.tolist())
                    logs_acc.setdefault("popart_sample_norm_val_targets", []).append(sample_norm_targ_val.tolist())
                    sample_old_val_norm = old_val_distilled_norm[valid_returns_mask][:5].cpu().numpy()
                    sample_old_val_denorm = old_val_distilled_denorm[valid_returns_mask][:5].cpu().numpy()
                    logs_acc.setdefault("popart_sample_old_val_norm", []).append(sample_old_val_norm.tolist())
                    logs_acc.setdefault("popart_sample_old_val_denorm_for_gae", []).append(sample_old_val_denorm.tolist())
                else:
                    for k_pa in ["popart_val_mu", "popart_val_sigma", "popart_base_mu", "popart_base_sigma",
                                 "popart_sample_raw_returns", "popart_sample_norm_val_targets",
                                 "popart_sample_old_val_norm", "popart_sample_old_val_denorm_for_gae"]:
                        logs_acc.setdefault(k_pa, []).append(0.0 if "mu" in k_pa or "sigma" in k_pa else [])

            valid_old_lp_mask = mask_btna.bool()
            if valid_old_lp_mask.any():
                valid_old_lp = old_logp_seq[valid_old_lp_mask]
                logs_acc["logp_old_mean"].append(valid_old_lp.mean().item())
                logs_acc["logp_old_min"].append(valid_old_lp.min().item())
                logs_acc["logp_old_max"].append(valid_old_lp.max().item())
            else:
                logs_acc["logp_old_mean"].append(0.0); logs_acc["logp_old_min"].append(0.0); logs_acc["logp_old_max"].append(0.0)

            raw_ext_mean = float(padded_rewards_seq[mask_bt1.bool()].mean()) if mask_bt1.any() else 0.0
            proc_rew_mean = float(rewards_seq[mask_bt1.bool()].mean()) if mask_bt1.any() else 0.0
            ret_vals = returns_seq[mask_bt1.bool()]; ret_mean = float(ret_vals.mean()) if ret_vals.numel() else 0.0; ret_std = float(ret_vals.std()) if ret_vals.numel() else 0.0
        # --- End of NO-GRAD PASS ---

        # --- PPO Training Loop ---
        idx = np.arange(B)
        for _epoch_ppo in range(self.epochs):
            np.random.shuffle(idx)
            for mb_start in range(0, B, self.mini_batch_size):
                mb_idx = idx[mb_start : mb_start + self.mini_batch_size]; M = mb_idx.shape[0]
                if M == 0: continue

                mask_mb = mask_bt[mb_idx]
                mask_mb1 = mask_mb.unsqueeze(-1)
                mask_mbna = mask_mb.unsqueeze(-1).expand(-1,-1,NA)
                mask_mbna1 = mask_mb.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,NA,1)

                M_T_flat_dim = M*T; M_T_NA_flat_dim = M*T*NA
                taus_for_ppo_mb_val = self._sample_taus(M_T_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None
                taus_for_ppo_mb_base = self._sample_taus(M_T_NA_flat_dim, self.num_quantiles) if self.enable_iqn_distillation else None

                emb_s_mb, _ = self.embedding_net.get_base_embedding({k: v[mb_idx] for k,v in padded_states_dict_seq.items()})
                init_h_mb = initial_hidden_states_batch[mb_idx] if initial_hidden_states_batch is not None and initial_hidden_states_batch.shape[0] == B else None
                if init_h_mb is not None and init_h_mb.shape[0] != M : init_h_mb = init_h_mb[:M]
                feats_s_mb = self._apply_memory_seq(emb_s_mb, init_h_mb)

                new_lp_mb, ent_mb = self._recompute_log_probs_from_features(feats_s_mb, padded_actions_seq[mb_idx])

                val_input_mb_for_ppo_flat = feats_s_mb.reshape(M_T_flat_dim, NA, -1)
                val_critic_args_mb = (val_input_mb_for_ppo_flat, taus_for_ppo_mb_val) if self.enable_iqn_distillation else (val_input_mb_for_ppo_flat,)
                new_val_output_mb = self.shared_critic.values(*val_critic_args_mb)
                new_val_distilled_norm_mb = new_val_output_mb[0] if self.enable_iqn_distillation else new_val_output_mb
                new_val_distilled_norm_mb = new_val_distilled_norm_mb.reshape(M,T,1)

                base_input_mb_comb = self.embedding_net.get_baseline_embeddings(feats_s_mb, padded_actions_seq[mb_idx])
                base_input_mb_for_ppo_flat = base_input_mb_comb.reshape(M_T_NA_flat_dim, NA, -1)
                base_critic_args_mb = (base_input_mb_for_ppo_flat, taus_for_ppo_mb_base) if self.enable_iqn_distillation else (base_input_mb_for_ppo_flat,)
                new_base_output_mb = self.shared_critic.baselines(*base_critic_args_mb)
                new_base_distilled_norm_mb = new_base_output_mb[0] if self.enable_iqn_distillation else new_base_output_mb
                new_base_distilled_norm_mb = new_base_distilled_norm_mb.reshape(M,T,NA,1)

                if self.enable_iqn_distillation and _epoch_ppo == 0 and mb_start == 0:
                    if self.enable_iqn_distillation and _epoch_ppo == 0 and mb_start == 0:
                        with torch.no_grad():
                            # For logging, get the PPO head output by calling it with its direct expected input
                            # (which is the mean-aggregated output of its attention module)
                            _v_ppo_input_agg_mb_log = self.shared_critic.value_attention(val_input_mb_for_ppo_flat)[0].mean(dim=1)
                            v_ppo_mb_norm_log = self.shared_critic.value_head_ppo(_v_ppo_input_agg_mb_log)
                            value_quantiles_mb_norm_log = new_val_output_mb[1].reshape(M_T_flat_dim, self.num_quantiles) if len(new_val_output_mb) > 1 else None

                            _b_ppo_input_agg_mb_log = self.shared_critic.baseline_attention(base_input_mb_for_ppo_flat)[0].mean(dim=1)
                            b_ppo_mb_norm_log = self.shared_critic.baseline_head_ppo(_b_ppo_input_agg_mb_log)
                            baseline_quantiles_mb_norm_log = new_base_output_mb[1].reshape(M_T_NA_flat_dim, self.num_quantiles) if len(new_base_output_mb) > 1 else None
                    
                    valid_v_ppo_mask_mb = mask_mb.reshape(-1).bool()
                    if valid_v_ppo_mask_mb.any():
                        logs_acc["distill_val_v_ppo_mean"].append(v_ppo_mb_norm_log[valid_v_ppo_mask_mb].mean().item())
                        logs_acc["distill_val_v_ppo_std"].append(v_ppo_mb_norm_log[valid_v_ppo_mask_mb].std().item())
                        if value_quantiles_mb_norm_log is not None:
                            logs_acc["distill_val_quantiles_mean"].append(value_quantiles_mb_norm_log[valid_v_ppo_mask_mb].mean().item())
                            logs_acc["distill_val_quantiles_std"].append(value_quantiles_mb_norm_log[valid_v_ppo_mask_mb].std().item())
                        logs_acc["distill_val_output_norm_mean"].append(new_val_distilled_norm_mb[mask_mb1.bool()].mean().item())
                        logs_acc["distill_val_output_norm_std"].append(new_val_distilled_norm_mb[mask_mb1.bool()].std().item())
                    else:
                        for k_dv in ["distill_val_v_ppo_mean", "distill_val_v_ppo_std", "distill_val_quantiles_mean", "distill_val_quantiles_std", "distill_val_output_norm_mean", "distill_val_output_norm_std"]: logs_acc[k_dv].append(0.0)

                    valid_b_ppo_mask_mb = mask_mbna.reshape(-1).bool()
                    if valid_b_ppo_mask_mb.any():
                        logs_acc["distill_base_b_ppo_mean"].append(b_ppo_mb_norm_log[valid_b_ppo_mask_mb].mean().item())
                        logs_acc["distill_base_b_ppo_std"].append(b_ppo_mb_norm_log[valid_b_ppo_mask_mb].std().item())
                        if baseline_quantiles_mb_norm_log is not None:
                            logs_acc["distill_base_quantiles_mean"].append(baseline_quantiles_mb_norm_log[valid_b_ppo_mask_mb].mean().item())
                            logs_acc["distill_base_quantiles_std"].append(baseline_quantiles_mb_norm_log[valid_b_ppo_mask_mb].std().item())
                        logs_acc["distill_base_output_norm_mean"].append(new_base_distilled_norm_mb[mask_mbna1.bool()].mean().item())
                        logs_acc["distill_base_output_norm_std"].append(new_base_distilled_norm_mb[mask_mbna1.bool()].std().item())
                    else:
                         for k_db in ["distill_base_b_ppo_mean", "distill_base_b_ppo_std", "distill_base_quantiles_mean", "distill_base_quantiles_std", "distill_base_output_norm_mean", "distill_base_output_norm_std"]: logs_acc[k_db].append(0.0)

                if self.action_space_type == "beta" and _epoch_ppo == 0 and mb_start == 0:
                    with torch.no_grad():
                        policy_input_flat_mb = feats_s_mb.reshape(M*T*NA, -1)
                        valid_policy_mask_flat_mb = mask_mbna.reshape(-1).bool()
                        if valid_policy_mask_flat_mb.any():
                            alpha_p, beta_p = self.policy_net.forward(policy_input_flat_mb[valid_policy_mask_flat_mb])
                            logs_acc["policy_alpha_mean"].append(alpha_p.mean().item()); logs_acc["policy_alpha_std"].append(alpha_p.std().item())
                            logs_acc["policy_beta_mean"].append(beta_p.mean().item()); logs_acc["policy_beta_std"].append(beta_p.std().item())
                        else:
                            for k_beta in ["policy_alpha_mean", "policy_alpha_std", "policy_beta_mean", "policy_beta_std"]: logs_acc[k_beta].append(0.0)

                old_lp_mb = old_logp_seq[mb_idx]
                old_val_distilled_mb_norm = old_val_distilled_norm[mb_idx]
                old_base_distilled_mb_norm = old_base_distilled_norm[mb_idx]
                returns_mb = returns_seq[mb_idx]

                new_base_denorm_distilled_mb = self.baseline_popart.denormalize_outputs(new_base_distilled_norm_mb) if self.enable_popart else new_base_distilled_norm_mb
                adv = (returns_mb.unsqueeze(2).expand_as(new_base_distilled_norm_mb) - new_base_denorm_distilled_mb).squeeze(-1)
                adv_mask_mbna = mask_mbna.bool()
                if self.normalize_adv:
                    valid_adv_data = adv[adv_mask_mbna]
                    if valid_adv_data.numel() > 0:
                        m,s = valid_adv_data.mean(), valid_adv_data.std()+1e-8; adv_norm_data = (valid_adv_data-m)/s
                        tmp_adv = torch.zeros_like(adv); tmp_adv[adv_mask_mbna] = adv_norm_data; adv = tmp_adv
                        if _epoch_ppo == 0 and mb_start == 0: logs_acc["adv_mean"].append(m.item()); logs_acc["adv_std"].append((s-1e-8).item())
                elif adv_mask_mbna.any() and _epoch_ppo == 0 and mb_start == 0:
                    logs_acc["adv_mean"].append(float(adv[adv_mask_mbna].mean())); logs_acc["adv_std"].append(float(adv[adv_mask_mbna].std()))

                pol_loss = self._ppo_clip_loss(new_lp_mb, old_lp_mb.detach(), adv.detach(), self.ppo_clip_range, mask_mbna)
                tgt_val_distilled_mb = returns_mb if not self.enable_popart else self.value_popart.normalize_targets(returns_mb)
                val_loss = self._clipped_value_loss(old_val_distilled_mb_norm.detach(), new_val_distilled_norm_mb, tgt_val_distilled_mb.detach(), self.value_clip_range, mask=mask_mb1)
                tgt_base_distilled_mb = returns_mb.unsqueeze(2).expand_as(new_base_distilled_norm_mb)
                if self.enable_popart: tgt_base_distilled_mb = self.baseline_popart.normalize_targets(tgt_base_distilled_mb)
                base_loss = self._clipped_value_loss(old_base_distilled_mb_norm.detach(), new_base_distilled_norm_mb, tgt_base_distilled_mb.detach(), self.value_clip_range, mask=mask_mbna1)
                ent_terms_masked = -ent_mb * mask_mbna
                ent_loss = ent_terms_masked.sum() / mask_mbna.sum().clamp(min=1e-8)
                ppo_total_loss = pol_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss + self.entropy_coeff * ent_loss

                rnd_loss_mb = torch.tensor(0.0, device=self.device)
                if self.enable_rnd and self.rnd_update_prop > 0.0:
                    valid_rnd_feats_mask_mbna_bool = mask_mbna.bool()
                    flat_feats_mb_for_rnd = feats_s_mb.reshape(M*T*NA, -1)
                    valid_feats_mb_for_rnd = flat_feats_mb_for_rnd[valid_rnd_feats_mask_mbna_bool.reshape(-1)]
                    if valid_feats_mb_for_rnd.numel() > 0:
                        num_rnd_samples = valid_feats_mb_for_rnd.shape[0]
                        k_rnd = int(num_rnd_samples * self.rnd_update_prop)
                        if k_rnd > 0:
                            sel_rnd_idx = torch.randperm(num_rnd_samples, device=self.device)[:k_rnd]
                            selected_feats_for_rnd = valid_feats_mb_for_rnd[sel_rnd_idx]
                            pred_rnd_val = self.rnd_predictor_network(selected_feats_for_rnd)
                            with torch.no_grad(): tgt_rnd_val = self.rnd_target_network(selected_feats_for_rnd)
                            rnd_loss_mb = F.mse_loss(pred_rnd_val, tgt_rnd_val); ppo_total_loss += rnd_loss_mb
                            logs_acc["grad_norm_rnd"].append(_get_avg_grad_mag(list(self.rnd_predictor_network.parameters())))

                self.trunk_opt.zero_grad(); self.policy_opt.zero_grad()
                self.value_distill_opt.zero_grad(); self.baseline_distill_opt.zero_grad()
                if self.rnd_opt: self.rnd_opt.zero_grad()
                ppo_total_loss.backward()

                with torch.no_grad():
                    logs_acc["grad_norm_trunk"].append(_get_avg_grad_mag(list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])))
                    logs_acc["grad_norm_policy"].append(_get_avg_grad_mag(list(self.policy_net.parameters())))
                    val_dist_params_list = list(self.shared_critic.value_head_ppo.parameters()) + list(self.shared_critic.value_attention.parameters())
                    if self.enable_iqn_distillation: val_dist_params_list += list(self.shared_critic.value_distill_net.parameters())
                    logs_acc["grad_norm_value_distill"].append(_get_avg_grad_mag(val_dist_params_list))
                    base_dist_params_list = list(self.shared_critic.baseline_head_ppo.parameters()) + list(self.shared_critic.baseline_attention.parameters())
                    if self.enable_iqn_distillation: base_dist_params_list += list(self.shared_critic.baseline_distill_net.parameters())
                    logs_acc["grad_norm_baseline_distill"].append(_get_avg_grad_mag(base_dist_params_list))

                ppo_path_params_list = self.parameters()
                gn_ppo = clip_grad_norm_(ppo_path_params_list, self.max_grad_norm)

                self.trunk_opt.step(); self.policy_opt.step()
                self.value_distill_opt.step(); self.baseline_distill_opt.step()
                if self.rnd_opt and rnd_loss_mb.requires_grad and rnd_loss_mb.abs().item() > 1e-9: self.rnd_opt.step()


                logs_acc["policy_loss"].append(pol_loss.item())
                logs_acc["value_distill_loss"].append(val_loss.item())
                logs_acc["baseline_distill_loss"].append(base_loss.item())
                valid_ent_mask_mbna = mask_mbna.bool()
                logs_acc["entropy_mean"].append(ent_mb[valid_ent_mask_mbna].mean().item() if valid_ent_mask_mbna.any() else 0.0)
                logs_acc["grad_norm_ppo"].append(gn_ppo.item() if torch.is_tensor(gn_ppo) else gn_ppo)
                if valid_ent_mask_mbna.any():
                    valid_new_lp = new_lp_mb[valid_ent_mask_mbna]
                    logs_acc["logp_new_mean"].append(valid_new_lp.mean().item())
                    logs_acc["logp_new_min"].append(valid_new_lp.min().item())
                    logs_acc["logp_new_max"].append(valid_new_lp.max().item())
                else:
                    logs_acc["logp_new_mean"].append(0.0); logs_acc["logp_new_min"].append(0.0); logs_acc["logp_new_max"].append(0.0)
                if self.enable_rnd: logs_acc["rnd_loss"].append(rnd_loss_mb.item())

            for sch_name_key in ["lr_trunk", "lr_policy", "lr_value_distill", "lr_baseline_distill"]:
                if self.schedulers.get(sch_name_key): self.schedulers[sch_name_key].step()
            if self.enable_rnd and self.schedulers.get("lr_rnd_predictor"): self.schedulers["lr_rnd_predictor"].step()
            if self.schedulers.get("entropy_coeff"): self.entropy_coeff = self.schedulers["entropy_coeff"].step()
            if self.schedulers.get("policy_clip"): self.ppo_clip_range = self.schedulers["policy_clip"].step()
            if self.schedulers.get("value_clip"): self.value_clip_range = self.schedulers["value_clip"].step()
            if self.schedulers.get("max_grad_norm"): self.max_grad_norm = self.schedulers["max_grad_norm"].step()
        # --- End of PPO Training Loop ---

        # --- IQN Training Loop ---
        if self.enable_iqn_distillation and hasattr(self, 'iqn_replay_buffer') and len(self.iqn_replay_buffer) >= self.iqn_batch_size :
            self._iqn_total_updates_counter +=1
            current_val_iqn_grad_norm_sum = 0.0
            current_base_iqn_grad_norm_sum = 0.0
            num_iqn_batches_processed = 0

            for _epoch_iqn in range(self.iqn_epochs):
                iqn_sample_tuple = self.iqn_replay_buffer.sample(self.iqn_batch_size)
                if iqn_sample_tuple is None: continue

                # Unpack features that are already processed and aggregated for IQN heads
                s_val_agg_iqn_mb, ns_val_agg_iqn_mb, \
                s_base_agg_iqn_mb, ns_base_agg_iqn_mb, \
                _actions_iqn_mb, r_iqn_mb, d_iqn_mb, mask_iqn_mb, \
                _init_h_iqn_mb, _next_init_h_iqn_mb = iqn_sample_tuple

                iqn_mb_B, iqn_mb_T = mask_iqn_mb.shape
                iqn_mb_NA = s_base_agg_iqn_mb.shape[2] # Num agents from baseline features

                # --- Value IQN Loss ---
                # Reshape aggregated value features: (B_iqn*T_valid, FeatureDim_Agg_Val)
                valid_val_mask_flat = mask_iqn_mb.reshape(-1).bool()
                s_val_agg_valid = s_val_agg_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, -1)[valid_val_mask_flat]
                ns_val_agg_valid = ns_val_agg_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, -1)[valid_val_mask_flat]
                r_val_iqn_valid = r_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, 1)[valid_val_mask_flat]
                d_val_iqn_valid = d_iqn_mb.reshape(iqn_mb_B * iqn_mb_T, 1)[valid_val_mask_flat]

                value_iqn_loss = torch.tensor(0.0, device=self.device)
                if s_val_agg_valid.shape[0] > 0:
                    eff_val_iqn_b_size = s_val_agg_valid.shape[0]
                    taus_current_val = self._sample_taus(eff_val_iqn_b_size, self.num_quantiles)
                    taus_next_val    = self._sample_taus(eff_val_iqn_b_size, self.num_quantiles_prime)

                    current_value_quantiles = self.shared_critic.value_iqn_net(s_val_agg_valid, taus_current_val)
                    with torch.no_grad():
                        next_value_quantiles_target = self.target_value_iqn_net(ns_val_agg_valid, taus_next_val)
                    value_bellman_targets = r_val_iqn_valid + self.gamma * (1.0 - d_val_iqn_valid) * next_value_quantiles_target.detach()
                    value_iqn_loss = self._compute_huber_quantile_loss(current_value_quantiles, value_bellman_targets, taus_current_val, self.iqn_kappa)
                logs_acc["value_iqn_loss"].append(value_iqn_loss.item())

                # --- Baseline IQN Loss ---
                # Reshape aggregated baseline features: (B_iqn*T_valid*NA, FeatureDim_Agg_Base)
                valid_base_mask_flat = mask_iqn_mb.unsqueeze(-1).expand(-1, -1, iqn_mb_NA).reshape(-1).bool()
                s_base_agg_valid = s_base_agg_iqn_mb.reshape(iqn_mb_B * iqn_mb_T * iqn_mb_NA, -1)[valid_base_mask_flat]
                ns_base_agg_valid = ns_base_agg_iqn_mb.reshape(iqn_mb_B * iqn_mb_T * iqn_mb_NA, -1)[valid_base_mask_flat]

                r_base_iqn_valid = r_iqn_mb.unsqueeze(2).expand(-1, -1, iqn_mb_NA, -1).reshape(iqn_mb_B*iqn_mb_T*iqn_mb_NA, 1)[valid_base_mask_flat]
                d_base_iqn_valid = d_iqn_mb.unsqueeze(2).expand(-1, -1, iqn_mb_NA, -1).reshape(iqn_mb_B*iqn_mb_T*iqn_mb_NA, 1)[valid_base_mask_flat]

                baseline_iqn_loss = torch.tensor(0.0, device=self.device)
                if s_base_agg_valid.shape[0] > 0:
                    eff_base_iqn_b_size = s_base_agg_valid.shape[0]
                    taus_current_base = self._sample_taus(eff_base_iqn_b_size, self.num_quantiles)
                    taus_next_base    = self._sample_taus(eff_base_iqn_b_size, self.num_quantiles_prime)

                    current_baseline_quantiles = self.shared_critic.baseline_iqn_net(s_base_agg_valid, taus_current_base)
                    with torch.no_grad():
                        next_baseline_quantiles_target = self.target_baseline_iqn_net(ns_base_agg_valid, taus_next_base)
                    baseline_bellman_targets = r_base_iqn_valid + self.gamma * (1.0 - d_base_iqn_valid) * next_baseline_quantiles_target.detach()
                    baseline_iqn_loss = self._compute_huber_quantile_loss(current_baseline_quantiles, baseline_bellman_targets, taus_current_base, self.iqn_kappa)
                logs_acc["baseline_iqn_loss"].append(baseline_iqn_loss.item())

                # Backward pass and optimizer steps for IQN components
                # Using separate loss coefficients
                if self.value_iqn_opt and value_iqn_loss.requires_grad and value_iqn_loss.abs().item() > 1e-9:
                    self.value_iqn_opt.zero_grad()
                    (self.value_iqn_loss_coeff * value_iqn_loss).backward()
                    gn_val_iqn = clip_grad_norm_(self.shared_critic.value_iqn_net.parameters(), self.iqn_max_grad_norm)
                    self.value_iqn_opt.step()
                    current_val_iqn_grad_norm_sum += (gn_val_iqn.item() if torch.is_tensor(gn_val_iqn) else gn_val_iqn) if gn_val_iqn is not None else 0.0

                if self.baseline_iqn_opt and baseline_iqn_loss.requires_grad and baseline_iqn_loss.abs().item() > 1e-9:
                    self.baseline_iqn_opt.zero_grad()
                    (self.baseline_iqn_loss_coeff * baseline_iqn_loss).backward()
                    gn_base_iqn = clip_grad_norm_(self.shared_critic.baseline_iqn_net.parameters(), self.iqn_max_grad_norm)
                    self.baseline_iqn_opt.step()
                    current_base_iqn_grad_norm_sum += (gn_base_iqn.item() if torch.is_tensor(gn_base_iqn) else gn_base_iqn) if gn_base_iqn is not None else 0.0
                num_iqn_batches_processed +=1

            if num_iqn_batches_processed > 0:
                logs_acc["grad_norm_value_iqn"].append(current_val_iqn_grad_norm_sum / num_iqn_batches_processed)
                logs_acc["grad_norm_baseline_iqn"].append(current_base_iqn_grad_norm_sum / num_iqn_batches_processed)
            else:
                if "grad_norm_value_iqn" not in logs_acc or not logs_acc["grad_norm_value_iqn"]: logs_acc["grad_norm_value_iqn"].append(0.0)
                if "grad_norm_baseline_iqn" not in logs_acc or not logs_acc["grad_norm_baseline_iqn"]: logs_acc["grad_norm_baseline_iqn"].append(0.0)

            if self.schedulers.get("lr_value_iqn"): self.schedulers["lr_value_iqn"].step()
            if self.schedulers.get("lr_baseline_iqn"): self.schedulers["lr_baseline_iqn"].step()
            if self._iqn_total_updates_counter % self.iqn_target_update_freq == 0:
                self._update_target_iqn_networks()
        # --- End of IQN Training Loop ---

        # --- Finalize Logs ---
        final_logs = {k: (np.mean(v) if v else 0.0) for k,v in logs_acc.items() if isinstance(v, list) and k not in ["popart_sample_raw_returns", "popart_sample_norm_val_targets", "popart_sample_old_val_norm", "popart_sample_old_val_denorm_for_gae"]}
        for k_pop_sample in ["popart_sample_raw_returns", "popart_sample_norm_val_targets", "popart_sample_old_val_norm", "popart_sample_old_val_denorm_for_gae"]:
            if k_pop_sample in logs_acc and logs_acc[k_pop_sample] and isinstance(logs_acc[k_pop_sample][0], (np.ndarray, list)):
                 final_logs[k_pop_sample] = np.mean([item for sublist in logs_acc[k_pop_sample] for item in sublist]).tolist() if logs_acc[k_pop_sample] else [] # Flatten list of lists/arrays
            elif k_pop_sample in logs_acc and logs_acc[k_pop_sample]:
                 final_logs[k_pop_sample] = np.mean(logs_acc[k_pop_sample])


        final_logs.update({
            "raw_reward_mean": raw_ext_mean, "processed_reward_mean": proc_rew_mean,
            "return_mean": ret_mean, "return_std": ret_std,
            "entropy_coeff_curr": self.entropy_coeff, "ppo_clip_curr": self.ppo_clip_range,
            "value_clip_curr": self.value_clip_range, "max_grad_norm_ppo_curr": self.max_grad_norm,
            "lr_trunk_curr": self.schedulers["lr_trunk"].get_last_lr()[0] if self.schedulers.get("lr_trunk") else lr_conf.get("lr_trunk",0),
            "lr_policy_curr": self.schedulers["lr_policy"].get_last_lr()[0] if self.schedulers.get("lr_policy") else lr_conf.get("lr_policy",0),
            "lr_value_distill_curr": self.schedulers["lr_value_distill"].get_last_lr()[0] if self.schedulers.get("lr_value_distill") else lr_conf.get("lr_value_distill",0),
            "lr_baseline_distill_curr": self.schedulers["lr_baseline_distill"].get_last_lr()[0] if self.schedulers.get("lr_baseline_distill") else lr_conf.get("lr_baseline_distill",0),
        })
        if self.enable_iqn_distillation:
            final_logs["lr_value_iqn_curr"] = self.schedulers["lr_value_iqn"].get_last_lr()[0] if self.schedulers.get("lr_value_iqn") else lr_conf.get("lr_value_iqn",0)
            final_logs["lr_baseline_iqn_curr"] = self.schedulers["lr_baseline_iqn"].get_last_lr()[0] if self.schedulers.get("lr_baseline_iqn") else lr_conf.get("lr_baseline_iqn",0)
            final_logs["iqn_max_grad_norm_curr"] = self.iqn_max_grad_norm
        if self.enable_rnd:
            final_logs["lr_rnd_curr"] = self.schedulers["lr_rnd_predictor"].get_last_lr()[0] if self.schedulers.get("lr_rnd_predictor") else lr_conf.get("lr_rnd",0)

        return final_logs

    def _ppo_clip_loss(self, new_lp, old_lp, adv, clip_range, mask):
        if mask.ndim < new_lp.ndim: mask = mask.unsqueeze(-1).expand_as(new_lp)
        ratio = torch.exp((new_lp - old_lp).clamp(-10,10)) # Clamping for stability
        s1 = ratio * adv; s2 = torch.clamp(ratio, 1.0-clip_range, 1.0+clip_range) * adv
        loss_terms = -torch.min(s1,s2) * mask
        return loss_terms.sum() / mask.sum().clamp(min=1e-8)

    def _clipped_value_loss(self, old_v_norm, new_v_norm, target_v_norm, clip_range, mask, reduction="mean"):
        if mask.ndim < new_v_norm.ndim: mask = mask.unsqueeze(-1).expand_as(new_v_norm)
        v_pred_clipped_norm = old_v_norm + (new_v_norm - old_v_norm).clamp(-clip_range, clip_range)
        vf_loss1 = (new_v_norm - target_v_norm).pow(2)
        vf_loss2 = (v_pred_clipped_norm - target_v_norm).pow(2)
        loss_terms = torch.max(vf_loss1, vf_loss2)
        masked_loss_terms = loss_terms * mask
        if reduction == "mean": return masked_loss_terms.sum() / mask.sum().clamp(min=1e-8)
        return masked_loss_terms

    def _recompute_log_probs_from_features(self, feats_seq, actions_seq):
        B,T,NA,F = feats_seq.shape; flat_feats = feats_seq.reshape(B*T*NA, F)

        if self.action_space_type == "discrete":
            action_dim_flat = self.total_action_dim_per_agent
        elif self.action_space_type in ["beta", "tanh_continuous", "gaussian"]:
            action_dim_flat = self.action_dim
        else:
            raise ValueError(f"Unknown action_space_type: {self.action_space_type} in _recompute_log_probs_from_features")

        flat_actions = actions_seq.reshape(B*T*NA, action_dim_flat)
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
        # Safely get learning rate, defaulting to 0 if optimizer or param_groups are missing
        if hasattr(self, 'policy_opt') and self.policy_opt and self.policy_opt.param_groups:
            return float(np.mean([pg.get("lr", 0.0) for pg in self.policy_opt.param_groups]))
        return 0.0