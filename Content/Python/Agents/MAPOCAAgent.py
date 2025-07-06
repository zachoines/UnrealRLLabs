from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
import copy
import random

from Source.Agent import Agent
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
    ForwardDynamicsModel
)
from Source.Memory import GRUSequenceMemory, RecurrentMemoryNetwork

# --- Helper Methods ---
def _get_avg_grad_mag(params_list: List[torch.nn.Parameter]) -> float:
    """Computes the average magnitude of gradients for a list of parameters."""
    total_abs_grad = 0.0
    num_params_with_grad = 0
    if not params_list: return 0.0
    for p in params_list:
        if p.grad is not None:
            total_abs_grad += p.grad.abs().mean().item()
            num_params_with_grad += 1
    return total_abs_grad / num_params_with_grad if num_params_with_grad > 0 else 0.0

# --- Replay Buffer for IQN ---
class IQNReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Tuple] = []
        self.position = 0

    def push_batch(self, *args):
        if self.capacity == 0: return
        cpu_args = [arg.detach().cpu() if torch.is_tensor(arg) else arg for arg in args]
        num_sequences_in_batch = cpu_args[0].shape[0]
        for i in range(num_sequences_in_batch):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None) # type: ignore
            sequence_data = tuple(arg[i] if arg is not None else None for arg in cpu_args)
            self.buffer[self.position] = sequence_data
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size: return None
        batch_tuples = random.sample(self.buffer, batch_size)
        return tuple(
            torch.stack([s[i] for s in batch_tuples if s[i] is not None]).to(self.device) if batch_tuples[0][i] is not None else None
            for i in range(len(batch_tuples[0]))
        )

    def __len__(self) -> int:
        return len(self.buffer)

class MAPOCAAgent(Agent):
    """
    Multi-Agent POCA Agent with optional auxiliary modules like RND, IQN, and Disagreement.
    """
    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__(cfg, device)
        self.device = device
        self.config = cfg

        # --- Parse Configurations ---
        a_cfg = cfg["agent"]["params"]
        t_cfg = cfg["train"]
        env_cfg = cfg.get("environment", {})
        env_shape = env_cfg.get("shape", {})
        env_params = env_cfg.get("params", {})
        net_cfg = a_cfg["networks"]

        # --- Core Hyperparameters ---
        self.gamma = a_cfg.get("gamma", 0.99)
        self.lmbda = a_cfg.get("lmbda", 0.95)
        self.value_loss_coeff = a_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff = a_cfg.get("baseline_loss_coeff", 0.5)
        self.disagreement_loss_coeff = a_cfg.get("disagreement_loss_coeff", 0.01)
        self.normalize_adv = a_cfg.get("normalize_advantages", True)
        self.lr_base = a_cfg.get("learning_rate", 3e-4)
        self.epochs = t_cfg.get("epochs", 4)
        self.mini_batch_size = t_cfg.get("mini_batch_size", 64)
        
        # --- Schedulable Parameters (initialized with their starting values) ---
        self.entropy_coeff = a_cfg.get("entropy_coeff", 0.01)
        self.ppo_clip_range = a_cfg.get("ppo_clip_range", 0.2)
        self.value_clip_range = a_cfg.get("value_clip_range", 0.2)
        self.max_grad_norm = a_cfg.get("max_grad_norm", 0.5)

        # --- Environment & Action Space Setup ---
        self.num_agents = env_params.get("MaxAgents", 1)
        if "agent" in env_shape.get("state", {}):
             self.num_agents = env_shape["state"]["agent"].get("max", self.num_agents)
        self._determine_action_space(env_shape.get("action", {}))

        # --- Core Network Definitions ---
        emb_main_cfg = net_cfg["MultiAgentEmbeddingNetwork"]
        self.embedding_net = MultiAgentEmbeddingNetwork(emb_main_cfg, environment_config_for_shapes=env_cfg).to(device)
        emb_out_dim = emb_main_cfg["cross_attention_feature_extractor"]["embed_dim"]

        self.enable_memory = a_cfg.get("enable_memory", False)
        if self.enable_memory:
            mem_type = a_cfg.get("memory_type", "gru")
            mem_specific_cfg = a_cfg.get(mem_type, {}).copy()
            mem_specific_cfg.setdefault("input_size", emb_out_dim)
            self.memory_hidden = mem_specific_cfg.get("hidden_size", 128)
            self.memory_layers = mem_specific_cfg.get("num_layers", 1)
            self.memory_module: Optional[RecurrentMemoryNetwork] = GRUSequenceMemory(**mem_specific_cfg).to(device)
            trunk_dim = self.memory_hidden
        else:
            self.memory_module = None
            trunk_dim = emb_out_dim

        pol_cfg = net_cfg["policy_network"].copy()
        policy_type_str = pol_cfg.pop("type", self.action_space_type_str_from_config)
        pol_cfg.update({"in_features": trunk_dim, "out_features": self.action_dim})
        policy_class_map = {"beta": BetaPolicyNetwork, "tanh_continuous": TanhContinuousPolicyNetwork, "gaussian": GaussianPolicyNetwork, "discrete": DiscretePolicyNetwork}
        self.policy_net = policy_class_map[policy_type_str](**pol_cfg).to(device)

        critic_full_config = net_cfg["critic_network"].copy()
        critic_full_config["feature_dim_for_iqn"] = trunk_dim
        self.shared_critic = SharedCritic(net_cfg=critic_full_config).to(device)

        # --- Auxiliary Modules Setup ---
        self.disagreement_cfg = a_cfg.get("disagreement_rewards")
        self.rnd_cfg = a_cfg.get("rnd_rewards")

        if self.disagreement_cfg:
            self._setup_disagreement(self.disagreement_cfg, trunk_dim)

        if self.rnd_cfg:
            self._setup_rnd(self.rnd_cfg, trunk_dim)

        self.enable_iqn_distillation = a_cfg.get("enable_iqn_distillation", False)
        self.enable_popart = a_cfg.get("enable_popart", False)

        if self.enable_iqn_distillation: self._setup_iqn(critic_full_config)
        if self.enable_popart: self._setup_popart(a_cfg)
        else: self.rewards_normalizer = RunningMeanStdNormalizer(**a_cfg.get("rewards_normalizer", {}), device=device) if a_cfg.get("rewards_normalizer") else None

        # --- Final Setup Steps ---
        self._setup_optimizers(a_cfg)
        self._setup_schedulers(a_cfg)

    # --- Core Agent Interface Methods ---
    @torch.no_grad()
    def get_actions(
        self,
        states: Dict[str, Any],
        dones: Optional[torch.Tensor] = None,
        truncs: Optional[torch.Tensor] = None,
        eval: bool = False,
        h_prev_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Inference-time method to get actions for the current state observations.

        Returns the UE-flattened actions along with log-probabilities, entropies,
        predicted values, and baselines used for PPO/MA-POCA updates.
        """
        s_mask = {"gridobject_sequence_mask": states.get("central", {}).get("gridobject_sequence_mask")}
        emb_SBEA, _ = self.embedding_net.get_base_embedding(states, central_component_padding_masks=s_mask)
        emb_E_A_Embed = emb_SBEA.squeeze(0)
        B_env, NA_runtime, _ = emb_E_A_Embed.shape

        h_next_gru_shaped = None
        feats_for_policy = emb_E_A_Embed

        if self.enable_memory and self.memory_module:
            current_h_for_gru = h_prev_batch.reshape(B_env * NA_runtime, self.memory_layers, self.memory_hidden).permute(1,0,2).contiguous() if h_prev_batch is not None else None
            flat_emb_for_gru = emb_E_A_Embed.reshape(B_env * NA_runtime, -1)
            feats_flat_from_gru, h_next_gru_flat = self.memory_module.forward_step(flat_emb_for_gru, current_h_for_gru)
            feats_for_policy = feats_flat_from_gru.reshape(B_env, NA_runtime, -1)
            h_next_gru_shaped = h_next_gru_flat.permute(1,0,2).reshape(B_env, NA_runtime, self.memory_layers, self.memory_hidden)
            if dones is not None and truncs is not None:
                # RLRunner zeros the hidden state before calling this method
                # when a new episode begins. We additionally mask the *next*
                # hidden state so that subsequent timesteps also start fresh.
                reset_mask = (dones.squeeze(0) > 0.5) | (truncs.squeeze(0) > 0.5)
                h_next_gru_shaped = h_next_gru_shaped * (1.0 - reset_mask.view(B_env, 1, 1, 1).float())

        policy_input_flat = feats_for_policy.reshape(B_env * NA_runtime, -1)
        actions_flat, log_probs_flat, entropies_flat = self.policy_net.get_actions(policy_input_flat, eval=eval)

        actions_shaped = actions_flat.view(B_env, NA_runtime, self.total_action_dim_per_agent)
        actions_ue_flat_output = actions_shaped.reshape(B_env, -1)
        log_probs_output = log_probs_flat.view(B_env, NA_runtime)
        entropies_output = entropies_flat.view(B_env, NA_runtime)

        with torch.no_grad():
            feats_seq = feats_for_policy.unsqueeze(1)  # (B,1,NA,F)
            act_seq = actions_shaped.unsqueeze(1)
            val_norm, base_norm, _, _, _ = self._get_critic_outputs(feats_seq, act_seq, B_env, 1, NA_runtime)

        values_output = val_norm.squeeze(1)
        baselines_output = base_norm.squeeze(1)

        return (
            actions_ue_flat_output,
            (log_probs_output, entropies_output, values_output, baselines_output),
            h_next_gru_shaped,
        )
    
    def update(
        self,
        padded_states_dict_seq: Dict[str, torch.Tensor],
        padded_actions_seq: torch.Tensor,
        padded_returns_seq: torch.Tensor,
        padded_rewards_seq: torch.Tensor,
        old_log_probs_seq: torch.Tensor,
        old_values_seq: torch.Tensor,
        old_baselines_seq: torch.Tensor,
        old_entropies_seq: torch.Tensor,
        padded_next_states_dict_seq: Dict[str, torch.Tensor],
        padded_dones_seq: torch.Tensor,
        padded_truncs_seq: torch.Tensor,
        initial_hidden_states_batch: Optional[torch.Tensor],
        attention_mask_batch: torch.Tensor,
    ) -> Dict[str, float]:
        logs_acc = self._initialize_logs()
        B, T = attention_mask_batch.shape
        NA = self.num_agents
        mask_btna = attention_mask_batch.unsqueeze(-1).expand(-1, -1, NA)

        valid_mask = attention_mask_batch.unsqueeze(-1).bool()
        if valid_mask.any():
            logs_acc["mean/return"].append(padded_returns_seq[valid_mask].mean().item())
            logs_acc["std/return"].append(padded_returns_seq[valid_mask].std().item())
            logs_acc["reward/raw_mean"].append(padded_rewards_seq[valid_mask].mean().item())
            logs_acc["reward/raw_std"].append(padded_rewards_seq[valid_mask].std().item())

            value_denorm = (
                self.value_popart.denormalize_outputs(old_values_seq)
                if self.enable_popart
                else old_values_seq
            )
            logs_acc["mean/value_prediction"].append(value_denorm[valid_mask].mean().item())
            logs_acc["std/value_prediction"].append(value_denorm[valid_mask].std().item())

        if mask_btna.any():
            logs_acc["mean/logp_old"].append(old_log_probs_seq[mask_btna.bool()].mean().item())

        with torch.no_grad():
            s_mask = {"gridobject_sequence_mask": padded_states_dict_seq.get("central", {}).get("gridobject_sequence_mask")}
            ns_mask = {"gridobject_sequence_mask": padded_next_states_dict_seq.get("central", {}).get("gridobject_sequence_mask")}
            emb_s_seq, _ = self.embedding_net.get_base_embedding(padded_states_dict_seq, central_component_padding_masks=s_mask)
            emb_ns_seq, _ = self.embedding_net.get_base_embedding(padded_next_states_dict_seq, central_component_padding_masks=ns_mask)
            feats_s_seq, h_n_for_next_buffer = self._apply_memory_seq(emb_s_seq, initial_hidden_states_batch, return_hidden=True)
            feats_ns_seq = self._apply_memory_seq(emb_ns_seq, h_n_for_next_buffer)

            _, _, _, s_val_feats_agg, s_base_feats_agg = self._get_critic_outputs(feats_s_seq, padded_actions_seq, B, T, NA)
            _, _, _, ns_val_feats_agg, ns_base_feats_agg = self._get_critic_outputs(feats_ns_seq, torch.zeros_like(padded_actions_seq), B, T, NA)

            intrinsic_rewards, intrinsic_stats = self._compute_intrinsic_rewards(feats_s_seq, padded_actions_seq, mask_btna)
            self._log_intrinsic_reward_stats(logs_acc, intrinsic_stats)
            padded_returns_seq = padded_returns_seq + intrinsic_rewards.mean(dim=-1, keepdim=True)

            if self.enable_popart:
                self._update_popart_stats(padded_returns_seq, attention_mask_batch.unsqueeze(-1), mask_btna.unsqueeze(-1))
                self._log_popart_stats(logs_acc)

            if self.enable_iqn_distillation:
                self._push_to_iqn_buffer(
                    s_val_feats_agg,
                    ns_val_feats_agg,
                    s_base_feats_agg,
                    ns_base_feats_agg,
                    padded_actions_seq,
                    padded_returns_seq,
                    padded_dones_seq,
                    attention_mask_batch,
                    initial_hidden_states_batch,
                h_n_for_next_buffer,
                )

        baseline_denorm = (
            self.baseline_popart.denormalize_outputs(old_baselines_seq)
            if self.enable_popart
            else old_baselines_seq
        )
        baseline_denorm = baseline_denorm.squeeze(-1)
        advantages_seq = self._calculate_advantages(padded_returns_seq, baseline_denorm, mask_btna, logs_acc)

        idx = np.arange(B)
        for _epoch in range(self.epochs):
            np.random.shuffle(idx)
            for mb_start in range(0, B, self.mini_batch_size):
                mb_idx = idx[mb_start : mb_start + self.mini_batch_size]; M = len(mb_idx)
                if M == 0: continue

                # --- Minibatch Data Preparation ---
                mb_actions = padded_actions_seq[mb_idx]
                mb_old_logp = old_log_probs_seq[mb_idx]
                mb_old_val_norm = old_values_seq[mb_idx]
                mb_old_base_norm = old_baselines_seq[mb_idx]
                mb_returns = padded_returns_seq[mb_idx]
                mb_adv = advantages_seq[mb_idx]
                mb_init_h = initial_hidden_states_batch[mb_idx] if initial_hidden_states_batch is not None else None
                mb_mask_btna = mask_btna[mb_idx]
                
                # Recompute features for the minibatch to ensure gradients flow correctly
                mb_states_dict = {"central": {k: v[mb_idx] for k, v in padded_states_dict_seq.get("central", {}).items() if v is not None}, "agent": padded_states_dict_seq.get("agent")[mb_idx] if "agent" in padded_states_dict_seq else None}
                s_mask_tensor = padded_states_dict_seq.get("central", {}).get("gridobject_sequence_mask")
                mb_s_mask = {"gridobject_sequence_mask": s_mask_tensor[mb_idx]} if s_mask_tensor is not None else None
                emb_s_mb, _ = self.embedding_net.get_base_embedding(mb_states_dict, central_component_padding_masks=mb_s_mask)
                feats_s_mb = self._apply_memory_seq(emb_s_mb, mb_init_h)

                # --- PPO Core Loss Calculation ---
                new_lp_mb, ent_mb = self._recompute_log_probs_from_features(feats_s_mb, mb_actions)
                new_val_norm, new_base_norm, _, _, _ = self._get_critic_outputs(feats_s_mb, mb_actions, M, T, NA)
                
                pol_loss = self._ppo_clip_loss(new_lp_mb, mb_old_logp.detach(), mb_adv.detach(), self.ppo_clip_range, mb_mask_btna)
                ent_loss = (-ent_mb * mb_mask_btna).sum() / mb_mask_btna.sum().clamp(min=1e-8)
                val_targets = mb_returns if not self.enable_popart else self.value_popart.normalize_targets(mb_returns)
                val_loss = self._clipped_value_loss(mb_old_val_norm.detach(), new_val_norm, val_targets.detach(), self.value_clip_range, attention_mask_batch[mb_idx].unsqueeze(-1))
                base_targets = mb_returns.unsqueeze(2).expand_as(new_base_norm)
                if self.enable_popart: base_targets = self.baseline_popart.normalize_targets(base_targets)
                base_loss = self._clipped_value_loss(mb_old_base_norm.detach(), new_base_norm, base_targets.detach(), self.value_clip_range, mb_mask_btna.unsqueeze(-1))
                
                # --- PPO Optimization Step ---
                ppo_total_loss = pol_loss + self.entropy_coeff * ent_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss
                
                ppo_opts = [self.optimizers["trunk"], self.optimizers["policy"], self.optimizers["value_path"], self.optimizers["baseline_path"]]
                for opt in ppo_opts: opt.zero_grad()
                ppo_total_loss.backward()
                total_grad_norm = self._clip_and_step_ppo_optimizers(ppo_opts)
                
                # --- Disagreement Ensemble Update Step ---
                if self.disagreement_cfg:
                    ns_mask_tensor = padded_next_states_dict_seq.get("central", {}).get("gridobject_sequence_mask")
                    mb_ns_mask = {"gridobject_sequence_mask": ns_mask_tensor[mb_idx]} if ns_mask_tensor is not None else None
                    mb_next_states_dict = {"central": {k: v[mb_idx] for k,v in padded_next_states_dict_seq.get("central", {}).items() if v is not None}, "agent": padded_next_states_dict_seq.get("agent")[mb_idx] if "agent" in padded_next_states_dict_seq else None}
                    emb_ns_mb, _ = self.embedding_net.get_base_embedding(mb_next_states_dict, central_component_padding_masks=mb_ns_mask)
                    feats_ns_mb = self._apply_memory_seq(emb_ns_mb, self._get_next_hidden_state(feats_s_mb, mb_init_h))
                    
                    disagreement_loss = self._compute_disagreement_loss(feats_s_mb.detach(), mb_actions.detach(), feats_ns_mb.detach(), mb_mask_btna)
                    if disagreement_loss.item() > 0:
                        self.optimizers["disagreement_ensemble"].zero_grad()
                        (self.disagreement_loss_coeff * disagreement_loss).backward() # Uses the coeff
                        self.optimizers["disagreement_ensemble"].step()
                else:
                    disagreement_loss = torch.tensor(0.0)

                # --- RND Predictor Update Step ---
                if self.rnd_cfg:
                    rnd_loss = self._compute_rnd_loss(feats_s_mb.detach(), mb_mask_btna)
                    self.optimizers["rnd_predictor"].zero_grad()
                    (self.rnd_loss_coeff * rnd_loss).backward()
                    self.optimizers["rnd_predictor"].step()
                else:
                    rnd_loss = torch.tensor(0.0)
                
                self._log_minibatch_stats(
                    logs_acc, pol_loss, val_loss, base_loss, ent_mb, new_lp_mb, 
                    mb_mask_btna, rnd_loss, disagreement_loss, feats_s_mb, 
                    total_grad_norm, new_val_norm, new_base_norm
                )

        if self.enable_iqn_distillation:
            self._train_iqn_networks(logs_acc)
        
        self._step_schedulers()
        
        return self._finalize_logs(logs_acc)
   
    # --- Initialization & Setup Helpers ---

    def _initialize_logs(self) -> Dict[str, List[float]]:
        """Initializes a dictionary to accumulate logs for one full update cycle."""
        logs_acc: Dict[str, List[Any]] = {
            "loss/policy": [], "loss/value": [], "loss/baseline": [], "loss/entropy": [],
            "mean/advantage_unnormalized": [], "std/advantage_unnormalized": [],
            "mean/logp_old": [], "mean/logp_new": [],
            "mean/return": [], "std/return": [],
            "reward/raw_mean": [], "reward/raw_std": [],
            "mean/value_prediction": [], "std/value_prediction": [],
            "grad_norm/total": [], "grad_norm/trunk": [], "grad_norm/policy": [],
            "grad_norm/value_path": [], "grad_norm/baseline_path": []
        }
        if self.rnd_cfg:
            logs_acc.update({"loss/rnd": [], "reward/rnd_raw_mean": [], "reward/rnd_raw_std": [], "reward/rnd_norm_mean": [], "reward/rnd_norm_std": [], "grad_norm/rnd_predictor": []})
        if self.disagreement_cfg:
            logs_acc.update({"loss/disagreement": [], "reward/disagreement_raw_mean": [], "reward/disagreement_raw_std": [], "reward/disagreement_norm_mean": [], "reward/disagreement_norm_std": [], "grad_norm/disagreement_ensemble": []})
        if self.enable_iqn_distillation:
            logs_acc.update({
                "loss/value_iqn": [], "loss/baseline_iqn": [],
                "mean/value_quantiles": [], "std/value_quantiles": [],
                "mean/baseline_quantiles": [], "std/baseline_quantiles": [],
                "mean/distilled_value": [], "std/distilled_value": [],
                "mean/distilled_baseline": [], "std/distilled_baseline": [],
                "grad_norm/value_iqn": [], "grad_norm/baseline_iqn": [],
                "grad_norm/distill_value": [], "grad_norm/distill_baseline": []
            })
        if self.enable_popart:
            logs_acc.update({"popart/value_mu": [], "popart/value_sigma": [], "popart/baseline_mu": [], "popart/baseline_sigma": []})
        if self.action_space_type_str_from_config == "beta":
            logs_acc.update({"policy/alpha_mean": [], "policy/alpha_std": [], "policy/beta_mean": [], "policy/beta_std": []})
        return logs_acc
    
    def _determine_action_space(self, action_config: Dict):
        policy_type_override = self.config["agent"]["params"]["networks"]["policy_network"].get("type")
        action_spec = action_config.get("agent", action_config.get("central", {}))
        if policy_type_override: self.action_space_type_str_from_config = policy_type_override
        elif "continuous" in action_spec: self.action_space_type_str_from_config = "beta"
        elif "discrete" in action_spec: self.action_space_type_str_from_config = "discrete"
        else: raise ValueError("Could not determine action space from config.")
        if self.action_space_type_str_from_config in ["beta", "tanh_continuous", "gaussian"]:
            self.action_dim = len(action_spec["continuous"])
            self.total_action_dim_per_agent = self.action_dim
        elif self.action_space_type_str_from_config == "discrete":
            self.action_dim = [d["num_choices"] for d in action_spec["discrete"]]
            self.total_action_dim_per_agent = len(self.action_dim)

    def _setup_iqn(self, critic_cfg):
        iqn_params = critic_cfg.get("iqn_params", {})
        self.num_quantiles = iqn_params.get("num_quantiles", 32)
        self.num_quantiles_prime = iqn_params.get("num_quantiles_prime", self.num_quantiles)
        self.iqn_kappa = iqn_params.get("kappa", 1.0)
        self.iqn_polyak = iqn_params.get("polyak_tau", 0.005)

        self.iqn_max_grad_norm = iqn_params.get("iqn_max_grad_norm", self.max_grad_norm)
        self.iqn_epochs = iqn_params.get("iqn_epochs", 1)

        self.value_iqn_loss_coeff = iqn_params.get("value_iqn_loss_coeff", 0.25)
        self.baseline_iqn_loss_coeff = iqn_params.get("baseline_iqn_loss_coeff", 0.25)
        self.iqn_target_update_freq = iqn_params.get("iqn_target_update_freq", 10)
        self.iqn_batch_size = iqn_params.get("iqn_batch_size", self.mini_batch_size)
        
        self._iqn_total_updates_counter = 0
        self.iqn_replay_buffer = IQNReplayBuffer(iqn_params.get("iqn_replay_buffer_size", 10000), self.device)
        self.target_value_iqn_net = copy.deepcopy(self.shared_critic.value_iqn_net).to(self.device)
        self.target_baseline_iqn_net = copy.deepcopy(self.shared_critic.baseline_iqn_net).to(self.device)
        for p in self.target_value_iqn_net.parameters(): p.requires_grad = False
        for p in self.target_baseline_iqn_net.parameters(): p.requires_grad = False

    def _setup_popart(self, a_cfg):
        beta, eps = a_cfg.get("popart_beta", 0.999), a_cfg.get("popart_epsilon", 1e-5)
        if self.enable_iqn_distillation:
            self.value_popart = PopArtNormalizer(self.shared_critic.value_distill_net.mlp[-1], beta, eps, self.device)
            self.baseline_popart = PopArtNormalizer(self.shared_critic.baseline_distill_net.mlp[-1], beta, eps, self.device)
        else:
            self.value_popart = PopArtNormalizer(self.shared_critic.value_head_ppo.output_layer, beta, eps, self.device)
            self.baseline_popart = PopArtNormalizer(self.shared_critic.baseline_head_ppo.output_layer, beta, eps, self.device)
        self.rewards_normalizer = None

    def _setup_rnd(self, rnd_cfg: Dict, trunk_dim: int):
        """Sets up the RND module from its dedicated configuration dictionary."""
        print("RND module ENABLED.")
        self.rnd_reward_coeff = rnd_cfg.get("reward_coeff", 0.01)
        self.rnd_loss_coeff = rnd_cfg.get("loss_coeff", 1.0) # New loss coefficient
        self.rnd_update_prop = rnd_cfg.get("update_proportion", 0.25)
        
        # Conditionally create a DEDICATED normalizer for RND
        self.normalize_rnd_reward = rnd_cfg.get("normalize_reward", True)
        self.rnd_intrinsic_reward_normalizer = None
        if self.normalize_rnd_reward:
            self.rnd_intrinsic_reward_normalizer = RunningMeanStdNormalizer(epsilon=1e-8, device=self.device)
        
        # Setup RND networks from sub-config
        network_cfg = rnd_cfg.get("network", {})
        output_size = network_cfg.get("output_size", 128)
        hidden_size = network_cfg.get("hidden_size", 256)
        
        self.rnd_target_network = RNDTargetNetwork(trunk_dim, output_size, hidden_size).to(self.device)
        for p in self.rnd_target_network.parameters(): p.requires_grad = False
        self.rnd_predictor_network = RNDPredictorNetwork(trunk_dim, output_size, hidden_size).to(self.device)

    def _setup_disagreement(self, disagreement_cfg: Dict, trunk_dim: int):
        """Sets up the disagreement module from its dedicated configuration dictionary."""
        print("Disagreement module ENABLED.")
        self.disagreement_reward_coeff = disagreement_cfg.get("reward_coeff", 0.01)
        self.disagreement_loss_coeff = disagreement_cfg.get("loss_coeff", 0.1)
        
        # Conditionally create the normalizer based on the toggle within the sub-config
        self.normalize_disagreement_reward = disagreement_cfg.get("normalize_reward", True)
        self.disagreement_intrinsic_reward_normalizer = None
        self.ensemble_size = disagreement_cfg.get("ensemble_size", 5)
        if self.normalize_disagreement_reward:
            self.disagreement_intrinsic_reward_normalizer = RunningMeanStdNormalizer(epsilon=1e-8, device=self.device)

        # Setup ensemble from network sub-config
        network_cfg = disagreement_cfg.get("network", {})
        hidden_size = network_cfg.get("hidden_size", 256)
        ensemble_size = disagreement_cfg.get("ensemble_size", 5)

        self.dynamics_ensemble = nn.ModuleList([
            ForwardDynamicsModel(trunk_dim, self.total_action_dim_per_agent, hidden_size=hidden_size)
            for _ in range(ensemble_size)
        ]).to(self.device)

    def _setup_optimizers(self, a_cfg):
        self.optimizers = {}
        lr_conf = {k: a_cfg.get(k, self.lr_base) for k in ["lr_trunk", "lr_policy", "lr_value_path", "lr_baseline_path", "lr_value_iqn", "lr_baseline_iqn", "lr_rnd_predictor", "lr_disagreement"]}
        beta_conf = a_cfg.get("optimizer_betas", {}); wd_conf = a_cfg.get("weight_decay", {})
        default_betas = tuple(beta_conf.get("default", (0.9, 0.999))); default_wd = wd_conf.get("default", 0.0)
        trunk_params = list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])
        self.optimizers["trunk"] = optim.AdamW(trunk_params, lr=lr_conf["lr_trunk"], betas=tuple(beta_conf.get("trunk", default_betas)), weight_decay=wd_conf.get("trunk", default_wd))
        self.optimizers["policy"] = optim.AdamW(self.policy_net.parameters(), lr=lr_conf["lr_policy"], betas=tuple(beta_conf.get("policy", default_betas)), weight_decay=wd_conf.get("policy", default_wd))
        value_path_params = list(self.shared_critic.value_head_ppo.parameters()) + list(self.shared_critic.value_attention.parameters()) + (list(self.shared_critic.value_distill_net.parameters()) if self.enable_iqn_distillation else [])
        self.optimizers["value_path"] = optim.AdamW(value_path_params, lr=lr_conf["lr_value_path"], betas=tuple(beta_conf.get("value", default_betas)), weight_decay=wd_conf.get("value", default_wd))
        baseline_path_params = list(self.shared_critic.baseline_head_ppo.parameters()) + list(self.shared_critic.baseline_attention.parameters()) + (list(self.shared_critic.baseline_distill_net.parameters()) if self.enable_iqn_distillation else [])
        self.optimizers["baseline_path"] = optim.AdamW(baseline_path_params, lr=lr_conf["lr_baseline_path"], betas=tuple(beta_conf.get("baseline", default_betas)), weight_decay=wd_conf.get("baseline", default_wd))
        if self.enable_iqn_distillation:
            self.optimizers["value_iqn"] = optim.AdamW(self.shared_critic.value_iqn_net.parameters(), lr=lr_conf["lr_value_iqn"], betas=tuple(beta_conf.get("iqn_value", default_betas)), weight_decay=wd_conf.get("iqn_value", 0.0))
            self.optimizers["baseline_iqn"] = optim.AdamW(self.shared_critic.baseline_iqn_net.parameters(), lr=lr_conf["lr_baseline_iqn"], betas=tuple(beta_conf.get("iqn_baseline", default_betas)), weight_decay=wd_conf.get("iqn_baseline", 0.0))
        if self.rnd_cfg: self.optimizers["rnd_predictor"] = optim.AdamW(self.rnd_predictor_network.parameters(), lr=lr_conf["lr_rnd_predictor"], betas=tuple(beta_conf.get("rnd", default_betas)), weight_decay=wd_conf.get("rnd", 0.0))
        if self.disagreement_cfg: self.optimizers["disagreement_ensemble"] = optim.AdamW(self.dynamics_ensemble.parameters(), lr=lr_conf["lr_disagreement"], betas=tuple(beta_conf.get("disagreement", default_betas)), weight_decay=wd_conf.get("disagreement", 0.0))

    def _setup_schedulers(self, a_cfg):
        sched_cfg = a_cfg.get("schedulers", {}); def_lin_sched = {"start_factor": 1.0, "end_factor": 1.0, "total_iters": 1}
        self.schedulers = {name: LinearLR(opt, **sched_cfg.get(f"lr_{name}", def_lin_sched)) for name, opt in self.optimizers.items()}
        self.schedulers["entropy_coeff"] = LinearValueScheduler(**sched_cfg.get("entropy_coeff", {"start_value":self.entropy_coeff, "end_value":self.entropy_coeff, "total_iters":1}))
        self.schedulers["policy_clip"] = LinearValueScheduler(**sched_cfg.get("policy_clip", {"start_value":self.ppo_clip_range, "end_value":self.ppo_clip_range, "total_iters":1}))
        self.schedulers["value_clip"] = LinearValueScheduler(**sched_cfg.get("value_clip", {"start_value":self.value_clip_range, "end_value":self.value_clip_range, "total_iters":1}))
        self.schedulers["max_grad_norm"] = LinearValueScheduler(**sched_cfg.get("max_grad_norm", {"start_value":self.max_grad_norm, "end_value":self.max_grad_norm, "total_iters":1}))
    
    def _calculate_advantages(self, returns_seq: torch.Tensor, baseline_seq: torch.Tensor, mask: torch.Tensor, logs_acc: Dict[str, List[float]]) -> torch.Tensor:
        """
        Calculates the advantages, logs stats for the unnormalized advantages,
        and then returns the optionally normalized advantages.
        """
        # Ensure inputs are shaped as (B,T) for returns and (B,T,NA) for baseline
        if returns_seq.dim() == 3:
            returns_seq = returns_seq.squeeze(-1)

        baseline_seq = baseline_seq.squeeze(-1) if baseline_seq.dim() == 4 else baseline_seq

        # The baseline is already denormalized if PopArt is used.
        advantages = returns_seq.unsqueeze(2).expand_as(baseline_seq) - baseline_seq
        
        # Log stats of unnormalized advantages
        valid_mask = mask.bool()
        if valid_mask.any():
            valid_advantages = advantages[valid_mask]
            # These keys were added to _initialize_logs
            logs_acc["mean/advantage_unnormalized"].append(valid_advantages.mean().item())
            logs_acc["std/advantage_unnormalized"].append(valid_advantages.std().item())

        if self.normalize_adv:
            # Re-fetch valid advantages in case it wasn't computed
            valid_advantages = advantages[valid_mask]
            if valid_advantages.numel() > 1:
                mean = valid_advantages.mean()
                std = valid_advantages.std() + 1e-8
                advantages[valid_mask] = (valid_advantages - mean) / std

        return advantages
    
    def _compute_gae_with_padding(self, r: torch.Tensor, v: torch.Tensor, v_next: torch.Tensor, d: torch.Tensor, tr: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        """
        Computes Generalized Advantage Estimation (GAE) for sequences with padding.
        The `mask_bt` ensures that padded steps do not contribute to the GAE calculation.
        """
        B, T, _ = r.shape
        returns = torch.zeros_like(r)
        gae = torch.zeros(B, 1, device=self.device)
        mask_bt1 = mask_bt.unsqueeze(-1)

        for t in reversed(range(T)):
            # Get masks for the current and next steps. Next step is invalid if current is the last.
            valid_mask_step = mask_bt1[:, t]
            valid_mask_next_step = mask_bt1[:, t + 1] if t < T - 1 else torch.zeros_like(valid_mask_step)

            # Bootstrapping is valid if the current step is not a 'done' terminal state
            # and the next step is a valid, non-padded part of the sequence.
            bootstrap_mask = (1.0 - d[:, t]) * valid_mask_next_step

            # TD-Error: delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = r[:, t] + self.gamma * v_next[:, t] * bootstrap_mask - v[:, t]
            
            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            gae = delta + self.gamma * self.lmbda * bootstrap_mask * gae

            # The return for the current step is GAE + V(s_t)
            current_return = gae + v[:, t]

            # Store returns only for valid steps; padded steps will remain zero.
            returns[:, t] = current_return * valid_mask_step
            
            # Reset GAE for the next iteration if the current step was padding.
            gae = gae * valid_mask_step
            
        return returns

    def compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        truncs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute episode returns using GAE on unpadded sequences."""
        v_denorm = self.value_popart.denormalize_outputs(values) if self.enable_popart else values

        if rewards.shape[-1] == 1 and v_denorm.shape[-1] > 1:
            rewards = rewards.expand_as(v_denorm)

        v_next = torch.zeros_like(v_denorm)
        v_next[:, :-1] = v_denorm[:, 1:]

        mask_bt = torch.ones(rewards.shape[0], rewards.shape[1], device=self.device)
        return self._compute_gae_with_padding(rewards, v_denorm, v_next, dones, truncs, mask_bt)

    def compute_bootstrapped_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        truncs: torch.Tensor,
        bootstrap_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute returns for a truncated rollout using the provided bootstrap value."""
        v_denorm = self.value_popart.denormalize_outputs(values) if self.enable_popart else values

        if rewards.shape[-1] == 1 and v_denorm.shape[-1] > 1:
            rewards = rewards.expand_as(v_denorm)

        v_next = torch.zeros_like(v_denorm)
        v_next[:, :-1] = v_denorm[:, 1:]
        v_next[:, -1] = bootstrap_value

        mask_bt = torch.ones(rewards.shape[0], rewards.shape[1], device=self.device)
        return self._compute_gae_with_padding(rewards, v_denorm, v_next, dones, truncs, mask_bt)

    def _ppo_clip_loss(self, new_lp, old_lp, adv, clip_range, mask):
        """Calculates the PPO clipped surrogate objective loss."""
        ratio = torch.exp((new_lp - old_lp).clamp(-10, 10))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        loss_unreduced = -torch.min(surr1, surr2)
        # Apply mask and compute mean only over valid elements
        masked_loss = loss_unreduced * mask
        return masked_loss.sum() / mask.sum().clamp(min=1e-8)

    def _clipped_value_loss(self, old_v, new_v, target_v, clip_range, mask):
        """Calculates the clipped value function loss."""
        v_clipped = old_v + (new_v - old_v).clamp(-clip_range, clip_range)
        vf_loss1 = (new_v - target_v).pow(2)
        vf_loss2 = (v_clipped - target_v).pow(2)
        loss_unreduced = torch.max(vf_loss1, vf_loss2)
        # Apply mask and compute mean only over valid elements
        masked_loss = loss_unreduced * mask
        return masked_loss.sum() / mask.sum().clamp(min=1e-8)

    # --- Auxiliary Module Helper Methods ---
    
    def _compute_rnd_loss(self, feats_seq, mask_btna) -> torch.Tensor:
        """Computes the loss for the RND predictor network."""
        if not self.rnd_cfg: return torch.tensor(0.0, device=self.device)

        valid_mask_flat = mask_btna.reshape(-1).bool()
        flat_feats = feats_seq.reshape(-1, feats_seq.shape[-1])
        valid_feats = flat_feats[valid_mask_flat]

        if valid_feats.numel() == 0: return torch.tensor(0.0, device=self.device)

        # Update only on a random subset of the valid data
        num_samples = int(valid_feats.shape[0] * self.rnd_update_prop)
        if num_samples == 0: return torch.tensor(0.0, device=self.device)
        
        indices = torch.randperm(valid_feats.shape[0], device=self.device)[:num_samples]
        selected_feats = valid_feats[indices]

        predicted_features = self.rnd_predictor_network(selected_feats)
        with torch.no_grad():
            target_features = self.rnd_target_network(selected_feats)
        
        return F.mse_loss(predicted_features, target_features)

    def _compute_disagreement_loss(self, feats_s_mb, actions_mb, feats_ns_mb, mask_mbna) -> torch.Tensor:
        """Computes the training loss for the forward dynamics ensemble."""
        if not self.disagreement_cfg: return torch.tensor(0.0, device=self.device)

        valid_mask_flat = mask_mbna.reshape(-1).bool()
        flat_feats_s = feats_s_mb.reshape(-1, feats_s_mb.shape[-1])[valid_mask_flat]
        flat_feats_ns = feats_ns_mb.reshape(-1, feats_ns_mb.shape[-1])[valid_mask_flat]
        flat_actions = actions_mb.reshape(-1, actions_mb.shape[-1])[valid_mask_flat]

        if flat_feats_s.numel() == 0: return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        for model in self.dynamics_ensemble:
            predicted_feats_ns = model(flat_feats_s, flat_actions)
            total_loss += F.mse_loss(predicted_feats_ns, flat_feats_ns.detach())
        
        return total_loss / self.ensemble_size

    def _push_to_iqn_buffer(
        self, 
        s_val_feats_agg: torch.Tensor, 
        ns_val_feats_agg: torch.Tensor, 
        s_base_feats_agg: torch.Tensor, 
        ns_base_feats_agg: torch.Tensor,
        actions_seq: torch.Tensor, 
        rewards_seq: torch.Tensor, 
        dones_seq: torch.Tensor, 
        attention_mask: torch.Tensor, 
        initial_h: Optional[torch.Tensor], 
        next_initial_h: Optional[torch.Tensor]
    ):
        """Pushes pre-computed features and trajectory data to the IQN replay buffer."""
        self.iqn_replay_buffer.push_batch(
            s_val_feats_agg, ns_val_feats_agg,
            s_base_feats_agg, ns_base_feats_agg,
            actions_seq, rewards_seq, dones_seq, attention_mask,
            initial_h, next_initial_h
        )

    def _compute_huber_quantile_loss(self, current_quantiles, target_quantiles_detached, taus_for_current):
        """
        Calculates the Huber quantile regression loss for IQN.
        Args:
            current_quantiles (Tensor): Shape (Batch, NumCurrentTaus)
            target_quantiles_detached (Tensor): Shape (Batch, NumTargetTaus)
            taus_for_current (Tensor): Shape (Batch, NumCurrentTaus, 1)
        """
        # Pairwise TD-errors
        td_errors = target_quantiles_detached.unsqueeze(1) - current_quantiles.unsqueeze(2)
        
        # Huber loss calculation
        abs_td_errors = torch.abs(td_errors)
        huber_loss_matrix = torch.where(
            abs_td_errors <= self.iqn_kappa, 
            0.5 * td_errors.pow(2), 
            self.iqn_kappa * (abs_td_errors - 0.5 * self.iqn_kappa)
        )
        
        # Quantile regression weighting
        indicator = (td_errors < 0).float()
        delta_weight = torch.abs(taus_for_current - indicator)
        
        # The final loss is the mean over all dimensions
        loss = (delta_weight * huber_loss_matrix).mean()
        return loss

    def _train_iqn_networks(self, logs_acc: Dict):
        """Performs off-policy training for the IQN value and baseline networks."""
        if not (self.enable_iqn_distillation and self.iqn_replay_buffer and len(self.iqn_replay_buffer) >= self.iqn_batch_size):
            return

        self._iqn_total_updates_counter += 1
        
        # --- FIX: Move accumulators inside, as this is one self-contained training step ---
        val_iqn_losses, base_iqn_losses = [], []
        val_q_means, val_q_stds = [], []
        base_q_means, base_q_stds = [], []
        val_iqn_grads, base_iqn_grads = [], []

        for _epoch_iqn in range(self.iqn_epochs):
            sample = self.iqn_replay_buffer.sample(self.iqn_batch_size)
            if sample is None: continue

            s_val_agg, ns_val_agg, s_base_agg, ns_base_agg, _, r_seq, d_seq, mask, _, _ = sample
            B, T, NA, _ = s_base_agg.shape
            
            # --- Value IQN Loss ---
            valid_mask_val_flat = mask.view(-1).bool()
            s_val_valid = s_val_agg.view(B * T, -1)[valid_mask_val_flat]
            
            if s_val_valid.shape[0] > 0:
                ns_val_valid = ns_val_agg.view(B * T, -1)[valid_mask_val_flat]
                r_val_valid = r_seq.view(B * T, -1)[valid_mask_val_flat]
                d_val_valid = d_seq.view(B * T, -1)[valid_mask_val_flat]
                
                eff_b_val = s_val_valid.shape[0]
                taus_current_val = self._sample_taus(eff_b_val, self.num_quantiles)
                taus_next_val = self._sample_taus(eff_b_val, self.num_quantiles_prime)

                current_value_quantiles = self.shared_critic.value_iqn_net(s_val_valid, taus_current_val)
                with torch.no_grad():
                    next_value_quantiles = self.target_value_iqn_net(ns_val_valid, taus_next_val)
                
                bellman_targets = r_val_valid + self.gamma * (1.0 - d_val_valid) * next_value_quantiles
                value_iqn_loss = self._compute_huber_quantile_loss(current_value_quantiles, bellman_targets.detach(), taus_current_val)
                
                val_iqn_losses.append(value_iqn_loss.item())
                val_q_means.append(current_value_quantiles.mean().item())
                val_q_stds.append(current_value_quantiles.std().item())
            
            # --- Baseline IQN Loss ---
            valid_mask_base_flat = mask.unsqueeze(-1).expand(-1, -1, NA).reshape(-1).bool()
            s_base_valid = s_base_agg.view(B * T * NA, -1)[valid_mask_base_flat]

            if s_base_valid.shape[0] > 0:
                ns_base_valid = ns_base_agg.view(B * T * NA, -1)[valid_mask_base_flat]
                r_base_valid = r_seq.unsqueeze(2).expand(-1, -1, NA, -1).reshape(B*T*NA, 1)[valid_mask_base_flat]
                d_base_valid = d_seq.unsqueeze(2).expand(-1, -1, NA, -1).reshape(B*T*NA, 1)[valid_mask_base_flat]

                eff_b_base = s_base_valid.shape[0]
                taus_current_base = self._sample_taus(eff_b_base, self.num_quantiles)
                taus_next_base = self._sample_taus(eff_b_base, self.num_quantiles_prime)

                current_baseline_quantiles = self.shared_critic.baseline_iqn_net(s_base_valid, taus_current_base)
                with torch.no_grad():
                    next_baseline_quantiles = self.target_baseline_iqn_net(ns_base_valid, taus_next_base)

                baseline_bellman_targets = r_base_valid + self.gamma * (1.0 - d_base_valid) * next_baseline_quantiles
                baseline_iqn_loss = self._compute_huber_quantile_loss(current_baseline_quantiles, baseline_bellman_targets.detach(), taus_current_base)
                
                base_iqn_losses.append(baseline_iqn_loss.item())
                base_q_means.append(current_baseline_quantiles.mean().item())
                base_q_stds.append(current_baseline_quantiles.std().item())

            # --- Optimization Step ---
            if 'value_iqn_loss' in locals() and 'baseline_iqn_loss' in locals():
                total_iqn_loss = (self.value_iqn_loss_coeff * value_iqn_loss) + (self.baseline_iqn_loss_coeff * baseline_iqn_loss)
                
                self.optimizers["value_iqn"].zero_grad()
                self.optimizers["baseline_iqn"].zero_grad()
                if total_iqn_loss > 0:
                    total_iqn_loss.backward()

                    # Combine parameters from both IQN optimizers for a single clipping operation
                    iqn_params = list(self.shared_critic.value_iqn_net.parameters()) + \
                                 list(self.shared_critic.baseline_iqn_net.parameters())
                    
                    # Clip the gradients of the IQN networks
                    clip_grad_norm_(iqn_params, self.iqn_max_grad_norm)

                    val_iqn_grads.append(_get_avg_grad_mag(self.shared_critic.value_iqn_net.parameters()))
                    base_iqn_grads.append(_get_avg_grad_mag(self.shared_critic.baseline_iqn_net.parameters()))
                    
                    self.optimizers["value_iqn"].step()
                    self.optimizers["baseline_iqn"].step()

        # --- Append logs for the entire IQN training phase ---
        if val_iqn_losses: logs_acc["loss/value_iqn"].append(np.mean(val_iqn_losses))
        if base_iqn_losses: logs_acc["loss/baseline_iqn"].append(np.mean(base_iqn_losses))
        if val_q_means: logs_acc["mean/value_quantiles"].append(np.mean(val_q_means))
        if val_q_stds: logs_acc["std/value_quantiles"].append(np.mean(val_q_stds))
        if base_q_means: logs_acc["mean/baseline_quantiles"].append(np.mean(base_q_means))
        if base_q_stds: logs_acc["std/baseline_quantiles"].append(np.mean(base_q_stds))
        if val_iqn_grads: logs_acc["grad_norm/value_iqn"].append(np.mean(val_iqn_grads))
        if base_iqn_grads: logs_acc["grad_norm/baseline_iqn"].append(np.mean(base_iqn_grads))

        # --- Target Network Update ---
        if self._iqn_total_updates_counter % self.iqn_target_update_freq == 0:
            self._update_target_iqn_networks()

    def _update_target_iqn_networks(self):
        """Performs a soft (Polyak) update of the IQN target networks."""
        if self.enable_iqn_distillation:
            for target_param, online_param in zip(self.target_value_iqn_net.parameters(), self.shared_critic.value_iqn_net.parameters()):
                target_param.data.copy_(self.iqn_polyak * online_param.data + (1.0 - self.iqn_polyak) * target_param.data)
            for target_param, online_param in zip(self.target_baseline_iqn_net.parameters(), self.shared_critic.baseline_iqn_net.parameters()):
                target_param.data.copy_(self.iqn_polyak * online_param.data + (1.0 - self.iqn_polyak) * target_param.data)

    def _update_popart_stats(self, returns_seq, mask_bt1, mask_btna1):
        """Updates PopArt normalizers for value and baseline networks."""
        if not self.enable_popart: return
        
        valid_returns_mask = mask_bt1.bool()
        if valid_returns_mask.any():
            self.value_popart.update_stats(returns_seq[valid_returns_mask])
            
            baseline_targets = returns_seq.unsqueeze(2).expand_as(mask_btna1)
            valid_baseline_targets = baseline_targets[mask_btna1.bool()]
            if valid_baseline_targets.numel() > 0:
                 self.baseline_popart.update_stats(valid_baseline_targets)
    
    def _compute_intrinsic_rewards(self, feats_seq, actions_seq, mask_btna) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes and combines all enabled intrinsic rewards (RND, Disagreement)."""
        B, T, NA, _ = feats_seq.shape
        total_intrinsic_rewards = torch.zeros(B, T, NA, device=self.device)
        intrinsic_stats = {}

        # --- RND Reward Calculation ---
        if self.rnd_cfg:
            flat_feats = feats_seq.reshape(B * T * NA, -1)
            with torch.no_grad():
                target_features = self.rnd_target_network(flat_feats)
                predictor_features = self.rnd_predictor_network(flat_feats)
            
            rnd_errors = F.mse_loss(predictor_features, target_features, reduction="none").mean(dim=-1)
            rnd_rewards = rnd_errors.reshape(B, T, NA)
            
            valid_mask = mask_btna.bool()
            if valid_mask.any():
                valid_rewards = rnd_rewards[valid_mask]
                intrinsic_stats['rnd_raw_mean'] = valid_rewards.mean().item()
                intrinsic_stats['rnd_raw_std'] = valid_rewards.std().item()

                final_rnd_rewards_for_agent = valid_rewards
                if self.normalize_rnd_reward and self.rnd_intrinsic_reward_normalizer:
                    self.rnd_intrinsic_reward_normalizer.update(valid_rewards.unsqueeze(-1))
                    normalized_rewards = self.rnd_intrinsic_reward_normalizer.normalize(valid_rewards.unsqueeze(-1)).squeeze(-1)
                    intrinsic_stats['rnd_norm_mean'] = normalized_rewards.mean().item()
                    intrinsic_stats['rnd_norm_std'] = normalized_rewards.std().item()
                    final_rnd_rewards_for_agent = normalized_rewards
                
                total_intrinsic_rewards[valid_mask] += self.rnd_reward_coeff * final_rnd_rewards_for_agent

        # --- Disagreement Reward Calculation ---
        if self.disagreement_cfg:
            flat_feats = feats_seq.reshape(B * T * NA, -1)
            flat_actions = actions_seq.reshape(B * T * NA, -1)
            
            with torch.no_grad():
                predictions = torch.stack([model(flat_feats, flat_actions) for model in self.dynamics_ensemble])
            
            disagreement = predictions.var(dim=0).mean(dim=-1)
            disagreement_rewards = disagreement.reshape(B, T, NA)

            valid_mask = mask_btna.bool()
            if valid_mask.any():
                valid_rewards = disagreement_rewards[valid_mask]
                intrinsic_stats['disagreement_raw_mean'] = valid_rewards.mean().item()
                intrinsic_stats['disagreement_raw_std'] = valid_rewards.std().item()

                final_disagreement_rewards_for_agent = valid_rewards
                if self.normalize_disagreement_reward and self.disagreement_intrinsic_reward_normalizer:
                    self.disagreement_intrinsic_reward_normalizer.update(valid_rewards.unsqueeze(-1))
                    normalized_rewards = self.disagreement_intrinsic_reward_normalizer.normalize(valid_rewards.unsqueeze(-1)).squeeze(-1)
                    intrinsic_stats['disagreement_norm_mean'] = normalized_rewards.mean().item()
                    intrinsic_stats['disagreement_norm_std'] = normalized_rewards.std().item()
                    final_disagreement_rewards_for_agent = normalized_rewards

                total_intrinsic_rewards[valid_mask] += self.disagreement_reward_coeff * final_disagreement_rewards_for_agent
                
        return total_intrinsic_rewards, intrinsic_stats
    
    def _log_intrinsic_reward_stats(self, logs_acc, intrinsic_stats):
        """Helper to append intrinsic reward stats to the log accumulator."""
        if self.rnd_cfg:
            logs_acc["reward/rnd_raw_mean"].append(intrinsic_stats.get('rnd_raw_mean', 0.0))
            logs_acc["reward/rnd_raw_std"].append(intrinsic_stats.get('rnd_raw_std', 0.0))
            logs_acc["reward/rnd_norm_mean"].append(intrinsic_stats.get('rnd_norm_mean', 0.0))
            logs_acc["reward/rnd_norm_std"].append(intrinsic_stats.get('rnd_norm_std', 0.0))
        if self.disagreement_cfg:
            logs_acc["reward/disagreement_raw_mean"].append(intrinsic_stats.get('disagreement_raw_mean', 0.0))
            logs_acc["reward/disagreement_raw_std"].append(intrinsic_stats.get('disagreement_raw_std', 0.0))
            logs_acc["reward/disagreement_norm_mean"].append(intrinsic_stats.get('disagreement_norm_mean', 0.0))
            logs_acc["reward/disagreement_norm_std"].append(intrinsic_stats.get('disagreement_norm_std', 0.0))
    
    # --- Utility Helper Methods ---
    
    def _apply_memory_seq(self, seq_feats, init_h, return_hidden=False):
        """Applies the recurrent memory module to a sequence of features."""
        if not self.memory_module: 
            return (seq_feats, None) if return_hidden else seq_feats

        B, T, NA, F = seq_feats.shape
        flat_feats = seq_feats.reshape(B * NA, T, F)
        
        h0_gru = init_h.reshape(B * NA, self.memory_layers, self.memory_hidden).permute(1, 0, 2).contiguous() if init_h is not None else None

        flat_out, h_n_gru = self.memory_module.forward_sequence(flat_feats, h0_gru)
        out = flat_out.reshape(B, T, NA, self.memory_hidden)
        
        if return_hidden:
            h_n_reshaped = h_n_gru.permute(1, 0, 2).reshape(B, NA, self.memory_layers, self.memory_hidden)
            return out, h_n_reshaped
        return out

    def _get_next_hidden_state(self, feats_s_mb, init_h_mb):
        """Computes the next hidden state without backpropagating through it."""
        if not self.memory_module: return None
        with torch.no_grad():
            _, h_next = self._apply_memory_seq(feats_s_mb, init_h_mb, return_hidden=True)
        return h_next

    def _recompute_log_probs_from_features(self, feats_seq, actions_seq):
        """Recomputes log probabilities and entropies for a given sequence."""
        B,T,NA,F = feats_seq.shape
        flat_feats = feats_seq.reshape(B*T*NA, F)
        action_dim = actions_seq.shape[-1]
        flat_actions = actions_seq.reshape(B*T*NA, action_dim)
        lp_flat, ent_flat = self.policy_net.recompute_log_probs(flat_feats, flat_actions)
        return lp_flat.reshape(B,T,NA), ent_flat.reshape(B,T,NA)

    def _sample_taus(self, batch_dim0_size: int, num_quantiles: int):
        """Generates random quantile fractions (taus) for IQN."""
        return torch.rand(batch_dim0_size, num_quantiles, 1, device=self.device)
    
    def _get_critic_outputs(self, feats_seq: torch.Tensor, actions_seq: torch.Tensor, B: int, T: int, NA: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Helper to get all critic-related outputs.
        
        Returns:
            A tuple of (
                value_distilled_or_ppo (Tensor),
                baseline_distilled_or_ppo (Tensor),
                value_quantiles (Tensor, optional),
                s_val_feats_agg (Tensor, optional), -> FOR REPLAY BUFFER
                s_base_feats_agg (Tensor, optional) -> FOR REPLAY BUFFER
            )
        """
        B_T_flat = B * T
        B_T_NA_flat = B * T * NA

        # --- Value Path ---
        val_input_flat = feats_seq.reshape(B_T_flat, NA, -1)
        s_val_attn_out, _ = self.shared_critic.value_attention(val_input_flat)
        s_val_feats_agg = s_val_attn_out.mean(dim=1)  # Shape: (B*T, H_embed)

        # --- Baseline Path ---
        base_input_comb = self.embedding_net.get_baseline_embeddings(feats_seq, actions_seq)
        base_input_flat = base_input_comb.reshape(B_T_NA_flat, NA, -1)
        s_base_attn_out, _ = self.shared_critic.baseline_attention(base_input_flat)
        s_base_feats_agg = s_base_attn_out.mean(dim=1) # Shape: (B*T*NA, H_embed)
        
        if self.enable_iqn_distillation:
            taus_val = self._sample_taus(B_T_flat, self.num_quantiles)
            taus_base = self._sample_taus(B_T_NA_flat, self.num_quantiles)
            
            v_ppo = self.shared_critic.value_head_ppo(s_val_feats_agg)
            value_quantiles = self.shared_critic.value_iqn_net(s_val_feats_agg, taus_val)
            val_distilled_norm = self.shared_critic.value_distill_net(v_ppo, value_quantiles)

            b_ppo = self.shared_critic.baseline_head_ppo(s_base_feats_agg)
            baseline_quantiles = self.shared_critic.baseline_iqn_net(s_base_feats_agg, taus_base)
            base_distilled_norm = self.shared_critic.baseline_distill_net(b_ppo, baseline_quantiles)
        else:
            val_distilled_norm = self.shared_critic.value_head_ppo(s_val_feats_agg)
            base_distilled_norm = self.shared_critic.baseline_head_ppo(s_base_feats_agg)
            value_quantiles = None
        
        # Reshape all outputs
        final_val = val_distilled_norm.reshape(B, T, 1)
        final_base = base_distilled_norm.reshape(B, T, NA, 1)
        final_s_val_feats = s_val_feats_agg.reshape(B, T, -1)
        final_s_base_feats = s_base_feats_agg.reshape(B, T, NA, -1)

        return final_val, final_base, value_quantiles, final_s_val_feats, final_s_base_feats
    
    # --- Optimization & Logging ---
    
    def _zero_all_grads(self):
        """Sets gradients of all optimizers to zero."""
        for opt in self.optimizers.values():
            opt.zero_grad()

    def _clip_and_step_ppo_optimizers(self, ppo_optimizers: List[optim.Optimizer]) -> float:
        """Clips gradients for PPO-related optimizers and steps them."""
        all_ppo_params = [p for opt in ppo_optimizers for group in opt.param_groups for p in group['params']]
        
        total_grad_norm = clip_grad_norm_(all_ppo_params, self.max_grad_norm)
        
        for opt in ppo_optimizers:
            opt.step()
            
        return total_grad_norm.item()
    
    def _step_schedulers(self):
        """Steps all learning rate and parameter schedulers."""
        for sched in self.schedulers.values():
            if isinstance(sched, torch.optim.lr_scheduler._LRScheduler):
                sched.step()
        
        self.entropy_coeff = self.schedulers["entropy_coeff"].step()
        self.ppo_clip_range = self.schedulers["policy_clip"].step()
        self.value_clip_range = self.schedulers["value_clip"].step()
        self.max_grad_norm = self.schedulers["max_grad_norm"].step()

    def _log_minibatch_stats(self, logs_acc, pol_loss, val_loss, base_loss, ent_mb, new_lp_mb, mask_mbna, rnd_loss, disagreement_loss, feats_s_mb, total_grad_norm, new_val_norm=None, new_base_norm=None):
        """Accumulates statistics from a single minibatch into the logs dictionary."""
        # Core PPO Losses
        logs_acc["loss/policy"].append(pol_loss.item())
        logs_acc["loss/value"].append(val_loss.item())
        logs_acc["loss/baseline"].append(base_loss.item())
        
        logs_acc["grad_norm/total"].append(total_grad_norm)

        # Policy Stats
        valid_mask = mask_mbna.bool()
        if valid_mask.any():
            logs_acc["loss/entropy"].append(ent_mb[valid_mask].mean().item())
            logs_acc["mean/logp_new"].append(new_lp_mb[valid_mask].mean().item())
        
        # Auxiliary Losses
        if self.rnd_cfg: logs_acc["loss/rnd"].append(rnd_loss.item())
        if self.disagreement_cfg: logs_acc["loss/disagreement"].append(disagreement_loss.item())
        
        # Log distilled values if IQN is enabled
        if self.enable_iqn_distillation and new_val_norm is not None and new_base_norm is not None:
            valid_val_mask = mask_mbna.any(dim=-1, keepdim=True).bool()
            if valid_val_mask.any():
                logs_acc["mean/distilled_value"].append(new_val_norm[valid_val_mask].mean().item())
                logs_acc["std/distilled_value"].append(new_val_norm[valid_val_mask].std().item())
            valid_base_mask = mask_mbna.unsqueeze(-1).bool()
            if valid_base_mask.any():
                logs_acc["mean/distilled_baseline"].append(new_base_norm[valid_base_mask].mean().item())
                logs_acc["std/distilled_baseline"].append(new_base_norm[valid_base_mask].std().item())

        # Policy-Specific Logging
        if self.action_space_type_str_from_config == "beta":
            with torch.no_grad():
                flat_feats = feats_s_mb.reshape(-1, feats_s_mb.shape[-1])
                valid_feats = flat_feats[valid_mask.reshape(-1)]
                if valid_feats.numel() > 0:
                    alpha, beta = self.policy_net.forward(valid_feats)
                    logs_acc["policy/alpha_mean"].append(alpha.mean().item())
                    logs_acc["policy/alpha_std"].append(alpha.std().item())
                    logs_acc["policy/beta_mean"].append(beta.mean().item())
                    logs_acc["policy/beta_std"].append(beta.std().item())

        # Component-wise Gradient Norms
        logs_acc["grad_norm/trunk"].append(_get_avg_grad_mag(list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])))
        logs_acc["grad_norm/policy"].append(_get_avg_grad_mag(list(self.policy_net.parameters())))
        
        value_path_params = list(self.shared_critic.value_head_ppo.parameters()) + list(self.shared_critic.value_attention.parameters())
        if self.enable_iqn_distillation: value_path_params += list(self.shared_critic.value_distill_net.parameters())
        logs_acc["grad_norm/value_path"].append(_get_avg_grad_mag(value_path_params))
        
        baseline_path_params = list(self.shared_critic.baseline_head_ppo.parameters()) + list(self.shared_critic.baseline_attention.parameters())
        if self.enable_iqn_distillation: baseline_path_params += list(self.shared_critic.baseline_distill_net.parameters())
        logs_acc["grad_norm/baseline_path"].append(_get_avg_grad_mag(baseline_path_params))
        
        if self.enable_iqn_distillation:
            logs_acc["grad_norm/distill_value"].append(_get_avg_grad_mag(self.shared_critic.value_distill_net.parameters()))
            logs_acc["grad_norm/distill_baseline"].append(_get_avg_grad_mag(self.shared_critic.baseline_distill_net.parameters()))

        if self.rnd_cfg: logs_acc["grad_norm/rnd_predictor"].append(_get_avg_grad_mag(self.rnd_predictor_network.parameters()))
        if self.disagreement_cfg: logs_acc["grad_norm/disagreement_ensemble"].append(_get_avg_grad_mag(self.dynamics_ensemble.parameters()))

    def _finalize_logs(self, logs_acc: Dict) -> Dict[str, float]:
        """
        Calculates the mean of all accumulated log values for the entire update step.

        Args:
            logs_acc (Dict): The dictionary containing lists of metrics collected
                             from each minibatch.

        Returns:
            Dict[str, float]: A dictionary with the final, averaged scalar values
                              ready for logging.
        """
        # Calculate the mean for any metric that has collected data.
        final_logs = {
            k: np.mean(v) if isinstance(v, list) and v else 0.0
            for k, v in logs_acc.items()
        }
        
        # Add the current values of any parameter schedulers to the logs.
        final_logs.update({
            f"lr/{name}_curr": opt.param_groups[0]['lr'] 
            for name, opt in self.optimizers.items()
        })
        final_logs.update({
            "param/entropy_coeff_curr": self.entropy_coeff,
            "param/ppo_clip_curr": self.ppo_clip_range,
            "param/value_clip_curr": self.value_clip_range,
            "param/max_grad_norm_curr": self.max_grad_norm,
        })
        
        return final_logs
    
    def _log_popart_stats(self, logs_acc: Dict[str, List[float]]):
        """Helper to log the current stats of the PopArt normalizers."""
        if not self.enable_popart: return
        
        logs_acc["popart/value_mu"].append(self.value_popart.mu.item())
        logs_acc["popart/value_sigma"].append(self.value_popart.sigma.item())
        logs_acc["popart/baseline_mu"].append(self.baseline_popart.mu.item())
        logs_acc["popart/baseline_sigma"].append(self.baseline_popart.sigma.item())
        
    # --- Save/Load ---

    def save(self, location: str) -> None:
        """Saves the agent's state_dict."""
        torch.save(self.state_dict(), location)

    def load(self, location: str) -> None:
        """Loads the agent's state_dict."""
        state = torch.load(location, map_location=self.device)
        self.load_state_dict(state)
