from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer, LinearValueScheduler, PopArtNormalizer, LinearLRDecay
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

def _get_l2_grad_norm(params_list: List[torch.nn.Parameter]) -> float:
    """Computes the L2 norm of gradients across a list of parameters."""
    total_sq = 0.0
    if not params_list:
        return 0.0
    for p in params_list:
        if p.grad is not None:
            g = p.grad
            total_sq += float(torch.sum(g * g).item())
    return float(total_sq ** 0.5) if total_sq > 0.0 else 0.0

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

        # --- FQF controls (optional) ---
        # If True, include trunk params in the second (fraction-only) phase update
        self.fqf_fraction_update_trunk = a_cfg.get("fqf_fraction_update_trunk", False)
        # Warmup iterations during which fraction entropy reg is disabled
        self.fraction_entropy_warmup_iters = int(a_cfg.get("fraction_entropy_warmup_iters", 0))
        # Optional clamp on fraction entropy term to prevent spikes (None disables)
        self.fraction_entropy_cap = a_cfg.get("fraction_entropy_cap", None)
        # Training step counter (minibatch steps)
        self.update_step = 0
        
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
        self.embedding_net = MultiAgentEmbeddingNetwork(
            emb_main_cfg, environment_config_for_shapes=env_cfg
        ).to(device)
        emb_out_dim = emb_main_cfg["cross_attention_feature_extractor"][
            "embed_dim"
        ]
        self.emb_out_dim = emb_out_dim

        self.enable_memory = a_cfg.get("enable_memory", False)
        if self.enable_memory:
            mem_type = a_cfg.get("memory_type", "gru")
            mem_specific_cfg = a_cfg.get(mem_type, {}).copy()
            mem_specific_cfg.setdefault("input_size", emb_out_dim)
            self.memory_hidden = mem_specific_cfg.get("hidden_size", 128)
            self.memory_layers = mem_specific_cfg.get("num_layers", 1)
            self.memory_module: Optional[RecurrentMemoryNetwork] = GRUSequenceMemory(**mem_specific_cfg).to(device)
            if self.memory_hidden != emb_out_dim:
                self.memory_to_embed_proj = nn.Linear(self.memory_hidden, emb_out_dim).to(device)
            else:
                self.memory_to_embed_proj = None
            trunk_dim = self.memory_hidden
        else:
            self.memory_module = None
            self.memory_to_embed_proj = None
            trunk_dim = emb_out_dim

        pol_cfg = net_cfg["policy_network"].copy()
        policy_type_str = pol_cfg.pop("type", self.action_space_type_str_from_config)
        pol_cfg.update({"in_features": trunk_dim, "out_features": self.action_dim})
        policy_class_map = {"beta": BetaPolicyNetwork, "tanh_continuous": TanhContinuousPolicyNetwork, "gaussian": GaussianPolicyNetwork, "discrete": DiscretePolicyNetwork}
        self.policy_net = policy_class_map[policy_type_str](**pol_cfg).to(device)

        critic_full_config = net_cfg["critic_network"].copy()
        # Ensure distributional network config has feature_dim set
        if "distributional_network_params" not in critic_full_config:
            critic_full_config["distributional_network_params"] = {}
        
        distrib_config = critic_full_config["distributional_network_params"]
        
        # Set feature_dim for whichever network is enabled
        if "fqf_network" in distrib_config:
            distrib_config["fqf_network"]["feature_dim"] = trunk_dim
        elif "iqn_network" in distrib_config:
            distrib_config["iqn_network"]["feature_dim"] = trunk_dim
        # If no distributional network is specified, we'll use traditional scalar value function
        # (SharedCritic will handle this case)
        
        self.shared_critic = SharedCritic(net_cfg=critic_full_config).to(device)
        
        # Store parameters for easier access based on which network is active
        if "fqf_network" in distrib_config:
            # FQF is enabled (distributional value function)
            network_config = distrib_config["fqf_network"]
            self.use_fqf = True
            self.fraction_loss_coeff = network_config.get("fraction_loss_coeff", 0.1)
            self.num_quantiles = network_config.get("num_quantiles", 32)
            self.iqn_kappa = network_config.get("iqn_kappa", 1.0)
            # Renamed for clarity: distributional value enabled
            self.enable_distributional = (self.num_quantiles > 0)
            self.enable_iqn = self.enable_distributional  # Backward compatibility alias
            # New optional controls
            self.fraction_entropy_coeff = network_config.get("fraction_entropy_coeff", 0.0)
            # Optional FQF controls can also be provided inside fqf_network
            self.fqf_fraction_update_trunk = network_config.get(
                "fqf_fraction_update_trunk", self.fqf_fraction_update_trunk)
            self.fraction_entropy_warmup_iters = int(network_config.get(
                "fraction_entropy_warmup_iters", self.fraction_entropy_warmup_iters))
            self.fraction_entropy_cap = network_config.get(
                "fraction_entropy_cap", self.fraction_entropy_cap)
        elif "iqn_network" in distrib_config:
            # IQN is enabled (distributional value function)
            network_config = distrib_config["iqn_network"]
            self.use_fqf = False
            self.fraction_loss_coeff = 0.0  # Not used for IQN
            self.num_quantiles = network_config.get("num_quantiles", 32)
            self.iqn_kappa = network_config.get("iqn_kappa", 1.0)
            self.enable_distributional = (self.num_quantiles > 0)
            self.enable_iqn = self.enable_distributional  # Backward compatibility alias
        else:
            # No distributional network - use traditional scalar value function
            self.use_fqf = False
            self.fraction_loss_coeff = 0.0
            self.num_quantiles = 0
            self.iqn_kappa = 1.0  # Not used, but set for consistency
            self.enable_distributional = False
            self.enable_iqn = False
            self.enable_iqn = False  # Traditional scalar value function

        # --- Auxiliary Modules Setup ---
        self.disagreement_cfg = a_cfg.get("disagreement_rewards")
        self.rnd_cfg = a_cfg.get("rnd_rewards")

        if self.disagreement_cfg:
            self._setup_disagreement(self.disagreement_cfg, trunk_dim)

        if self.rnd_cfg:
            self._setup_rnd(self.rnd_cfg, trunk_dim)
        
        # --- Normalization Setup ---
        self.enable_popart = a_cfg.get("enable_popart", False)
        if self.enable_popart:
            self._setup_popart(a_cfg)
        else:
            self.rewards_normalizer = RunningMeanStdNormalizer(**a_cfg.get("rewards_normalizer", {}), device=device) if a_cfg.get("rewards_normalizer") else None

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
        emb_SBEA, _ = self.embedding_net.get_base_embedding(states)
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
            # Get the counterfactual embeddings needed for the baseline
            baseline_feats_seq = self.embedding_net.get_baseline_embeddings(
                common_emb=feats_for_policy.unsqueeze(1), # Add time dim: (B, 1, NA, F)
                actions=actions_shaped.unsqueeze(1)      # Add time dim: (B, 1, NA, Act)
            )

            val_norm, base_norm, _ = self.shared_critic.get_values_and_baselines(
                value_feats_seq=feats_for_policy.unsqueeze(1),
                baseline_feats_seq=baseline_feats_seq
            )

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

            intrinsic_rewards, intrinsic_stats = self._compute_intrinsic_rewards(feats_s_seq, padded_actions_seq, mask_btna)
            self._log_intrinsic_reward_stats(logs_acc, intrinsic_stats)
            padded_returns_seq = padded_returns_seq + intrinsic_rewards.mean(dim=-1, keepdim=True)

            if self.enable_popart:
                self._update_popart_stats(padded_returns_seq, attention_mask_batch.unsqueeze(-1), mask_btna.unsqueeze(-1))
                self._log_popart_stats(logs_acc)

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
                emb_s_mb, _ = self.embedding_net.get_base_embedding(mb_states_dict)
                feats_s_mb = self._apply_memory_seq(emb_s_mb, mb_init_h)

                # Generate the counterfactual baseline embeddings for the mini-batch
                baseline_feats_s_mb = self.embedding_net.get_baseline_embeddings(
                    common_emb=feats_s_mb,
                    actions=mb_actions
                )

                # --- PPO Core Loss Calculation ---
                new_lp_mb, ent_mb = self._recompute_log_probs_from_features(feats_s_mb, mb_actions)
                # Pass both sets of features to the corrected critic
                # If using IQN (not FQF), sample taus explicitly to align loss weighting with forward
                taus_bt = None
                if self.enable_distributional and not self.use_fqf:
                    B_mb, T_mb = feats_s_mb.shape[0], feats_s_mb.shape[1]
                    taus_bt = torch.rand(B_mb * T_mb, self.num_quantiles, 1, device=self.device)

                new_val_norm, new_base_norm, new_val_quantiles = self.shared_critic.get_values_and_baselines(
                    value_feats_seq=feats_s_mb,
                    baseline_feats_seq=baseline_feats_s_mb,
                    taus=taus_bt
                )
                
                pol_loss = self._ppo_clip_loss(new_lp_mb, mb_old_logp.detach(), mb_adv.detach(), self.ppo_clip_range, mb_mask_btna)
                # Robust entropy masking using boolean mask and safe denom
                ent_mask_bool = mb_mask_btna.bool() if mb_mask_btna.dtype != torch.bool else mb_mask_btna
                ent_loss = torch.where(ent_mask_bool, -ent_mb, torch.zeros_like(ent_mb)).sum() / ent_mask_bool.sum().clamp(min=1)
                
                # Value function loss (FQF/IQN or clipped depending on configuration)
                if self.enable_distributional:
                    val_targets = mb_returns if not self.enable_popart else self.value_popart.normalize_targets(mb_returns)
                    # Reshape val_targets from (B, T, 1) to (B*T, 1) to match quantiles shape (B*T, num_quantiles)
                    val_targets_reshaped = val_targets.reshape(-1, val_targets.shape[-1])
                    # Reshape attention mask from (B, T) to (B*T,) to match
                    mask_reshaped = attention_mask_batch[mb_idx].reshape(-1)
                    
                    if self.use_fqf:
                        # FQF: compute quantile and fraction losses separately for two-phase update
                        learned_taus = self.shared_critic.last_learned_taus  # Set during forward pass
                        if learned_taus is not None:
                            q_loss, frac_loss = self._compute_fqf_losses(
                                new_val_quantiles, learned_taus, val_targets_reshaped, mask_reshaped
                            )
                            # Phase-1 uses only quantile loss; fraction loss handled in phase-2
                            val_loss = q_loss
                        else:
                            # Fallback to IQN if no learned taus available
                            q_loss = self._compute_iqn_quantile_loss(new_val_quantiles, val_targets_reshaped, mask_reshaped, taus_bt.squeeze(-1) if taus_bt is not None else None)
                            frac_loss = torch.tensor(0.0, device=self.device)
                            val_loss = q_loss
                        # Entropy regularizer + fraction diagnostics
                        with torch.no_grad():
                            val_feats_agg_ng = getattr(self.shared_critic, 'last_value_feats_agg', None)
                            if val_feats_agg_ng is not None:
                                if val_feats_agg_ng.device != self.device:
                                    val_feats_agg_ng = val_feats_agg_ng.to(self.device, non_blocking=True)
                                T = getattr(self.shared_critic.value_iqn_net, 'softmax_temperature', 1.0)
                                logits_ng = self.shared_critic.value_iqn_net.fraction_net(val_feats_agg_ng)
                                probs_ng = torch.softmax(logits_ng / max(T, 1e-8), dim=1)
                                ent_ng = -(probs_ng * (probs_ng.clamp(min=1e-8).log())).sum(dim=1)
                                valid_mask_float = mask_reshaped.float()
                                denom = valid_mask_float.sum().clamp(min=1e-8)
                                logs_acc["fraction/entropy"].append((ent_ng * valid_mask_float).sum().item() / denom.item())
                                logs_acc["fraction/prob_max"].append((probs_ng.max(dim=1).values * valid_mask_float).sum().item() / denom.item())
                                logs_acc["fraction/prob_min"].append((probs_ng.min(dim=1).values * valid_mask_float).sum().item() / denom.item())
                                top1 = torch.topk(probs_ng, k=1, dim=1).values.squeeze(-1)
                                logs_acc["fraction/top1_mass"].append((top1 * valid_mask_float).sum().item() / denom.item())
                        # Note: fraction entropy regularizer is applied in phase-2 only now
                    else:
                        # Standard IQN Quantile Regression Loss
                        val_loss = self._compute_iqn_quantile_loss(new_val_quantiles, val_targets_reshaped, mask_reshaped, taus_bt.squeeze(-1) if taus_bt is not None else None)
                else:
                    # Standard clipped value loss when IQN is disabled
                    val_targets = mb_returns if not self.enable_popart else self.value_popart.normalize_targets(mb_returns)
                    val_loss = self._clipped_value_loss(mb_old_val_norm.detach(), new_val_norm, val_targets.detach(), self.value_clip_range, attention_mask_batch[mb_idx].unsqueeze(-1))
                
                # Scalar baseline loss (unchanged)  
                base_targets = mb_returns.unsqueeze(2).expand_as(new_base_norm)
                if self.enable_popart: base_targets = self.baseline_popart.normalize_targets(base_targets)
                base_loss = self._clipped_value_loss(mb_old_base_norm.detach(), new_base_norm, base_targets.detach(), self.value_clip_range, mb_mask_btna.unsqueeze(-1))
                
                # --- Two-phase PPO step for FQF to isolate fraction updates ---
                if self.use_fqf and "value_quantile" in self.optimizers and "value_fraction" in self.optimizers:
                    # Phase 1: policy + trunk + quantile + baseline
                    ppo_opts_phase1 = [
                        self.optimizers["trunk"], self.optimizers["policy"],
                        self.optimizers["value_quantile"], self.optimizers["baseline_path"]
                    ]
                    for opt in ppo_opts_phase1: opt.zero_grad()
                    ppo_total_loss_phase1 = pol_loss + self.entropy_coeff * ent_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss
                    ppo_total_loss_phase1.backward()
                    total_grad_norm = self._clip_and_step_ppo_optimizers(ppo_opts_phase1)
                    # Capture trunk grad stats right after phase-1 update so that
                    # if we do not update trunk in phase-2, we can still report
                    # meaningful trunk gradients for this minibatch.
                    trunk_params_list = list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])
                    trunk_grad_phase1 = _get_avg_grad_mag(trunk_params_list)
                    trunk_grad_l2_phase1 = _get_l2_grad_norm(trunk_params_list)

                    # Phase 2: fraction net only
                    # Prepare grads for fraction phase
                    self.optimizers["value_fraction"].zero_grad()
                    if self.fqf_fraction_update_trunk and "trunk" in self.optimizers:
                        self.optimizers["trunk"].zero_grad()
                    # Compute fresh fraction loss and apply optional entropy regularizer
                    current_quantiles_detached = new_val_quantiles.detach()
                    # Build features for phase-2: recompute trunk if updating it, else detach to cut trunk graph
                    if self.fqf_fraction_update_trunk:
                        emb_s_mb_p2, _ = self.embedding_net.get_base_embedding(mb_states_dict)
                        feats_s_mb_p2 = self._apply_memory_seq(emb_s_mb_p2, mb_init_h)
                    else:
                        feats_s_mb_p2 = feats_s_mb.detach()
                    frac_loss_p2, logits_p2, probs_p2 = self._compute_fqf_fraction_loss_phase2(
                        feats_s_mb_p2, current_quantiles_detached, mask_reshaped
                    )
                    eff_coeff = float(getattr(self, 'fraction_entropy_coeff', 0.0))
                    if self.fraction_entropy_warmup_iters and (self.update_step < self.fraction_entropy_warmup_iters):
                        eff_coeff = 0.0
                    if eff_coeff != 0.0:
                        ent = -(probs_p2 * (probs_p2.clamp(min=1e-8).log())).sum(dim=1)
                        ent_mean = (ent * mask_reshaped.float()).sum() / mask_reshaped.float().sum().clamp(min=1e-8)
                        if self.fraction_entropy_cap is not None:
                            try:
                                cap_val = float(self.fraction_entropy_cap)
                                ent_mean = torch.clamp(ent_mean, max=cap_val)
                            except Exception:
                                pass
                        frac_loss_p2 = frac_loss_p2 - eff_coeff * ent_mean

                    (self.fraction_loss_coeff * frac_loss_p2).backward()
                    self.optimizers["value_fraction"].step()
                    if self.fqf_fraction_update_trunk and "trunk" in self.optimizers:
                        # Step trunk with fraction gradients (optional co-adaptation)
                        self.optimizers["trunk"].step()
                else:
                    # Single-phase (IQN or scalar value)
                    ppo_opts = [self.optimizers["trunk"], self.optimizers["policy"], self.optimizers["value_path"], self.optimizers["baseline_path"]]
                    for opt in ppo_opts: opt.zero_grad()
                    ppo_total_loss = pol_loss + self.entropy_coeff * ent_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss
                    ppo_total_loss.backward()
                    total_grad_norm = self._clip_and_step_ppo_optimizers(ppo_opts)
                    trunk_grad_phase1 = None
                    trunk_grad_l2_phase1 = None
                
                # --- Disagreement Ensemble Update Step ---
                if self.disagreement_cfg:
                    mb_next_states_dict = {"central": {k: v[mb_idx] for k,v in padded_next_states_dict_seq.get("central", {}).items() if v is not None}, "agent": padded_next_states_dict_seq.get("agent")[mb_idx] if "agent" in padded_next_states_dict_seq else None}
                    emb_ns_mb, _ = self.embedding_net.get_base_embedding(mb_next_states_dict)
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
                    total_grad_norm, new_val_norm, new_base_norm, new_val_quantiles if self.enable_distributional else None,
                    # When trunk is not co-adapted in phase-2, reuse phase-1 trunk grads
                    trunk_grad_override=(trunk_grad_phase1 if (self.use_fqf and not self.fqf_fraction_update_trunk) else None),
                    trunk_grad_l2_override=(trunk_grad_l2_phase1 if (self.use_fqf and not self.fqf_fraction_update_trunk) else None)
                )
                # Step schedulers per minibatch (LR + scalar schedulers)
                self._step_schedulers()
                # Advance minibatch update counter
                self.update_step += 1
        
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
            "grad_norm/value_path": [], "grad_norm/baseline_path": [],
            "grad_norm/value_quantile_net": [], "grad_norm/value_fraction_net": [], "grad_norm/value_attention": [],
            "fqf/min_tau_diff": [], "fqf/mean_tau_diff": [], "fqf/max_tau_diff": [],
            "fraction/entropy": [], "fraction/prob_max": [], "fraction/prob_min": [], "fraction/top1_mass": []
        }
        # L2 gradient logs live under the 'grads/' namespace
        logs_acc.update({
            "grads/trunk_l2": [], "grads/policy_l2": [], "grads/value_attention_l2": [],
            "grads/value_path_l2": [], "grads/value_quantile_net_l2": [], "grads/value_fraction_net_l2": [],
            "grads/baseline_path_l2": []
        })
        if self.rnd_cfg:
            logs_acc.update({"loss/rnd": [], "reward/rnd_raw_mean": [], "reward/rnd_raw_std": [], "reward/rnd_norm_mean": [], "reward/rnd_norm_std": [], "grad_norm/rnd_predictor": [], "grads/rnd_predictor_l2": []})
        if self.disagreement_cfg:
            logs_acc.update({"loss/disagreement": [], "reward/disagreement_raw_mean": [], "reward/disagreement_raw_std": [], "reward/disagreement_norm_mean": [], "reward/disagreement_norm_std": [], "grad_norm/disagreement_ensemble": [], "grads/disagreement_ensemble_l2": []})

        # Add distributional RL logs
        logs_acc.update({
            "loss/iqn_value": [],
            "loss/fqf_value": [],
            "mean/value_quantiles": [], "std/value_quantiles": []
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

    def _setup_popart(self, a_cfg):
        beta, eps = a_cfg.get("popart_beta", 0.999), a_cfg.get("popart_epsilon", 1e-5)
        
        if self.enable_distributional:
            # PopArt with IQN is complex since IQN outputs distributions, not scalars
            print("Warning: PopArt normalization with IQN is not fully implemented in the clean approach.")
            print("Consider using standard reward normalization instead.")
            self.value_popart = None  # IQN doesn't need PopArt normalization
        else:
            # Standard PopArt for scalar value function
            self.value_popart = PopArtNormalizer(self.shared_critic.value_head.output_layer, beta, eps, self.device)
        
        # Baseline PopArt (always scalar)
        self.baseline_popart = PopArtNormalizer(self.shared_critic.baseline_head.output_layer, beta, eps, self.device)
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
        lr_conf = {k: a_cfg.get(k, self.lr_base) for k in [
            "lr_trunk", "lr_policy", "lr_value_path", "lr_baseline_path",
            "lr_value_iqn", "lr_baseline_iqn", "lr_rnd_predictor", "lr_disagreement",
            "lr_value_quantile", "lr_value_fraction"
        ]}
        beta_conf = a_cfg.get("optimizer_betas", {}); wd_conf = a_cfg.get("weight_decay", {})
        default_betas = tuple(beta_conf.get("default", (0.9, 0.999))); default_wd = wd_conf.get("default", 0.0)
        trunk_params = list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])
        self.optimizers["trunk"] = optim.AdamW(trunk_params, lr=lr_conf["lr_trunk"], betas=tuple(beta_conf.get("trunk", default_betas)), weight_decay=wd_conf.get("trunk", default_wd))
        self.optimizers["policy"] = optim.AdamW(self.policy_net.parameters(), lr=lr_conf["lr_policy"], betas=tuple(beta_conf.get("policy", default_betas)), weight_decay=wd_conf.get("policy", default_wd))
        
        # Value path
        if self.shared_critic.value_iqn_net is not None:
            if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'fraction_net'):
                # Split optimizers: quantile+attention vs fraction
                quantile_params = list(self.shared_critic.value_iqn_net.quantile_net.parameters()) + \
                                  list(self.shared_critic.value_attention.parameters())
                fraction_params = list(self.shared_critic.value_iqn_net.fraction_net.parameters())
                base_vlr = a_cfg.get("lr_value_path", self.lr_base)
                lr_vq = a_cfg.get("lr_value_quantile", base_vlr)
                lr_vf = a_cfg.get("lr_value_fraction", base_vlr * 2.0)
                self.optimizers["value_quantile"] = optim.AdamW(quantile_params, lr=lr_vq, betas=tuple(beta_conf.get("value_quantile", default_betas)), weight_decay=wd_conf.get("value_quantile", default_wd))
                self.optimizers["value_fraction"] = optim.AdamW(fraction_params, lr=lr_vf, betas=tuple(beta_conf.get("value_fraction", default_betas)), weight_decay=wd_conf.get("value_fraction", default_wd))
            else:
                # Single optimizer for IQN value path (quantile + attention)
                value_path_params = (list(self.shared_critic.value_iqn_net.parameters()) +
                                   list(self.shared_critic.value_attention.parameters()))
                self.optimizers["value_path"] = optim.AdamW(value_path_params, lr=lr_conf["lr_value_path"], betas=tuple(beta_conf.get("value", default_betas)), weight_decay=wd_conf.get("value", default_wd))
        else:
            # Scalar value path (no distributional network): value_head + attention
            value_scalar_params = list(self.shared_critic.value_head.parameters()) + \
                                  list(self.shared_critic.value_attention.parameters())
            self.optimizers["value_path"] = optim.AdamW(value_scalar_params, lr=lr_conf["lr_value_path"], betas=tuple(beta_conf.get("value", default_betas)), weight_decay=wd_conf.get("value", default_wd))
        
        # Baseline path: scalar baseline + attention
        baseline_path_params = (list(self.shared_critic.baseline_head.parameters()) + 
                              list(self.shared_critic.baseline_attention.parameters()))
        self.optimizers["baseline_path"] = optim.AdamW(baseline_path_params, lr=lr_conf["lr_baseline_path"], betas=tuple(beta_conf.get("baseline", default_betas)), weight_decay=wd_conf.get("baseline", default_wd))
        if self.rnd_cfg: self.optimizers["rnd_predictor"] = optim.AdamW(self.rnd_predictor_network.parameters(), lr=lr_conf["lr_rnd_predictor"], betas=tuple(beta_conf.get("rnd", default_betas)), weight_decay=wd_conf.get("rnd", 0.0))
        if self.disagreement_cfg: self.optimizers["disagreement_ensemble"] = optim.AdamW(self.dynamics_ensemble.parameters(), lr=lr_conf["lr_disagreement"], betas=tuple(beta_conf.get("disagreement", default_betas)), weight_decay=wd_conf.get("disagreement", 0.0))

    def _setup_schedulers(self, a_cfg):
        sched_cfg = a_cfg.get("schedulers", {})
        def_lin_sched = {"start_factor": 1.0, "end_factor": 1.0, "total_iters": 1}

        self.schedulers = {}
        self._initial_scheduler_states = {}

        for name, opt in self.optimizers.items():
            key_primary = f"lr_{name}"
            lr_conf = sched_cfg.get(key_primary)
            if lr_conf is None and name == "disagreement_ensemble":
                lr_conf = sched_cfg.get("lr_disagreement")
            if lr_conf is None:
                lr_conf = def_lin_sched
            self.schedulers[name] = LinearLRDecay(opt, **lr_conf)
            self._record_initial_scheduler_state(name)

        self.schedulers["entropy_coeff"] = LinearValueScheduler(**sched_cfg.get(
            "entropy_coeff",
            {"start_value": self.entropy_coeff, "end_value": self.entropy_coeff, "total_iters": 1}
        ))
        self._record_initial_scheduler_state("entropy_coeff")

        self.schedulers["policy_clip"] = LinearValueScheduler(**sched_cfg.get(
            "policy_clip",
            {"start_value": self.ppo_clip_range, "end_value": self.ppo_clip_range, "total_iters": 1}
        ))
        self._record_initial_scheduler_state("policy_clip")

        self.schedulers["value_clip"] = LinearValueScheduler(**sched_cfg.get(
            "value_clip",
            {"start_value": self.value_clip_range, "end_value": self.value_clip_range, "total_iters": 1}
        ))
        self._record_initial_scheduler_state("value_clip")

        self.schedulers["max_grad_norm"] = LinearValueScheduler(**sched_cfg.get(
            "max_grad_norm",
            {"start_value": self.max_grad_norm, "end_value": self.max_grad_norm, "total_iters": 1}
        ))
        self._record_initial_scheduler_state("max_grad_norm")

        self.schedulers["fraction_entropy_coeff"] = LinearValueScheduler(**sched_cfg.get(
            "fraction_entropy_coeff",
            {"start_value": getattr(self, 'fraction_entropy_coeff', 0.0), "end_value": getattr(self, 'fraction_entropy_coeff', 0.0), "total_iters": 1}
        ))
        self._record_initial_scheduler_state("fraction_entropy_coeff")

        self.schedulers["fraction_loss_coeff"] = LinearValueScheduler(**sched_cfg.get(
            "fraction_loss_coeff",
            {"start_value": getattr(self, 'fraction_loss_coeff', 0.0), "end_value": getattr(self, 'fraction_loss_coeff', 0.0), "total_iters": 1}
        ))
        self._record_initial_scheduler_state("fraction_loss_coeff")

        init_temp = 1.0
        if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'softmax_temperature'):
            init_temp = float(self.shared_critic.value_iqn_net.softmax_temperature)
        self.schedulers["fqf_temperature"] = LinearValueScheduler(**sched_cfg.get(
            "fqf_temperature",
            {"start_value": init_temp, "end_value": init_temp, "total_iters": 1}
        ))
        self._record_initial_scheduler_state("fqf_temperature")

        init_alpha = 0.0
        if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'prior_blend_alpha'):
            init_alpha = float(self.shared_critic.value_iqn_net.prior_blend_alpha)
        self.schedulers["fqf_prior_blend_alpha"] = LinearValueScheduler(**sched_cfg.get(
            "fqf_prior_blend_alpha",
            {"start_value": init_alpha, "end_value": init_alpha, "total_iters": 1}
        ))
        self._record_initial_scheduler_state("fqf_prior_blend_alpha")
        self._sync_schedulable_scalars()

    def _record_initial_scheduler_state(self, name: str) -> None:
        if not hasattr(self, "_initial_scheduler_states"):
            self._initial_scheduler_states = {}
        sched = self.schedulers.get(name)
        snapshot = None
        if sched is not None and hasattr(sched, "state_dict"):
            try:
                snapshot = copy.deepcopy(sched.state_dict())
            except Exception:
                snapshot = None
        self._initial_scheduler_states[name] = snapshot

    def _sync_schedulable_scalars(self) -> None:
        try:
            if hasattr(self, "schedulers") and isinstance(self.schedulers, dict):
                if "entropy_coeff" in self.schedulers and hasattr(self.schedulers["entropy_coeff"], "current_value"):
                    self.entropy_coeff = float(self.schedulers["entropy_coeff"].current_value())
                if "policy_clip" in self.schedulers and hasattr(self.schedulers["policy_clip"], "current_value"):
                    self.ppo_clip_range = float(self.schedulers["policy_clip"].current_value())
                if "value_clip" in self.schedulers and hasattr(self.schedulers["value_clip"], "current_value"):
                    self.value_clip_range = float(self.schedulers["value_clip"].current_value())
                if "max_grad_norm" in self.schedulers and hasattr(self.schedulers["max_grad_norm"], "current_value"):
                    self.max_grad_norm = float(self.schedulers["max_grad_norm"].current_value())
                if "fraction_entropy_coeff" in self.schedulers and hasattr(self.schedulers["fraction_entropy_coeff"], "current_value"):
                    self.fraction_entropy_coeff = float(self.schedulers["fraction_entropy_coeff"].current_value())
                if "fraction_loss_coeff" in self.schedulers and hasattr(self.schedulers["fraction_loss_coeff"], "current_value"):
                    self.fraction_loss_coeff = float(self.schedulers["fraction_loss_coeff"].current_value())
                if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'softmax_temperature') and \
                   "fqf_temperature" in self.schedulers and hasattr(self.schedulers["fqf_temperature"], "current_value"):
                    self.shared_critic.value_iqn_net.softmax_temperature = float(self.schedulers["fqf_temperature"].current_value())
                if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'prior_blend_alpha') and \
                   "fqf_prior_blend_alpha" in self.schedulers and hasattr(self.schedulers["fqf_prior_blend_alpha"], "current_value"):
                    self.shared_critic.value_iqn_net.prior_blend_alpha = float(self.schedulers["fqf_prior_blend_alpha"].current_value())
        except Exception:
            pass

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
    
    def _compute_gae_with_padding(
        self,
        r: torch.Tensor,
        v: torch.Tensor,
        v_next: torch.Tensor,
        d: torch.Tensor,
        tr: torch.Tensor,
        mask_bt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        r : torch.Tensor
            Reward tensor with padding.
        v : torch.Tensor
            Current value estimates.
        v_next : torch.Tensor
            Next-step value estimates.
        d : torch.Tensor
            Done flags for episode termination.
        tr : torch.Tensor
            Truncation flags. When ``tr`` is ``1`` the GAE is reset and
            bootstrapping does not cross this boundary.
        mask_bt : torch.Tensor
            Mask indicating valid (non-padded) steps.
        """
        B, T, _ = r.shape
        returns = torch.zeros_like(r)
        gae = torch.zeros(B, 1, device=self.device)
        mask_bt1 = mask_bt.unsqueeze(-1)

        for t in reversed(range(T)):
            valid_mask_step = mask_bt1[:, t]
            if t == T - 1:
                valid_mask_next_step = torch.ones_like(valid_mask_step)
            else:
                valid_mask_next_step = mask_bt1[:, t + 1]

            not_done = 1.0 - d[:, t]
            not_trunc = 1.0 - tr[:, t]
            bootstrap_mask = not_done * not_trunc * valid_mask_next_step

            delta = r[:, t] + self.gamma * v_next[:, t] * bootstrap_mask - v[:, t]

            gae = delta + self.gamma * self.lmbda * bootstrap_mask * gae

            current_return = gae + v[:, t]

            returns[:, t] = current_return * valid_mask_step

            gae = gae * valid_mask_step * not_trunc
            
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
        if truncs.numel() > 1:
            v_next[:, :-1] = v_next[:, :-1] * (1.0 - truncs[:, 1:])

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
        # For bootstrapped rollouts we propagate across truncation boundaries,
        # so provide a zero tensor for ``tr``.
        zeros_tr = torch.zeros_like(truncs)
        return self._compute_gae_with_padding(rewards, v_denorm, v_next, dones, zeros_tr, mask_bt)

    def _ppo_clip_loss(self, new_lp, old_lp, adv, clip_range, mask):
        """Calculates the PPO clipped surrogate objective loss."""
        ratio = torch.exp((new_lp - old_lp).clamp(-10, 10))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        loss_unreduced = -torch.min(surr1, surr2)
        # Apply boolean mask with where to avoid NaN propagation from masked entries
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        loss_masked = torch.where(mask_bool, loss_unreduced, torch.zeros_like(loss_unreduced))
        denom = mask_bool.sum().clamp(min=1)
        return loss_masked.sum() / denom

    def _clipped_value_loss(self, old_v, new_v, target_v, clip_range, mask):
        """Calculates the clipped value function loss."""
        v_clipped = old_v + (new_v - old_v).clamp(-clip_range, clip_range)
        # vf_loss1 = (new_v - target_v).pow(2)
        # vf_loss2 = (v_clipped - target_v).pow(2)
        # Compute elementwise losses so masking is effective on padded steps.
        vf_loss1 = F.smooth_l1_loss(new_v, target_v, reduction='none')
        vf_loss2 = F.smooth_l1_loss(v_clipped, target_v, reduction='none')
        loss_unreduced = torch.max(vf_loss1, vf_loss2)
        # Apply boolean mask with where to avoid NaN propagation from masked entries
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        loss_masked = torch.where(mask_bool, loss_unreduced, torch.zeros_like(loss_unreduced))
        denom = mask_bool.sum().clamp(min=1)
        return loss_masked.sum() / denom

    # --- Auxiliary Module Helper Methods ---
    def _huber_loss(self, pred, target, delta=1.0):
        diff = pred - target
        return torch.where(
            diff.abs() <= delta,
            0.5 * diff.pow(2),
            delta * (diff.abs() - 0.5 * delta)
        )

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

    def _compute_iqn_quantile_loss(self, current_quantiles: torch.Tensor, 
                                  target_values: torch.Tensor, mask: torch.Tensor,
                                  taus_used: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute IQN quantile regression loss for on-policy value function training.
        
        Args:
            current_quantiles: IQN quantile predictions, shape (B*T, num_quantiles)
            target_values: Target values (returns), shape (B*T, 1) or (B*T,)
            mask: Valid timestep mask, shape (B*T,)
            
        Returns:
            Scalar loss value
        """
        B_T, num_quantiles = current_quantiles.shape
        
        # Expand target values to match quantiles: (B*T, 1) -> (B*T, num_quantiles)  
        if target_values.dim() == 1:
            target_values = target_values.unsqueeze(-1)
        target_expanded = target_values.expand(-1, num_quantiles)
        
        # Compute TD errors: target - prediction
        td_errors = target_expanded - current_quantiles  # (B*T, num_quantiles)
        
        # Generate quantile fractions (τ) for current quantiles
        # Use uniform spacing as approximation: τ_i = (i + 0.5) / N
        taus = torch.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, 
                             num_quantiles, device=current_quantiles.device)
        taus = taus.unsqueeze(0).expand(B_T, -1)  # (B*T, num_quantiles)
        # If explicit taus were used to generate current_quantiles, prefer them for correct weighting
        if taus_used is not None:
            taus = taus_used
        
        # Asymmetric Huber quantile loss
        abs_errors = torch.abs(td_errors)
        huber_loss = torch.where(
            abs_errors <= self.iqn_kappa,
            0.5 * td_errors.pow(2),
            self.iqn_kappa * (abs_errors - 0.5 * self.iqn_kappa)
        )
        
        # Quantile regression weighting: |τ - I{δ < 0}|
        indicators = (td_errors < 0).float()
        quantile_weights = torch.abs(taus - indicators)
        
        # Weighted loss
        weighted_loss = quantile_weights * huber_loss  # (B*T, num_quantiles)
        
        # Apply mask and average
        mask_expanded = mask.unsqueeze(-1).expand_as(weighted_loss)
        masked_loss = weighted_loss * mask_expanded
        
        quantile_loss = masked_loss.sum() / mask_expanded.sum().clamp(min=1e-8)
        # Normalize by number of quantiles to keep scale comparable
        return quantile_loss / max(float(num_quantiles), 1.0)
    
    def _compute_fqf_losses(self, current_quantiles: torch.Tensor,
                             learned_taus: torch.Tensor,
                             target_values: torch.Tensor,
                             mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paper-aligned FQF objective split into two components:
        - quantile Huber loss at tau_hat midpoints (for value/quantile net)
        - fraction loss using gradient-of-t surrogate (for fraction net)

        Returns (quantile_loss, fraction_loss)
        """
        B_T, num_quantiles = current_quantiles.shape  # N quantiles

        if learned_taus is not None and learned_taus.device != current_quantiles.device:
            learned_taus = learned_taus.to(current_quantiles.device, non_blocking=True)

        # Expand targets to match quantiles
        if target_values.dim() == 1:
            target_values = target_values.unsqueeze(-1)
        target_expanded = target_values.expand(-1, num_quantiles)

        # TD errors and Huber
        td_errors = target_expanded - current_quantiles
        abs_errors = torch.abs(td_errors)
        huber_loss = torch.where(
            abs_errors <= self.iqn_kappa,
            0.5 * td_errors.pow(2),
            self.iqn_kappa * (abs_errors - 0.5 * self.iqn_kappa)
        )

        # Reconstruct t^ from learned internal boundaries: [0, t_1..t_{N-1}, 1]
        learned_taus_2d = learned_taus.squeeze(-1) if learned_taus.dim() == 3 else learned_taus  # (B*T, N-1)
        zeros = torch.zeros(B_T, 1, device=current_quantiles.device, dtype=current_quantiles.dtype)
        ones = torch.ones(B_T, 1, device=current_quantiles.device, dtype=current_quantiles.dtype)
        tau_boundaries = torch.cat([zeros, learned_taus_2d, ones], dim=1)  # (B*T, N+1)
        interval_widths = tau_boundaries[:, 1:] - tau_boundaries[:, :-1]  # (B*T, N)
        tau_prev = tau_boundaries[:, :-1]  # (B*T, N)
        tau_hat = tau_prev + 0.5 * interval_widths  # (B*T, N)

        # Quantile regression weighting at t^
        indicators = (td_errors < 0).float()
        quantile_weights = torch.abs(tau_hat - indicators)

        # Weighted quantile loss with mask normalization
        qloss = (quantile_weights * huber_loss)
        mask_expanded = mask.unsqueeze(-1).expand_as(qloss)
        quantile_loss = (qloss * mask_expanded).sum() / mask_expanded.sum().clamp(min=1e-8)
        # Normalize by number of quantiles to keep value loss scale comparable
        quantile_loss = quantile_loss / max(float(num_quantiles), 1.0)

        # Fraction loss: need quantiles at boundaries (internal only)
        # Compute WITH gradients for fraction loss - critical for FQF optimization!
        val_feats_agg = getattr(self.shared_critic, 'last_value_feats_agg', None)
        if val_feats_agg is None:
            return quantile_loss, torch.tensor(0.0, device=self.device)  # fallback
        if val_feats_agg.device != current_quantiles.device:
            val_feats_agg = val_feats_agg.to(current_quantiles.device, non_blocking=True)
        sa_quantiles = self.shared_critic.value_iqn_net.quantile_net(
            val_feats_agg, learned_taus_2d.unsqueeze(-1))  # Keep gradients for tau optimization!
        sa_quantile_hats = current_quantiles.detach()  # Only detach the midpoint quantiles

        values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
        signs_1 = sa_quantiles > torch.cat([sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
        values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
        signs_2 = sa_quantiles < torch.cat([sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
        gradient_of_taus = (torch.where(signs_1, values_1, -values_1)
                            + torch.where(signs_2, values_2, -values_2))  # (B*T, N-1)

        mask_tau = mask.unsqueeze(-1).expand_as(gradient_of_taus)
        fraction_loss = ((gradient_of_taus * learned_taus_2d) * mask_tau).sum() / mask_tau.sum().clamp(min=1e-8)

        return quantile_loss, fraction_loss

    def _compute_fqf_fraction_loss_phase2(self,
                                          feats_seq: torch.Tensor,
                                          current_quantiles_detached: torch.Tensor,
                                          mask_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute fraction-only loss on a fresh graph to avoid in-place/version issues
        after phase-1 updates. Returns (fraction_loss, logits, probs).
        """
        B, T, NA, F = feats_seq.shape
        # Compute aggregated value features (B*T, H)
        # If we want the fraction update to also update the trunk, we must allow
        # gradients to flow through the attention/trunk path. Otherwise, keep it
        # under no_grad to avoid unnecessary memory use.
        if getattr(self, 'fqf_fraction_update_trunk', False):
            val_attn_out, _ = self.shared_critic.value_attention(feats_seq.reshape(B * T, NA, F))
            val_feats_agg = val_attn_out.mean(dim=1)
        else:
            with torch.no_grad():
                val_attn_out, _ = self.shared_critic.value_attention(feats_seq.reshape(B * T, NA, F))
                val_feats_agg = val_attn_out.mean(dim=1)

        # Fraction proposal (interval probs) and internal boundaries
        fqf_net = self.shared_critic.value_iqn_net
        logits = fqf_net.fraction_net(val_feats_agg)
        Ttemp = float(getattr(fqf_net, 'softmax_temperature', 1.0))
        probs = torch.softmax(logits / max(Ttemp, 1e-8), dim=1)
        tau_cum = torch.cumsum(probs, dim=1)
        taus_internal = tau_cum[:, :-1] if probs.shape[1] > 1 else tau_cum[:, :0]

        # Quantiles at boundaries and gradient surrogate
        sa_quantiles = fqf_net.quantile_net(val_feats_agg, taus_internal.unsqueeze(-1))  # (B*T, N-1)
        sa_quantile_hats = current_quantiles_detached  # (B*T, N)
        values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
        signs_1 = sa_quantiles > torch.cat([sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
        values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
        signs_2 = sa_quantiles < torch.cat([sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
        gradient_of_taus = (torch.where(signs_1, values_1, -values_1)
                            + torch.where(signs_2, values_2, -values_2))

        mask_tau = mask_flat.unsqueeze(-1).expand_as(gradient_of_taus)
        fraction_loss = ((gradient_of_taus * taus_internal) * mask_tau).sum() / mask_tau.sum().clamp(min=1e-8)
        return fraction_loss, logits, probs

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
    
    def _get_critic_outputs(self, feats_seq: torch.Tensor, actions_seq: torch.Tensor, B: int, T: int, NA: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Clean helper to get critic outputs using IQN value function and scalar baselines.
        
        Returns:
            values: Scalar values from IQN expectation, shape (B, T, 1)
            baselines: Scalar baselines, shape (B, T, NA, 1)  
            value_quantiles: IQN quantiles for loss computation, shape (B*T, num_quantiles)
        """
        # Use the simplified SharedCritic interface
        values, baselines, value_quantiles = self.shared_critic.get_values_and_baselines(
            feats_seq, actions_seq, taus=None  # Let IQN sample random quantiles
        )
        
        return values, baselines, value_quantiles
    
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
        """Steps all learning rate and parameter schedulers (call per minibatch)."""
        # Step all LR schedulers
        for sched in self.schedulers.values():
            if isinstance(sched, torch.optim.lr_scheduler._LRScheduler):
                sched.step()

        # Step scalar schedulers and assign back
        self.entropy_coeff = self.schedulers["entropy_coeff"].step()
        self.ppo_clip_range = self.schedulers["policy_clip"].step()
        self.value_clip_range = self.schedulers["value_clip"].step()
        self.max_grad_norm = self.schedulers["max_grad_norm"].step()

        # FQF extras
        if "fraction_entropy_coeff" in self.schedulers:
            self.fraction_entropy_coeff = self.schedulers["fraction_entropy_coeff"].step()
        if "fraction_loss_coeff" in self.schedulers:
            self.fraction_loss_coeff = self.schedulers["fraction_loss_coeff"].step()
        if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'softmax_temperature'):
            new_temp = self.schedulers["fqf_temperature"].step()
            self.shared_critic.value_iqn_net.softmax_temperature = float(new_temp)
        if self.use_fqf and hasattr(self.shared_critic.value_iqn_net, 'prior_blend_alpha'):
            new_alpha = self.schedulers["fqf_prior_blend_alpha"].step()
            self.shared_critic.value_iqn_net.prior_blend_alpha = float(new_alpha)

    def _clear_fqf_cache(self) -> None:
        """Release cached tensors used for FQF auxiliary losses/logging."""
        if hasattr(self.shared_critic, "last_value_feats_agg"):
            self.shared_critic.last_value_feats_agg = None
        if hasattr(self.shared_critic, "last_learned_taus"):
            self.shared_critic.last_learned_taus = None

    def _log_minibatch_stats(self, logs_acc, pol_loss, val_loss, base_loss, ent_mb, new_lp_mb, mask_mbna, rnd_loss, disagreement_loss, feats_s_mb, total_grad_norm, new_val_norm=None, new_base_norm=None, new_val_quantiles=None, trunk_grad_override: Optional[float] = None, trunk_grad_l2_override: Optional[float] = None):
        """Accumulates statistics from a single minibatch into the logs dictionary."""
        # Core PPO Losses
        logs_acc["loss/policy"].append(pol_loss.item())
        
        # Log value loss under appropriate category
        if self.enable_distributional:
            if self.use_fqf:
                logs_acc["loss/fqf_value"].append(val_loss.item())
            else:
                logs_acc["loss/iqn_value"].append(val_loss.item())
        else:
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
        
        # Log IQN quantiles and loss (only if IQN is enabled)
        if self.enable_distributional and new_val_quantiles is not None:
            with torch.no_grad():
                logs_acc["mean/value_quantiles"].append(new_val_quantiles.detach().mean().item())
                logs_acc["std/value_quantiles"].append(new_val_quantiles.detach().std().item())

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
        trunk_params_for_log = list(self.embedding_net.parameters()) + (list(self.memory_module.parameters()) if self.enable_memory else [])
        if trunk_grad_override is not None:
            logs_acc["grad_norm/trunk"].append(float(trunk_grad_override))
        else:
            logs_acc["grad_norm/trunk"].append(_get_avg_grad_mag(trunk_params_for_log))
        # L2 versions under 'grads/' prefix
        if trunk_grad_l2_override is not None:
            logs_acc.setdefault("grads/trunk_l2", []).append(float(trunk_grad_l2_override))
        else:
            logs_acc.setdefault("grads/trunk_l2", []).append(_get_l2_grad_norm(trunk_params_for_log))
        # Policy
        policy_params_for_log = list(self.policy_net.parameters())
        logs_acc["grad_norm/policy"].append(_get_avg_grad_mag(policy_params_for_log))
        logs_acc.setdefault("grads/policy_l2", []).append(_get_l2_grad_norm(policy_params_for_log))
        
        # Value path gradient norms (handle scalar vs distributional)
        value_attention_params = list(self.shared_critic.value_attention.parameters())
        logs_acc["grad_norm/value_attention"].append(_get_avg_grad_mag(value_attention_params))
        logs_acc.setdefault("grads/value_attention_l2", []).append(_get_l2_grad_norm(value_attention_params))
        if self.shared_critic.value_iqn_net is not None:
            value_path_params = (list(self.shared_critic.value_iqn_net.parameters()) +
                               list(self.shared_critic.value_attention.parameters()))
            logs_acc["grad_norm/value_path"].append(_get_avg_grad_mag(value_path_params))
            logs_acc.setdefault("grads/value_path_l2", []).append(_get_l2_grad_norm(value_path_params))
            if hasattr(self.shared_critic.value_iqn_net, 'quantile_net'):
                quantile_params = list(self.shared_critic.value_iqn_net.quantile_net.parameters())
                logs_acc["grad_norm/value_quantile_net"].append(_get_avg_grad_mag(quantile_params))
                logs_acc.setdefault("grads/value_quantile_net_l2", []).append(_get_l2_grad_norm(quantile_params))
            if hasattr(self.shared_critic.value_iqn_net, 'fraction_net'):
                fraction_params = list(self.shared_critic.value_iqn_net.fraction_net.parameters())
                logs_acc["grad_norm/value_fraction_net"].append(_get_avg_grad_mag(fraction_params))
                logs_acc.setdefault("grads/value_fraction_net_l2", []).append(_get_l2_grad_norm(fraction_params))
        else:
            # Scalar path: value_head + attention
            value_scalar_params = list(self.shared_critic.value_head.parameters()) + \
                                  list(self.shared_critic.value_attention.parameters())
            logs_acc["grad_norm/value_path"].append(_get_avg_grad_mag(value_scalar_params))
            logs_acc.setdefault("grads/value_path_l2", []).append(_get_l2_grad_norm(value_scalar_params))
        
        baseline_path_params = (list(self.shared_critic.baseline_head.parameters()) + 
                              list(self.shared_critic.baseline_attention.parameters()))
        logs_acc["grad_norm/baseline_path"].append(_get_avg_grad_mag(baseline_path_params))
        logs_acc.setdefault("grads/baseline_path_l2", []).append(_get_l2_grad_norm(baseline_path_params))

        if self.rnd_cfg:
            rnd_params = list(self.rnd_predictor_network.parameters())
            logs_acc["grad_norm/rnd_predictor"].append(_get_avg_grad_mag(rnd_params))
            logs_acc.setdefault("grads/rnd_predictor_l2", []).append(_get_l2_grad_norm(rnd_params))
        if self.disagreement_cfg:
            dis_params = list(self.dynamics_ensemble.parameters())
            logs_acc["grad_norm/disagreement_ensemble"].append(_get_avg_grad_mag(dis_params))
            logs_acc.setdefault("grads/disagreement_ensemble_l2", []).append(_get_l2_grad_norm(dis_params))

        # FQF tau spacing stats
        if self.enable_distributional and self.use_fqf and getattr(self.shared_critic, 'last_learned_taus', None) is not None:
            with torch.no_grad():
                taus_internal = self.shared_critic.last_learned_taus.squeeze(-1)
                B_T = taus_internal.shape[0]
                zeros = torch.zeros(B_T, 1, device=taus_internal.device, dtype=taus_internal.dtype)
                ones = torch.ones(B_T, 1, device=taus_internal.device, dtype=taus_internal.dtype)
                tau_boundaries = torch.cat([zeros, taus_internal, ones], dim=1)
                tau_diffs = tau_boundaries[:, 1:] - tau_boundaries[:, :-1]
                logs_acc["fqf/min_tau_diff"].append(tau_diffs.min().item())
                logs_acc["fqf/mean_tau_diff"].append(tau_diffs.mean().item())
                logs_acc["fqf/max_tau_diff"].append(tau_diffs.max().item())

        self._clear_fqf_cache()

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
        # Log fraction loss coeff when available
        try:
            final_logs["param/fraction_loss_coeff_curr"] = float(getattr(self, 'fraction_loss_coeff', 0.0))
        except Exception:
            pass
        # Optional FQF parameter logs (when enabled)
        if getattr(self, 'use_fqf', False):
            try:
                final_logs.update({
                    "param/fraction_entropy_coeff_curr": float(getattr(self, 'fraction_entropy_coeff', 0.0)),
                })
            except Exception:
                pass
            try:
                if hasattr(self.shared_critic, 'value_iqn_net'):
                    if hasattr(self.shared_critic.value_iqn_net, 'softmax_temperature'):
                        final_logs["param/fqf_temperature_curr"] = float(self.shared_critic.value_iqn_net.softmax_temperature)
                    if hasattr(self.shared_critic.value_iqn_net, 'prior_blend_alpha'):
                        final_logs["param/fqf_prior_blend_alpha_curr"] = float(self.shared_critic.value_iqn_net.prior_blend_alpha)
            except Exception:
                pass
        
        return final_logs
    
    def _log_popart_stats(self, logs_acc: Dict[str, List[float]]):
        """Helper to log the current stats of the PopArt normalizers."""
        if not self.enable_popart: return
        
        logs_acc["popart/value_mu"].append(self.value_popart.mu.item())
        logs_acc["popart/value_sigma"].append(self.value_popart.sigma.item())
        logs_acc["popart/baseline_mu"].append(self.baseline_popart.mu.item())
        logs_acc["popart/baseline_sigma"].append(self.baseline_popart.sigma.item())
        
    # --- Save/Load ---

    def save(self, location: str, include_optimizers: bool = False) -> None:
        """Save model parameters and optionally optimizer states."""
        super().save(location, include_optimizers=include_optimizers)

    def load(
        self,
        location: str,
        load_optimizers: bool = False,
        load_schedulers: bool = True,
        reset_schedulers: bool = False,
    ) -> None:
        """Load model parameters and optionally optimizer/scheduler states."""
        super().load(
            location,
            load_optimizers=load_optimizers,
            load_schedulers=load_schedulers,
            reset_schedulers=reset_schedulers,
        )
        self._sync_schedulable_scalars()
