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
)
from Source.Memory import GRUSequenceMemory

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _bool_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise OR that works on float or bool tensors."""
    return (a > 0.5) | (b > 0.5)

# -----------------------------------------------------------------------------
class MAPOCAAgent(Agent):
    """Multi-Agent PPO / MA-POCA agent with shared embedding + GRU trunk."""

    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__(cfg, device)
        self.device = device
        a_cfg      = cfg["agent"]["params"]
        t_cfg      = cfg["train"]
        env_shape  = cfg["environment"]["shape"]
        env_params = cfg["environment"]["params"]

        # ---------- hyper-parameters ----------
        self.gamma               = a_cfg.get("gamma", 0.99)
        self.lmbda               = a_cfg.get("lambda", 0.95)
        self.entropy_coeff       = a_cfg.get("entropy_coeff", 0.01)
        self.value_loss_coeff    = a_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff = a_cfg.get("baseline_loss_coeff", 0.5)
        self.max_grad_norm       = a_cfg.get("max_grad_norm", 0.5)
        self.normalize_adv       = a_cfg.get("normalize_advantages", True)
        self.ppo_clip_range      = a_cfg.get("ppo_clip_range", 0.2)
        self.value_clip_range    = a_cfg.get("value_clip_range", 0.2)
        self.lr_base             = a_cfg.get("learning_rate", 3e-4)
        self.epochs              = t_cfg.get("epochs", 4)
        self.mini_batch_size     = t_cfg.get("mini_batch_size", 64)
        self.no_grad_bs          = a_cfg.get("no_grad_forward_batch_size", self.mini_batch_size)

        # ---------- environment counts --------
        self.num_agents = env_params.get("MaxAgents", 1)
        if "agent" in env_shape["state"]:
            self.num_agents = env_shape["state"]["agent"].get("max", self.num_agents)

        # ---------- recorder ----------
        self.state_recorder = StateRecorder(cfg.get("StateRecorder")) if cfg.get("StateRecorder") else None

        # ---------- action space ----------
        self._determine_action_space(env_shape["action"])

        # ---------- networks ----------
        emb_cfg         = a_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        self.embedding_net = MultiAgentEmbeddingNetwork(emb_cfg).to(device)
        emb_out         = emb_cfg["cross_attention_feature_extractor"]["embed_dim"]

        self.enable_memory = a_cfg.get("enable_memory", False)
        if self.enable_memory:
            mem_cfg = a_cfg.get(a_cfg.get("memory_type", "gru"), {})
            mem_cfg.setdefault("input_size", emb_out)
            self.memory_hidden = mem_cfg.get("hidden_size", 128)
            self.memory_layers = mem_cfg.get("num_layers", 1)
            self.memory_module = GRUSequenceMemory(**mem_cfg).to(device)
            trunk_dim = self.memory_hidden
        else:
            self.memory_module = None
            self.memory_hidden = self.memory_layers = 0
            trunk_dim = emb_out

        pol_cfg = a_cfg["networks"]["policy_network"].copy()
        pol_cfg.update({"in_features": trunk_dim, "out_features": self.action_dim})
        if self.action_space_type == "continuous":
            self.policy_net = BetaPolicyNetwork(**pol_cfg).to(device)
        else:
            self.policy_net = DiscretePolicyNetwork(**pol_cfg).to(device)

        self.shared_critic = SharedCritic(a_cfg["networks"]["critic_network"]).to(device)

        # ---------- PopArt / reward norm ----------
        self.enable_popart = a_cfg.get("enable_popart", False)
        if self.enable_popart:
            beta, eps = a_cfg.get("popart_beta", 0.999), a_cfg.get("popart_epsilon", 1e-5)
            self.value_popart    = PopArtNormalizer(self.shared_critic.value_head.output_layer,    beta, eps, device).to(device)
            self.baseline_popart = PopArtNormalizer(self.shared_critic.baseline_head.output_layer, beta, eps, device).to(device)
            self.rewards_normalizer = None
        else:
            self.value_popart = self.baseline_popart = None
            rnorm_cfg = a_cfg.get("rewards_normalizer")
            self.rewards_normalizer = RunningMeanStdNormalizer(**rnorm_cfg, device=device) if rnorm_cfg else None

        # ---------- RND ----------
        self.enable_rnd = a_cfg.get("enable_rnd", False)
        if self.enable_rnd:
            rnd_cfg = a_cfg.get("rnd_params", {})
            rnd_in, rnd_out = trunk_dim, rnd_cfg.get("output_size", 128)
            rnd_hid = rnd_cfg.get("hidden_size", 256)
            self.rnd_target_network = RNDTargetNetwork(rnd_in, rnd_out, rnd_hid).to(device)
            self.rnd_target_network.eval()
            for p in self.rnd_target_network.parameters(): p.requires_grad_(False)
            self.rnd_predictor_network = RNDPredictorNetwork(rnd_in, rnd_out, rnd_hid).to(device)
            self.intrinsic_reward_normalizer = RunningMeanStdNormalizer(epsilon=1e-8, device=device)
            self.intrinsic_reward_coeff = rnd_cfg.get("intrinsic_reward_coeff", 0.01)
            self.rnd_update_prop        = rnd_cfg.get("rnd_update_proportion", 0.25)
        else:
            self.rnd_target_network    = self.rnd_predictor_network = None
            self.intrinsic_reward_normalizer = None
            self.intrinsic_reward_coeff      = 0.0
            self.rnd_update_prop             = 0.0

        # ---------- optimisers ----------
        lr_pol = a_cfg.get("lr_policy",    self.lr_base)
        lr_val = a_cfg.get("lr_value",     self.lr_base)
        lr_bas = a_cfg.get("lr_baseline",  self.lr_base)
        lr_rnd = a_cfg.get("lr_rnd_predictor", self.lr_base)

        # read optimizer betas and weight decay from config
        opt_betas = a_cfg.get("optimizer_betas", {})
        wd_cfg    = a_cfg.get("weight_decay", {})
        trunk_betas  = tuple(opt_betas.get("trunk",    (0.9, 0.98)))
        policy_betas = tuple(opt_betas.get("policy",   (0.9, 0.98)))
        value_betas  = tuple(opt_betas.get("value",    (0.9, 0.95)))
        base_betas   = tuple(opt_betas.get("baseline", (0.9, 0.95)))
        rnd_betas    = tuple(opt_betas.get("rnd",      (0.9, 0.98)))
        trunk_wd     = wd_cfg.get("trunk",    1e-5)
        policy_wd    = wd_cfg.get("policy",   1e-5)
        value_wd     = wd_cfg.get("value",    1e-4)
        base_wd      = wd_cfg.get("baseline", 1e-4)
        rnd_wd       = wd_cfg.get("rnd",      0.0)

        trunk_params = list(self.embedding_net.parameters())
        if self.memory_module:
            trunk_params += list(self.memory_module.parameters())

        # 1) trunk optimizer
        self.trunk_opt  = optim.Adam(trunk_params, lr=lr_pol, betas=trunk_betas, eps=1e-7, weight_decay=trunk_wd)
        # 2) policy optimizer
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr_pol, betas=policy_betas, eps=1e-7, weight_decay=policy_wd)
        # 3) value head optimizer
        self.value_opt  = optim.Adam(
            list(self.shared_critic.value_head.parameters()) + list(self.shared_critic.value_attention.parameters()),
            lr=lr_val * 0.5, betas=value_betas, eps=1e-6, weight_decay=value_wd
        )
        # 4) baseline head optimizer
        self.base_opt   = optim.Adam(
            list(self.shared_critic.baseline_head.parameters()) + list(self.shared_critic.baseline_attention.parameters()),
            lr=lr_bas, betas=base_betas, eps=1e-6, weight_decay=base_wd
        )
        # 5) rnd predictor optimizer
        self.rnd_opt    = optim.Adam(
            self.rnd_predictor_network.parameters(),
            lr=lr_rnd * 2.0, betas=rnd_betas, eps=1e-7, weight_decay=rnd_wd
        ) if self.enable_rnd else None

        # ---------- schedulers ----------
        sched_cfg = a_cfg.get("schedulers", {})
        self.policy_lr_sched = LinearLR(self.policy_opt, **sched_cfg.get("lr", {})) if "lr" in sched_cfg else None
        self.entropy_sched   = LinearValueScheduler(**sched_cfg.get("entropy_coeff", {})) if "entropy_coeff" in sched_cfg else None
        if self.entropy_sched:
            self.entropy_coeff = self.entropy_sched.start_value
        self.clip_sched      = LinearValueScheduler(**sched_cfg.get("policy_clip", {})) if "policy_clip" in sched_cfg else None
        self.val_clip_sched  = LinearValueScheduler(**sched_cfg.get("value_clip", {})) if "value_clip" in sched_cfg else None
        self.grad_sched      = LinearValueScheduler(**sched_cfg.get("max_grad_norm", {})) if "max_grad_norm" in sched_cfg else None

    # ------------------------------------------------------------------
    # action-space helper
    # ------------------------------------------------------------------
    def _determine_action_space(self, cfg: Dict[str, Any]):
        if "agent" not in cfg:
            raise ValueError("Missing 'action.agent' block")
        a_cfg = cfg["agent"]
        if "continuous" in a_cfg:
            self.action_space_type = "continuous"
            self.action_dim        = len(a_cfg["continuous"])
            self.total_action_dim_per_agent = self.action_dim
        elif "discrete" in a_cfg:
            self.action_space_type = "discrete"
            self.action_dim = [d["num_choices"] for d in a_cfg["discrete"]]
            self.total_action_dim_per_agent = len(self.action_dim)
        else:
            raise ValueError("Specify discrete or continuous")

    # ------------------------------------------------------------------
    # memory utilities
    # ------------------------------------------------------------------
    def _apply_memory_seq(
        self,
        seq_feats: torch.Tensor,
        init_h: Optional[torch.Tensor],
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.memory_module:
            if return_hidden:
                return seq_feats, None
            return seq_feats

        B, T, NA, F = seq_feats.shape
        flat = seq_feats.reshape(B * NA, T, F)

        if init_h is None:
            init_h = torch.zeros(B, NA, self.memory_layers, self.memory_hidden, device=self.device)
        # reshape to (layers, batch, hidden)
        h0   = init_h.reshape(B * NA, self.memory_layers, self.memory_hidden).permute(1, 0, 2).contiguous()
        flat_out, h_n = self.memory_module.forward_sequence(flat, h0)
        out = flat_out.reshape(B, T, NA, self.memory_hidden)

        if return_hidden:
            # reshape h_n back to (B, NA, layers, hidden)
            h_n = h_n.permute(1, 0, 2).reshape(B, NA, self.memory_layers, self.memory_hidden)
            return out, h_n
        return out

    # ------------------------------------------------------------------
    # ACTION SELECTION
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_actions(
        self,
        states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        truncs: torch.Tensor,
        h_prev_batch: Optional[torch.Tensor] = None,
        eval: bool = False,
        record: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B_env = states["agent"].shape[1] if "agent" in states else 1
        emb, _ = self.embedding_net.get_base_embedding(states)
        emb = emb.squeeze(0)

        if self.memory_module:
            if h_prev_batch is None:
                h_prev_batch = torch.zeros(
                    B_env, self.num_agents, self.memory_layers, self.memory_hidden, device=self.device
                )
            flat         = emb.reshape(B_env * self.num_agents, -1)
            h_prev_flat  = h_prev_batch.reshape(B_env * self.num_agents, self.memory_layers, self.memory_hidden)\
                                     .permute(1, 0, 2).contiguous()
            flat_out, h_next_flat = self.memory_module.forward_step(flat, h_prev_flat)
            feats        = flat_out.reshape(B_env, self.num_agents, -1)
            h_next       = h_next_flat.permute(1, 0, 2).reshape(
                B_env, self.num_agents, self.memory_layers, self.memory_hidden
            )
            reset = _bool_or(dones, truncs).view(B_env, 1, 1, 1).float()
            h_next = h_next * (1.0 - reset)
        else:
            feats  = emb
            h_next = None

        pol_in = feats.reshape(B_env * self.num_agents, -1)
        act_flat, lp_flat, ent_flat = self.policy_net.get_actions(pol_in, eval)
        actions = act_flat.reshape(B_env, self.num_agents * self.total_action_dim_per_agent)
        logp    = lp_flat.reshape(B_env, self.num_agents)
        ent     = ent_flat.reshape(B_env, self.num_agents)

        if record and self.state_recorder and "central" in states and states["central"].ndim >= 2:
            self.state_recorder.record_frame(states["central"][0, 0].cpu().numpy().flatten())

        return actions, (logp, ent), h_next

    # ------------------------------------------------------------------
    # PPO UPDATE
    # ------------------------------------------------------------------
    def update(
        self,
        padded_states_dict_seq: Dict[str, torch.Tensor],
        padded_actions_seq: torch.Tensor,
        padded_rewards_seq: torch.Tensor,
        padded_next_states_dict_seq: Dict[str, torch.Tensor],
        padded_dones_seq: torch.Tensor,
        padded_truncs_seq: torch.Tensor,
        initial_hidden_states_batch: Optional[torch.Tensor],
        attention_mask_batch: torch.Tensor,
    ) -> Dict[str, float]:
        B, T = attention_mask_batch.shape
        NA    = self.num_agents
        mask_bt    = attention_mask_batch
        mask_bt1   = mask_bt.unsqueeze(-1)
        mask_btna1 = mask_bt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, NA, 1)

        # ----- reward normalization -----
        rewards_seq = padded_rewards_seq.clone()
        if self.rewards_normalizer is not None:
            valid_r = rewards_seq[mask_bt1.bool()]
            if valid_r.numel():
                self.rewards_normalizer.update(valid_r.unsqueeze(-1))
                rewards_seq[mask_bt1.bool()] = self.rewards_normalizer.normalize(valid_r.unsqueeze(-1)).squeeze(-1)

        # =============================================================
        # NO-GRAD PASS: compute old logprobs, values, intrinsic rewards, returns & update PopArt
        # =============================================================
        with torch.no_grad():
            emb_seq, _      = self.embedding_net.get_base_embedding(padded_states_dict_seq)
            emb_next_seq, _ = self.embedding_net.get_base_embedding(padded_next_states_dict_seq)

            if self.memory_module:
                feats_seq, h_n = self._apply_memory_seq(
                    emb_seq, initial_hidden_states_batch, return_hidden=True
                )
                feats_next_seq = self._apply_memory_seq(emb_next_seq, h_n)
            else:
                feats_seq      = self._apply_memory_seq(emb_seq, None)
                feats_next_seq = self._apply_memory_seq(emb_next_seq, None)

            old_logp_seq, _ = self._recompute_log_probs_from_features(feats_seq, padded_actions_seq)
            old_val_norm    = self.shared_critic.values(feats_seq)
            next_val_norm   = self.shared_critic.values(feats_next_seq)
            base_in_seq     = self.embedding_net.get_baseline_embeddings(feats_seq, padded_actions_seq)
            old_base_norm   = self.shared_critic.baselines(base_in_seq)

            # intrinsic reward
            intrinsic_na = torch.zeros(B, T, NA, device=self.device)
            if self.enable_rnd:
                flat_feats = feats_seq.reshape(B * T * NA, -1)
                tgt        = self.rnd_target_network(flat_feats)
                pred       = self.rnd_predictor_network(flat_feats)
                mse        = F.mse_loss(pred, tgt, reduction="none").mean(-1)
                intrinsic_na = mse.reshape(B, T, NA)
                valid_intr = intrinsic_na[mask_bt.unsqueeze(-1).expand_as(intrinsic_na).bool()]
                if valid_intr.numel():
                    self.intrinsic_reward_normalizer.update(valid_intr.unsqueeze(-1))
                    intrinsic_na[mask_bt.unsqueeze(-1).expand_as(intrinsic_na).bool()] = \
                        self.intrinsic_reward_normalizer.normalize(valid_intr.unsqueeze(-1)).squeeze(-1).to(intrinsic_na.dtype)
            intrinsic_seq = intrinsic_na.mean(-1, keepdim=True)

            # returns & GAE
            rew_gae     = rewards_seq + self.intrinsic_reward_coeff * intrinsic_seq
            old_val     = self.value_popart.denormalize_outputs(old_val_norm)  if self.enable_popart else old_val_norm
            nxt_val     = self.value_popart.denormalize_outputs(next_val_norm) if self.enable_popart else next_val_norm
            returns_seq = self._compute_gae_with_padding(
                rew_gae, old_val, nxt_val, padded_dones_seq, padded_truncs_seq, mask_bt
            )

            # stats for logging
            raw_ext_mean  = float(padded_rewards_seq[mask_bt1.bool()].mean()) if mask_bt1.any() else 0.0
            proc_rew_mean = float(rewards_seq[mask_bt1.bool()].mean())     if mask_bt1.any() else 0.0
            ret_vals      = returns_seq[mask_bt1.bool()]
            ret_mean      = float(ret_vals.mean()) if ret_vals.numel() else 0.0
            ret_std       = float(ret_vals.std())  if ret_vals.numel() else 0.0

            # PopArt stats update (once)
            if self.enable_popart:
                self.value_popart.update_stats(returns_seq[mask_bt1.bool()])
                self.baseline_popart.update_stats(
                    returns_seq.unsqueeze(2).expand(-1, -1, NA, -1)[mask_btna1.bool()]
                )
        # =============================================================

        # TRAINING LOOP
        logs_acc: Dict[str, List[float]] = {k: [] for k in [
            "policy", "value", "baseline", "entropy", "grad", "adv_mean", "adv_std"
        ]}
        if self.enable_rnd:
            logs_acc["rnd"] = []

        idx = np.arange(B)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for mb_start in range(0, B, self.mini_batch_size):
                mb_idx = idx[mb_start: mb_start + self.mini_batch_size]
                if mb_idx.size == 0:
                    continue
                mask_mb = mask_bt[mb_idx]  # (M,T)

                # fresh forward for grads
                emb_mb, _        = self.embedding_net.get_base_embedding({
                    k: v[mb_idx] for k, v in padded_states_dict_seq.items()
                })
                feats_mb         = self._apply_memory_seq(
                    emb_mb,
                    initial_hidden_states_batch[mb_idx] if initial_hidden_states_batch is not None else None
                )
                new_lp_mb, ent_mb = self._recompute_log_probs_from_features(feats_mb, padded_actions_seq[mb_idx])
                new_val_norm_mb   = self.shared_critic.values(feats_mb)
                base_in_mb        = self.embedding_net.get_baseline_embeddings(feats_mb, padded_actions_seq[mb_idx])
                new_base_norm_mb  = self.shared_critic.baselines(base_in_mb)

                old_lp_mb        = old_logp_seq[mb_idx]
                old_val_mb_norm  = old_val_norm[mb_idx]
                old_base_mb_norm = old_base_norm[mb_idx]
                returns_mb       = returns_seq[mb_idx]

                # advantages
                new_base_denorm = (
                    self.baseline_popart.denormalize_outputs(new_base_norm_mb)
                    if self.enable_popart else new_base_norm_mb
                )
                adv = (returns_mb.unsqueeze(2) - new_base_denorm).squeeze(-1)
                valid_adv_mask = mask_mb.unsqueeze(-1).expand_as(adv).bool()
                if self.normalize_adv:
                    valid_adv = adv[valid_adv_mask]
                    if valid_adv.numel():
                        m, s = valid_adv.mean(), valid_adv.std() + 1e-8
                        adv_norm = (valid_adv - m) / s
                        tmp = torch.zeros_like(adv)
                        tmp[valid_adv_mask] = adv_norm
                        adv = tmp
                        logs_acc["adv_mean"].append(m.item())
                        logs_acc["adv_std"].append((s - 1e-8).item())
                else:
                    if valid_adv_mask.any():
                        logs_acc["adv_mean"].append(float(adv[valid_adv_mask].mean()))
                        logs_acc["adv_std"].append(float(adv[valid_adv_mask].std()))

                # policy loss
                pol_loss = self._ppo_clip_loss(new_lp_mb, old_lp_mb, adv, self.ppo_clip_range)

                # value loss
                tgt_val   = returns_mb if not self.enable_popart else self.value_popart.normalize_targets(returns_mb)
                val_terms = self._clipped_value_loss(
                    old_val_mb_norm, new_val_norm_mb, tgt_val, self.value_clip_range, reduction="none"
                )
                mask_val  = mask_mb.unsqueeze(-1)
                val_loss  = (val_terms * mask_val).sum() / mask_val.sum().clamp(min=1)

                # baseline loss
                tgt_base   = returns_mb.unsqueeze(2).expand_as(new_base_norm_mb)
                if self.enable_popart:
                    tgt_base = self.baseline_popart.normalize_targets(tgt_base)
                base_terms = self._clipped_value_loss(
                    old_base_mb_norm, new_base_norm_mb, tgt_base, self.value_clip_range, reduction="none"
                )
                mask_base = mask_mb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, NA, 1)
                base_loss  = (base_terms * mask_base).sum() / mask_base.sum().clamp(min=1)

                # entropy loss
                mask_ent = mask_mb.unsqueeze(-1)
                ent_loss = (-ent_mb * mask_ent).sum() / mask_ent.sum().clamp(min=1)

                # total loss
                loss = (
                    pol_loss
                    + self.value_loss_coeff * val_loss
                    + self.baseline_loss_coeff * base_loss
                    + self.entropy_coeff * ent_loss
                )

                # RND update on subsample
                rnd_loss = torch.tensor(0.0, device=self.device)
                if self.enable_rnd and self.rnd_update_prop > 0.0:
                    flat_f = feats_mb.reshape(-1, feats_mb.shape[-1])
                    n      = flat_f.shape[0]
                    k      = int(n * self.rnd_update_prop)
                    if k > 0:
                        sel     = torch.randperm(n, device=self.device)[:k]
                        pred_rnd= self.rnd_predictor_network(flat_f[sel])
                        with torch.no_grad():
                            tgt_rnd = self.rnd_target_network(flat_f[sel])
                        rnd_loss = F.mse_loss(pred_rnd, tgt_rnd)
                        loss    += rnd_loss

                # backward and optimization
                for opt in (self.trunk_opt, self.policy_opt, self.value_opt, self.base_opt, self.rnd_opt):
                    if opt: opt.zero_grad()
                loss.backward()
                gn = clip_grad_norm_(self.parameters(), self.max_grad_norm)
                for opt in (self.trunk_opt, self.policy_opt, self.value_opt, self.base_opt, self.rnd_opt):
                    if opt: opt.step()

                # logs
                logs_acc["policy"].append(pol_loss.item())
                logs_acc["value"].append(val_loss.item())
                logs_acc["baseline"].append(base_loss.item())
                logs_acc["entropy"].append(ent_mb.mean().item())
                logs_acc["grad"].append(gn.item() if torch.is_tensor(gn) else gn)
                if self.enable_rnd:
                    logs_acc["rnd"].append(rnd_loss.item())

            # schedulers
            if self.policy_lr_sched: self.policy_lr_sched.step()
            if self.entropy_sched:   self.entropy_coeff    = self.entropy_sched.step()
            if self.clip_sched:      self.ppo_clip_range   = self.clip_sched.step()
            if self.val_clip_sched:  self.value_clip_range = self.val_clip_sched.step()
            if self.grad_sched:      self.max_grad_norm    = self.grad_sched.step()

        # aggregate logs
        logs = {k: (np.mean(v) if v else 0.0) for k, v in logs_acc.items()}
        logs.update({
            "raw_reward_mean":       raw_ext_mean,
            "processed_reward_mean": proc_rew_mean,
            "return_mean":           ret_mean,
            "return_std":            ret_std,
            "entropy_coeff":         self.entropy_coeff,
            "ppo_clip":              self.ppo_clip_range,
            "lr_policy":             self._get_policy_lr(),
        })
        # --- Add PopArt running stats ---
        if self.enable_popart:
            logs.update({
                # value head running moments
                "popart_value_mu": float(self.value_popart.mu),
                "popart_value_sigma": float(self.value_popart.sigma),
                # baseline head running moments
                "popart_baseline_mu": float(self.baseline_popart.mu),
                "popart_baseline_sigma": float(self.baseline_popart.sigma),
            })
        return logs

    # ------------------------------------------------------------------
    # GAE with padding
    # ------------------------------------------------------------------
    def _compute_gae_with_padding(self, r, v, v_next, d, tr, mask_bt):
        B, T, _ = r.shape
        out = torch.zeros_like(r)
        gae = torch.zeros(B, 1, device=self.device)
        term = (d + tr).clamp(max=1.0)
        for t in reversed(range(T)):
            valid     = mask_bt[:, t:t+1]
            nxt_valid = mask_bt[:, t+1:t+2] if t < T-1 else torch.zeros_like(valid)
            non_term  = (1.0 - term[:, t]) * nxt_valid
            delta     = r[:, t] + self.gamma * v_next[:, t] * non_term - v[:, t]
            gae       = delta + self.gamma * self.lmbda * non_term * gae
            out[:, t] = gae + v[:, t]
            gae       = gae * valid
        return out

    # ------------------------------------------------------------------
    # PPO clipping helpers
    # ------------------------------------------------------------------
    def _ppo_clip_loss(self, new_lp, old_lp, adv, clip_range):
        ratio = torch.exp((new_lp - old_lp).clamp(-10, 10))
        s1    = ratio * adv
        s2    = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
        return -torch.min(s1, s2).mean()

    def _clipped_value_loss(self, old, new, target, clip_range, reduction="mean"):
        clipped = old + (new - old).clamp(-clip_range, clip_range)
        loss    = torch.max((new - target).pow(2), (clipped - target).pow(2))
        if reduction == "none":
            return loss
        return loss.mean()

    # ------------------------------------------------------------------
    # recompute log-probs
    # ------------------------------------------------------------------
    def _recompute_log_probs_from_features(self, feats, actions):
        flat_f = feats.reshape(-1, feats.shape[-1])
        flat_a = actions.reshape(-1, actions.shape[-1])
        lp, ent = self.policy_net.recompute_log_probs(flat_f, flat_a)
        leading = feats.shape[:-1]
        return lp.reshape(*leading), ent.reshape(*leading)

    # ------------------------------------------------------------------
    # gather parameters for grad clipping
    # ------------------------------------------------------------------
    def parameters(self):
        params = (
            list(self.embedding_net.parameters())
            + list(self.policy_net.parameters())
            + list(self.shared_critic.parameters())
        )
        if self.memory_module:
            params += list(self.memory_module.parameters())
        if self.enable_rnd:
            params += list(self.rnd_predictor_network.parameters())
        return params

    # ------------------------------------------------------------------
    # current policy learning rate
    # ------------------------------------------------------------------
    def _get_policy_lr(self):
        return float(np.mean([pg["lr"] for pg in self.policy_opt.param_groups]))
