from typing import Dict
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from Source.StateRecorder import StateRecorder

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer, LinearValueScheduler, OneCycleLRWithMin,OneCycleCosineScheduler
from Source.Networks import (
    MultiAgentEmbeddingNetwork,
    SharedCritic,
    TanhContinuousPolicyNetwork,
    DiscretePolicyNetwork
)

class MAPOCAAgent(Agent):
    """
    Multi-agent PPO (MAPOCA) style agent. 
    - TanhContinuous or Discrete policy networks
    - clipped value & baseline (optional)
    - single-pass approach for policy/value/baseline
    """

    def __init__(self, config: Dict, device: torch.device):
        super().__init__(config, device)

        agent_cfg = config["agent"]["params"]
        train_cfg = config["train"]
        shape_cfg = config["environment"]["shape"]
        action_cfg= shape_cfg["action"]
        rec_cfg = config.get("StateRecorder", None)

        # PPO / GAE Hyperparams
        self.gamma   = agent_cfg.get("gamma", 0.99)
        self.lmbda   = agent_cfg.get("lambda", 0.95)
        self.lr      = agent_cfg.get("learning_rate", 3e-4)
        self.value_loss_coeff    = agent_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff = agent_cfg.get("baseline_loss_coeff", 0.5)
        self.entropy_coeff       = agent_cfg.get("entropy_coeff", 0.01)
        self.max_grad_norm       = agent_cfg.get("max_grad_norm", 0.5)
        self.normalize_adv       = agent_cfg.get("normalize_advantages", False)
        self.ppo_clip_range      = agent_cfg.get("ppo_clip_range", 0.1)

        # Value clipping
        self.clipped_value_loss  = agent_cfg.get("clipped_value_loss", False)
        self.value_clip_range    = agent_cfg.get("value_clip_range", 0.2)

        # training
        self.epochs         = train_cfg.get("epochs", 4)
        self.mini_batch_size= train_cfg.get("mini_batch_size", 128)

        # Recording
        self.state_recorder = None
        if rec_cfg:
            self.state_recorder = StateRecorder(rec_cfg)

        # Determine action space
        if "agent" in action_cfg:
            if "discrete" in action_cfg["agent"]:
                self.action_space_type = "discrete"
                d_list = action_cfg["agent"]["discrete"]
                if len(d_list) > 1:
                    self.action_dim = [d["num_choices"] for d in d_list]
                else:
                    self.action_dim = d_list[0]["num_choices"]
            elif "continuous" in action_cfg["agent"]:
                self.action_space_type = "continuous"
                c_ranges = action_cfg["agent"]["continuous"]
                self.action_dim = len(c_ranges)
            else:
                raise ValueError("No discrete/continuous under action/agent.")
        elif "central" in action_cfg:
            # single-agent fallback
            if "discrete" in action_cfg["central"]:
                self.action_space_type = "discrete"
                d_list = action_cfg["central"]["discrete"]
                if len(d_list) > 1:
                    self.action_dim = [item["num_choices"] for item in d_list]
                else:
                    self.action_dim = d_list[0]["num_choices"]
            elif "continuous" in action_cfg["central"]:
                self.action_space_type = "continuous"
                c_ranges = action_cfg["central"]["continuous"]
                self.action_dim = len(c_ranges)
            else:
                raise ValueError("No discrete/continuous under action/central.")
        else:
            raise ValueError("No agent or central block in action config.")

        # Build networks
        self.embedding_network = MultiAgentEmbeddingNetwork(
            agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        ).to(device)

        self.shared_critic = SharedCritic(
            agent_cfg["networks"]['critic_network']
        ).to(device)

        # policy net
        pol_cfg = agent_cfg["networks"]["policy_network"]
        if self.action_space_type == "continuous":
            self.policy_net = TanhContinuousPolicyNetwork(
                in_features=pol_cfg["in_features"],
                out_features=self.action_dim,
                hidden_size=pol_cfg["hidden_size"],
                mean_scale=pol_cfg.get("mean_scale", -4.0),
                log_std_min=pol_cfg.get("log_std_min", -20.0),
                log_std_max=pol_cfg.get("log_std_max", 2.0),
                entropy_method=pol_cfg.get("entropy_method", "analytic"),
                n_entropy_samples=pol_cfg.get("n_entropy_samples", 5)
            ).to(device)
        else:
            self.policy_net = DiscretePolicyNetwork(
                in_features=pol_cfg["in_features"],
                out_features=self.action_dim,
                hidden_size=pol_cfg["hidden_size"]
            ).to(device)

        # Optim & schedulers
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        lr_sched_cfg = agent_cfg["schedulers"].get("lr", None)
        self.lr_scheduler = None
        if lr_sched_cfg:
            # self.lr_scheduler = OneCycleLRWithMin(self.optimizer, **lr_sched_cfg)
            self.lr_scheduler = LinearLR(self.optimizer, **lr_sched_cfg)

        ent_sched_cfg = agent_cfg["schedulers"].get("entropy_coeff", None)
        self.entropy_scheduler = None
        if ent_sched_cfg:
            self.entropy_scheduler = LinearValueScheduler(**ent_sched_cfg)
            # self.entropy_scheduler = OneCycleCosineScheduler(**ent_sched_cfg)

        policy_clip_sched_cfg = agent_cfg["schedulers"].get("policy_clip", None)
        self.policy_clip_scheduler = None
        if policy_clip_sched_cfg:
            self.policy_clip_scheduler = LinearValueScheduler(**policy_clip_sched_cfg)

        value_clip_sched_cfg = agent_cfg["schedulers"].get("value_clip", None)
        self.value_clip_scheduler = None
        if value_clip_sched_cfg:
            self.value_clip_scheduler = LinearValueScheduler(**value_clip_sched_cfg)

        grad_norm_sched_cfg = agent_cfg["schedulers"].get("max_grad_norm", None)
        self.max_grad_norm_scheduler = None
        if grad_norm_sched_cfg:
            self.max_grad_norm_scheduler = LinearValueScheduler(**grad_norm_sched_cfg)

        # Normalizers
        rewards_normalization_cfg = agent_cfg.get('rewards_normalizer', None)
        self.rewards_normalizer = None
        if rewards_normalization_cfg:
            self.rewards_normalizer = RunningMeanStdNormalizer(
                **rewards_normalization_cfg,
                device=self.device
            )

    def parameters(self):
        return (
            list(self.embedding_network.parameters())
            + list(self.shared_critic.parameters())
            + list(self.policy_net.parameters())
        )

    # ------------------------------------------------------------
    #  get_actions()
    # ------------------------------------------------------------
    @torch.no_grad()
    def get_actions(self, states: dict, dones: torch.Tensor, truncs: torch.Tensor, eval: bool=False, record: bool=True):
        """
        Build embeddings => pass to self.policy_net => reshape
        Return => actions, (log_probs, entropies)
        """
        common_emb, _ = self.embedding_network.get_base_embedding(states)
        S,E,NA,H = common_emb.shape
        emb_2d   = common_emb.reshape(S*E*NA, H)

        actions_flat, lp_flat, ent_flat = self.policy_net.get_actions(emb_2d, eval=eval)

        if self.action_space_type == "continuous":
            act_dim = actions_flat.shape[-1]
            actions = actions_flat.reshape(S,E,NA, act_dim)
        else:
            # discrete => single-cat or multi-cat
            if actions_flat.dim()==1:
                # single-cat => shape => (B,)
                actions = actions_flat.reshape(S,E,NA,1)
            else:
                # multi-discrete => shape => (B,#branches)
                nb = actions_flat.shape[-1]
                actions = actions_flat.reshape(S,E,NA, nb)

        log_probs = lp_flat.reshape(S,E,NA)
        entropies = ent_flat.reshape(S,E,NA)


        # 2) If we want to record the central state for the "first environment":
        if self.state_recorder is not None:
            c_1d = states["central"][0,1,:].cpu().numpy()
            self.state_recorder.record_frame(c_1d)

        return actions, (log_probs, entropies)

    # ------------------------------------------------------------
    #  update()
    # ------------------------------------------------------------
    def update(self,
           states: dict,
           next_states: dict,
           actions: torch.Tensor,
           rewards: torch.Tensor,
           dones: torch.Tensor,
           truncs: torch.Tensor):
        """
        1) Possibly normalize rewards
        2) Gather old values/baselines/log_probs
        3) Compute returns
        4) Mini-batch => Compute new => Clipped losses => Step
        5) Return logs
        """

        # shape check
        if "agent" in states:
            NS, NE, NA, _ = states["agent"].shape
        else:
            NS, NE, _ = states["central"].shape
            NA = 1

        # Maybe normalize rewards
        rewards_mean = rewards.mean()
        rewards_norm_mean = rewards_mean
        if self.rewards_normalizer:
            self.rewards_normalizer.update(rewards)
            rewards = self.rewards_normalizer.normalize(rewards)
            rewards_norm_mean = rewards.mean()

        # ------------------------------------------------
        # 1) Gather old data
        # ------------------------------------------------
        with torch.no_grad():
            # Get base embeddings and attention weights
            base_emb, _ = self.embedding_network.get_base_embedding(states)
            baseline_emb = self.embedding_network.get_baseline_embeddings(base_emb, actions)

            # Compute old values, baselines using attention weights
            old_vals = self.shared_critic.values(base_emb)  # => (NS,NE,1)
            old_bases = self.shared_critic.baselines(baseline_emb)  # => (NS,NE,NA,1)

            # Compute old log_probs
            old_lp, _ = self._recompute_log_probs(base_emb, actions)

            # Compute next state embeddings and values
            next_base_emb, _ = self.embedding_network.get_base_embedding(next_states)
            next_vals = self.shared_critic.values(next_base_emb)

            # Compute returns using TD(lambda)
            returns = self._compute_td_lambda_returns(
                rewards, old_vals, next_vals, dones, truncs
            )

        # ------------------------------------------------
        # 2) Prepare arrays to store logs
        # ------------------------------------------------
        idxes = np.arange(NS)
        pol_losses, val_losses, base_losses = [], [], []
        entropies = []
        advantages_list, advantages_mean_list, advantages_std_list = [], [], []
        returns_list, grad_norm_list, lrs_list = [], [], []

        # For debugging log-prob
        logprob_means, logprob_mins, logprob_maxs = [], [], []

        # Track layer-wise gradient norms for averaging
        grad_norms_sum_per_layer = {}
        grad_norms_count = 0

        # Single pass stats
        returns_list.append(returns.mean().detach().cpu())

        # ------------------------------------------------
        # 3) Training loop over epochs + mini-batches
        # ------------------------------------------------
        for _ in range(self.epochs):
            np.random.shuffle(idxes)
            nB = (NS + self.mini_batch_size - 1) // self.mini_batch_size

            for b in range(nB):
                # Step schedulers
                if self.entropy_scheduler:
                    self.entropy_coeff = self.entropy_scheduler.step()
                if self.policy_clip_scheduler:
                    self.ppo_clip_range = self.policy_clip_scheduler.step()
                if self.value_clip_scheduler:
                    self.value_clip_range = self.value_clip_scheduler.step()
                if self.max_grad_norm_scheduler:
                    self.max_grad_norm = self.max_grad_norm_scheduler.step()

                start = b * self.mini_batch_size
                end = min((b + 1) * self.mini_batch_size, NS)
                mb_idx = idxes[start:end]
                MB = len(mb_idx)

                st_mb = self._slice_state_dict(states, mb_idx)
                act_mb = actions[mb_idx]
                ret_mb = returns[mb_idx]  # => (MB,NE,1)
                olp_mb = old_lp[mb_idx]  # => (MB,NE,NA)
                oval_mb = old_vals[mb_idx]  # => (MB,NE,1)
                obase_mb = old_bases[mb_idx]  # => (MB,NE,NA,1)

                # Forward pass on new states
                base_emb_mb, _ = self.embedding_network.get_base_embedding(st_mb)
                baseline_emb_mb = self.embedding_network.get_baseline_embeddings(base_emb_mb, act_mb)

                # Compute new log_probs
                new_lp_mb, ent_mb = self._recompute_log_probs(base_emb_mb, act_mb)

                # Log-prob stats
                lp_det = new_lp_mb.detach()
                logprob_means.append(lp_det.mean().cpu())
                logprob_mins.append(lp_det.min().cpu())
                logprob_maxs.append(lp_det.max().cpu())

                # Compute new values, baselines using new attention weights
                new_vals_mb = self.shared_critic.values(base_emb_mb)  # => (MB,NE,1)
                new_base_mb = self.shared_critic.baselines(baseline_emb_mb)  # => (MB,NE,NA,1)

                # Advantage Calculation
                ret_mb_exp = ret_mb.unsqueeze(2).expand(MB, NE, NA, 1)
                adv_mb = (ret_mb_exp - new_base_mb).squeeze(-1)  # => (MB,NE,NA)

                # Advantage stats
                adv_mean = adv_mb.mean()
                adv_std = adv_mb.std() 
                advantages_mean_list.append(adv_mean.detach().cpu())
                advantages_std_list.append(adv_std.detach().cpu())
                if self.normalize_adv:
                    adv_mb = (adv_mb - adv_mean) / (adv_std + 1e-8)
                    adv_clip_factor = 5.0
                    adv_mb = adv_mb.clamp(-adv_clip_factor, adv_clip_factor)

                # Detach advantage for policy gradient
                detached_adv = adv_mb.detach()
                advantages_list.append(detached_adv.mean().cpu())

                # Policy PPO Clip Loss
                pol_loss = self._ppo_clip_loss(new_lp_mb, olp_mb, detached_adv, self.ppo_clip_range)

                # Value Function Loss (with optional clipping)
                if self.clipped_value_loss:
                    val_loss = self._clipped_value_loss(oval_mb, new_vals_mb, ret_mb, self.value_clip_range)
                else:
                    val_loss = 0.5 * F.mse_loss(new_vals_mb, ret_mb)

                # Baseline Loss (with optional clipping)
                if self.clipped_value_loss:
                    base_loss = self._clipped_value_loss(obase_mb, new_base_mb, ret_mb_exp, self.value_clip_range)
                else:
                    base_loss = 0.5 * F.mse_loss(new_base_mb, ret_mb_exp)

                # Final loss
                ent_mean = ent_mb.mean()
                total_loss = pol_loss + self.value_loss_coeff * val_loss + self.baseline_loss_coeff * base_loss - self.entropy_coeff * ent_mean

                self.optimizer.zero_grad()
                total_loss.backward()

                # Measure per-layer grad norms before clipping
                self._accumulate_per_layer_grad_norms(grad_norms_sum_per_layer)

                gn = clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                pol_losses.append(pol_loss.detach().cpu())
                val_losses.append(val_loss.detach().cpu())
                base_losses.append(base_loss.detach().cpu())
                entropies.append(ent_mean.detach().cpu())
                grad_norm_list.append(gn)
                lrs_list.append(self._get_avg_lr(self.optimizer))

                grad_norms_count += 1

        # ------------------------------------------------
        # final logs
        # ------------------------------------------------
        # average the per-layer grad norms over all mini-batches
        layer_grad_norm_logs = {}
        for pname, accum in grad_norms_sum_per_layer.items():
            mean_norm = accum / float(grad_norms_count)
            layer_grad_norm_logs[f"GradNorm/{pname}"] = mean_norm

        logs = {
            "Policy Loss": torch.stack(pol_losses).mean(),
            "Value Loss": torch.stack(val_losses).mean(),
            "Baseline Loss": torch.stack(base_losses).mean(),
            "Entropy": torch.stack(entropies).mean(),
            "Entropy coeff": torch.tensor(self.entropy_coeff),
            "Avg. Adv": torch.stack(advantages_list).mean(),
            "Avg. Returns": torch.stack(returns_list).mean(),
            "Avg. Grad Norm": torch.stack(grad_norm_list).mean(),
            "Ave. Rewards": rewards_mean,
            "Ave. Rewards norm": rewards_norm_mean,
            "Learning Rate": torch.stack(lrs_list).mean(),
            "Logprob Min": torch.stack(logprob_mins).mean(),
            "Logprob Max": torch.stack(logprob_maxs).mean()
        }
        logs.update(layer_grad_norm_logs)
        return logs

    # ------------------------------------------------------------
    #  HELPER METHODS
    # ------------------------------------------------------------
    def _compute_td_lambda_returns(self,
                                   rewards: torch.Tensor,
                                   values:  torch.Tensor,
                                   next_values: torch.Tensor,
                                   dones:  torch.Tensor,
                                   truncs: torch.Tensor):
        """
        Standard GAE-like approach => shapes => (NS,NE,1)
        """
        NS, NE, _ = rewards.shape
        out_rets = torch.zeros_like(rewards)
        gae = torch.zeros(NE,1, device=rewards.device)

        for t in reversed(range(NS)):
            if t == NS-1:
                nv = next_values[t]*(1-dones[t])*(1-truncs[t])
            else:
                nv = values[t+1]*(1-dones[t+1])*(1-truncs[t+1])
            delta = rewards[t] + self.gamma*nv - values[t]
            gae   = delta + self.gamma*self.lmbda*(1-dones[t])*(1-truncs[t])*gae
            out_rets[t] = gae + values[t]
        return out_rets

    def _ppo_clip_loss(self, new_lp, old_lp, adv, clip_range):
        diff = (new_lp - old_lp).clamp(-10.0, 10.0) # to avoid extreme exponent
        ratio = torch.exp(diff)
        clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        obj1 = ratio * adv
        obj2 = clipped * adv
        return -torch.mean(torch.min(obj1, obj2))

    def _clipped_value_loss(self,
                            old_values: torch.Tensor,
                            new_values: torch.Tensor,
                            targets:   torch.Tensor,
                            clip_range: float):
        """
        old_values, new_values, targets => shape => (MB,NE,1) or (MB,NE,NA,1)
        Return => scalar clipped value MSE
        """
        v_clipped    = old_values + torch.clamp( (new_values - old_values),
                                                 -clip_range, clip_range )
        mse_unclipped= (new_values - targets).pow(2)
        mse_clipped  = (v_clipped - targets).pow(2)
        max_mse      = torch.max(mse_unclipped, mse_clipped)
        return 0.5 * max_mse.mean()

    def _recompute_log_probs(self, comm_att: torch.Tensor, actions: torch.Tensor):
        """
        Flatten => call self.policy_net.recompute_log_probs => reshape => (MB,NE,NA)
        Return => (log_probs, entropies)
        """
        MB, NE, NA, H = comm_att.shape
        emb_2d = comm_att.reshape(MB*NE*NA, H)

        if self.action_space_type == "continuous":
            # shape => (MB,NE,NA, actDim)
            if actions.dim() == 4:
                a_resh = actions.reshape(MB*NE*NA, actions.shape[-1])
            else:
                a_resh = actions.reshape(MB*NE*NA, -1)
        else:
            # discrete => single-cat or multi-cat
            if actions.dim() == 4:
                # multi-cat => (MB,NE,NA,#branches)
                a_resh = actions.reshape(MB*NE*NA, actions.shape[-1])
            else:
                # single-cat => (MB,NE,NA)
                a_resh = actions.reshape(MB*NE*NA)

        lp_2d, ent_2d = self.policy_net.recompute_log_probs(emb_2d, a_resh)
        return lp_2d.reshape(MB,NE,NA), ent_2d.reshape(MB,NE,NA)

    def _slice_state_dict(self, states_dict: dict, indices):
        """
        Slices dict along dim=0 => (NS, NE, ...).
        """
        new_dict = {}
        idx_t = torch.as_tensor(indices, device=self.device)
        for k, v in states_dict.items():
            new_dict[k] = v.index_select(0, idx_t).contiguous()
        return new_dict

    def _get_avg_lr(self, optimizer) -> torch.Tensor:
        lr_sum, count = 0.0, 0
        for pg in optimizer.param_groups:
            lr_sum += pg['lr']
            count  += 1
        return torch.tensor(lr_sum/max(count,1.0))

    def _accumulate_per_layer_grad_norms(self, grad_norms_sum_per_layer: dict):
        """
        Accumulate gradient L2-norms for each parameter name. Typically, param 'name'
        is unique because named_parameters() includes sub-module prefixes.

        If a collision occurs (extremely rare), we append a suffix index.
        """
        for param_name, param in self.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.data.norm(2).item()
                # ensure uniqueness if name collision
                tmp_name = param_name
                idx_sfx = 0
                while tmp_name in grad_norms_sum_per_layer:
                    # append an index
                    idx_sfx += 1
                    tmp_name = f"{param_name}__dup{idx_sfx}"
                
                if tmp_name not in grad_norms_sum_per_layer:
                    grad_norms_sum_per_layer[tmp_name] = 0.0
                
                grad_norms_sum_per_layer[tmp_name] += gnorm