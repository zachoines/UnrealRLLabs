from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer, LinearValueScheduler
from Source.Networks import (
    MultiAgentEmbeddingNetwork,
    SharedCritic,
    ContinuousPolicyNetwork,
    DiscretePolicyNetwork
)

class MAPOCAAgent(Agent):
    """
    MA-POCA Agent using a single common embedding pass. Key changes:
      - We call `common_emb = embedding_network.get_common_embedding(states)`
      - Then we call `embed_for_policy`, `embed_for_value`, `embed_for_baseline` on it,
        so that we only parse the environment dictionary once per forward pass.
      - For baseline, we also pass the actions to do groupmate-splitting.

    The rest is standard PPO-like code for multi-agent posthumous credit assignment.
    """

    def __init__(self, config: Dict, device: torch.device):
        super(MAPOCAAgent, self).__init__(config, device)

        agent_cfg = config["agent"]["params"]
        train_cfg = config["train"]
        env_shape_cfg = config["environment"]["shape"]
        action_cfg = env_shape_cfg["action"]

        # --------------------
        #    Hyperparams
        # --------------------
        self.gamma   = agent_cfg.get("gamma",  0.99)
        self.lmbda   = agent_cfg.get("lambda", 0.95)
        self.lr      = agent_cfg.get("learning_rate", 3e-4)

        self.value_loss_coeff     = agent_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff  = agent_cfg.get("baseline_loss_coeff", 0.5)
        self.entropy_coeff        = agent_cfg.get("entropy_coeff", 0.01)
        self.base_entropy_coeff   = agent_cfg.get("entropy_coeff", 0.01)
        self.max_grad_norm        = agent_cfg.get("max_grad_norm", 0.5)
        self.normalize_rewards    = agent_cfg.get("normalize_rewards", False)
        self.normalize_advantages = agent_cfg.get("normalize_advantages", False)
        self.ppo_clip_range       = agent_cfg.get("ppo_clip_range", 0.1)

        # Training config
        self.epochs          = train_cfg.get("epochs", 4)
        self.mini_batch_size = train_cfg.get("mini_batch_size", 128)

        # --------------------
        #   Action Space
        # --------------------
        # Checking whether 'agent' or 'central' block is used:
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
            # single-agent version
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

        # --------------------
        #   Networks
        # --------------------
        self.embedding_network = MultiAgentEmbeddingNetwork(
            agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        ).to(device)

        self.shared_critic = SharedCritic(config).to(device)

        # policy net: discrete or continuous
        if self.action_space_type == "continuous":
            self.policy_net = ContinuousPolicyNetwork(
                **agent_cfg["networks"]["policy_network"]
            ).to(device)
        else:
            self.policy_net = DiscretePolicyNetwork(
                **agent_cfg["networks"]["policy_network"],
                out_features=self.action_dim
            ).to(device)

        # --------------------
        #   Optim & schedulers
        # --------------------
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        lr_sched_cfg = agent_cfg["schedulers"].get("lr", None)
        self.lr_scheduler = None
        if lr_sched_cfg:
            self.lr_scheduler = LinearLR(self.optimizer, **lr_sched_cfg)

        ent_sched_cfg = agent_cfg["schedulers"].get("entropy_coeff", None)
        self.entropy_scheduler = None
        if ent_sched_cfg:
            self.entropy_scheduler = LinearValueScheduler(**ent_sched_cfg)

        # Optional normalizers
        if self.normalize_rewards:
            self.reward_normalizer = RunningMeanStdNormalizer(device=device)
        else:
            self.reward_normalizer = None

        if self.normalize_advantages:
            self.advantage_normalizer = RunningMeanStdNormalizer(device=device)
        else:
            self.advantage_normalizer = None

    def parameters(self):
        return (
            list(self.embedding_network.parameters()) +
            list(self.shared_critic.parameters()) +
            list(self.policy_net.parameters())
        )

    # ------------------------------------------------------------
    #             ACTION SELECTION
    # ------------------------------------------------------------
    def get_actions(self,
                    states: dict,
                    dones: torch.Tensor,
                    truncs: torch.Tensor,
                    eval: bool = False):
        """
        states => e.g. {"central":(S,E,cDim), "agent":(S,E,NA,aDim)}
        Return => (actions, (log_probs, entropies)) => shapes => (S,E,NA,...)
        """
        with torch.no_grad():
            # Single pass => produce a "common_emb"
            common_emb = self.embedding_network.apply_attention(self.embedding_network.get_common_embedding(states))
            S, E, NA, H = common_emb.shape
            flat_emb = common_emb.contiguous().reshape(S*E*NA, H)

            if self.action_space_type == "continuous":
                mean, std = self.policy_net(flat_emb)
                dist = torch.distributions.Normal(mean, std)
                if eval:
                    actions_flat = mean
                else:
                    actions_flat = dist.sample()

                logp_flat = dist.log_prob(actions_flat).sum(dim=-1)
                ent_flat  = dist.entropy().sum(dim=-1)

                act_dim   = actions_flat.shape[-1]
                actions   = actions_flat.reshape(S,E,NA, act_dim)
                log_probs = logp_flat.reshape(S,E,NA)
                entropy   = ent_flat.reshape(S,E,NA)

            else:
                # discrete or multi-discrete
                logits_out = self.policy_net(flat_emb)
                if isinstance(logits_out, list):
                    # multi-discrete
                    actions_list, lp_list, ent_list = [], [], []
                    for branch_logits in logits_out:
                        dist_b = torch.distributions.Categorical(logits=branch_logits)
                        if eval:
                            a_b = branch_logits.argmax(dim=-1)
                        else:
                            a_b = dist_b.sample()
                        lp_b  = dist_b.log_prob(a_b)
                        ent_b = dist_b.entropy()
                        actions_list.append(a_b)
                        lp_list.append(lp_b)
                        ent_list.append(ent_b)

                    acts_stacked = torch.stack(actions_list, dim=-1)
                    lp_stacked   = torch.stack(lp_list, dim=-1).sum(dim=-1)
                    ent_stacked  = torch.stack(ent_list, dim=-1).sum(dim=-1)

                    nb = len(logits_out)
                    actions   = acts_stacked.reshape(S,E,NA, nb)
                    log_probs = lp_stacked.reshape(S,E,NA)
                    entropy   = ent_stacked.reshape(S,E,NA)
                else:
                    # single-cat
                    dist_c = torch.distributions.Categorical(logits=logits_out)
                    if eval:
                        a_flat = logits_out.argmax(dim=-1)
                    else:
                        a_flat = dist_c.sample()
                    lp_flat = dist_c.log_prob(a_flat)
                    ent_flat= dist_c.entropy()

                    actions   = a_flat.reshape(S,E,NA, 1)
                    log_probs = lp_flat.reshape(S,E,NA)
                    entropy   = ent_flat.reshape(S,E,NA)

            return actions, (log_probs, entropy)

    # ------------------------------------------------------------
    #                   MAIN UPDATE
    # ------------------------------------------------------------
    def update(self,
               states: dict,
               next_states: dict,
               actions: torch.Tensor,
               rewards: torch.Tensor,
               dones: torch.Tensor,
               truncs: torch.Tensor):
        """
        states => { "central":(NS,NE,cSize), "agent":(NS,NE,NA,aSize) }
        next_states => same
        actions => (NS,NE,NA, actDim or #branches)
        rewards => (NS,NE,1)
        ...
        """
        # deduce (NS,NE,NA)
        if "agent" in states:
            NS, NE, NA, _ = states["agent"].shape
        else:
            # single-agent => shape => (NS,NE,cDim)
            NS, NE, _ = states["central"].shape
            NA = 1

        # Optionally normalize rewards
        rewards_mean = rewards.mean()
        if self.reward_normalizer:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        # ---------------
        #   1) Collect old log_probs, old value, old baseline
        # ---------------
        with torch.no_grad():
            # For states => values
            common_emb       = self.embedding_network.apply_attention(self.embedding_network.get_common_embedding(states))
            old_vals         = self.get_value(common_emb)

            # for next_states => next_value
            next_common_emb  = self.embedding_network.apply_attention(self.embedding_network.get_common_embedding(next_states))
            next_vals        = self.get_value(next_common_emb)

            # old log_probs => from policy
            old_log_probs, _ = self.recompute_log_probs(common_emb, actions)

            # old baseline => embed_for_baseline
            # baseline_emb     = self.embedding_network.embed_for_baseline(common_emb, actions)
            # old_baselines    = self.shared_critic.baselines(baseline_emb)

            # compute returns => GAE
            returns = self.compute_td_lambda_returns(
                rewards, old_vals, next_vals, dones, truncs
            )

        # ---------------
        #   2) Mini-batch loop
        # ---------------
        indices = np.arange(NS)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            num_batches = (NS + self.mini_batch_size - 1) // self.mini_batch_size

            policy_losses, value_losses, baseline_losses = [], [], []
            entropies, returns_list, lrs_list = [], [], []

            for b in range(num_batches):
                self.optimizer.zero_grad()

                start = b*self.mini_batch_size
                end   = min((b+1)*self.mini_batch_size, NS)
                mb_idx= indices[start:end]
                MB    = len(mb_idx)

                # slice sub-batch from states
                states_mb    = self._slice_state_dict(states, mb_idx)
                actions_mb   = actions[mb_idx]
                rets_mb      = returns[mb_idx]
                old_lp_mb    = old_log_probs[mb_idx]
                # old_bases_mb = old_baselines[mb_idx]

                # 2.1) build new embeddings
                common_mb        = self.embedding_network.apply_attention(self.embedding_network.get_common_embedding(states_mb))
                baseline_mb_emb  = self.embedding_network.embed_for_baseline(common_mb, actions_mb)

                # 2.2) new log_probs
                new_log_probs_mb, ent_mb = self.recompute_log_probs(common_mb, actions_mb)

                # new value & baseline
                new_vals_mb      = self.get_value(common_mb)
                new_baselines_mb = self.shared_critic.baselines(baseline_mb_emb)

                # advantage => returns - baseline
                # shape => (MB,NE,NA,1)
                rets_mb_xpd = rets_mb.unsqueeze(2).expand(MB, NE, NA, 1)
                adv_mb = (rets_mb_xpd - new_baselines_mb).squeeze(-1)

                if self.normalize_advantages:
                    adv_mean = adv_mb.mean()
                    adv_std  = adv_mb.std() + 1e-8
                    adv_mb   = (adv_mb - adv_mean)/adv_std

                # 2.3) compute losses
                pol_loss = self.policy_loss(
                    new_log_probs_mb,
                    old_lp_mb,
                    adv_mb,
                    self.ppo_clip_range
                )
                val_loss = F.mse_loss(new_vals_mb, rets_mb)
                base_loss= F.mse_loss(new_baselines_mb, rets_mb_xpd)
                ent_mean = ent_mb.mean()

                if self.entropy_scheduler is not None:
                    self.entropy_coeff = self.entropy_scheduler.step()

                total_loss = pol_loss \
                           + (self.value_loss_coeff * val_loss
                             + self.baseline_loss_coeff * base_loss) \
                           - self.entropy_coeff * ent_mean

                total_loss.backward()
                clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # stats
                policy_losses.append(pol_loss.detach().cpu())
                value_losses.append(val_loss.detach().cpu())
                baseline_losses.append(base_loss.detach().cpu())
                entropies.append(ent_mean.detach().cpu())
                returns_list.append(rets_mb.mean().detach().cpu())
                lrs_list.append(self.get_average_lr(self.optimizer).detach().cpu())

        return {
            "Policy Loss":    torch.stack(policy_losses).mean(),
            "Value Loss":     torch.stack(value_losses).mean(),
            "Baseline Loss":  torch.stack(baseline_losses).mean(),
            "Entropy":        torch.stack(entropies).mean(),
            "Entropy coeff" : torch.tensor(self.entropy_coeff),
            "Average Returns":torch.stack(returns_list).mean(),
            "Learning Rate":  torch.stack(lrs_list).mean(),
            "Average Rewards":rewards_mean
        }

    # ------------------------------------------------------------
    #             Aux Methods
    # ------------------------------------------------------------

    def compute_td_lambda_returns(self,
                                  rewards: torch.Tensor,
                                  values:  torch.Tensor,
                                  next_values: torch.Tensor,
                                  dones:  torch.Tensor,
                                  truncs: torch.Tensor):
        """
        Standard GAE/TD(λ). All shapes => (NS,NE,1).
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

    def get_value(self, emb: torch.Tensor):
        return self.shared_critic.values(emb)

    def policy_loss(self,
                    new_log_probs: torch.Tensor,
                    old_log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    clip_range: float):
        """
        Standard PPO clip objective
        new_log_probs => shape (MB,NE,NA)
        old_log_probs => shape (MB,NE,NA)
        advantages    => shape (MB,NE,NA)
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0-clip_range, 1.0+clip_range)
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        return -torch.mean(torch.min(obj1, obj2))

    def recompute_log_probs(self, policy_emb: torch.Tensor, actions: torch.Tensor):
        """
        Recompute log_probs & entropy with the current policy net.
        policy_emb => (MB,NE,NA,H).
        actions    => (MB,NE,NA,...).
        """
        MB, NE, NA, H = policy_emb.shape
        flat_emb = policy_emb.reshape(MB*NE*NA, H)

        if self.action_space_type == "continuous":
            mean, std = self.policy_net(flat_emb)
            dist = torch.distributions.Normal(mean, std)
            # shape => (MB,NE,NA, actDim)
            if actions.dim() == 4:
                # e.g. (MB,NE,NA, actDim)
                a_resh = actions.reshape(MB*NE*NA, actions.shape[-1])
            else:
                a_resh = actions.reshape(MB*NE*NA, -1)

            lp = dist.log_prob(a_resh).sum(dim=-1)
            ent= dist.entropy().sum(dim=-1)
            return lp.view(MB,NE,NA), ent.view(MB,NE,NA)

        else:
            # discrete or multi
            logits_list = self.policy_net(flat_emb)
            if isinstance(logits_list, list):
                # multi-discrete
                # shape => (MB,NE,NA, len(logits_list))
                a_resh = actions.reshape(MB*NE*NA, len(logits_list))

                branch_lp, branch_ent = [], []
                for i, branch_logits in enumerate(logits_list):
                    dist_b = torch.distributions.Categorical(logits=branch_logits)
                    a_b = a_resh[:, i]
                    lp_b = dist_b.log_prob(a_b)
                    ent_b= dist_b.entropy()
                    branch_lp.append(lp_b)
                    branch_ent.append(ent_b)

                log_probs = torch.stack(branch_lp, dim=-1).sum(dim=-1)
                entropy   = torch.stack(branch_ent, dim=-1).sum(dim=-1)
                return (log_probs.view(MB,NE,NA),
                        entropy.view(MB,NE,NA))

            else:
                # single-cat
                dist_c = torch.distributions.Categorical(logits=logits_list)
                a_resh = actions.reshape(MB*NE*NA)
                lp = dist_c.log_prob(a_resh)
                ent= dist_c.entropy()
                return (lp.view(MB,NE,NA),
                        ent.view(MB,NE,NA))

    def _slice_state_dict(self, states_dict: dict, indices):
        """
        Slices the dictionary along dimension=0 (the time/batch dimension).
        """
        new_dict = {}
        idx_t = torch.as_tensor(indices, device=self.device)
        for k, v in states_dict.items():
            # v => shape (NS, NE, ...) or (NS,NE,NA,...)
            new_dict[k] = v.index_select(0, idx_t).contiguous()
        return new_dict
