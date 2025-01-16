from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer
from Source.Networks import (
    MultiAgentEmbeddingNetwork,
    SharedCritic,
    ContinuousPolicyNetwork,
    DiscretePolicyNetwork
)

class MAPOCAAgent(Agent):
    """
    MA-POCA Agent that can handle discrete or continuous actions, using a PPO-style update.
    
    In line with the MA-POCA paper:
      - pi(...) uses RSA(g(obs))
      - V(...) uses RSA(g(obs))
      - Q(...) uses RSA([g(obs_j), f(obs_i, act_i)]) i != j for the baseline
    """

    def __init__(self, config: Dict, device: torch.device):
        super(MAPOCAAgent, self).__init__(config, device)

        agent_cfg = config['agent']['params']
        env_cfg   = config['environment']['shape']
        train_cfg = config['train']

        # -- Hyperparameters
        self.gamma   = agent_cfg.get('gamma', 0.99)
        self.lmbda   = agent_cfg.get('lambda', 0.95)
        self.lr      = agent_cfg.get('learning_rate', 3e-4)
        self.hidden_size = agent_cfg.get('hidden_size', 128)

        self.value_loss_coeff     = agent_cfg.get('value_loss_coeff', 0.5)
        self.baseline_loss_coeff  = agent_cfg.get('baseline_loss_coeff', 0.5)
        self.entropy_coeff        = agent_cfg.get('entropy_coeff', 0.01)
        self.max_grad_norm        = agent_cfg.get('max_grad_norm', 0.5)
        self.normalize_rewards    = agent_cfg.get('normalize_rewards', False)
        self.normalize_advantages = agent_cfg.get('normalize_advantages', False)
        self.ppo_clip_range       = agent_cfg.get('ppo_clip_range', 0.1)

        # -- (Optional) primal-dual for entropy
        self.adaptive_entropy = agent_cfg.get('adaptive_entropy', False)
        if self.adaptive_entropy:
            self.target_entropy   = agent_cfg.get('target_entropy', -1.0)
            self.entropy_lambda_lr = agent_cfg.get('entropy_lambda_lr', 3e-4)
            self.entropy_lambda  = agent_cfg.get('entropy_lambda_initial', 0.0)

        # -- Training config
        self.epochs          = train_cfg.get('epochs', 4)
        self.mini_batch_size = train_cfg.get('mini_batch_size', 128)

        # -- Environment action space info
        action_space = env_cfg['action_space']
        self.action_space_type = action_space['type']
        if self.action_space_type == 'continuous':
            self.action_dim = action_space['size']
        else:
            # discrete => single or multi
            self.action_branches = action_space.get('agent_actions', [])
            if self.action_branches:
                # multi-discrete
                self.action_dim = [b['num_choices'] for b in self.action_branches]
            else:
                # single discrete
                self.action_dim = action_space['size']

        # -- Networks
        # Embedding network: g, f, RSA
        self.embedding_network = MultiAgentEmbeddingNetwork(
            agent_cfg['networks']['MultiAgentEmbeddingNetwork']
        ).to(device)

        # Critic: has separate heads for value & baseline
        self.shared_critic = SharedCritic(config).to(device)

        # Policy net: either continuous or discrete
        if self.action_space_type == 'continuous':
            self.policy_net = ContinuousPolicyNetwork(
                **agent_cfg['networks']['policy_network']
            ).to(device)
        else:
            self.policy_net = DiscretePolicyNetwork(
                **agent_cfg['networks']['policy_network'],
                out_features=self.action_dim,
            ).to(device)

        # -- Optimizer & LR schedule
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        self.lr_scheduler = LinearLR(
            self.optimizer,
            **agent_cfg['schedulers']['lr']
        )

        # -- Optional reward normalization
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

    # ---------------------------------------------------
    # ACTION SELECTION (Policy)
    # ---------------------------------------------------
    def get_actions(self, states: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, eval: bool = False):
        """
        states: (S, E, NA, obs_dim)

        Return:
          actions => (S, E, NA, #branches or action_dim)
          log_probs => (S, E, NA)
          entropies => (S, E, NA)
        """
        with torch.no_grad():
            # 1) embeddings for policy = RSA(g(obs))
            policy_emb = self.embedding_network.embed_obs_for_policy(states)
            S, E, NA, H = policy_emb.shape

            # 2) Flatten => (S*E*NA, H)
            flat_emb = policy_emb.view(S * E * NA, H)

            # 3) pass through policy net => distribution => sample
            if self.action_space_type == 'continuous':
                mean, std = self.policy_net(flat_emb)
                dist = torch.distributions.Normal(mean, std)
                if eval:
                    actions_flat = mean
                else:
                    actions_flat = dist.sample()
                log_probs_flat = dist.log_prob(actions_flat).sum(-1)
                entropy_flat   = dist.entropy().sum(-1)

                # reshape
                act_dim = actions_flat.shape[-1]
                actions = actions_flat.view(S, E, NA, act_dim)
                log_probs = log_probs_flat.view(S, E, NA)
                entropy  = entropy_flat.view(S, E, NA)

            else:
                # discrete
                logits = self.policy_net(flat_emb)
                if isinstance(logits, list):
                    # multi-discrete
                    actions_list, lp_list, ent_list = [], [], []
                    for branch_logits in logits:
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

                    # stack
                    acts_stacked  = torch.stack(actions_list, dim=-1)
                    lp_stacked    = torch.stack(lp_list, dim=-1).sum(dim=-1)
                    ent_stacked   = torch.stack(ent_list, dim=-1).sum(dim=-1)

                    # reshape
                    nb = len(logits)
                    actions  = acts_stacked.view(S, E, NA, nb)
                    log_probs= lp_stacked.view(S, E, NA)
                    entropy  = ent_stacked.view(S, E, NA)
                else:
                    # single cat
                    dist_cat = torch.distributions.Categorical(logits=logits)
                    if eval:
                        actions_flat = logits.argmax(dim=-1)
                    else:
                        actions_flat = dist_cat.sample()
                    log_probs_flat = dist_cat.log_prob(actions_flat)
                    entropy_flat   = dist_cat.entropy()

                    actions = actions_flat.view(S, E, NA, 1)
                    log_probs = log_probs_flat.view(S, E, NA)
                    entropy  = entropy_flat.view(S, E, NA)

            return actions, (log_probs, entropy)

    # ---------------------------------------------------
    # MAIN UPDATE: PPO / MA-POCA
    # ---------------------------------------------------
    def update(self, states, next_states, actions, rewards, dones, truncs):
        """
        states, next_states: (NS, NE, NA, obs_dim)
        actions: (NS, NE, NA, ...)
        rewards: (NS, NE, 1)
        """
        NS, NE, NA, obs_dim = states.shape
        rewards_average = rewards.mean()
        if self.reward_normalizer:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        # 1) Embeddings for value => RSA(g(obs))
        with torch.no_grad():
            v_emb      = self.embedding_network.embed_obs_for_value(states)
            v_emb_next = self.embedding_network.embed_obs_for_value(next_states)
            values     = self.get_value(v_emb)
            next_vals  = self.get_value(v_emb_next)

            # compute returns
            returns    = self.compute_td_lambda_returns(rewards, values, next_vals, dones, truncs)

            # old log probs => from policy embeddings
            policy_emb = self.embedding_network.embed_obs_for_policy(states)
            old_log_probs, _ = self.recompute_log_probs(policy_emb, actions)

        # 2) Shuffle mini-batches
        indices = np.arange(NS)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            num_batches = (NS + self.mini_batch_size - 1) // self.mini_batch_size

            total_policy_losses   = []
            total_value_losses    = []
            total_baseline_losses = []
            total_entropies       = []
            total_returns         = []
            total_lrs             = []

            for b in range(num_batches):
                start = b*self.mini_batch_size
                end   = min((b+1)*self.mini_batch_size, NS)
                mb_idx= indices[start:end]
                MB    = len(mb_idx)

                # gather mini-batch data
                states_mb   = states[mb_idx]
                actions_mb  = actions[mb_idx]
                returns_mb  = returns[mb_idx]
                old_lp_mb   = old_log_probs[mb_idx]

                # policy embeddings for new log_probs
                pol_emb_mb  = self.embedding_network.embed_obs_for_policy(states_mb)
                log_probs_mb, entropy_mb = self.recompute_log_probs(pol_emb_mb, actions_mb)

                # value embeddings for new value
                val_emb_mb  = self.embedding_network.embed_obs_for_value(states_mb)
                values_mb   = self.get_value(val_emb_mb)

                # baseline embeddings => eqn(8)
                base_emb_mb = self.embedding_network.embed_obs_actions_for_baseline(states_mb, actions_mb)
                predicted_baselines = self.shared_critic.baselines(base_emb_mb)

                # compute advantages => returns - baseline
                returns_mb_expanded = returns_mb.unsqueeze(2).expand(MB, NE, NA, 1)
                adv_mb = (returns_mb_expanded - predicted_baselines).squeeze(-1)
                
                if self.normalize_advantages:
                    # Compute mean & std over the entire mini-batch
                    mean_adv = adv_mb.mean()
                    std_adv = adv_mb.std() + 1e-8

                    # Shift and scale
                    adv_mb = (adv_mb - mean_adv) / std_adv
                    
                # PPO policy loss
                mb_policy_loss = self.policy_loss(
                    log_probs=log_probs_mb,
                    old_log_probs=old_lp_mb,
                    advantages=adv_mb,
                    clip_range=self.ppo_clip_range
                )

                # value loss => MSE( V(obs), returns )
                mb_value_loss    = F.mse_loss(values_mb, returns_mb)
                # baseline loss => MSE( Q(obs, act), returns ) 
                mb_baseline_loss = F.mse_loss(predicted_baselines, returns_mb_expanded)
                mb_ent_mean      = entropy_mb.mean()

                # combine
                if self.adaptive_entropy:
                    total_loss = (
                        mb_policy_loss +
                        self.value_loss_coeff * mb_value_loss +
                        self.baseline_loss_coeff * mb_baseline_loss +
                        self.entropy_lambda * (mb_ent_mean - self.target_entropy)
                    )
                else:
                    total_loss = (
                        mb_policy_loss +
                        (self.value_loss_coeff * mb_value_loss +
                        self.baseline_loss_coeff * mb_baseline_loss) -
                        self.entropy_coeff * mb_ent_mean
                    )

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()

                # stats
                total_policy_losses.append(mb_policy_loss.detach().cpu())
                total_value_losses.append(mb_value_loss.detach().cpu())
                total_baseline_losses.append(mb_baseline_loss.detach().cpu())
                total_entropies.append(mb_ent_mean.detach().cpu())
                total_returns.append(returns_mb.mean().detach().cpu())
                total_lrs.append(self.get_average_lr(self.optimizer).detach().cpu())

                # update lambda if adaptive
                if self.adaptive_entropy:
                    with torch.no_grad():
                        ent_error = (mb_ent_mean.detach().item() - self.target_entropy)
                        self.entropy_lambda = max(0.0, self.entropy_lambda + self.entropy_lambda_lr*ent_error)

        return {
            "Policy Loss":    torch.stack(total_policy_losses).mean(),
            "Value Loss":     torch.stack(total_value_losses).mean(),
            "Baseline Loss":  torch.stack(total_baseline_losses).mean(),
            "Entropy":        torch.stack(total_entropies).mean(),
            "Average Returns":torch.stack(total_returns).mean(),
            "Learning Rate":  torch.stack(total_lrs).mean(),
            "Average Rewards":rewards_average
        }

    # ---------------------------------------------------
    # AUXILIARY UTILS
    # ---------------------------------------------------
    def compute_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        """
        Standard GAE/TD(λ)-style return computation. 
        """
        NS, NE, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(NE,1, device=rewards.device)

        for t in reversed(range(NS)):
            if t == NS-1:
                next_val = next_values[t]*(1 - dones[t])*(1 - truncs[t])
            else:
                next_val = values[t+1]*(1 - dones[t+1])*(1 - truncs[t+1])
            delta = rewards[t] + self.gamma*next_val - values[t]
            gae   = delta + self.gamma*self.lmbda*(1 - dones[t])*(1 - truncs[t])*gae
            returns[t] = gae + values[t]
        return returns

    def get_value(self, embeddings: torch.Tensor):
        return self.shared_critic.values(embeddings)

    def policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                    advantages: torch.Tensor, clip_range: float) -> torch.Tensor:
        """
        PPO clip objective
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        return -torch.mean(torch.min(obj1, obj2))

    def recompute_log_probs(self, embeddings, actions):
        """
        Recompute log_probs (and entropies) with the current policy net, 
        given 'policy embeddings' = RSA(g(obs)).
        """
        S, E, NA, H = embeddings.shape
        flat_emb = embeddings.view(S*E*NA, H)

        if self.action_space_type == 'continuous':
            mean, std = self.policy_net(flat_emb)
            dist = torch.distributions.Normal(mean, std)
            if actions.dim() == 4:
                # (S,E,NA,action_dim)
                act_reshaped = actions.view(S*E*NA, actions.shape[-1])
            else:
                act_reshaped = actions.view(S*E*NA, -1)

            log_probs = dist.log_prob(act_reshaped).sum(-1)
            entropy   = dist.entropy().sum(-1)
            return log_probs.view(S,E,NA), entropy.view(S,E,NA)

        # discrete
        logits_list = self.policy_net(flat_emb)
        if isinstance(logits_list, list):
            # multi-discrete
            a_reshaped = actions.view(S*E*NA, len(logits_list))
            branch_lp, branch_ent = [], []
            for b_idx, branch_logits in enumerate(logits_list):
                dist_b = torch.distributions.Categorical(logits=branch_logits)
                a_b = a_reshaped[:, b_idx]
                lp_b = dist_b.log_prob(a_b)
                ent_b= dist_b.entropy()
                branch_lp.append(lp_b)
                branch_ent.append(ent_b)
            log_probs = torch.stack(branch_lp, dim=-1).sum(-1)
            entropy   = torch.stack(branch_ent, dim=-1).sum(-1)
            return log_probs.view(S,E,NA), entropy.view(S,E,NA)
        else:
            # single-cat
            dist_c = torch.distributions.Categorical(logits=logits_list)
            a_reshaped = actions.view(S*E*NA)
            lp = dist_c.log_prob(a_reshaped)
            ent= dist_c.entropy()
            return lp.view(S,E,NA), ent.view(S,E,NA)
