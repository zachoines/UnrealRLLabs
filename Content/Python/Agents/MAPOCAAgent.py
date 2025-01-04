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
    MA-POCA Agent that can handle discrete or continuous actions, using PPO-style updates.
    """

    def __init__(self, config: Dict, device: torch.device):
        super(MAPOCAAgent, self).__init__(config, device)

        agent_cfg = config['agent']['params']
        env_cfg = config['environment']['shape']
        train_cfg = config['train']

        # Agent params
        self.gamma = agent_cfg.get('gamma', 0.99)
        self.lmbda = agent_cfg.get('lambda', 0.95)
        self.lr = agent_cfg.get('learning_rate', 3e-4)
        self.hidden_size = agent_cfg.get('hidden_size', 128)
        self.value_loss_coeff = agent_cfg.get('value_loss_coeff', 0.5)
        self.baseline_loss_coeff = agent_cfg.get('baseline_loss_coeff', 0.5)
        self.entropy_coeff = agent_cfg.get('entropy_coeff', 0.01)
        self.max_grad_norm = agent_cfg.get('max_grad_norm', 0.5)
        self.normalize_rewards = agent_cfg.get('normalize_rewards', False)
        self.ppo_clip_range = agent_cfg.get('ppo_clip_range', 0.1)

        # Adaptive Entropy (primal-dual)
        self.adaptive_entropy = agent_cfg.get('adaptive_entropy', False)
        if self.adaptive_entropy:
            self.target_entropy = agent_cfg.get('target_entropy', -1.0)
            self.entropy_lambda_lr = agent_cfg.get('entropy_lambda_lr', 3e-4)
            self.entropy_lambda = agent_cfg.get('entropy_lambda_initial', 0.0)

        # Train params
        self.epochs = train_cfg.get('epochs', 4)
        self.mini_batch_size = train_cfg.get('mini_batch_size', 128)

        # Env info
        action_space = env_cfg['action_space']
        self.action_space_type = action_space['type']
        if self.action_space_type == 'continuous':
            self.action_dim = action_space['size']
        else:
            # multi-discrete or single discrete
            self.action_branches = action_space.get('agent_actions', [])
            if self.action_branches:
                # multi-discrete
                self.action_dim = [b['num_choices'] for b in self.action_branches]
            else:
                # single discrete
                self.action_dim = action_space['size']

        # Embedding & Critic
        self.embedding_network = MultiAgentEmbeddingNetwork(
            agent_cfg['networks']['MultiAgentEmbeddingNetwork']
        ).to(device)
        self.shared_critic = SharedCritic(config).to(device)

        # Policy
        if self.action_space_type == 'continuous':
            self.policy_net = ContinuousPolicyNetwork(
                **agent_cfg['networks']['policy_network']
            ).to(device)
        else:
            # discrete
            self.policy_net = DiscretePolicyNetwork(
                **agent_cfg['networks']['policy_network'],
                out_features=self.action_dim,
            ).to(device)

        # Optimizer, Scheduler
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        self.lr_scheduler = LinearLR(
            self.optimizer,
            **agent_cfg['schedulers']['lr']
        )

        # Optional reward normalization
        if self.normalize_rewards:
            self.reward_normalizer = RunningMeanStdNormalizer(device=device)
        else:
            self.reward_normalizer = None

    def parameters(self):
        return (
            list(self.embedding_network.parameters()) +
            list(self.shared_critic.parameters()) +
            list(self.policy_net.parameters())
        )

    def get_actions(self, states: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, eval: bool = False):
        """
        Produce an action for each agent in each environment for all steps (S).
        
        states: (S, E, NA, obs_dim)
        dones, truncs: (S, E, NA or something similar) -- typically only used 
                    for memory-based or LSTM-based policies, 
                    but shown here for completeness.
        eval: Boolean. If True, use a deterministic action (mean or argmax).
        
        returns: 
        actions => (S, E, NA, #action_branches or action_dim)
        (log_probs => (S, E, NA), entropies => (S, E, NA))
        """
        with torch.no_grad():
            # 1) Embeddings => shape (S, E, NA, H)
            embeddings = self.embedding_network(states)
            S, E, NA, H = embeddings.shape

            # 2) Flatten out (S, E, NA) => (S*E*NA, H)
            flat = embeddings.view(S * E * NA, H)

            # 3) Dist depends on continuous or discrete
            if self.action_space_type == 'continuous':
                mean, std = self.policy_net(flat)
                dist = torch.distributions.Normal(mean, std)
                if eval:
                    actions_flat = mean  # shape => (S*E*NA, action_dim)
                else:
                    actions_flat = dist.sample()
                log_probs_flat = dist.log_prob(actions_flat).sum(-1)  # => (S*E*NA)
                entropy_flat = dist.entropy().sum(-1)                # => (S*E*NA)

                # reshape back => (S, E, NA, ...)
                action_dim = actions_flat.shape[-1]
                actions = actions_flat.view(S, E, NA, action_dim)
                log_probs = log_probs_flat.view(S, E, NA)
                entropy = entropy_flat.view(S, E, NA)

            else:
                # discrete: single or multi
                logits = self.policy_net(flat)  
                if isinstance(logits, list):
                    # multi-discrete => list[Tensor], each => (S*E*NA, branch_dim)
                    actions_list = []
                    log_probs_list = []
                    entropy_list = []
                    for branch_logits in logits:
                        d_b = torch.distributions.Categorical(logits=branch_logits)
                        if eval:
                            a_b = branch_logits.argmax(dim=-1)  # (S*E*NA)
                        else:
                            a_b = d_b.sample()
                        lp_b = d_b.log_prob(a_b)  # (S*E*NA)
                        ent_b = d_b.entropy()     # (S*E*NA)
                        actions_list.append(a_b)
                        log_probs_list.append(lp_b)
                        entropy_list.append(ent_b)

                    # stack => shape (S*E*NA, #branches)
                    # then reshape => (S, E, NA, #branches)
                    actions_stacked = torch.stack(actions_list, dim=-1)
                    log_probs_stacked = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
                    entropy_stacked = torch.stack(entropy_list, dim=-1).sum(dim=-1)

                    # reshape
                    num_branches = len(logits)
                    actions = actions_stacked.view(S, E, NA, num_branches)
                    log_probs = log_probs_stacked.view(S, E, NA)
                    entropy = entropy_stacked.view(S, E, NA)
                else:
                    # single cat => shape => (S*E*NA, #acts)
                    dist_cat = torch.distributions.Categorical(logits=logits)
                    if eval:
                        actions_flat = logits.argmax(dim=-1)  # => (S*E*NA)
                    else:
                        actions_flat = dist_cat.sample()      # => (S*E*NA)
                    log_probs_flat = dist_cat.log_prob(actions_flat)   # => (S*E*NA)
                    entropy_flat = dist_cat.entropy()                  # => (S*E*NA)

                    # reshape => (S,E,NA,...)
                    actions = actions_flat.view(S, E, NA, 1)
                    log_probs = log_probs_flat.view(S, E, NA)
                    entropy = entropy_flat.view(S, E, NA)

            return actions, (log_probs, entropy)

    def update(self, states, next_states, actions, rewards, dones, truncs):
        """
        Perform PPO updates for MA-POCA
        """
        NS, NE, NA, _ = states.shape
        if self.reward_normalizer:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        with torch.no_grad():
            states_emb = self.embedding_network(states)
            next_states_emb = self.embedding_network(next_states)
            values = self.get_value(states_emb)
            next_values = self.get_value(next_states_emb)
            returns = self.compute_td_lambda_returns(rewards, values, next_values, dones, truncs)
            old_log_probs, _ = self.recompute_log_probs(states_emb, actions)

        # Shuffle mini-batches
        indices = np.arange(NS)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            num_batches = (NS + self.mini_batch_size - 1) // self.mini_batch_size

            total_policy_losses = []
            total_value_losses = []
            total_baseline_losses = []
            total_entropies = []
            total_returns = []
            total_lrs = []

            for b in range(num_batches):
                start = b*self.mini_batch_size
                end = min((b+1)*self.mini_batch_size, NS)
                mb_idx = indices[start:end]
                MB = len(mb_idx)

                # gather mini-batch data
                states_mb = states[mb_idx]
                actions_mb = actions[mb_idx]
                returns_mb = returns[mb_idx]
                old_log_probs_mb = old_log_probs[mb_idx]

                # forward pass
                embeddings_mb = self.embedding_network(states_mb)
                baseline_inputs = self.get_baseline_inputs(embeddings_mb, actions_mb)
                predicted_baselines = self.shared_critic.baselines(baseline_inputs)
                values_mb = self.get_value(embeddings_mb)
                log_probs_mb, entropy_mb = self.recompute_log_probs(embeddings_mb, actions_mb)

                # shape => (MB, NE, NA, 1)
                returns_mb_expanded = returns_mb.unsqueeze(2).expand(MB, NE, NA, 1)
                adv_mb = returns_mb_expanded - predicted_baselines
                adv_mb = adv_mb.squeeze(-1)  # (MB,NE,NA)

                # [CHANGED] separate method for policy loss
                mb_policy_loss = self.policy_loss(
                    log_probs=log_probs_mb,
                    old_log_probs=old_log_probs_mb,
                    advantages=adv_mb,
                    clip_range=self.ppo_clip_range
                )

                mb_value_loss = F.mse_loss(values_mb, returns_mb)
                mb_baseline_loss = F.mse_loss(predicted_baselines, returns_mb_expanded)
                mb_ent_mean = entropy_mb.mean()

                if self.adaptive_entropy:
                    # L = policy + value + baseline + λ*(H - targetH)
                    total_loss = (mb_policy_loss +
                                  self.value_loss_coeff*mb_value_loss +
                                  mb_baseline_loss +
                                  self.entropy_lambda*(mb_ent_mean - self.target_entropy))
                else:
                    total_loss = mb_policy_loss + self.value_loss_coeff*(mb_value_loss) + self.baseline_loss_coeff*(mb_baseline_loss) - (self.entropy_coeff*mb_ent_mean)

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()

                # track stats
                total_policy_losses.append(mb_policy_loss.detach().cpu())
                total_value_losses.append(mb_value_loss.detach().cpu())
                total_baseline_losses.append(mb_baseline_loss.detach().cpu())
                total_entropies.append(mb_ent_mean.detach().cpu())
                total_returns.append(returns_mb.mean().detach().cpu())
                total_lrs.append(self.get_average_lr(self.optimizer).detach().cpu())

                # if adaptive entropy => update lambda
                if self.adaptive_entropy:
                    with torch.no_grad():
                        ent_error = (mb_ent_mean.detach().item() - self.target_entropy)
                        self.entropy_lambda = max(0.0, self.entropy_lambda + self.entropy_lambda_lr*ent_error)

        return {
            "Policy Loss": torch.stack(total_policy_losses).mean(),
            "Value Loss": torch.stack(total_value_losses).mean(),
            "Baseline Loss": torch.stack(total_baseline_losses).mean(),
            "Entropy": torch.stack(total_entropies).mean(),
            "Average Returns": torch.stack(total_returns).mean(),
            "Average Rewards": rewards.mean(),
            "Learning Rate": torch.stack(total_lrs).mean(),
        }

    def policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                    advantages: torch.Tensor, clip_range: float) -> torch.Tensor:
        """
        Compute PPO clip objective for policy, for both discrete or continuous.
        log_probs, old_log_probs => shape (MB,NE,NA)
        advantages => shape (MB,NE,NA)
        clip_range => e.g. 0.1 or 0.2
        """
        ratio = torch.exp(log_probs - old_log_probs)  # shape => (MB,NE,NA)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.min(obj1, obj2))  # scalar
        return policy_loss

    # ---------------------------------------------------------------------
    #                     Others remain the same
    # ---------------------------------------------------------------------
    def compute_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        NS, NE, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(NE,1, device=rewards.device)

        for t in reversed(range(NS)):
            if t == NS-1:
                next_val = next_values[t]*(1 - dones[t])*(1 - truncs[t])
            else:
                next_val = values[t+1]*(1 - dones[t+1])*(1 - truncs[t+1])
            delta = rewards[t] + self.gamma*next_val - values[t]
            gae = delta + self.gamma*self.lmbda*(1 - dones[t])*(1 - truncs[t])*gae
            returns[t] = gae + values[t]
        return returns

    def get_value(self, embeddings: torch.Tensor):
        return self.shared_critic.values(embeddings)

    def get_baseline_inputs(self, agent_embeddings, actions):
        agent_embeds, groupmates_embeds, groupmates_actions = \
            self.embedding_network.split_agent_obs_groupmates_obs_actions(agent_embeddings, actions)
        groupmates_embeds_with_actions = self.embedding_network.encode_groupmates_obs_actions(
            groupmates_embeds, groupmates_actions
        )
        x = torch.cat([agent_embeds, groupmates_embeds_with_actions], dim=3).contiguous()
        return x

    def recompute_log_probs(self, embeddings, actions):
        S, E, NA, H = embeddings.shape
        flat = embeddings.view(S*E, NA, H)

        # If continuous
        if self.action_space_type == 'continuous':
            mean, std = self.policy_net(flat)
            dist = torch.distributions.Normal(mean, std)
            if actions.dim() == 4:
                # shape => (S,E,NA,action_dim)
                a_reshaped = actions.view(S*E, NA, actions.shape[-1])
            else:
                a_reshaped = actions.view(S*E, NA, -1)
            log_probs = dist.log_prob(a_reshaped).sum(-1)
            entropy = dist.entropy().sum(-1)
            log_probs = log_probs.view(S, E, NA)
            entropy = entropy.view(S, E, NA)
            return log_probs, entropy

        # If discrete (single or multi)
        logits_list = self.policy_net(flat)  # either (B,#acts) or list
        if isinstance(logits_list, list):
            # multi-discrete
            # shape => (S,E,NA,#branches)
            a_reshaped = actions.view(S*E, NA, len(logits_list))
            branch_log_probs = []
            branch_ent = []
            for b_idx, branch_logits in enumerate(logits_list):
                dist_b = torch.distributions.Categorical(logits=branch_logits)
                a_b = a_reshaped[:, :, b_idx]
                lp_b = dist_b.log_prob(a_b)
                ent_b = dist_b.entropy()
                branch_log_probs.append(lp_b)
                branch_ent.append(ent_b)
            # sum over branches
            log_probs = torch.stack(branch_log_probs, dim=-1).sum(dim=-1)
            entropy = torch.stack(branch_ent, dim=-1).sum(dim=-1)
            log_probs = log_probs.view(S,E,NA)
            entropy = entropy.view(S,E,NA)
            return log_probs, entropy
        else:
            # single cat => shape => (B,#acts)
            dist_c = torch.distributions.Categorical(logits=logits_list)
            # actions => (S,E,NA,1) possibly
            a_reshaped = actions.view(S*E, NA)
            lp = dist_c.log_prob(a_reshaped)
            ent = dist_c.entropy()
            lp = lp.view(S, E, NA)
            ent = ent.view(S, E, NA)
            return lp, ent
