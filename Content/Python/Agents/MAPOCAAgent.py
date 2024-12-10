from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer
from Source.Networks import MultiAgentEmbeddingNetwork, SharedCritic, ContinuousPolicyNetwork

class MAPOCAAgent(Agent):
    """
    MA-POCA Agent for Continuous Actions
    
    Following the MA-POCA paper (equations 5-8), we approximate Q^\pi(s,a) with y^(λ) (TD(λ) returns),
    and train:
    - The policy (Eq.5) using Adv_i = y^(λ) - b_i(s,a^{-i})
    - The value function (Eqs.6&7) with MSE(V(s), y^(λ))
    - The baseline function (Eq.8) with MSE(Q_ψ(...), y^(λ))

    The integrals and exact expectations from equations 1-3 are not directly computed.
    Instead, we rely on standard RL bootstrapping techniques and TD(λ) returns as stable training targets.

    We handle multi-agent embeddings via MultiAgentEmbeddingNetwork and RSA attention,
    and compute TD(λ) returns accounting for terminals and truncated episodes.
    """

    def __init__(self, config: Dict, device: torch.device):
        super(MAPOCAAgent, self).__init__(config, device)

        agent_cfg = config['agent']['params']
        env_cfg = config['environment']['shape']
        train_cfg = config['train']

        self.gamma = agent_cfg.get('gamma', 0.99)
        self.lmbda = agent_cfg.get('lambda', 0.95)
        self.lr = agent_cfg.get('learning_rate', 3e-4)
        self.hidden_size = agent_cfg.get('hidden_size', 128)
        self.value_loss_coeff = agent_cfg.get('value_loss_coeff', 0.5)
        self.entropy_coeff = agent_cfg.get('entropy_coeff', 0.01)
        self.max_grad_norm = agent_cfg.get('max_grad_norm', 0.5)
        self.normalize_rewards = agent_cfg.get('normalize_rewards', False)

        self.single_agent_obs_size = env_cfg['single_agent_obs_size']
        self.max_agents = env_cfg['max_agents']
        action_space = env_cfg['action_space']
        assert action_space['type'] == 'continuous', "We assume continuous actions"
        self.action_dim = action_space['size']

        # Initialize networks
        self.embedding_network = MultiAgentEmbeddingNetwork(config).to(self.device)
        self.shared_critic = SharedCritic(config).to(self.device)
        self.policy_net = ContinuousPolicyNetwork(
            in_features=self.hidden_size,
            out_features=self.action_dim,
            hidden_size=self.hidden_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if self.normalize_rewards:
            self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        else:
            self.reward_normalizer = None

        self.mini_batch_size = train_cfg.get('mini_batch_size', 128)

    def parameters(self):
        return list(self.embedding_network.parameters()) + \
               list(self.shared_critic.parameters()) + \
               list(self.policy_net.parameters())

    def get_actions(self, states: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, eval: bool = False):
        """
        states: (1, NE, NA, obs_dim)
        1) Embed states -> (1,NE,NA,H)
        2) Policy net -> mean,std -> sample continuous actions
        """
        with torch.no_grad():
            embeddings = self.embedding_network(states)  # (1,NE,NA,H)
            B,NE,NA,H = embeddings.shape  # B=1
            flat = embeddings.view(NE*NA,H)
            mean, std = self.policy_net(flat)
            dist = torch.distributions.Normal(mean, std)
            if eval:
                actions = mean
            else:
                actions = dist.sample() # (NE*NA,action_dim)
            log_probs = dist.log_prob(actions).sum(-1)  # (NE*NA)
            entropy = dist.entropy().sum(-1)  # (NE*NA)

            actions = actions.view(NE,NA,self.action_dim)
            log_probs = log_probs.view(NE,NA)
            entropy = entropy.view(NE,NA)
            return actions, (log_probs, entropy)

    def update(self, states, next_states, actions, rewards, dones, truncs):
        """
        states: (NS,NE,NA,obs_dim)
        next_states: (NS,NE,NA,obs_dim)
        actions: (NS,NE,NA,action_dim)
        rewards: (NS,NE,1)
        dones,truncs: (NS,NE,1)

        Steps:
        1) Compute y^(λ) returns as targets.
        2) Embed states to get values and baselines inputs.
        3) Compute predicted baselines, values.
        4) Compute advantages and log_probs.
        5) Mini-batch update combining policy, value, baseline losses.
        """
        NS,NE,NA,obs_dim = states.shape

        # Normalize rewards if needed
        if self.reward_normalizer:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        # Compute embeddings for value bootstrap
        with torch.no_grad():
            states_emb = self.embedding_network(states)         # (NS,NE,NA,H)
            next_states_emb = self.embedding_network(next_states)
            values = self.get_value(states_emb)                 # (NS,NE,1)
            next_values = self.get_value(next_states_emb)

        # Compute y^(λ) returns:
        returns = self.compute_td_lambda_returns(rewards, values, next_values, dones, truncs)
        # returns: (NS,NE,1)

        with torch.no_grad():
            # Baseline inputs: exclude agent's own action
            agent_obs, groupmates_obs, groupmates_actions = self.embedding_network.split_agent_obs_groupmates_obs_actions(states, actions)
            baseline_inputs = self.get_baseline_inputs(states_emb, groupmates_actions)
            predicted_baselines = self.shared_critic.baselines(baseline_inputs) # (NS,NE,NA,1)

            log_probs, entropy = self.recompute_log_probs(states_emb, actions)

        # Advantages: Adv_i = y^(λ) - b_i(s,a^{-i})
        adv = (returns.unsqueeze(2) - predicted_baselines).squeeze(-1) # (NS,NE,NA)

        # For mini-batch updates:
        NSNE = NS*NE
        B_total = NSNE*NA
        indices = np.arange(B_total)
        np.random.shuffle(indices)

        # Flatten relevant tensors:
        log_probs_vec = log_probs.view(B_total)
        adv_vec = adv.view(B_total)
        returns_vec = returns.repeat_interleave(NA, dim=2).view(B_total,1)
        values_vec = values.repeat_interleave(NA, dim=2).view(B_total,1)
        baseline_vec = predicted_baselines.view(B_total,1)
        ent_vec = entropy.view(B_total)

        # Equation references as comments inside mini-batch loop:
        total_policy_losses = []
        total_value_losses = []
        total_baseline_losses = []
        total_entropies = []
        total_returns = []

        for start in range(0, B_total, self.mini_batch_size):
            end = min(start+self.mini_batch_size, B_total)
            mb_idx = indices[start:end]

            mb_log_probs = log_probs_vec[mb_idx]
            mb_adv = adv_vec[mb_idx]
            mb_returns = returns_vec[mb_idx]
            mb_values = values_vec[mb_idx]
            mb_baselines = baseline_vec[mb_idx]
            mb_ent = ent_vec[mb_idx]

            # Baseline Loss (Eq.8): L_baseline = MSE(Q_ψ(...) - y^(λ))
            mb_baseline_loss = F.mse_loss(mb_baselines, mb_returns)

            # Value Loss (Eq.6 & 7): L_value = MSE(V(s), y^(λ))
            mb_value_loss = F.mse_loss(mb_values, mb_returns)

            # Policy Loss (Eq.5): L_policy = -E[logπ(a|s)*Adv]
            mb_policy_loss = -(mb_log_probs * mb_adv).mean()

            mb_ent_mean = mb_ent.mean()

            # Combined Loss:
            # L = L_policy + value_loss_coeff*L_value + L_baseline - entropy_coeff*Entropy
            # Entropy is included as negative to promote exploration
            mb_loss = mb_policy_loss + self.value_loss_coeff*mb_value_loss + mb_baseline_loss - self.entropy_coeff*mb_ent_mean

            self.optimizer.zero_grad()
            mb_loss.backward()
            clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_losses.append(mb_policy_loss.detach().cpu())
            total_value_losses.append(mb_value_loss.detach().cpu())
            total_baseline_losses.append(mb_baseline_loss.detach().cpu())
            total_entropies.append(mb_ent_mean.detach().cpu())
            total_returns.append(mb_returns.mean().detach().cpu())

        logs = {
            "Policy Loss": torch.stack(total_policy_losses).mean(),
            "Value Loss": torch.stack(total_value_losses).mean(),
            "Baseline Loss": torch.stack(total_baseline_losses).mean(),
            "Entropy": torch.stack(total_entropies).mean(),
            "Average Returns": torch.stack(total_returns).mean()
        }

        return logs

    def compute_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        """
        Compute y^(λ) returns using a TD(λ)-like approach (similar to GAE).
        Handle terminals by zeroing out next_val when episode ends.

        Eq.6 & 7 use y^(λ) as target.
        """
        NS,NE,_ = rewards.shape
        returns = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards)

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
        # Compute V(s) from embeddings:
        return self.shared_critic.values(embeddings)

    def get_baseline_inputs(self, agent_embeddings, groupmates_actions):
        """
        Construct baseline inputs by excluding agent's own action:
        Eq.8: Q_ψ(...) approximates baseline from scenario where only groupmates' info is present.

        final shape: (NS,NE,NA,NA,H)
        """
        agent_embeds, groupmates_embeds = self.embedding_network.split_agent_groupmates_obs(agent_embeddings)
        groupmates_embeds_with_actions = self.embedding_network.encode_groupmates_obs_actions(groupmates_embeds, groupmates_actions)
        x = torch.cat([agent_embeds, groupmates_embeds_with_actions], dim=3).contiguous()
        return x

    def recompute_log_probs(self, embeddings, actions):
        """
        Recompute log_probs and entropy for actions taken:
        Needed for policy gradient (Eq.5).
        """
        NS,NE,NA,H = embeddings.shape
        flat = embeddings.view(NS*NE*NA,H)
        mean, std = self.policy_net(flat)
        dist = torch.distributions.Normal(mean, std)
        A = actions.view(NS*NE*NA, self.action_dim)
        log_probs = dist.log_prob(A).sum(-1).view(NS,NE,NA)
        entropy = dist.entropy().sum(-1).view(NS,NE,NA)
        return log_probs, entropy
