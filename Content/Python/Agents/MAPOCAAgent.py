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

    Equations 5-8 from MA-POCA:
    - Policy (Eq.5): L_policy = -E[logπ(a|s)*Adv]
    - Value (Eq.6,7): L_value = MSE(V(s), y^(λ))
    - Baseline (Eq.8): L_baseline = MSE(Q_ψ(...), y^(λ))

    y^(λ): TD(λ) returns approximating Q^\pi(s,a).
    Adv_i = y^(λ) - b_i(s,a^{-i})

    We recompute forward passes during mini-batch updates so that gradients can flow.
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
            actions = mean if eval else dist.sample() # (NE*NA,action_dim)
            log_probs = dist.log_prob(actions).sum(-1)  # (NE*NA)
            entropy = dist.entropy().sum(-1)  # (NE*NA)

            actions = actions.view(NE,NA,self.action_dim)
            log_probs = log_probs.view(NE,NA)
            entropy = entropy.view(NE,NA)
            return actions, (log_probs, entropy)

    def update(self, states, next_states, actions, rewards, dones, truncs):
        """
        Perform an update step after collecting trajectories.

        Shapes:
        states, next_states: (NS,NE,NA,obs_dim)
        actions: (NS,NE,NA,action_dim)
        rewards, dones, truncs: (NS,NE,1)
        NS = number of steps collected, NE = number of environments, NA = number of agents.

        Steps:
        1) Compute y^(λ) returns as targets (no_grad) using TD(λ)-like method.
        2) Shuffle the step indices and split into mini-batches of steps.
        3) For each mini-batch of steps:
        - Extract states_mb, actions_mb, returns_mb
        - Run forward passes (with grad enabled) to get values, baselines, log_probs, entropy
        - Compute advantages: Adv_i = returns - baselines
        - Compute losses (policy, value, baseline, entropy)
        - Backprop and update parameters.
        """

        NS,NE,NA,obs_dim = states.shape

        # Normalize rewards if needed
        if self.reward_normalizer:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        # Compute embeddings and values for TD(λ) returns calculation:
        with torch.no_grad():
            states_emb = self.embedding_network(states)   # (NS,NE,NA,H)
            next_states_emb = self.embedding_network(next_states)
            values = self.get_value(states_emb)           # (NS,NE,1)
            next_values = self.get_value(next_states_emb)
            returns = self.compute_td_lambda_returns(rewards, values, next_values, dones, truncs) # (NS,NE,1)

        indices = np.arange(NS)
        np.random.shuffle(indices)

        # Determine the number of mini-batches:
        num_batches = (NS + self.mini_batch_size - 1) // self.mini_batch_size

        total_policy_losses = []
        total_value_losses = []
        total_baseline_losses = []
        total_entropies = []
        total_returns = []

        # Mini-batch loop: we pick subsets of steps. Each mini-batch has shape (MB,NE,NA,...)
        for b in range(num_batches):
            start = b * self.mini_batch_size
            end = min((b+1)*self.mini_batch_size, NS)
            mb_idx = indices[start:end]
            MB = len(mb_idx)

            # Extract mini-batch steps:
            states_mb = states[mb_idx]       # (MB,NE,NA,obs_dim)
            actions_mb = actions[mb_idx]     # (MB,NE,NA,action_dim)
            returns_mb = returns[mb_idx]     # (MB,NE, 1)

            # Forward passes with grad:
            # Embeddings
            embeddings_mb = self.embedding_network(states_mb)  # (MB,NE,NA,H)

            # Baseliness
            baseline_inputs = self.get_baseline_inputs(embeddings_mb, actions_mb)
            predicted_baselines = self.shared_critic.baselines(baseline_inputs) # (MB,NE,NA,1)

            # Values
            values_mb = self.get_value(embeddings_mb)  # (MB,NE,1)

            # Policy log_probs and entropy:
            log_probs_mb, entropy_mb = self.recompute_log_probs(embeddings_mb, actions_mb) # (MB,NE,NA)

            # Compute advantages:
            # Unsqueeze returns to (MB,NE,1,1) so it broadcasts over NA dimension (where Adv_i = returns_mb - baselines)
            adv_mb = returns_mb.unsqueeze(2) - predicted_baselines  # (MB,NE,NA,1)
            adv_mb = adv_mb.squeeze(-1) # (MB,NE,NA)

            # Compute losses:
            # Baseline Loss (Eq.8): MSE(baseline, returns)
            mb_baseline_loss = F.mse_loss(
                predicted_baselines,
                returns_mb.unsqueeze(2).repeat_interleave(NA, dim=2) # Repeated on the agent dimention
            )

            # Value Loss (Eq.6 & 7): MSE(V(s), returns)
            mb_value_loss = F.mse_loss(values_mb, returns_mb)

            # Policy Loss (Eq.5): L_policy = -E[logπ(a|s)*Adv]
            mb_policy_loss = -(log_probs_mb * adv_mb.detach()).mean()

            # Entropy mean:
            mb_ent_mean = entropy_mb.mean()

            # Combined weighted losses:
            mb_loss = mb_policy_loss + self.value_loss_coeff*mb_value_loss + mb_baseline_loss - self.entropy_coeff*mb_ent_mean

            self.optimizer.zero_grad()
            mb_loss.backward()
            clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_losses.append(mb_policy_loss.detach().cpu())
            total_value_losses.append(mb_value_loss.detach().cpu())
            total_baseline_losses.append(mb_baseline_loss.detach().cpu())
            total_entropies.append(mb_ent_mean.detach().cpu())
            total_returns.append(returns_mb.mean().detach().cpu())

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
        
        rewards, values, next_values, dones, truncs: (NS,NE,1)
        returns: (NS,NE,1)

        We'll do a backward pass:
        gae is (NE,1), updated at each step t:
            delta_t = r_t + gamma * next_val - v_t
            gae = delta_t + gamma*lambda*(1 - done_t)*(1 - trunc_t)*gae
            returns[t] = gae + values[t]
        """
        NS,NE,_ = rewards.shape
        returns = torch.zeros_like(rewards)  # (NS,NE,1)
        gae = torch.zeros(NE,1, device=rewards.device) # (NE,1)

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

    def get_baseline_inputs(self, agent_embeddings, actions):
        """
        Construct baseline inputs by excluding agent's own action:
        final shape: (NS,NE,NA,NA,H)

        We must implement a similar split as before but watch dimension usage carefully.
        Note: We must do a similar trick as in update mini-batch step if needed.
        """
        # NOTE: In the final approach above, we handle baseline input construction in mini-batch loop directly.
        # Here we keep the same function for consistency. We assume agent_embeddings=(S,E,NA,H) 
        # and actions=(S,E,NA,action_dim).
        agent_embeds, groupmates_embeds, groupmates_actions = self.embedding_network.split_agent_obs_groupmates_obs_actions(agent_embeddings, actions)
        groupmates_embeds_with_actions = self.embedding_network.encode_groupmates_obs_actions(groupmates_embeds, groupmates_actions)
        x = torch.cat([agent_embeds, groupmates_embeds_with_actions], dim=3).contiguous()
        return x

    def recompute_log_probs(self, embeddings, actions):
        """
        Recompute log_probs and entropy for actions taken:
        embeddings: (S,E,NA,H)
        actions: (S,E,NA,action_dim)
        """
        S,E,NA,H = embeddings.shape
        flat = embeddings.view(S*E*NA,H)
        mean, std = self.policy_net(flat)
        dist = torch.distributions.Normal(mean, std)
        A = actions.view(S*E*NA, self.action_dim)
        log_probs = dist.log_prob(A).sum(-1).view(S,E,NA)
        entropy = dist.entropy().sum(-1).view(S,E,NA)
        return log_probs, entropy
