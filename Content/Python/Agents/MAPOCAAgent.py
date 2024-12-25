from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR

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

    Returns Calculation:
    - y^(λ) approximates Q^\pi(s,a) via TD(λ) returns.
    - Adv_i = y^(λ) - b_i(s,a^{-i})

    PPO objective:
    - L_policy = E[min(r_t * Adv, clip(r_t, 1-ε, 1+ε)*Adv)]

    Using primal-dual for entropy constraint:
    - target_entropy specified
    - entropy_lambda updated via dual ascent
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
        self.entropy_coeff = agent_cfg.get('entropy_coeff', 0.01)  # used if not adaptive
        self.max_grad_norm = agent_cfg.get('max_grad_norm', 0.5)
        self.normalize_rewards = agent_cfg.get('normalize_rewards', False)
        self.ppo_clip_range = agent_cfg.get('ppo_clip_range', 0.1)

        # Adaptive entropy via primal-dual
        self.adaptive_entropy = agent_cfg.get('adaptive_entropy', False)
        if self.adaptive_entropy:
            self.target_entropy = agent_cfg.get('target_entropy', -env_cfg['action_space']['size'])
            self.entropy_lambda_lr = agent_cfg.get('entropy_lambda_lr', 3e-4)
            self.entropy_lambda = agent_cfg.get('entropy_lambda_initial', 0.0)

        # Train params
        self.epochs = train_cfg.get('epochs', 4)
        self.mini_batch_size = train_cfg.get('mini_batch_size', 128)

        # Env info
        action_space = env_cfg['action_space']
        assert action_space['type'] == 'continuous', "We assume continuous actions"
        self.action_dim = action_space['size']

        # Initialize networks
        self.embedding_network = MultiAgentEmbeddingNetwork(config).to(self.device)
        self.shared_critic = SharedCritic(config).to(self.device)
        self.policy_net = ContinuousPolicyNetwork(
            **agent_cfg['networks']['policy_network']
        ).to(self.device)

        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay = 1e-3)
        self.rl_scheduler = ExponentialLR(self.optimizer, gamma=0.999)

        if self.normalize_rewards:
            self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        else:
            self.reward_normalizer = None

    def parameters(self):
        base_params = list(self.embedding_network.parameters()) + \
                      list(self.shared_critic.parameters()) + \
                      list(self.policy_net.parameters())
        return base_params

    def get_actions(self, states: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, eval: bool = False):
        with torch.no_grad():
            embeddings = self.embedding_network(states)
            B,NE,NA,H = embeddings.shape
            flat = embeddings.view(NE*NA,H)
            mean, std = self.policy_net(flat)
            dist = torch.distributions.Normal(mean, std)
            actions = mean if eval else dist.sample()  
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

            actions = actions.view(NE,NA,self.action_dim)
            log_probs = log_probs.view(NE,NA)
            entropy = entropy.view(NE,NA)
            return actions, (log_probs, entropy)

    def update(self, states, next_states, actions, rewards, dones, truncs):
        NS,NE,NA,obs_dim = states.shape

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

        for _ in range(self.epochs):
            indices = np.arange(NS)
            np.random.shuffle(indices)

            num_batches = (NS + self.mini_batch_size - 1) // self.mini_batch_size

            total_policy_losses = []
            total_value_losses = []
            total_baseline_losses = []
            total_entropies = []
            total_returns = []
            total_lrs = []
            entropy_coeff_values = []

            for b in range(num_batches):
                start = b * self.mini_batch_size
                end = min((b+1)*self.mini_batch_size, NS)
                mb_idx = indices[start:end]
                MB = len(mb_idx)

                states_mb = states[mb_idx]
                actions_mb = actions[mb_idx]
                returns_mb = returns[mb_idx]
                old_log_probs_mb = old_log_probs[mb_idx]

                embeddings_mb = self.embedding_network(states_mb)
                baseline_inputs = self.get_baseline_inputs(embeddings_mb, actions_mb)
                predicted_baselines = self.shared_critic.baselines(baseline_inputs)
                values_mb = self.get_value(embeddings_mb)
                log_probs_mb, entropy_mb = self.recompute_log_probs(embeddings_mb, actions_mb)

                ratio = torch.exp(log_probs_mb - old_log_probs_mb)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range)

                returns_mb_expanded = returns_mb.unsqueeze(2).expand(MB,NE,NA,1)
                adv_mb = returns_mb_expanded - predicted_baselines
                adv_mb = adv_mb.squeeze(-1)
                adv_detached = adv_mb.detach()

                obj1 = ratio * adv_detached
                obj2 = clipped_ratio * adv_detached
                mb_policy_loss = -torch.mean(torch.min(obj1, obj2))

                mb_value_loss = F.smooth_l1_loss(values_mb, returns_mb)
                mb_baseline_loss = F.smooth_l1_loss(predicted_baselines, returns_mb_expanded)
                mb_ent_mean = entropy_mb.mean() 

                if self.adaptive_entropy:
                    # Lagrangian: L = PPOObj + entropy_lambda*(H_current - H_target)
                    mb_loss = mb_policy_loss + self.value_loss_coeff*mb_value_loss + mb_baseline_loss + self.entropy_lambda*(mb_ent_mean - self.target_entropy)
                else:
                    mb_loss = mb_policy_loss + self.value_loss_coeff*(mb_value_loss + 0.5 * mb_baseline_loss) - self.entropy_coeff*mb_ent_mean

                self.optimizer.zero_grad()
                mb_loss.backward()
                clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.rl_scheduler.step()

                if self.adaptive_entropy:
                    # Dual gradient ascent on entropy_lambda:
                    # For updating lambda, we don't want policy gradients, so we detach here:
                    with torch.no_grad():
                        ent_error = (mb_ent_mean.detach().item() - self.target_entropy)
                        self.entropy_lambda = max(0.0, self.entropy_lambda + self.entropy_lambda_lr * ent_error)
                    entropy_coeff_values.append(torch.tensor(self.entropy_lambda).cpu())
                else:
                    entropy_coeff_values.append(torch.tensor(self.entropy_coeff).cpu())

                total_policy_losses.append(mb_policy_loss.detach().cpu())
                total_value_losses.append(mb_value_loss.detach().cpu())
                total_baseline_losses.append(mb_baseline_loss.detach().cpu())
                total_entropies.append(mb_ent_mean.detach().cpu())
                total_returns.append(returns_mb.mean().detach().cpu())
                total_lrs.append(self.get_average_lr(self.optimizer).detach().cpu())
                
        return {
            "Policy Loss": torch.stack(total_policy_losses).mean(),
            "Value Loss": torch.stack(total_value_losses).mean(),
            "Baseline Loss": torch.stack(total_baseline_losses).mean(),
            "Entropy": torch.stack(total_entropies).mean(),
            "Average Returns": torch.stack(total_returns).mean(),
            "Average Rewards": rewards.mean(),
            "Learning Rate": torch.stack(total_lrs).mean(),
            # "Entropy Coeff (Lambda)": torch.stack(entropy_coeff_values).mean(),
        }

    def compute_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        NS,NE,_ = rewards.shape
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
        agent_embeds, groupmates_embeds, groupmates_actions = self.embedding_network.split_agent_obs_groupmates_obs_actions(agent_embeddings, actions)
        groupmates_embeds_with_actions = self.embedding_network.encode_groupmates_obs_actions(groupmates_embeds, groupmates_actions)
        x = torch.cat([agent_embeds, groupmates_embeds_with_actions], dim=3).contiguous()
        return x

    def recompute_log_probs(self, embeddings, actions):
        S,E,NA,H = embeddings.shape
        flat = embeddings.view(S*E, NA, H)
        mean, std = self.policy_net(flat)
        dist = torch.distributions.Normal(mean, std)
        A = actions.view(S*E, NA, self.action_dim)
        log_probs = dist.log_prob(A).sum(-1).view(S,E,NA)
        entropy = dist.entropy().sum(-1).view(S,E,NA)
        return log_probs, entropy