import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Tuple
from Agents.Agent import Agent
from Config import MAPOCAConfig
from Networks import *
from Utility import RunningMeanStdNormalizer, CosineAnnealingScheduler, ModifiedOneCycleLR
from torch.optim.lr_scheduler import _LRScheduler
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, state_embedding_size, max_seq_length):
        super(PositionalEncoding, self).__init__()

        self.encoding: torch.Tensor = torch.zeros(max_seq_length, state_embedding_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, state_embedding_size, 2) * -(math.log(10000.0) / state_embedding_size))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        *initial_dims, num_agents, embed_size = x.shape
        encodings = self.encoding[:num_agents, :].expand_as(x).detach().to(x.device)
        return (x + encodings).view(*initial_dims, num_agents, embed_size)
    
    def to(self, device=None):
        self.encoding.to(device=device)

class SharedCritic(nn.Module):
    def __init__(self, config):
        super(SharedCritic, self).__init__()
        self.config = config
        self.RSA = RSA(**self.config.networks["RSA"])
        self.value = ValueNetwork(**self.config.networks["value_network"])
        self.baseline = ValueNetwork(**self.config.networks["value_network"])
        self.state_encoder = StatesEncoder2d(**self.config.networks["state_encoder2d"])
        self.state_action_encoder = StatesActionsEncoder2d(**self.config.networks["state_action_encoder2d"])
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.config.embed_size, max_seq_length=self.config.max_agents)

    def baselines(self, batched_agent_states: torch.Tensor, batched_groupmates_states: torch.Tensor, batched_groupmates_actions: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Reshape to combine steps and envs to simple batched dim
        num_steps, num_envs, num_agents, state_size = states.shape
        _, _, _, num_actions = actions.shape
        batched_agent_states = batched_agent_states.contiguous().view(num_steps * num_envs * num_agents, 1, state_size)
        batched_groupmates_states = batched_groupmates_states.contiguous().view(num_steps * num_envs * num_agents, num_agents - 1, state_size)
        batched_groupmates_actions = batched_groupmates_actions.contiguous().view(num_steps * num_envs * num_agents, num_agents - 1, num_actions)

        # Encode the groupmates' states and actions together
        groupmates_states_actions_encoded = self.state_action_encoder(batched_groupmates_states, batched_groupmates_actions)

        # Encode the agent's state
        agent_state_encoded = self.state_encoder(batched_agent_states).contiguous().view(num_steps * num_envs * num_agents, 1, -1)

        # Combine agent state encoding with groupmates' state-action encodings
        combined_states_actions = torch.cat([agent_state_encoded, groupmates_states_actions_encoded], dim=1).contiguous()

        # Pass the combined tensor through RSA
        rsa_out = self.RSA(self.positional_encodings(combined_states_actions))
        
        # Mean pool over the second dimension (num_agents)
        rsa_output_mean = rsa_out.mean(dim=1)
    
        # Get the baselines for all agents from the value network
        baselines = self.baseline(rsa_output_mean)
        return baselines.contiguous().view(num_steps, num_envs, num_agents, 1)
    
    def values(self, x: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs, num_agents, state_size = x.shape
        x = x.contiguous().view(num_steps * num_envs * num_agents, state_size)
        x = self.state_encoder(x)
        x = x.view(num_steps * num_envs, num_agents, self.config.embed_size)
        x = self.positional_encodings(x)
        x = self.RSA(x)
        x = torch.mean(x, dim=1).squeeze()
        values = self.value(x)
        return values.contiguous().view(num_steps, num_envs, 1)
    
    def forward(self, x):
        return self.values(x)

class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        self.config = config
        if self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
            if isinstance(self.config.networks["policy"]["out_features"], list):
                raise NotImplementedError("Only support for one discrete branch currently!")
            
            self.policy = DiscretePolicyNetwork(
                **self.config.networks["policy"]
            )
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        self.state_encoder = StatesEncoder2d(**self.config.networks["state_encoder2d"])
        self.RSA = RSA(**self.config.networks["RSA"])
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.config.embed_size, max_seq_length=self.config.max_agents)

    def probabilities(self, x: torch.Tensor):
        num_steps, num_envs, num_agents, state_size = x.shape
        x = self.state_encoder(
            x.contiguous().view(num_steps * num_envs * num_agents, state_size)
        )
        x = x.contiguous().view(num_steps * num_envs, num_agents, self.config.embed_size)
        x = self.RSA(self.positional_encodings(x))
        x = x.contiguous().view(num_steps, num_envs, num_agents, self.config.embed_size)
        return self.policy(x)
    
    def log_probs(self, states: torch.Tensor, actions: torch.Tensor):
        m = dist.Categorical(probs=self.probabilities(states))
        return m.log_prob(actions.squeeze(-1)), m.entropy()

    def forward(self, x):
        return self.probabilities(x)
    
class MultiAgentICM(nn.Module):
    def __init__(self, config):
        super(MultiAgentICM, self).__init__()
        self.config = config
        
        # RSA for extracting inter-agent feature representations for both forward and inverse models
        self.rsa = RSA(**self.config.networks["ICM"]["forward_rsa"])
        
        self.state_encoder = StatesEncoder(**self.config.networks["ICM"]["state_encoder"])
        self.state_action_encoder = StatesActionsEncoder(**self.config.networks["ICM"]["state_action_encoder"])
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.config.embed_size, max_seq_length=self.config.max_agents)
        self.inverse_head = LinearNetwork(**self.config.networks["ICM"]["inverse_head"])
        self.inverse_body = LinearNetwork(**self.config.networks["ICM"]["inverse_body"])

    def forward_rsa_network(self, states: torch.Tensor, actions: torch.Tensor):
        x = self.state_action_encoder(states, actions)
        num_steps, num_envs, num_agents, embed_size = x.shape
        x = self.positional_encodings(x)
        x = x.contiguous().view(num_steps * num_envs, num_agents, embed_size)
        x = self.rsa(x)
        return x

    def inverse_rsa_network(self, state_features: torch.Tensor, next_state_features: torch.Tensor):
        num_steps, num_envs, num_agents, embed_size = state_features.shape
        x = torch.cat([
            self.positional_encodings(state_features.contiguous().view(num_steps * num_envs, num_agents, embed_size)),
            self.positional_encodings(next_state_features.contiguous().view(num_steps * num_envs, num_agents, embed_size)),
        ], dim=1)
        x = self.inverse_body(x)
        x = self.rsa(x)
        action_logits = self.inverse_head(x)
        return action_logits

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.LongTensor, n: float = 0.5, beta: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_steps, num_envs, num_agents, action_dim = actions.shape

        # Feature Network
        state_features = self.state_encoder(states)
        next_state_features = self.state_encoder(next_states)

        # Forward Model
        predicted_next_state_features = self.forward_rsa_network(state_features, actions)

        # Inverse Model
        predicted_action_logits = self.inverse_rsa_network(state_features, next_state_features)

        # Calculate losses
        forward_loss_per_state = F.mse_loss(predicted_next_state_features, next_state_features.detach(), reduction='none')
        forward_loss = beta * forward_loss_per_state.mean()
        inverse_loss = (1 - beta) * F.cross_entropy(predicted_action_logits.view(num_steps * num_envs * num_agents, action_dim), actions.view(-1))  # Using cross-entropy for discrete actions

        # Intrinsic reward
        intrinsic_reward = n * forward_loss_per_state.mean(-1).squeeze(-1).detach()

        return forward_loss, inverse_loss, intrinsic_reward

class MAPocaAgent(Agent):
    def __init__(self, config: MAPOCAConfig):
        super(MAPocaAgent, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        self.policy = PolicyNetwork(self.config).to(self.device)
        self.shared_critic = SharedCritic(self.config).to(self.device)

        self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.advantage_normalizer = RunningMeanStdNormalizer(device=self.device)
        
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.config.policy_learning_rate)
        self.policy_scheduler = ModifiedOneCycleLR(
            self.policy_optimizer, 
            max_lr=self.config.policy_learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=0.3   
        ) 

        self.shared_critic_optimizer = optim.AdamW(self.shared_critic.parameters(), lr=self.config.policy_learning_rate)
        self.shared_critic_scheduler = ModifiedOneCycleLR(
            self.shared_critic_optimizer, 
            max_lr=self.config.value_learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=0.3
        )
        hold_steps = int(self.config.anneal_steps * 0.3)
        self.entropy_scheduler = CosineAnnealingScheduler(
            hold_steps=hold_steps,
            T=self.config.anneal_steps-hold_steps, 
            H=self.config.entropy_coefficient, 
            L=self.config.entropy_coefficient/100
        )
        self.policy_clip_scheduler = CosineAnnealingScheduler(
            hold_steps=hold_steps,
            T=self.config.anneal_steps-hold_steps, 
            H=self.config.policy_clip, 
            L=self.config.policy_clip/4
        )
        self.value_clip_scheduler = CosineAnnealingScheduler(
            hold_steps=hold_steps,
            T=self.config.anneal_steps-hold_steps, 
            H=self.config.value_clip, 
            L=self.config.value_clip/4
        )

    def get_actions(self, states: torch.Tensor, eval: bool=False):
        if self.config.action_space.has_discrete():
            probabilities = self.policy.probabilities(states)
            m = dist.Categorical(probs=probabilities)
            
            if eval:
                actions_indices = m.probs.argmax(dim=-1)
            else:
                actions_indices = m.sample()

            return actions_indices, m.log_prob(actions_indices), m.entropy()
        else:
            raise NotImplementedError("No support for continuous actions yet!")

    def lambda_returns(self, rewards, values_next, dones, truncs):
        num_steps, _, _ = rewards.shape
        returns = torch.zeros_like(rewards).to(self.device)
        returns[-1] = rewards[-1] + self.config.gamma * values_next[-1] * (1 - dones[-1]) * (1 - truncs[-1])
        
        for t in reversed(range(num_steps - 1)):
            non_terminal = 1.0 - torch.clamp(dones[t] + truncs[t], 0.0, 1.0)
            bootstrapped_value = self.config.gamma * (self.config.lambda_ * returns[t + 1] + (1 - self.config.lambda_) * values_next[t])
            returns[t] = rewards[t] + non_terminal * bootstrapped_value
        
        return returns
    
    def AgentGroupMatesStatesActions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = states.shape
        
        agent_states_list = []
        groupmates_states_list = []
        groupmates_actions_list = []

        for agent_idx in range(num_agents):
            agent_states_list.append(states[:, :, agent_idx, :].unsqueeze(2))
            groupmates_states = torch.cat([states[:, :, :agent_idx, :], states[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
            groupmates_actions = torch.cat([actions[:, :, :agent_idx, :], actions[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
            groupmates_states_list.append(groupmates_states)
            groupmates_actions_list.append(groupmates_actions)

        batched_agent_states = torch.cat(agent_states_list, dim=2).unsqueeze(3).contiguous()
        batched_groupmates_states = torch.cat(groupmates_states_list, dim=2).contiguous()
        batched_groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()
        return batched_agent_states, batched_groupmates_states, batched_groupmates_actions
    
    def trust_region_value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, epsilon=0.1) -> torch.Tensor:
        value_pred_clipped = old_values + (values - old_values).clamp(-epsilon, epsilon)
        loss_clipped = (returns - value_pred_clipped)**2
        loss_unclipped = (returns - values)**2
        value_loss = torch.mean(torch.max(loss_clipped, loss_unclipped))
        return value_loss

    def value_loss(self, old_values, states, next_states, rewards, dones, truncs):
        values = self.shared_critic.values(states)
        with torch.no_grad():
            targets = self.lambda_returns(rewards, self.shared_critic.values(next_states), dones, truncs)
        loss = self.trust_region_value_loss(
            values,
            old_values,
            targets.detach(),
            self.value_clip_scheduler.value()
        )
        return loss, targets
    
    def baseline_loss(self, states: torch.Tensor, actions: torch.Tensor, old_baselines: torch.Tensor, agent_states: torch.Tensor, groupmates_states: torch.Tensor, groupmates_actions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = states.shape
        total_loss = torch.tensor(0.0).to(self.device)
        advantages = torch.zeros_like(old_baselines).to(self.device)
        baselines = self.shared_critic.baselines(agent_states, groupmates_states, groupmates_actions, states, actions)
        
        # Calculate loss and advantages for all agents
        for agent_idx in range(num_agents):
            agent_baselines = baselines[:, :, agent_idx, :]
            old_agent_baselines = old_baselines[:, :, agent_idx, :]
            loss = self.trust_region_value_loss(
                agent_baselines,
                old_agent_baselines,
                targets.detach(),
                self.value_clip_scheduler.value()
            )
            total_loss += loss
            advantages[:, :, agent_idx, :] = targets - agent_baselines

        return total_loss / num_agents, advantages
    
    def policy_loss(self, prev_log_probs: torch.Tensor, advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        log_probs, entropy = self.policy.log_probs(states, actions)
        ratio = torch.exp(log_probs - prev_log_probs)
        loss_policy = -torch.mean(
            torch.min(
                ratio * advantages.squeeze().detach(),
                ratio.clip(1.0 - self.policy_clip_scheduler.value(), 1.0 + self.policy_clip_scheduler.value()) * advantages.squeeze().detach(),
            )
        )
        return loss_policy, entropy

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor):
        with torch.no_grad():
            batch_rewards = rewards
            if self.config.normalize_rewards:
                self.reward_normalizer.update(rewards)
                rewards = self.reward_normalizer.normalize(rewards)

            # For trust region loss
            old_log_probs, _ = self.policy.log_probs(states, actions)
            batched_agent_states, batched_groupmates_states, batched_groupmates_actions = self.AgentGroupMatesStatesActions(states, actions)
            old_baselines = self.shared_critic.baselines(batched_agent_states, batched_groupmates_states, batched_groupmates_actions, states, actions)
            old_values = self.shared_critic.values(states)

        _, _, num_agents, _ = states.shape
        num_steps, _, _ = rewards.shape 
        mini_batch_size = num_steps // self.config.num_mini_batches

        total_loss_combined, total_loss_policy, total_loss_value, total_loss_baseline, total_entropy, = [], [], [], [], []
        for _ in range(self.config.num_epocs):

            # Generate a random order of mini-batches
            mini_batch_start_indices = list(range(0, states.size(0), mini_batch_size))
            np.random.shuffle(mini_batch_start_indices)

            # Process each mini-batch
            for start_idx in mini_batch_start_indices:

                # Mini-batch indices
                end_idx = min(start_idx + mini_batch_size, states.size(0))
                ids = slice(start_idx, end_idx)

                # Extract mini-batch data
                with torch.no_grad():
                    mb_batched_agent_states = batched_agent_states[ids]
                    mb_batched_groupmates_states = batched_groupmates_states[ids]
                    mb_batched_groupmates_actions = batched_groupmates_actions[ids]
                    mb_states = states[ids]
                    mb_actions = actions[ids]
                    mb_dones = dones[ids]
                    mb_truncs = truncs[ids]
                    mb_next_states = next_states[ids]
                    mb_rewards = rewards[ids]
                    mb_old_log_probs = old_log_probs[ids]
                    mb_old_values = old_values[ids]
                    mb_old_baselines = old_baselines[ids]

                # Clear gradiants for the next batch
                self.policy_optimizer.zero_grad()
                self.shared_critic_optimizer.zero_grad()

                # Calculate losses
                loss_value, mb_targets = self.value_loss(mb_old_values, mb_states, mb_next_states, mb_rewards, mb_dones, mb_truncs)
                loss_baseline, mb_advantages = self.baseline_loss(
                    mb_states,
                    mb_actions,
                    mb_old_baselines,
                    mb_batched_agent_states,
                    mb_batched_groupmates_states,
                    mb_batched_groupmates_actions,
                    mb_targets
                )
                
                if self.config.normalize_advantages:
                    self.advantage_normalizer.update(mb_advantages)
                    mb_advantages = self.advantage_normalizer.normalize(mb_advantages)
                loss_policy, entropy = self.policy_loss(mb_old_log_probs, mb_advantages, mb_states, mb_actions)

                # Perform backpropagations
                policy_loss = (loss_policy - self.entropy_scheduler.value() * entropy.mean())
                shared_critic_loss = loss_value + (1 / num_agents * loss_baseline)
                total_loss = policy_loss + shared_critic_loss 
                policy_loss.backward()
                shared_critic_loss.backward()

                clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                clip_grad_norm_(self.shared_critic.parameters(), self.config.max_grad_norm)

                self.policy_optimizer.step() 
                self.shared_critic_optimizer.step()

                self.policy_scheduler.step() 
                self.shared_critic_scheduler.step()
                self.entropy_scheduler.step()
                self.value_clip_scheduler.step()
                self.policy_clip_scheduler.step()

                # Accumulate Metrics
                total_loss_combined += [total_loss]
                total_loss_policy += [loss_policy]
                total_loss_value += [loss_value]
                total_loss_baseline += [loss_baseline]
                total_entropy += [entropy.mean()]

        total_loss_policy = torch.stack(total_loss_policy).mean().cpu()
        total_loss_value = torch.stack(total_loss_value).mean().cpu()
        total_entropy = torch.stack(total_entropy).mean().cpu()
        total_loss_baseline = torch.stack(total_loss_baseline).mean().cpu()
        total_loss_combined = torch.stack(total_loss_combined).mean().cpu()
        
        return {
            "Total Loss": total_loss_combined,
            "Policy Loss": total_loss_policy,
            "Value Loss": total_loss_value,
            "Baseline Loss": total_loss_baseline,
            "Entropy": total_entropy,
            "Average Rewards": batch_rewards.mean(),
            "Normalized Rewards": rewards.mean(),
            "Shared Critic Learning Rate": self.shared_critic_scheduler.get_last_lr()[0],
            "Policy Learning Rate": self.policy_scheduler.get_last_lr()[0],
            "Entropy Coeff": torch.tensor(self.entropy_scheduler.value()),
            "Policy Clip": torch.tensor(self.policy_clip_scheduler.value()),
            "Value Clip": torch.tensor(self.value_clip_scheduler.value())
        }
