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
from Utility import RunningMeanStdNormalizer, OneCycleCosineScheduler, ModifiedOneCycleLR
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts, CyclicLR
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
        self.state_encoder = StatesEncoder(**self.config.networks["state_encoder"])
        self.state_action_encoder = StatesActionsEncoder(**self.config.networks["state_action_encoder"])
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.config.embed_size, max_seq_length=self.config.max_agents)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        rsa_out = self.RSA(combined_states_actions)
        
        # Mean pool over the second dimension (num_agents)
        rsa_output_mean = rsa_out.mean(dim=1)
    
        # Get the baselines for all agents from the value network
        num_agents_tensor = torch.full((num_steps * num_envs * num_agents, 1), num_agents/self.config.max_agents).to(self.device)
        baselines = self.baseline(torch.cat([rsa_output_mean, num_agents_tensor], dim=-1))
        return baselines.contiguous().view(num_steps, num_envs, num_agents, 1)
    
    def values(self, x: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs, num_agents, state_size = x.shape
        x = x.contiguous().view(num_steps * num_envs * num_agents, state_size)
        x = self.state_encoder(x)
        x = x.view(num_steps * num_envs, num_agents, self.config.embed_size)
        x = self.RSA(x)
        x = torch.mean(x, dim=1).squeeze()
        num_agents_tensor = torch.full((num_steps * num_envs, 1), num_agents/self.config.max_agents).to(self.device)
        values = self.value(torch.cat([x, num_agents_tensor], dim=-1))
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
        self.state_encoder = StatesEncoder(**self.config.networks["state_encoder"])
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
    
class DiscreteMultiAgentICM(nn.Module):
    def __init__(self, config):
        super(DiscreteMultiAgentICM, self).__init__()
        self.config = config
        
        # RSA for extracting inter-agent feature representations for both forward and inverse models,
        self.icm_gain = self.config.networks["ICM"]["icm_gain"]
        self.icm_beta = self.config.networks["ICM"]["icm_beta"]
        self.embed_size = self.config.networks["ICM"]["embed_size"] 
        self.rsa = RSA(**self.config.networks["ICM"]["rsa"])
        self.state_encoder = StatesEncoder2d(**self.config.networks["ICM"]["state_encoder2d"])
        self.forward_head = LinearNetwork(**self.config.networks["ICM"]["forward_head"])
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.embed_size, max_seq_length=self.config.max_agents)
        self.inverse_head = LinearNetwork(**self.config.networks["ICM"]["inverse_head"])

    def forward_rsa_network(self, state_features: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([state_features, actions], dim=-1)
        x = self.forward_head(x)
        return x

    def inverse_rsa_network(self, state_features: torch.Tensor, next_state_features: torch.Tensor):
        x = torch.cat([
            state_features,
            next_state_features
        ], dim=-1)
        action_logits = self.inverse_head(x)
        return action_logits

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_steps, num_envs, num_agents, action_dim = actions.shape

        # Feature Network
        num_steps, num_envs, num_agents, _ = states.shape
        _, _, _, num_actions = actions.shape
        actions = actions.contiguous().view(num_steps * num_envs, num_agents, num_actions)
        state_features = self.rsa(
            self.positional_encodings(
                self.state_encoder(states).contiguous().view(num_steps * num_envs, num_agents, self.embed_size)
            )
        )
        next_state_features = self.rsa(
            self.positional_encodings(
                self.state_encoder(next_states).contiguous().view(num_steps * num_envs, num_agents, self.embed_size)
            )
        )

        # Calculate forward loss
        predicted_next_state_features = self.forward_rsa_network(state_features, actions)
        forward_loss_per_state = F.mse_loss(predicted_next_state_features, next_state_features.detach(), reduction='none')
        forward_loss = self.icm_beta * forward_loss_per_state.mean()

        # Calculate inverse loss
        predicted_action_logits = self.inverse_rsa_network(state_features, next_state_features)
        *_, num_logits = predicted_action_logits.shape
        inverse_loss = (1 - self.icm_beta) * F.cross_entropy(predicted_action_logits.view(num_steps * num_envs * num_agents, num_logits), actions.view(-1).to(torch.long))
                
        # Intrinsic reward
        intrinsic_reward = self.icm_gain * forward_loss_per_state.view(num_steps, num_envs, num_agents, self.embed_size).mean(dim=2).mean(-1) # Average rewards along agent dimention

        return forward_loss, inverse_loss, intrinsic_reward.unsqueeze(-1)

class MAPocaAgent(Agent):
    def __init__(self, config: MAPOCAConfig):
        super(MAPocaAgent, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
    
        self.policy = PolicyNetwork(self.config).to(self.device)
        self.shared_critic = SharedCritic(self.config).to(self.device)

        self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.advantage_normalizer = RunningMeanStdNormalizer(device=self.device)
        
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.config.policy_learning_rate)
        # self.policy_scheduler = CyclicLR(
        #     self.policy_optimizer, 
        #     base_lr=self.config.policy_learning_rate/100,  # Lower boundary in the cycle
        #     max_lr=self.config.policy_learning_rate,    # Upper boundary in the cycle
        #     step_size_up=100, 
        #     step_size_down=None,  # If None, it is set to step_size_up
        #     mode='exp_range',
        #     gamma=0.99994,  # Factor by which the learning rate decreases each step
        #     cycle_momentum=False  # Set to False if the optimizer does not use momentum
        # )
        # self.policy_scheduler = CosineAnnealingWarmRestarts(
        #     self.policy_optimizer, 
        #     T_0=100, 
        #     T_mult=2, 
        #     eta_min=self.config.policy_learning_rate/1000
        # )
        self.policy_scheduler = ModifiedOneCycleLR(
            self.policy_optimizer, 
            max_lr=self.config.policy_learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=.2,
            final_div_factor= 300,
        ) 

        self.shared_critic_optimizer = optim.AdamW(self.shared_critic.parameters(), lr=self.config.value_learning_rate)
        # self.shared_critic_scheduler = CyclicLR(
        #     self.shared_critic_optimizer, 
        #     base_lr=self.config.value_learning_rate/100,  # Lower boundary in the cycle
        #     max_lr=self.config.value_learning_rate,    # Upper boundary in the cycle
        #     step_size_up=100, 
        #     step_size_down=None,  # If None, it is set to step_size_up
        #     mode='exp_range',
        #     gamma=0.99994,  # Factor by which the learning rate decreases each step
        #     cycle_momentum=False  # Set to False if the optimizer does not use momentum
        # )
        # self.shared_critic_scheduler = CosineAnnealingWarmRestarts(
        #     self.policy_optimizer, 
        #     T_0=100, 
        #     T_mult=2, 
        #     eta_min=self.config.value_learning_rate/1000
        # )
        self.shared_critic_scheduler = ModifiedOneCycleLR(
            self.shared_critic_optimizer, 
            max_lr=self.config.value_learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=.2,
            final_div_factor= 300,
        )
        self.entropy_scheduler = OneCycleCosineScheduler(
            self.config.entropy_coefficient, self.config.anneal_steps, pct_start=.2, div_factor=2, final_div_factor=100
        )
        self.policy_clip_scheduler = OneCycleCosineScheduler(
            self.config.policy_clip, self.config.anneal_steps, pct_start=.2, div_factor=2.0, final_div_factor=4.0
        )
        self.value_clip_scheduler = OneCycleCosineScheduler(
            self.config.value_clip, self.config.anneal_steps, pct_start=.2, div_factor=2.0, final_div_factor=4.0
        )

        # TODO: Add toggle for continous ICM rewards.
        if self.config.icm_enabled:
            self.icm = DiscreteMultiAgentICM(self.config).to(self.device)
            self.icm_optimizer = optim.AdamW(self.icm.parameters(), lr=self.config.policy_learning_rate)
            self.icm_scheduler = ModifiedOneCycleLR(
                self.icm_optimizer  , 
                max_lr=self.config.policy_learning_rate,
                total_steps=self.config.anneal_steps,
                anneal_strategy='cos',
                pct_start=.2,
                final_div_factor= 300,
            )

    def to_gpu(self, x):
        if isinstance(x, tuple):
            return tuple(item.to(device=self.gpu_device) for item in x)
        return x.to(device=self.device)

    def to_cpu(self, x):
        if isinstance(x, tuple):
            return tuple(item.to(device=self.cpu_device) for item in x)
        return x.to(device=self.cpu_device)

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
 
    # def trust_region_value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, epsilon=0.1) -> torch.Tensor:
    #     # Calculate mean and standard deviation of the differences
    #     differences = values - old_values
    #     mean_diff = torch.mean(differences)
    #     std_dev_diff = torch.std(differences)

    #     # Dynamic epsilon based on mean and standard deviation, but capped by a fixed epsilon
    #     dynamic_epsilon = min(mean_diff + std_dev_diff, epsilon)

    #     # Clipping the value predictions
    #     value_pred_clipped = old_values + differences.clamp(-dynamic_epsilon, dynamic_epsilon)
        
    #     # Calculating the clipped and unclipped losses
    #     loss_clipped = (returns - value_pred_clipped) ** 2
    #     loss_unclipped = (returns - values) ** 2

    #     # Combining the losses
    #     value_loss = torch.mean(torch.max(loss_clipped, loss_unclipped))

    #     return value_loss

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
            batch_rewards = rewards.mean()
            
            # For trust region loss
            old_log_probs, _ = self.policy.log_probs(states, actions)
            batched_agent_states, batched_groupmates_states, batched_groupmates_actions = self.AgentGroupMatesStatesActions(states, actions)
            old_baselines = self.shared_critic.baselines(batched_agent_states, batched_groupmates_states, batched_groupmates_actions, states, actions)
            old_values = self.shared_critic.values(states)
            if self.config.icm_enabled:
                _, _, icm_rewards = self.icm(states, next_states, actions)
                rewards += icm_rewards

            if self.config.normalize_rewards:
                self.reward_normalizer.update(rewards)
                rewards = self.reward_normalizer.normalize(rewards)

            # Put on RAM. Multi-agent computations are GPU memory taxing, and should be moved over when needed.
            states = self.to_cpu(states)
            next_states = self.to_cpu(next_states)
            actions = self.to_cpu(actions)
            rewards = self.to_cpu(rewards)
            dones = self.to_cpu(dones)
            truncs = self.to_cpu(truncs)
            batched_agent_states = self.to_cpu(batched_agent_states)
            batched_groupmates_states = self.to_cpu(batched_groupmates_states)
            batched_groupmates_actions = self.to_cpu(batched_groupmates_actions)

            old_log_probs = self.to_cpu(old_log_probs)
            old_baselines = self.to_cpu(old_baselines)
            old_values = self.to_cpu(old_values)

            if self.config.icm_enabled:
                icm_rewards = self.to_cpu(icm_rewards)

        num_steps, _, _ = rewards.shape 
        mini_batch_size = num_steps // self.config.num_mini_batches

        total_loss_combined, total_loss_policy, total_loss_value, total_loss_baseline, total_entropy = [], [], [], [], []
        if self.config.icm_enabled:
            total_icm_forward_loss, total_icm_inverse_loss = [], []

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
                    # mb_batched_agent_states = batched_agent_states[ids]
                    # mb_batched_groupmates_states = batched_groupmates_states[ids]
                    # mb_batched_groupmates_actions = batched_groupmates_actions[ids]
                    # mb_states = states[ids]
                    # mb_next_states = next_states[ids]
                    # mb_actions = actions[ids]
                    # mb_dones = dones[ids]
                    # mb_truncs = truncs[ids]
                    # mb_next_states = next_states[ids]
                    # mb_rewards = rewards[ids]
                    # mb_old_log_probs = old_log_probs[ids]
                    # mb_old_values = old_values[ids]
                    # mb_old_baselines = old_baselines[ids]

                    mb_batched_agent_states = self.to_gpu(batched_agent_states[ids])
                    mb_batched_groupmates_states = self.to_gpu(batched_groupmates_states[ids])
                    mb_batched_groupmates_actions = self.to_gpu(batched_groupmates_actions[ids])
                    mb_states = self.to_gpu(states[ids])
                    mb_next_states = self.to_gpu(next_states[ids])
                    mb_actions = self.to_gpu(actions[ids])
                    mb_dones = self.to_gpu(dones[ids])
                    mb_truncs = self.to_gpu(truncs[ids])
                    mb_next_states = self.to_gpu(next_states[ids])
                    mb_rewards = self.to_gpu(rewards[ids])
                    mb_old_log_probs = self.to_gpu(old_log_probs[ids])
                    mb_old_values = self.to_gpu(old_values[ids])
                    mb_old_baselines = self.to_gpu(old_baselines[ids])

                # Clear gradiants for the next batch
                self.policy_optimizer.zero_grad()
                self.shared_critic_optimizer.zero_grad()

                # Calculate losses
                loss_value, mb_targets = self.value_loss(
                    mb_old_values, 
                    mb_states, 
                    mb_next_states, 
                    mb_rewards, 
                    mb_dones, 
                    mb_truncs
                )
                
                loss_baseline, mb_advantages = self.baseline_loss( # Long comp time
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
                
                loss_policy, entropy = self.policy_loss(
                    mb_old_log_probs, 
                    mb_advantages, 
                    mb_states, 
                    mb_actions
                )

                if self.config.icm_enabled:
                    forward_loss, inverse_loss, _ = self.icm(
                        mb_states, 
                        mb_next_states, 
                        mb_actions
                    )
                    total_icm_loss = forward_loss + inverse_loss
                    total_icm_loss.backward()
                    clip_grad_norm_(self.icm.parameters(), self.config.max_grad_norm)
                    self.icm_optimizer.step()
                    self.icm_scheduler.step()
                    total_icm_forward_loss.append(forward_loss.cpu())
                    total_icm_inverse_loss.append(inverse_loss.cpu())

                # Perform backpropagations
                policy_loss = (loss_policy - self.entropy_scheduler.value() * entropy.mean())
                shared_critic_loss = loss_value + (0.5 * loss_baseline)
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
        if self.config.icm_enabled:
            total_icm_forward_loss = torch.stack(total_icm_forward_loss).mean().cpu()
            total_icm_inverse_loss = torch.stack(total_icm_inverse_loss).mean().cpu()
        
        return {
            "Total Loss": total_loss_combined,
            "Policy Loss": total_loss_policy,
            "Value Loss": total_loss_value,
            "Baseline Loss": total_loss_baseline,
            "Entropy": total_entropy,
            "Average Rewards": batch_rewards,
            "Normalized Rewards": rewards.mean(),
            "Shared Critic Learning Rate": self.shared_critic_scheduler.get_last_lr()[0],
            "Policy Learning Rate": self.policy_scheduler.get_last_lr()[0],
            "Entropy Coeff": torch.tensor(self.entropy_scheduler.value()),
            "Policy Clip": torch.tensor(self.policy_clip_scheduler.value()),
            "Value Clip": torch.tensor(self.value_clip_scheduler.value()),
            **(
                {
                    "ICM Rewards": icm_rewards.mean(), # type: ignore 
                    "ICM Forward Loss": total_icm_forward_loss, # type: ignore 
                    "ICM Inverse Loss": total_icm_inverse_loss, # type: ignore 
                } 
                if self.config.icm_enabled 
                else {}
            ),
        }
