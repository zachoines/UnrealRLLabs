import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from typing import Dict
from Agents.Agent import Agent
from Config import MAPOCAConfig
from Networks import *

class Baseline(nn.Module):
    def __init__(self, config: MAPOCAConfig):
        super(Baseline, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_encoder = StatesEncoder(
            **self.config.networks["state_encoder"]
        ).to(self.device)

        self.state_action_encoder = StatesActionsEncoder(
            **self.config.networks["state_action_encoder"]
        ).to(self.device)
        
        self.RSA = RSA(
            **self.config.networks["RSA"]
        ).to(self.device)

        self.value_network = ValueNetwork(
            **self.config.networks["value_network"]
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.policy_learning_rate)

    def forward(self, batched_agent_states, batched_groupmates_states, batched_groupmates_actions):
        # Encode the groupmates' states and actions together
        groupmates_states_actions_encoded = self.state_action_encoder(batched_groupmates_states, batched_groupmates_actions)

        # Encode the agent's state
        agent_state_encoded = self.state_encoder(batched_agent_states)

        # Combine agent state encoding with groupmates' state-action encodings
        combined_states_actions = torch.cat([agent_state_encoded, groupmates_states_actions_encoded], dim=1)

        # Pass the combined tensor through RSA
        rsa_out = self.RSA(combined_states_actions)
        
        # Mean pool over the second dimension (num_agents)
        rsa_output_mean = rsa_out.mean(dim=1)

        # Get the baselines for all agents from the value network
        baselines = self.value_network(rsa_output_mean)

        return baselines
    
    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Policy(nn.Module):
    def __init__(self, config: MAPOCAConfig):
        super(Policy, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_encoder = StatesEncoder(
            **self.config.networks["state_encoder"]
        ).to(self.device)

        self.RSA = RSA(
            **self.config.networks["RSA"]
        ).to(self.device)

        if self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
            if len(self.config.action_space.discrete_actions) > 1:
                raise NotImplementedError("Only support for one discrete branch currently!")
            
            self.policy = DiscretePolicyNetwork(
                **self.config.networks["policy"]
            ).to(self.device)
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.policy_learning_rate)

    def forward(self, states: torch.tensor):
        states = self.state_encoder(states)
        # self.RSA(states_emb)
        return self.policy(states)
    
    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Value(nn.Module):
    def __init__(self, config: MAPOCAConfig):
        super(Value, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_encoder = StatesEncoder(
            **self.config.networks["state_encoder"]
        ).to(self.device)
        
        self.RSA = RSA(
            **self.config.networks["RSA"]
        ).to(self.device)

        self.value_network = ValueNetwork(
            **self.config.networks["value_network"]
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.policy_learning_rate)

    def forward(self, states: torch.Tensor):
        states_encoded = self.state_encoder(states)
        rsa_out = self.RSA(states_encoded)
        rsa_output_mean = torch.mean(rsa_out, dim=1).squeeze()
        values = self.value_network(rsa_output_mean)
        return values
    
    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class MAPocaAgent(Agent):
    def __init__(self, config: MAPOCAConfig):
        super(MAPocaAgent, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Value Network
        self.value = Value(self.config)

        # Policy Network
        self.policy = Policy(self.config)

        # Baseline Network
        self.baseline = Baseline(self.config)


    def y_lambda(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> torch.Tensor:
        lambda_return = torch.zeros_like(rewards)
        n = torch.ones_like(rewards[0])
        for t in range(rewards.size(0)):
            lambda_power = self.config.lambda_ ** (n - 1)
            lambda_return += lambda_power * self.G_t(rewards, next_values, dones, truncs, t + 1)
            n = (n * (1 - dones[t]) * (1 - truncs[t])) + 1  # increment or reset lambda power if done or trunc
        return (1 - self.config.lambda_) * lambda_return
    
    def G_t(self, rewards: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, n: int) -> torch.Tensor:
        gamma_powers = torch.ones_like(rewards[:n])
        running_gamma_power = torch.ones_like(rewards[0])
        bootstrap_values = torch.zeros_like(rewards[:n])  # Initialize bootstrap values

        for t in range(n):
            gamma_powers[t] = self.config.gamma ** running_gamma_power
            # If done or trunc, add the bootstrap value for that trajectory at the appropriate gamma power
            bootstrap_values[t] = gamma_powers[t] * torch.clamp(dones[t] + truncs[t], 0, 1) * next_values[t]
            # Reset gamma power if done or trunc
            running_gamma_power = (running_gamma_power * (1 - dones[t]) * (1 - truncs[t])) + 1

        # Multiply gamma_powers against rewards and masks
        sum_rewards = torch.sum(rewards[:n] * (gamma_powers - 1) * (1 - dones[:n, :]) * (1 - truncs[:n, :]), dim=0)

        # Sum the bootstrap values
        total_bootstrap_value = torch.sum(bootstrap_values, dim=0)

        # Total return value
        G_t_val = sum_rewards + total_bootstrap_value

        return G_t_val


    def get_actions(self, states: torch.Tensor, eval: bool=False):
        num_steps, num_envs, num_agents, _ = states.shape
        states_flat = self._flatten_multienv(states)
        probabilities = self.policy(states_flat)
        
        if self.config.action_space.has_discrete():
            m = dist.Categorical(probabilities)
            
            if eval:
                actions_indices = m.probs.argmax(dim=-1)
            else:
                actions_indices = m.sample()

            log_probs = m.log_prob(actions_indices)
            entropy = m.entropy()
            return actions_indices.view(num_steps, num_envs, num_agents, 1), log_probs.view(num_steps, num_envs, num_agents, 1), entropy.view(num_steps, num_envs, num_agents, 1)
        else:
            raise NotImplementedError("No support for continuous actions yet!")

    def baseline_loss(self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor):
        total_loss = torch.tensor(0.0).to(states.device)  # Initialize total loss to zero
        num_steps, num_envs, num_agents, state_size = states.shape
        _, _, _, num_actions = actions.shape
        advantages = torch.zeros((num_steps, num_envs, num_agents, 1)).to(states.device)
        
        # Initialize lists to store states and actions for all agents
        agent_states_list = []
        groupmates_states_list = []
        groupmates_actions_list = []

        # Accumulate states and actions for all agents
        for agent_idx in range(num_agents):
            agent_states_list.append(states[:, :, agent_idx, :].unsqueeze(2))
            groupmates_states = torch.cat([states[:, :, :agent_idx, :], states[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
            groupmates_actions = torch.cat([actions[:, :, :agent_idx, :], actions[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
            groupmates_states_list.append(groupmates_states)
            groupmates_actions_list.append(groupmates_actions)

        # Concatenate lists to form batched tensors
        batched_agent_states = torch.cat(agent_states_list, dim=2).unsqueeze(3)
        batched_groupmates_states = torch.cat(groupmates_states_list, dim=2)
        batched_groupmates_actions = torch.cat(groupmates_actions_list, dim=2)

        # Reshape to combine steps and envs to simple batched dim
        batched_agent_states = batched_agent_states.view(num_steps * num_envs * num_agents, 1, state_size) 
        batched_groupmates_states = batched_groupmates_states.view(num_steps * num_envs * num_agents, num_agents - 1, state_size)
        batched_groupmates_actions = batched_groupmates_actions.view(num_steps * num_envs * num_agents, num_agents - 1, num_actions)

        # Get baselines for all agents at once
        baselines = self.baseline(batched_agent_states, batched_groupmates_states, batched_groupmates_actions)
        baselines = baselines.view(num_steps, num_envs, num_agents, 1)

        # Calculate loss and advantages for all agents
        for agent_idx in range(num_agents):
            agent_baselines = baselines[:, :, agent_idx, :]
            loss = F.smooth_l1_loss(agent_baselines, targets.detach())
            total_loss += loss
            advantages[:, :, agent_idx, :] = targets - agent_baselines

        return total_loss, advantages

    def _flatten_multienv(self, t: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs, num_agents = t.shape[0], t.shape[1], t.shape[2]
        t_flat = t.view(num_steps * num_envs, num_agents, -1)
        return t_flat
    
    def get_values(self, states: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs = states.shape[0], states.shape[1]
        states_flattened = self._flatten_multienv(states)
        values_flattened = self.value(states_flattened)
        values = values_flattened.view(num_steps, num_envs, 1)
        return values
    
    def value_loss(self, states, next_states, rewards, dones, truncs):
        with torch.no_grad():
            values = self.get_values(states)
            next_values = self.get_values(next_states)
            targets = self.y_lambda(values, next_values, rewards, dones, truncs)
            # targets, _ = self.compute_gae_and_targets(rewards, dones, truncs, values, next_values)
        loss = F.smooth_l1_loss(self.get_values(states), targets)
        return loss, targets
    
    def policy_loss(self, log_probs: torch.Tensor, entropy: torch.Tensor, advantages: torch.Tensor):
        # Combine the policy gradient loss and the entropy bonus
        pg_loss = -log_probs * advantages.detach()
        loss = (pg_loss - (self.config.entropy_coefficient * entropy))
        return loss.mean()

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor ):
        with torch.no_grad():
            perm = torch.randperm(states.size(0))
            states = states[perm]
            next_states = next_states[perm]
            actions = actions[perm]
            rewards = rewards[perm]
            dones = dones[perm]
            truncs = truncs[perm]

        value_loss, targets = self.value_loss(states, next_states, rewards, dones, truncs)
        baseline_loss, advantages = self.baseline_loss(states, actions, targets)
        _, log_probs, entropy = self.get_actions(states)
        policy_loss = self.policy_loss(log_probs, entropy, advantages)

        self.baseline.update(baseline_loss)
        self.policy.update(policy_loss)
        self.value.update(value_loss)

        return {
            "Total Loss": policy_loss + value_loss + baseline_loss,
            "Policy Loss": policy_loss,
            "Value Loss": value_loss,
            "Baseline Loss": baseline_loss,
            "Entropy": entropy.mean(),
            "Average Rewards": rewards.mean()
        }

# class MAPocaAgentV2(Agent):
#     def __init__(self, config: MAPOCAConfig):
#         super(MAPocaAgent, self).__init__(config)
#         self.config = config
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#         # Encoders
#         # self.state_encoder = StatesEncoder(
#         #     **self.config.networks["state_encoder"]
#         # ).to(self.device)
        
#         # self.state_action_encoder = StatesActionsEncoder(
#         #     **self.config.networks["state_action_encoder"]
#         # ).to(self.device)

#         self.state_encoder = StatesEncoder2d(
#             **self.config.networks["state_encoder2d"]
#         ).to(self.device)

#         self.state_action_encoder = StatesActionsEncoder2d(
#             **self.config.networks["state_action_encoder2d"]
#         ).to(self.device)
        
#         # Residual Self Attention Block
#         self.RSA = RSA(
#             **self.config.networks["RSA"]
#         ).to(self.device)

#         self.POLICY_RSA = RSA(
#             **self.config.networks["RSA"]
#         ).to(self.device)

#         # Value Network
#         self.value_network = ValueNetwork(
#             **self.config.networks["value_network"]
#         ).to(self.device)

#         # Policy Network
#         if self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
#             if len(self.config.action_space.discrete_actions) > 1:
#                 raise NotImplementedError("Only support for one discrete branch currently!")
            
#             self.policy = DiscretePolicyNetwork(
#                 **self.config.networks["policy"]
#             ).to(self.device)
#         else:
#             raise NotImplementedError("No support for continuous actions yet!")


#         self.optimizer = optim.Adam(self.parameters(), lr=self.config.policy_learning_rate)

#     def get_actions(self, states: torch.Tensor, eval: bool=False):
#         num_steps, num_envs, num_agents, _ = states.shape
#         states_flat = self._flatten_multienv(states)
#         states_flat_emb = self.state_encoder(states_flat)
#         # states_emb_flat_rsa = self.POLICY_RSA(states_flat_emb)
#         probabilities = self.policy(states_flat_emb)[0]
        
#         if self.config.action_space.has_discrete():
#             m = dist.Categorical(probabilities)
            
#             if eval:
#                 actions_indices = m.probs.argmax(dim=-1)
#             else:
#                 actions_indices = m.sample()

#             log_probs = m.log_prob(actions_indices)
#             entropy = m.entropy()
#             return actions_indices.view(num_steps, num_envs, num_agents, 1), log_probs.view(num_steps, num_envs, num_agents, 1), entropy.view(num_steps, num_envs, num_agents, 1)
#         else:
#             raise NotImplementedError("No support for continuous actions yet!")

#     def get_baseline(self, batched_agent_states, batched_groupmates_states, batched_groupmates_actions):

#         # Encode the groupmates' states and actions together
#         groupmates_states_actions_encoded = self.state_action_encoder(batched_groupmates_states, batched_groupmates_actions)

#         # Encode the agent's state
#         agent_state_encoded = self.state_encoder(batched_agent_states)

#         # Combine agent state encoding with groupmates' state-action encodings
#         combined_states_actions = torch.cat([agent_state_encoded, groupmates_states_actions_encoded], dim=1)

#         # Pass the combined tensor through RSA
#         rsa_out = self.RSA(combined_states_actions)
        
#         # Mean pool over the second dimension (num_agents)
#         rsa_output_mean = rsa_out.mean(dim=1)

#         # Get the baselines for all agents from the value network
#         baselines = self.value_network(rsa_output_mean)

#         return baselines

#     def baseline_loss(self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor):
#         total_loss = torch.tensor(0.0).to(states.device)  # Initialize total loss to zero
#         num_steps, num_envs, num_agents, state_size = states.shape
#         _, _, _, num_actions = actions.shape
#         advantages = torch.zeros((num_steps, num_envs, num_agents, 1)).to(states.device)
        
#         # Initialize lists to store states and actions for all agents
#         agent_states_list = []
#         groupmates_states_list = []
#         groupmates_actions_list = []

#         # Accumulate states and actions for all agents
#         for agent_idx in range(num_agents):
#             agent_states_list.append(states[:, :, agent_idx, :].unsqueeze(2))
#             groupmates_states = torch.cat([states[:, :, :agent_idx, :], states[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
#             groupmates_actions = torch.cat([actions[:, :, :agent_idx, :], actions[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
#             groupmates_states_list.append(groupmates_states)
#             groupmates_actions_list.append(groupmates_actions)

#         # Concatenate lists to form batched tensors
#         batched_agent_states = torch.cat(agent_states_list, dim=2).unsqueeze(3)
#         batched_groupmates_states = torch.cat(groupmates_states_list, dim=2)
#         batched_groupmates_actions = torch.cat(groupmates_actions_list, dim=2)

#         # Reshape to combine steps and envs to simple batched dim
#         batched_agent_states = batched_agent_states.view(num_steps * num_envs * num_agents, 1, state_size) 
#         batched_groupmates_states = batched_groupmates_states.view(num_steps * num_envs * num_agents, num_agents - 1, state_size)
#         batched_groupmates_actions = batched_groupmates_actions.view(num_steps * num_envs * num_agents, num_agents - 1, num_actions)

#         # Get baselines for all agents at once
#         baselines = self.get_baseline(batched_agent_states, batched_groupmates_states, batched_groupmates_actions)
#         baselines = baselines.view(num_steps, num_envs, num_agents, 1)

#         # Calculate loss and advantages for all agents
#         for agent_idx in range(num_agents):
#             agent_baselines = baselines[:, :, agent_idx, :]
#             loss = F.mse_loss(agent_baselines, targets.detach())
#             total_loss += loss
#             advantages[:, :, agent_idx, :] = targets - agent_baselines

#         return total_loss / num_agents, advantages

#     def _flatten_multienv(self, t: torch.Tensor) -> torch.Tensor:
#         num_steps, num_envs, num_agents = t.shape[0], t.shape[1], t.shape[2]
#         t_flat = t.view(num_steps * num_envs, num_agents, -1)
#         return t_flat
    
#     def get_values(self, states: torch.Tensor) -> torch.Tensor:
#         num_steps, num_envs = states.shape[0], states.shape[1]
#         states_flattened = self._flatten_multienv(states)
#         states_encoded = self.state_encoder(states_flattened)
#         rsa_out = self.RSA(states_encoded)
#         rsa_output_mean = torch.mean(rsa_out, dim=1).squeeze()
#         values_flattened = self.value_network(rsa_output_mean)
#         values = values_flattened.view(num_steps, num_envs, 1)
#         return values
    
#     def y_lambda(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> torch.Tensor:
#         lambda_return = torch.zeros_like(rewards)
#         n = torch.ones_like(rewards[0])
#         for t in range(rewards.size(0)):
#             lambda_power = self.config.lambda_ ** (n - 1)
#             lambda_return += lambda_power * self.G_t(rewards, next_values, dones, truncs, t + 1)
#             n = (n * (1 - dones[t]) * (1 - truncs[t])) + 1  # increment or reset lambda power if done or trunc
#         return (1 - self.config.lambda_) * lambda_return
    
#     def G_t(self, rewards: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, n: int) -> torch.Tensor:
#         gamma_powers = torch.ones_like(rewards[:n])
#         running_gamma_power = torch.ones_like(rewards[0])
#         bootstrap_values = torch.zeros_like(rewards[:n])  # Initialize bootstrap values

#         for t in range(n):
#             gamma_powers[t] = self.config.gamma ** running_gamma_power
#             # If done or trunc, add the bootstrap value for that trajectory at the appropriate gamma power
#             bootstrap_values[t] = gamma_powers[t] * torch.clamp(dones[t] + truncs[t], 0, 1) * next_values[t]
#             # Reset gamma power if done or trunc
#             running_gamma_power = (running_gamma_power * (1 - dones[t]) * (1 - truncs[t])) + 1

#         # Multiply gamma_powers against rewards and masks
#         sum_rewards = torch.sum(rewards[:n] * (gamma_powers - 1) * (1 - dones[:n, :]) * (1 - truncs[:n, :]), dim=0)

#         # Sum the bootstrap values
#         total_bootstrap_value = torch.sum(bootstrap_values, dim=0)

#         # Total return value
#         G_t_val = sum_rewards + total_bootstrap_value

#         return G_t_val

#     def value_loss(self, states, next_states, rewards, dones, truncs):
#         with torch.no_grad():
#             values = self.get_values(states)
#             next_values = self.get_values(next_states)
#             # targets = self.y_lambda(values, next_values, rewards, dones, truncs)
#             targets, _ = self.compute_gae_and_targets(rewards, dones, truncs, values, next_values)
#         loss = F.mse_loss(self.get_values(states), targets)
#         return loss, targets
    
#     def policy_loss(self, log_probs: torch.Tensor, entropy: torch.Tensor, advantages: torch.Tensor):
#         # Combine the policy gradient loss and the entropy bonus
#         pg_loss = -log_probs * advantages.detach()
#         loss = (pg_loss - (self.config.entropy_coefficient * entropy))
#         return loss.mean()

#     def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor ):
#         with torch.no_grad():
#             perm = torch.randperm(states.size(0))
#             states = states[perm]
#             next_states = next_states[perm]
#             actions = actions[perm]
#             rewards = rewards[perm]
#             dones = dones[perm]
#             truncs = truncs[perm]

#         value_loss, targets = self.value_loss(states, next_states, rewards, dones, truncs)
#         baseline_loss, advantages = self.baseline_loss(states, actions, targets)
#         _, log_probs, entropy = self.get_actions(states)
#         policy_loss = self.policy_loss(log_probs, entropy, advantages)

#         loss = policy_loss + 0.5 * (value_loss + 0.5 * baseline_loss)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return {
#             "Total Loss": loss,
#             "Policy Loss": policy_loss,
#             "Value Loss": value_loss,
#             "Baseline Loss": baseline_loss,
#             "Entropy": entropy.mean(),
#             "Average Rewards": rewards.mean()
#         }