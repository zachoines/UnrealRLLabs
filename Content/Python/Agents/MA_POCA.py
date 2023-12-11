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
from Utility import RunningMeanStdNormalizer

class MAPocaAgentV1(Agent):
    def __init__(self, config: MAPOCAConfig):
        super(MAPocaAgentV1, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
            if isinstance(self.config.networks["policy"]["out_features"], list):
                raise NotImplementedError("Only support for one discrete branch currently!")
            
            self.policy = DiscretePolicyNetwork(
                **self.config.networks["policy"]
            ).to(self.device)
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        
        self.RSA = RSA(**self.config.networks["RSA"]).to(self.device)
        self.value = ValueNetwork(**self.config.networks["value_network"]).to(self.device)
        self.baseline = ValueNetwork(**self.config.networks["value_network"]).to(self.device)
        self.state_encoder = StatesEncoder(**self.config.networks["state_encoder"]).to(self.device)
        self.state_action_encoder = StatesActionsEncoder(**self.config.networks["state_action_encoder"]).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.policy_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.999)
        self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)

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
        agent_state_encoded = self.state_encoder(batched_agent_states)

        # Combine agent state encoding with groupmates' state-action encodings
        combined_states_actions = torch.cat([agent_state_encoded, groupmates_states_actions_encoded], dim=1).contiguous()

        # Pass the combined tensor through RSA
        rsa_out = self.RSA(combined_states_actions)
        
        # Mean pool over the second dimension (num_agents)
        rsa_output_mean = rsa_out.mean(dim=1)
    
        # Get the baselines for all agents from the value network
        baselines = self.baseline(F.leaky_relu(rsa_output_mean))
        return baselines.contiguous().view(num_steps, num_envs, num_agents, 1)

    def policies(self, x: torch.Tensor):
        num_steps, num_envs, num_agents, state_size = x.shape
        x = x.view(num_steps * num_envs * num_agents, state_size)
        x = self.state_encoder(x)
        x = self.RSA(x)
        x = x.contiguous().view(num_steps, num_envs, num_agents, self.config.embed_size)
        x = F.leaky_relu(x)
        return self.policy(x)

    def get_actions(self, states: torch.Tensor, eval: bool=False):
        if self.config.action_space.has_discrete():
            probabilities = self.policies(states)
            m = dist.Categorical(probs=probabilities)
            
            if eval:
                actions_indices = m.probs.argmax(dim=-1)
            else:
                actions_indices = m.sample()

            return actions_indices, m.log_prob(actions_indices), m.entropy()
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        
    def get_log_probs(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Calculate the log probability of taking given actions in given states under the current policy.
        """
        probabilities = self.policies(states)
        m = dist.Categorical(probs=probabilities)

        return m.log_prob(actions.squeeze(-1)), m.entropy()

    def lambda_returns(self, rewards, values_next, dones, truncs):
        num_steps, _, _ = rewards.shape
        returns = torch.zeros_like(rewards).to(self.device)
        returns[-1] = rewards[-1] + self.config.gamma * values_next[-1] * (1 - dones[-1]) * (1 - truncs[-1])
        
        for t in reversed(range(num_steps - 1)):
            non_terminal = (1 - dones[t]) * (1 - truncs[t])
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
    
    def get_values(self, states: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs, num_agents, state_size = states.shape
        x = self.state_encoder(states.view(num_steps * num_envs, num_agents, state_size))
        x = self.RSA(x)
        x = torch.mean(x, dim=1).squeeze()
        x = F.leaky_relu(x)
        values = self.value(x)
        return values.contiguous().view(num_steps, num_envs, 1)
    
    def trust_region_value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, epsilon=0.1) -> torch.Tensor:
        value_pred_clipped = old_values + (values - old_values).clamp(-epsilon, epsilon)
        loss_clipped = (returns - value_pred_clipped).pow(2)
        loss_unclipped = (returns - values).pow(2)
        value_loss = torch.max(loss_unclipped, loss_clipped).mean()
        return value_loss

    def value_loss(self, old_values, states, next_states, rewards, dones, truncs):
        values = self.get_values(states)
        with torch.no_grad():
            targets = self.lambda_returns(rewards, self.get_values(next_states), dones, truncs)
        loss = self.trust_region_value_loss(
            values,
            old_values,
            targets.detach(),
            self.config.clip_epsilon
        )
        return loss, targets
    
    def baseline_loss(self, states: torch.Tensor, actions: torch.Tensor, old_baselines: torch.Tensor, agent_states: torch.Tensor, groupmates_states: torch.Tensor, groupmates_actions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = states.shape
        total_loss = torch.tensor(0.0).to(self.device)
        advantages = torch.zeros_like(old_baselines).to(self.device)
        baselines = self.baselines(agent_states, groupmates_states, groupmates_actions, states, actions)
        
        # Calculate loss and advantages for all agents
        for agent_idx in range(num_agents):
            agent_baselines = baselines[:, :, agent_idx, :]
            old_agent_baselines = old_baselines[:, :, agent_idx, :]
            loss = self.trust_region_value_loss(
                agent_baselines,
                old_agent_baselines,
                targets,
                self.config.clip_epsilon
            )
            total_loss += loss
            advantages[:, :, agent_idx, :] = targets - agent_baselines

        return total_loss / num_agents, advantages
    
    def policy_loss(self, prev_log_probs: torch.Tensor, advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        log_probs, entropy = self.get_log_probs(states, actions)
        ratio = torch.exp(log_probs - prev_log_probs)
        loss_policy = -torch.mean(
            torch.min(
                ratio * advantages.squeeze().detach(),
                ratio.clip(1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages.squeeze().detach(),
            )
        )
        return loss_policy, entropy

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor):
        with torch.no_grad():
            batch_rewards = rewards
            # self.reward_normalizer.update(rewards)
            # rewards = self.reward_normalizer.normalize(rewards)

            # For trust region policy loss
            old_log_probs, _ = self.get_log_probs(states, actions)

            # For trust region baseline and value loss
            batched_agent_states, batched_groupmates_states, batched_groupmates_actions = self.AgentGroupMatesStatesActions(states, actions)
            old_baselines = self.baselines(batched_agent_states, batched_groupmates_states, batched_groupmates_actions, states, actions)
            old_values = self.get_values(states)

        _, _, num_agents, _ = states.shape
        num_steps, _, _ = rewards.shape 
        num_rounds = 4
        mini_batch_size = num_steps // 8

        total_loss_combined, total_loss_policy, total_loss_value, total_loss_baseline, total_entropy, = [], [], [], [], []
        for _ in range(num_rounds):

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
                self.optimizer.zero_grad()

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
                
                loss_policy, entropy = self.policy_loss(mb_old_log_probs, mb_advantages, mb_states, mb_actions)

                # Perform backpropagations
                total_loss = (loss_policy - self.config.entropy_coefficient * entropy.mean()) + 0.5 * (loss_value + ((1.0 / num_agents) * loss_baseline ))
                total_loss.backward()
                clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

                self.optimizer.step() 
                self.scheduler.step()

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
            # "Normalized Rewards" : rewards.mean(),
            "Learning Rate": self.scheduler.get_last_lr()[0]
        }

class MAPocaAgentV2(Agent):
    def __init__(self, config: MAPOCAConfig):
        super(MAPocaAgentV2, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(self.config).to(self.device)
        self.shared_critic = SharedCritic(self.config).to(self.device)

        self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.advantage_normalizer = RunningMeanStdNormalizer(device=self.device)

        self.critic_optimizer = optim.Adam(self.shared_critic.parameters(), lr=self.config.value_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config.policy_learning_rate)

        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.999)
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=100, gamma=0.999)

    def get_actions(self, states: torch.Tensor, eval: bool=False):
        num_steps, num_envs, num_agents, state_size = states.shape
        probabilities = self.policy(states.view(num_steps * num_envs * num_agents, state_size))
        
        if self.config.action_space.has_discrete():
            m = dist.Categorical(probs=probabilities)
            
            if eval:
                actions_indices = m.probs.argmax(dim=-1)
            else:
                actions_indices = m.sample()

            log_probs = m.log_prob(actions_indices)
            entropy = m.entropy()
            return actions_indices.contiguous().view(num_steps, num_envs, num_agents, 1), log_probs.contiguous().view(num_steps, num_envs, num_agents, 1), entropy.contiguous().view(num_steps, num_envs, num_agents, 1)
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        
    def get_log_probs(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Calculate the log probability of taking given actions in given states under the current policy.

        :param states: Tensor representing the states.
        :param actions: Tensor representing the actions taken in those states.
        :return: Tensor representing the log probabilities of these actions under the current policy.
        """
        # Flatten the states for processing
        num_steps, num_envs, num_agents, state_size = states.shape
        flat_states = states.view(num_steps * num_envs * num_agents, state_size)

        # Get the probability distribution over actions from the policy network
        probabilities = self.policy(flat_states)

        # Create a categorical distribution over the probabilities
        m = dist.Categorical(probs=probabilities)

        # Flatten actions for processing
        flat_actions = actions.view(-1)

        # Calculate log probabilities of the given actions
        log_probs = m.log_prob(flat_actions)

        # Reshape log_probs back to the original shape of states (minus the state size dimension)
        return log_probs.view(num_steps, num_envs, num_agents, 1), m.entropy().view(num_steps, num_envs, num_agents, 1)

    def lambda_returns(self, rewards, values_next, dones, truncs):
        num_steps, _, _ = rewards.shape
        returns = torch.zeros_like(rewards).to(self.device)
        
        # Set the return for the final timestep
        returns[-1] = rewards[-1] + self.config.gamma * values_next[-1] * (1 - dones[-1]) * (1 - truncs[-1])
        
        # Loop through each step backwards
        for t in reversed(range(num_steps - 1)):
            # Calculate the mask for non-terminal states
            non_terminal = (1 - dones[t]) * (1 - truncs[t])
            # Calculate the bootstrapped value
            bootstrapped_value = self.config.gamma * (self.config.lambda_ * returns[t + 1] + (1 - self.config.lambda_) * values_next[t])
            # Calculate the return
            returns[t] = rewards[t] + non_terminal * bootstrapped_value
        
        return returns

    def eligibility_trace_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        num_steps, _, _ = rewards.shape
        td_lambda_returns = torch.zeros_like(rewards)
        
        # Initialize eligibility traces
        eligibility_traces = torch.zeros_like(rewards)

        # Initialize the next_value for the calculation of returns at T + 1
        next_value = next_values[-1]
        
        for t in reversed(range(num_steps - 1)):
            # Clamp the sum of dones and truncs to ensure it doesn't exceed 1
            masks = 1.0 - torch.clamp(dones[t] + truncs[t], 0, 1)
            rewards_t = rewards[t]
            values_t = values[t]
            
            # Update the eligibility traces
            eligibility_traces[t] = self.config.gamma * self.config.lambda_ * eligibility_traces[t + 1] * masks + 1.0
            
            # Calculate TD Error
            td_error = rewards_t + self.config.gamma * next_value * masks - values_t
            
            # Calculate TD(λ) return for time t
            td_lambda_returns[t] = td_lambda_returns[t] + eligibility_traces[t] * td_error
            
            # Update the next_value for the next iteration of the loop
            next_value = values_t

        # Handle the last step separately as it does not have a next step
        td_lambda_returns[-1] = rewards[-1] + self.config.gamma * next_value * (1.0 - torch.clamp(dones[-1] + truncs[-1], 0, 1)) - values[-1]

        return td_lambda_returns

    def AgentGroupMatesStatesActions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = states.shape
        
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
        batched_agent_states = torch.cat(agent_states_list, dim=2).unsqueeze(3).contiguous()
        batched_groupmates_states = torch.cat(groupmates_states_list, dim=2).contiguous()
        batched_groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()
        return batched_agent_states, batched_groupmates_states, batched_groupmates_actions
    
    def get_values(self, states: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs, num_agents, state_size = states.shape
        states_flattened = states.view(num_steps * num_envs, num_agents, state_size)
        values_flattened = self.shared_critic.values(states_flattened).contiguous()
        values = values_flattened.view(num_steps, num_envs, 1)
        return values
    
    def trust_region_value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, epsilon=0.1) -> torch.Tensor:

        # Calculate clipped and uncliplled value loss
        value_pred_clipped = old_values + (values - old_values).clamp(-epsilon, epsilon)
        loss_clipped = (returns - value_pred_clipped).pow(2)
        loss_unclipped = (returns - values).pow(2)

        # Take the maximum of clipped and unclipped loss
        value_loss = torch.max(loss_unclipped, loss_clipped).mean()

        return value_loss

    def value_loss(self, old_values, states, next_states, rewards, dones, truncs):
        values = self.get_values(states)
        with torch.no_grad():
            # targets = self.lambda_returns(rewards, self.get_values(next_states), dones, truncs)
            targets = self.eligibility_trace_td_lambda_returns(rewards, values.clone(), self.get_values(next_states), dones, truncs)
            # targets, _ = self.compute_gae_and_targets(rewards, dones, truncs, values, next_values)
        loss = self.trust_region_value_loss(
            self.get_values(states),
            old_values,
            targets.detach(),
            self.config.clip_epsilon
        )
        return loss, targets
    
    def baseline_loss(self, states: torch.Tensor, actions: torch.Tensor, old_baselines: torch.Tensor, agent_states: torch.Tensor, groupmates_states: torch.Tensor, groupmates_actions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = states.shape
        total_loss = torch.tensor(0.0).to(self.device)
        advantages = torch.zeros_like(old_baselines).to(self.device)
        baselines = self.shared_critic.baselines(agent_states, groupmates_states, groupmates_actions, states, actions)
        
        for agent_idx in range(num_agents):
            agent_baselines = baselines[:, :, agent_idx, :]
            old_agent_baselines = old_baselines[:, :, agent_idx, :]
            loss = self.trust_region_value_loss(
                agent_baselines,
                old_agent_baselines,
                targets,
                self.config.clip_epsilon
            )
            total_loss += loss
            advantages[:, :, agent_idx, :] =  targets - agent_baselines

        return total_loss / num_agents, advantages
    
    def policy_loss(self, prev_log_probs: torch.Tensor, advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        log_probs, entropy = self.get_log_probs(states, actions)
        ratio = torch.exp(log_probs - prev_log_probs)
        loss_policy = -torch.mean(
            torch.min(
                ratio * advantages.detach(),
                ratio.clip(1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages.detach(),
            )
        )
        return loss_policy, entropy

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor):
        with torch.no_grad():
            batch_rewards = rewards
            #self.reward_normalizer.update(rewards)
            #rewards = self.reward_normalizer.normalize(rewards)

            # For trust region policy loss
            old_log_probs, _ = self.get_log_probs(states, actions)

            # For trust region baseline and value loss
            batched_agent_states, batched_groupmates_states, batched_groupmates_actions = self.AgentGroupMatesStatesActions(states, actions)
            old_baselines = self.shared_critic.baselines(batched_agent_states, batched_groupmates_states, batched_groupmates_actions, states, actions)
            old_values = self.get_values(states)

        _, _, num_agents, _ = states.shape
        num_steps, _, _ = rewards.shape 
        num_rounds = 4
        mini_batch_size = num_steps // 4

        total_loss_combined, total_loss_policy, total_loss_value, total_loss_baseline, total_entropy, = [], [], [], [], []
        for _ in range(num_rounds):

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
                self.critic_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()

                # Calculate losses and backprop
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
                
                # normalized_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # self.advantage_normalizer.update(mb_advantages)
                # normalized_advantages = self.advantage_normalizer.normalize(mb_advantages)
                loss_policy, entropy = self.policy_loss(mb_old_log_probs, mb_advantages, mb_states, mb_actions)

                # Perform backpropagations
                (loss_policy - self.config.entropy_coefficient * entropy.mean()).backward()
                (loss_value + ((1.0 / num_agents) * loss_baseline )).backward()

                clip_grad_norm_(self.shared_critic.parameters(), self.config.max_grad_norm)
                clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)

                self.critic_optimizer.step() 
                self.policy_optimizer.step()
                self.critic_scheduler.step()
                self.policy_scheduler.step()

                # Accumulate Metrics
                total_loss_combined += [(loss_baseline + loss_value + loss_policy)]
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
            "Policy Learning Rate": self.policy_scheduler.get_last_lr()[0],
            "Critic Learning Rate": self.critic_scheduler.get_last_lr()[0]
        }

class SharedCritic(nn.Module):
    def __init__(self, config: MAPOCAConfig):
        super(SharedCritic, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_encoder = StatesEncoder(**self.config.networks["state_encoder"]).to(self.device)
        self.state_action_encoder = StatesActionsEncoder(**self.config.networks["state_action_encoder"]).to(self.device)
        self.RSA = RSA(**self.config.networks["RSA"]).to(self.device)
        self.value = ValueNetwork(**self.config.networks["value_network"]).to(self.device)
        self.baseline = ValueNetwork(**self.config.networks["value_network"]).to(self.device)

    def values(self, states: torch.Tensor) -> torch.Tensor:
        x = self.state_encoder(states)
        x = self.RSA(x)
        x = torch.mean(x, dim=1).squeeze()
        x = F.leaky_relu(x)
        return self.value(x)
    
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
        agent_state_encoded = self.state_encoder(batched_agent_states)

        # Combine agent state encoding with groupmates' state-action encodings
        combined_states_actions = torch.cat([agent_state_encoded, groupmates_states_actions_encoded], dim=1).contiguous()

        # Pass the combined tensor through RSA
        rsa_out = self.RSA(combined_states_actions)
        
        # Mean pool over the second dimension (num_agents)
        rsa_output_mean = rsa_out.mean(dim=1)
    
        # Get the baselines for all agents from the value network
        baselines = self.baseline(F.leaky_relu(rsa_output_mean))
        return baselines.contiguous().view(num_steps, num_envs, num_agents, 1)
    
    def forward():
        pass
    
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
            if isinstance(self.config.networks["policy"]["out_features"], list):
                raise NotImplementedError("Only support for one discrete branch currently!")
            
            self.policy = DiscretePolicyNetwork(
                **self.config.networks["policy"]
            ).to(self.device)
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        
    def forward(self, states: torch.tensor):
        x = self.state_encoder(states)
        x = self.RSA(x)
        return self.policy(F.leaky_relu(x))
