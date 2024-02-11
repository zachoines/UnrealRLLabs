import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Tuple
from Agents.Agent import Agent
from Agents.MA_POCA import * 
from Config import MAPOCAConfig
from Networks import *
from Utility import RunningMeanStdNormalizer, OneCycleCosineScheduler, ModifiedOneCycleLR

class SharedCriticLSTM(nn.Module):
    def __init__(self, config):
        super(SharedCriticLSTM, self).__init__()
        self.config = config
        self.RSA = RSA(**self.config.networks["RSA"])
        self.value = ValueNetwork(**self.config.networks["value_network"])
        self.baseline = ValueNetwork(**self.config.networks["value_network"])
        self.state_encoder = StatesEncoder(**self.config.networks["state_encoder"])
        self.state_action_encoder = StatesActionsEncoder(**self.config.networks["state_action_encoder"])
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.config.embed_size, max_seq_length=self.config.max_agents)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm_layer_value = LSTMNetwork(**self.config.networks["Value_LSTM"])
        self.lstm_layer_baseline = LSTMNetwork(**self.config.networks["Value_LSTM"])

    def baselines(self, batched_agent_states: torch.Tensor, batched_groupmates_states: torch.Tensor, batched_groupmates_actions: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None) -> torch.Tensor:
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
    
        # Inform currrent number of agents
        # num_agents_tensor = torch.full((num_steps * num_envs * num_agents, 1), num_agents/self.config.max_agents).to(self.device)
        # rsa_output_mean = torch.cat([rsa_output_mean, num_agents_tensor], dim=-1)

        # Temporal procesing through LSTM
        rsa_output_mean = rsa_output_mean.contiguous().view(num_steps, num_envs * num_agents, self.config.embed_size)
        agent_terminals = terminals.repeat_interleave(num_agents, dim=2).view(num_steps, num_envs * num_agents, 1)
        lstm_out, memory_states = self.lstm_layer_baseline(rsa_output_mean, dones=agent_terminals, input_hidden=memory_state)
        
        # Get the baselines for all agents from the value network
        baselines = self.baseline(lstm_out)
        return baselines.contiguous().view(num_steps, num_envs, num_agents, 1), memory_states
        
    def values(self, x: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None) -> torch.Tensor:
        num_steps, num_envs, num_agents, state_size = x.shape
        x = self.state_encoder(x)
        # num_agents_tensor = torch.full((num_steps * num_envs, 1), num_agents/self.config.max_agents).to(self.device)
        # x = torch.cat([x, num_agents_tensor], dim=-1)
        x = x.contiguous().view(num_steps, num_envs * num_agents, self.config.embed_size)
        agent_terminals = terminals.repeat_interleave(num_agents, dim=2).view(num_steps, num_envs * num_agents, 1)
        x, memory_states = self.lstm_layer_value(
            x, 
            dones=agent_terminals, 
            input_hidden=memory_state
        )
        x = x.view(num_steps * num_envs, num_agents, self.config.embed_size)
        x = self.RSA(x)
        x = torch.mean(x, dim=1).squeeze()
        values = self.value(x)
        return values.contiguous().view(num_steps, num_envs, 1), memory_states 
    
    def forward(self, x, terminals: torch.Tensor=None, memory_state: torch.Tensor=None):
        return self.values(x, terminals, memory_state)

class PolicyNetworkLSTM(nn.Module):
    def __init__(self, config):
        super(PolicyNetworkLSTM, self).__init__()
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
        self.lstm_layer = LSTMNetwork(**self.config.networks["Policy_LSTM"])

    def probabilities(self, x: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None):
        num_steps, num_envs, num_agents, state_size = x.shape
        x = self.state_encoder(x)
        x = x.contiguous().view(num_steps, num_envs * num_agents, self.config.embed_size)
        agent_terminals = terminals.repeat_interleave(num_agents, dim=2).view(num_steps, num_envs * num_agents, 1)
        x, memory_states = self.lstm_layer(
            x, 
            dones=agent_terminals, 
            input_hidden=memory_state
        )
        x = x.contiguous().view(num_steps * num_envs, num_agents, self.config.embed_size)
        x = self.RSA(self.positional_encodings(x))
        x = x.contiguous().view(num_steps, num_envs, num_agents, self.config.embed_size)
        return self.policy(x), memory_states
    
    def log_probs(self, states: torch.Tensor, actions: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None):
        probs, memory_states = self.probabilities(states, terminals, memory_state)
        m = dist.Categorical(probs=probs)
        return m.log_prob(actions.squeeze(-1)), m.entropy(), memory_states

    def forward(self, x, terminals: torch.Tensor=None, memory_state: torch.Tensor=None):
        return self.probabilities(x, terminals, memory_state)
    
class MAPocaLSTMAgent(Agent):
    def __init__(self, config: MAPOCAConfig):
        super(MAPocaLSTMAgent, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        self.policy = PolicyNetworkLSTM(self.config).to(self.device)
        self.shared_critic = SharedCriticLSTM(self.config).to(self.device)
        self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.advantage_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.config.policy_learning_rate)
        self.policy_scheduler = ModifiedOneCycleLR(
            self.policy_optimizer, 
            max_lr=self.config.policy_learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=0.20,
            final_div_factor=1000,
        ) 
        self.shared_critic_optimizer = optim.AdamW(self.shared_critic.parameters(), lr=self.config.value_learning_rate)
        self.shared_critic_scheduler = ModifiedOneCycleLR(
            self.shared_critic_optimizer, 
            max_lr=self.config.value_learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=0.20,
            final_div_factor=1000,
        )
        self.entropy_scheduler = OneCycleCosineScheduler(
            self.config.entropy_coefficient, self.config.anneal_steps, pct_start=0.20, div_factor=1.0, final_div_factor=1000
        )
        self.policy_clip_scheduler = OneCycleCosineScheduler(
            self.config.policy_clip, self.config.anneal_steps, pct_start=0.20, div_factor=2.0, final_div_factor=10.0
        )
        self.value_clip_scheduler = OneCycleCosineScheduler(
            self.config.value_clip, self.config.anneal_steps, pct_start=0.20, div_factor=2.0, final_div_factor=10.0
        )
        self.last_memory_state = None, None, None

    def to_gpu(self, x):
        if isinstance(x, tuple):
            return tuple(item.to(device=self.gpu_device) for item in x)
        return x.to(device=self.device)

    def to_cpu(self, x):
        if isinstance(x, tuple):
            return tuple(item.to(device=self.cpu_device) for item in x)
        return x.to(device=self.cpu_device)

    def get_actions(self, states: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, eval: bool=False):
        if self.config.action_space.has_discrete():
            self.policy.eval() # To disable dropout 
            probabilities, _ = self.policy.probabilities(states, torch.logical_or(dones, truncs))
            m = dist.Categorical(probs=probabilities)
            
            if eval:  
                actions_indices = m.probs.argmax(dim=-1)
            else:
                actions_indices = m.sample()

            return actions_indices, m.log_prob(actions_indices), m.entropy()
        else:
            raise NotImplementedError("No support for continuous actions yet!")

    def lambda_returns(self, rewards, values_next, terminals):
        num_steps, _, _ = rewards.shape
        returns = torch.zeros_like(rewards).to(self.device)
        # The last step's return is its reward plus the discounted next value, adjusted for terminals
        returns[-1] = rewards[-1] + self.config.gamma * values_next[-1] * (1 - terminals[-1])
        for t in reversed(range(num_steps - 1)):
            non_terminal = 1.0 - terminals[t]
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

    def value_loss(self, old_values: torch.Tensor, states: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None):
        # Compute values for states and next states
        with torch.no_grad():
            self.eval()
            states_plus_one = torch.cat((states, next_states[-1:, :]), dim=0)
            terminals_plus_one = torch.cat((terminals, torch.zeros_like(terminals[-1:, :])), dim = 0)
            values_plus_one, _ = self.shared_critic.values(states_plus_one, terminals_plus_one, memory_state)
            targets = self.compute_targets(rewards, terminals, values_plus_one[:-1], values_plus_one[1:])
            self.train()

        loss = self.trust_region_value_loss(
            self.shared_critic.values(states, terminals, memory_state)[0],
            old_values,
            targets.detach(),
            self.value_clip_scheduler.value()
        )
        return loss, targets
    
    def baseline_loss(self, states: torch.Tensor, actions: torch.Tensor, old_baselines: torch.Tensor, agent_states: torch.Tensor, groupmates_states: torch.Tensor, groupmates_actions: torch.Tensor, targets: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = states.shape
        total_loss = torch.tensor(0.0).to(self.device)
        advantages = torch.zeros_like(old_baselines).to(self.device)
        baselines, _ = self.shared_critic.baselines(
            agent_states, 
            groupmates_states, 
            groupmates_actions, 
            states, 
            actions, 
            terminals=terminals, 
            memory_state=memory_state
        )
        
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
    
    def policy_loss(self, prev_log_probs: torch.Tensor, advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None):
        log_probs, entropy, _ = self.policy.log_probs(states, actions, terminals, memory_state)
        ratio = torch.exp(log_probs - prev_log_probs)
        loss_policy = -torch.mean(
            torch.min(
                ratio * advantages.squeeze().detach(),
                ratio.clip(1.0 - self.policy_clip_scheduler.value(), 1.0 + self.policy_clip_scheduler.value()) * advantages.squeeze().detach(),
            )
        )
        return loss_policy, entropy
    
    def get_current_memory_state(self):
        return (
            self.shared_critic.lstm_layer_baseline.get_hidden(),
            self.shared_critic.lstm_layer_value.get_hidden(),
            self.policy.lstm_layer.get_hidden()
        )
    
    def init_memory_state(self, num_envs: int, num_agents: int):
        self.shared_critic.lstm_layer_baseline.init_hidden(num_envs * num_agents),
        self.shared_critic.lstm_layer_value.init_hidden(num_envs * num_agents),
        self.policy.lstm_layer.init_hidden(num_envs * num_agents)

    def set_memory_state(self, baseline: torch.Tensor, value: torch.Tensor, policy: torch.Tensor):
        self.shared_critic.lstm_layer_baseline.set_hidden(baseline)
        self.shared_critic.lstm_layer_value.set_hidden(value)
        self.policy.lstm_layer.set_hidden(policy)

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor):
        self.eval()

        num_steps, num_envs, num_agents, _ = states.shape 

        # Manually manage current LSTM memory states so to not disrupt online training 
        last_baseline_memory, last_value_memory, last_policy_memory = self.last_memory_state
        if last_baseline_memory == None or last_value_memory == None or last_policy_memory == None:
            self.init_memory_state(num_envs, num_agents)

        with torch.no_grad():
            batch_rewards = rewards.mean()
         
            # For trust region loss
            terminals = torch.logical_or(dones.to(dtype=torch.bool), truncs.to(dtype=torch.bool)).float()
            old_log_probs, _, old_policy_memory = self.policy.log_probs(
                states, 
                actions, 
                terminals, 
                last_policy_memory
            )
            batched_agent_states, batched_groupmates_states, batched_groupmates_actions = self.AgentGroupMatesStatesActions(states, actions)
            old_baselines, old_baseline_memory = self.shared_critic.baselines(
                batched_agent_states, 
                batched_groupmates_states, 
                batched_groupmates_actions, 
                states, 
                actions,
                terminals,
                last_baseline_memory
            )
            old_values, old_value_memory = self.shared_critic.values(
                states, 
                terminals, 
                last_value_memory
            )

            self.last_memory_state = self.get_current_memory_state()

            if self.config.normalize_rewards:
                self.reward_normalizer.update(rewards)
                rewards = self.reward_normalizer.normalize(rewards)

            # Put on RAM. Multi-agent computations are very GPU memory taxing, and should be moved over when only when needed.
            states = self.to_cpu(states)
            next_states = self.to_cpu(next_states)
            actions = self.to_cpu(actions)
            rewards = self.to_cpu(rewards)
            terminals = self.to_cpu(terminals)
            batched_agent_states = self.to_cpu(batched_agent_states)
            batched_groupmates_states = self.to_cpu(batched_groupmates_states)
            batched_groupmates_actions = self.to_cpu(batched_groupmates_actions)

            old_policy_memory = self.to_cpu(old_policy_memory)
            old_value_memory = self.to_cpu(old_value_memory)
            old_baseline_memory = self.to_cpu(old_baseline_memory)
            old_log_probs = self.to_cpu(old_log_probs)
            old_baselines = self.to_cpu(old_baselines)
            old_values = self.to_cpu(old_values)

        self.train()
        mini_batch_size = num_steps // self.config.num_mini_batches
        total_loss_combined, total_loss_policy, total_loss_value, total_loss_baseline, total_entropy = [], [], [], [], []

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
                    mb_batched_agent_states = self.to_gpu(batched_agent_states[ids])
                    mb_batched_groupmates_states = self.to_gpu(batched_groupmates_states[ids])
                    mb_batched_groupmates_actions = self.to_gpu(batched_groupmates_actions[ids])
                    mb_states = self.to_gpu(states[ids])
                    mb_next_states = self.to_gpu(next_states[ids])
                    mb_actions = self.to_gpu(actions[ids])
                    mb_terminals = self.to_gpu(terminals[ids])
                    mb_next_states = self.to_gpu(next_states[ids])
                    mb_rewards = self.to_gpu(rewards[ids])
                    mb_old_log_probs = self.to_gpu(old_log_probs[ids])
                    mb_old_values = self.to_gpu(old_values[ids])
                    mb_old_baselines = self.to_gpu(old_baselines[ids])
                    mb_old_policy_memory = self.to_gpu(old_policy_memory[ids])
                    mb_old_value_memory = self.to_gpu(old_value_memory[ids])
                    mb_old_baseline_memory = self.to_gpu(old_baseline_memory[ids])

                # Clear gradiants for the next batch
                self.policy_optimizer.zero_grad()
                self.shared_critic_optimizer.zero_grad()

                # Calculate losses
                loss_value, mb_targets = self.value_loss(
                    mb_old_values, 
                    mb_states, 
                    mb_next_states, 
                    mb_rewards, 
                    terminals=mb_terminals,
                    memory_state=mb_old_value_memory[0, :],
                )
                
                loss_baseline, mb_advantages = self.baseline_loss( # Long comp time
                    mb_states,
                    mb_actions,
                    mb_old_baselines,
                    mb_batched_agent_states,
                    mb_batched_groupmates_states,
                    mb_batched_groupmates_actions,
                    mb_targets,
                    terminals=mb_terminals,
                    memory_state=mb_old_baseline_memory[0, :],
                )
                
                if self.config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-6)
                    # self.advantage_normalizer.update(mb_advantages)
                    # mb_advantages = self.advantage_normalizer.normalize(mb_advantages)
                
                loss_policy, entropy = self.policy_loss(
                    mb_old_log_probs, 
                    mb_advantages, 
                    mb_states, 
                    mb_actions,
                    terminals=mb_terminals,
                    memory_state=mb_old_policy_memory[0, :],
                )

                # Perform backpropagations
                policy_loss = (loss_policy - (self.entropy_scheduler.value() * entropy.mean()))
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
        
        self.set_memory_state(*self.last_memory_state)
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
            "Value Clip": torch.tensor(self.value_clip_scheduler.value())
        }
