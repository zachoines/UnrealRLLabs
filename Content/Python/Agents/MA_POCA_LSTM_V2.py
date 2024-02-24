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
from Config import MAPOCA_LSTM_Light_Config
from Networks import *
from Utility import RunningMeanStdNormalizer, OneCycleCosineScheduler, ModifiedOneCycleLR

class SharedCriticLight(nn.Module):
    def __init__(self, config):
        super(SharedCriticLight, self).__init__()
        self.config = config
        self.value = ValueNetwork(**self.config.networks["critic_network"]["value_head"])
        self.baseline = ValueNetwork(**self.config.networks["critic_network"]["baseline_head"])
        self.RSA = RSA(**self.config.networks["critic_network"]["value_RSA"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def baselines(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to combine steps and envs to simple batched dim
        num_steps, num_envs, num_agents, _, embed_size = x.shape

        # Get the baselines for all agents from the value network
        x = x.view(num_steps * num_envs * num_agents, num_agents, embed_size)
        x = self.RSA(x)
        x = torch.mean(x, dim=1)
        return self.baseline(x).view(num_steps, num_envs, num_agents, 1)
        
    def values(self, x: torch.Tensor) -> torch.Tensor:
        num_steps, num_envs, num_agents, embed_size = x.shape
        x = torch.mean(x, dim=2)
        values = self.value(x)
        return values.contiguous().view(num_steps, num_envs, 1)
    
    def forward(self, x):
        pass  # TODO

class PolicyNetworkLight(nn.Module):
    def __init__(self, config):
        super(PolicyNetworkLight, self).__init__()
        self.config = config
        if self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
            if isinstance(self.config.networks["policy_network"]["policy_head"]["out_features"], list):
                raise NotImplementedError("Only support for one discrete branch currently!")
            
            self.policy = DiscretePolicyNetwork(
                **self.config.networks["policy_network"]["policy_head"]
            )
        else:
            raise NotImplementedError("No support for continuous actions yet!")
        
    def probabilities(self, obs: torch.Tensor):
        return self.policy(obs)
    
    def log_probs(self, obs: torch.Tensor, actions: torch.Tensor):
        probs = self.probabilities(obs)
        m = dist.Categorical(probs=probs)
        return m.log_prob(actions.squeeze(-1)), m.entropy()

    def forward(self, x):
        pass # TODO

class MultiAgentEmbeddingNetwork(nn.Module):
    def __init__(self, config):
        super(MultiAgentEmbeddingNetwork, self).__init__()
        self.config = config
        self.agent_embedding_encoder = StatesEncoder(**self.config.networks["MultiAgentEmbeddingNetwork"]["agent_embedding_encoder"])
        self.agent_obs_encoder = StatesEncoder(**self.config.networks["MultiAgentEmbeddingNetwork"]["agent_obs_encoder"])
        self.obs_action_encoder = StatesActionsEncoder(**self.config.networks["MultiAgentEmbeddingNetwork"]["obs_actions_encoder"])
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def split_agent_groupmates_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, num_envs, num_agents, _ = obs.shape
        
        agent_obs_list = []
        groupmates_obs_list = []

        for agent_idx in range(num_agents):
            agent_obs_list.append(obs[:, :, agent_idx, :].unsqueeze(2))
            groupmates_obs_list.append(torch.cat([obs[:, :, :agent_idx, :], obs[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2))

        agent_obs = torch.cat(agent_obs_list, dim=2).unsqueeze(3).contiguous()
        groupmates_obs = torch.cat(groupmates_obs_list, dim=2).contiguous()
        return agent_obs, groupmates_obs
    
    def split_agent_obs_groupmates_obs_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, num_agents, _ = obs.shape
        
        agent_obs_list = []
        groupmates_obs_list = []
        groupmates_actions_list = []

        for agent_idx in range(num_agents):
            agent_obs_list.append(obs[:, :, agent_idx, :].unsqueeze(2))
            groupmates_obs = torch.cat([obs[:, :, :agent_idx, :], obs[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
            groupmates_actions = torch.cat([actions[:, :, :agent_idx, :], actions[:, :, agent_idx+1:, :]], dim=2).unsqueeze(2)
            groupmates_obs_list.append(groupmates_obs)
            groupmates_actions_list.append(groupmates_actions)

        agent_obs = torch.cat(agent_obs_list, dim=2).unsqueeze(3).contiguous()
        groupmates_obs = torch.cat(groupmates_obs_list, dim=2).contiguous()
        groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()
        return agent_obs, groupmates_obs, groupmates_actions
        
    def encode_groupmates_obs_actions(self, groupmates_embeddings: torch.Tensor, groupmates_actions: torch.Tensor) -> torch.Tensor:
        
        # Reshape to combine steps and envs to simple batched dim
        num_steps, num_envs, num_agents, num_groupmates, embed_size = groupmates_embeddings.shape

        # Groupmates' obs and actions together
        return self.obs_action_encoder(groupmates_embeddings, groupmates_actions)
    
    def encode_state(self, agent_obs: torch.Tensor):
        num_steps, num_envs, num_agents, obs_size = agent_obs.shape

        # Encode the agent's obs
        agent_obs_encoded = self.agent_obs_encoder(agent_obs)
        return agent_obs_encoded

class MAPocaLSTMAgentLight(Agent): # 'Light' as in more memory efficient
    def __init__(self, config: MAPOCA_LSTM_Light_Config):
        super(MAPocaLSTMAgentLight, self).__init__(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        self.embedding_network = MultiAgentEmbeddingNetwork(self.config).to(self.device)
        self.policy = PolicyNetworkLight(self.config).to(self.device)
        self.shared_critic = SharedCriticLight(self.config).to(self.device)
        self.Shared_RSA = RSA(**self.config.networks["RSA"]).to(self.device)
        self.Shared_LSTM = LSTMNetwork(**self.config.networks["LSTM"]).to(self.device)
        self.positional_encodings = PositionalEncoding(state_embedding_size=self.config.networks["RSA"]["embed_size"], max_seq_length=self.config.max_agents)

        self.reward_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.advantage_normalizer = RunningMeanStdNormalizer(device=self.device)
        self.optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        self.scheduler = ModifiedOneCycleLR(
            self.optimizer, 
            max_lr=self.config.learning_rate,
            total_steps=self.config.anneal_steps,
            anneal_strategy='cos',
            pct_start=0.10,
            div_factor=25,
            final_div_factor=1000,
        ) 
        self.entropy_scheduler = OneCycleCosineScheduler(
            self.config.entropy_coefficient, 
            self.config.anneal_steps,
            pct_start=0.10, 
            div_factor=1.0,
            final_div_factor=50
        )
        self.policy_clip_scheduler = OneCycleCosineScheduler(
            self.config.policy_clip, 
            self.config.anneal_steps, 
            pct_start=0.10, 
            div_factor=1.0, 
            final_div_factor=4.0
        )
        self.value_clip_scheduler = OneCycleCosineScheduler(
            self.config.value_clip, 
            self.config.anneal_steps, 
            pct_start=0.10, 
            div_factor=1.0, 
            final_div_factor=10.0
        )
        self.last_memory_state = None
        
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
            probabilities = self.policy.probabilities(
                self.embed_states(
                    states, 
                    terminals=torch.logical_or(dones, truncs)
                )[0]
            )
            m = dist.Categorical(probs=probabilities)
            
            if eval:  
                actions_indices = m.probs.argmax(dim=-1)
            else:
                actions_indices = m.sample()

            return actions_indices, m.log_prob(actions_indices), m.entropy()
        else:
            raise NotImplementedError("No support for continuous actions yet!")

    def trust_region_value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, epsilon=0.10) -> torch.Tensor:
        diffs = values - old_values
        value_pred_clipped = old_values + (diffs).clamp(-epsilon, epsilon)
        loss_clipped = (returns - value_pred_clipped)**2
        loss_unclipped = (returns - values)**2
        value_loss = torch.mean(torch.max(loss_clipped, loss_unclipped))
        return value_loss

    def value_loss(
            self, 
            agent_obs_embeddings: torch.tensor, 
            states: torch.Tensor, 
            next_states: torch.Tensor,
            old_values: torch.Tensor, 
            rewards: torch.Tensor, 
            terminals: torch.Tensor=None, 
            memory_state: torch.Tensor=None
        ):
        
        with torch.no_grad():
            self.eval()
            values_plus_one = self.shared_critic.values(
                self.embed_states(
                    torch.cat((states, next_states[-1:, :]), dim=0), 
                    torch.cat((terminals, torch.zeros_like(terminals[-1:, :])), dim = 0), 
                    memory_state
                )[0]
            )
            targets = self.compute_targets(rewards, terminals, values_plus_one[:-1], values_plus_one[1:])
            self.train()

        loss = self.trust_region_value_loss(
            self.shared_critic.values(agent_obs_embeddings),
            old_values,
            targets.detach(),
            self.value_clip_scheduler.value()
        )
        return loss, targets

    def baseline_loss(self, embeddings: torch.Tensor, old_baselines: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, num_agents, _ = embeddings.shape
        baseline_losses = []
        advantages = torch.zeros_like(old_baselines).to(self.device)
        baselines = self.shared_critic.baselines(embeddings)
        
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
            baseline_losses.append(loss)
            advantages[:, :, agent_idx, :] = targets - agent_baselines
        average_baseline_loss = torch.mean(
            torch.stack(baseline_losses, dim=0)
        )
        return average_baseline_loss, advantages
    
    def policy_loss(self, embeddings: torch.Tensor, prev_log_probs: torch.Tensor, advantages: torch.Tensor, actions: torch.Tensor):
        log_probs, entropy = self.policy.log_probs(
            embeddings,
            actions
        )
        ratio = torch.exp(log_probs - prev_log_probs)
        loss_policy = -torch.mean(
            torch.min(
                ratio * advantages.squeeze().detach(),
                ratio.clip(1.0 - self.policy_clip_scheduler.value(), 1.0 + self.policy_clip_scheduler.value()) * advantages.squeeze().detach(),
            )
        )
        return loss_policy, entropy
    
    def get_current_memory_state(self):
        return self.Shared_LSTM.get_hidden()
    
    def init_memory_state(self, num_envs: int, num_agents: int):
        self.Shared_LSTM.init_hidden(num_envs * num_agents)

    def set_memory_state(self, mem: torch.Tensor):
        self.Shared_LSTM.set_hidden(mem)
    
    def get_baseline_inputs(self, agent_embeddings, groupmates_actions):
        agent_embeds, groupmates_embeds = self.embedding_network.split_agent_groupmates_obs(
            self.embedding_network.agent_embedding_encoder(agent_embeddings)
        )
        groupmates_embeds_with_actions = self.embedding_network.encode_groupmates_obs_actions(groupmates_embeds, groupmates_actions)
        x = torch.cat(
            [agent_embeds, groupmates_embeds_with_actions], 
            dim=3
        ).contiguous()
        return x

    def apply_lstm(self, encoded_agent_obs: torch.Tensor, terminals: torch.Tensor=None, memory_state: torch.Tensor=None) -> torch.Tensor:
        # Reshape to combine steps and envs to simple batched dim
        num_steps, num_envs, num_agents, embed_size = encoded_agent_obs.shape
    
        # Temporal procesing through LSTM
        x = encoded_agent_obs.contiguous().view(num_steps, num_envs * num_agents, embed_size)
        agent_terminals = terminals.repeat_interleave(num_agents, dim=2).view(num_steps, num_envs * num_agents, 1)
        x, memory_states = self.Shared_LSTM(x, dones=agent_terminals, input_hidden=memory_state)
        return x.contiguous().view(num_steps, num_envs, num_agents, -1), memory_states
          
    def apply_attention(self, embeddings):
        num_steps, num_envs, num_agents, embed_size = embeddings.shape

        # Combine agent state encoding with groupmates' obs-action encodings
        x = embeddings.contiguous().view(num_steps * num_envs, num_agents, embed_size)

        # Pass the combined tensor through RSA
        x = self.Shared_RSA(self.positional_encodings(x))
        return x.contiguous().view(num_steps, num_envs, num_agents, embed_size)
    
    def embed_states(self, states, terminals=None, memory_state=None):
        states_embedded = self.embedding_network.encode_state(states)
        states_embedded, old_memory = self.apply_lstm(states_embedded, terminals=terminals, memory_state=memory_state)
        states_embedded = self.apply_attention(states_embedded)
        return states_embedded, old_memory

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor):
        self.eval()

        num_steps, num_envs, num_agents, _ = states.shape 

        # Manually manage current LSTM memory states so to not disrupt online training 
        last_memory = self.last_memory_state
        if last_memory == None:
            self.init_memory_state(num_envs, num_agents)

        with torch.no_grad():
            batch_rewards = rewards.mean()
            terminals = torch.logical_or(dones.to(dtype=torch.bool), truncs.to(dtype=torch.bool)).float()
            
            # Extract agent's groupmates actions
            _, _, groupmates_actions = self.embedding_network.split_agent_obs_groupmates_obs_actions(states, actions)
            
            # Apply spatial attention then lstm for all agent's to create shared embeddings for Policy, Value, and Baseline Networks
            states_embedded, old_memory = self.embed_states(states, terminals=terminals, memory_state=last_memory)
            
            old_log_probs, _ = self.policy.log_probs(
                states_embedded,
                actions
            )
            old_baselines = self.shared_critic.baselines(
                self.get_baseline_inputs(states_embedded, groupmates_actions)
            )
            old_values = self.shared_critic.values(
                states_embedded
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

            old_memory = self.to_cpu(old_memory)
            old_log_probs = self.to_cpu(old_log_probs)
            old_baselines = self.to_cpu(old_baselines)
            old_values = self.to_cpu(old_values)

            groupmates_actions = self.to_cpu(groupmates_actions)

        self.train()
        mini_batch_size = num_steps // self.config.num_mini_batches
        total_loss_combined, total_loss_policy, total_loss_value, total_loss_baseline, total_entropy = [], [], [], [], []

        for _ in range(self.config.num_epocs):

            # Generate a random order of mini-batches
            mini_batch_start_indices = list(range(0, states.size(0), mini_batch_size))
            np.random.shuffle(mini_batch_start_indices)

            # Process each mini-batch
            for start_idx in mini_batch_start_indices:
                end_idx = min(start_idx + mini_batch_size, states.size(0))
                ids = slice(start_idx, end_idx)

                # Extract mini-batch data; Place onto device
                with torch.no_grad():
                    mb_states = self.to_gpu(states[ids])
                    mb_next_states = self.to_gpu(next_states[ids])
                    mb_actions = self.to_gpu(actions[ids])
                    mb_terminals = self.to_gpu(terminals[ids])
                    mb_next_states = self.to_gpu(next_states[ids])
                    mb_rewards = self.to_gpu(rewards[ids])
                    mb_old_log_probs = self.to_gpu(old_log_probs[ids])
                    mb_old_values = self.to_gpu(old_values[ids])
                    mb_old_baselines = self.to_gpu(old_baselines[ids])
                    mb_old_memory = self.to_gpu(old_memory[ids])
                    mb_groupmates_actions = self.to_gpu(groupmates_actions[ids])

                # Clear gradiants for the next batch
                self.optimizer.zero_grad()
                mb_states_embed, _ = self.embed_states(mb_states, terminals=mb_terminals, memory_state=mb_old_memory[0, :])

                # Calculate losses
                loss_value, mb_targets = self.value_loss(
                    mb_states_embed,
                    mb_states, 
                    mb_next_states, 
                    mb_old_values,
                    mb_rewards, 
                    terminals=mb_terminals,
                    memory_state=mb_old_memory[0, :],
                )
                
                loss_baseline, mb_advantages = self.baseline_loss(
                    self.get_baseline_inputs(
                        mb_states_embed, 
                        mb_groupmates_actions
                    ),
                    mb_old_baselines,
                    mb_targets,
                )
                
                if self.config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-6)
 
                loss_policy, entropy = self.policy_loss(
                    mb_states_embed,
                    mb_old_log_probs, 
                    mb_advantages, 
                    mb_actions
                )

                # Perform backpropagations
                policy_loss = (loss_policy - (self.entropy_scheduler.value() * entropy.mean()))
                total_loss = policy_loss + 0.5 * (loss_value + (0.5 * loss_baseline))
                total_loss.backward()

                clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step() 
                self.scheduler.step() 

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
        
        self.set_memory_state(self.last_memory_state)
        return {
            "Total Loss": total_loss_combined,
            "Policy Loss": total_loss_policy,
            "Value Loss": total_loss_value,
            "Baseline Loss": total_loss_baseline,
            "Entropy": total_entropy,
            "Average Rewards": batch_rewards,
            "Normalized Rewards": rewards.mean(),
            "Learning Rate": self.scheduler.get_last_lr()[0],
            "Entropy Coeff": torch.tensor(self.entropy_scheduler.value()),
            "Policy Clip": torch.tensor(self.policy_clip_scheduler.value()),
            "Value Clip": torch.tensor(self.value_clip_scheduler.value())
        }
