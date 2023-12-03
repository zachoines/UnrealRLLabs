import torch
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torch.distributions import Normal, Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import Tensor
from typing import Dict
from Networks import *
from Config import A2CConfig
from Agents.Agent import Agent 

class A2C(Agent):
    def __init__(self, config: A2CConfig):
        super().__init__(config)
        """
        A class that represents an advantage actor critic agent

        Attributes:
            config (Config): Contains various network, training, and environment parameters
        """

        self.config: A2CConfig = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.config.action_space.has_continuous() and not self.config.action_space.has_discrete():
            self.actor = ContinousPolicyNetwork(**self.config.networks["policy"], device=self.device)    
        elif self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
            self.actor = DiscretePolicyNetwork(**self.config.networks["policy"], device=self.device)
        else:
            raise NotImplementedError

        self.critic = ValueNetwork(**self.config.networks["value"], device=self.device)
        self.target_critic = ValueNetwork(**self.config.networks["value"], device=self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())  
        self.optimizers = self.get_optimizers()
        
    def get_actions(self, states: torch.Tensor, eval=False, **kwargs) -> tuple[Tensor, Tensor]:
        states.to(device=self.device)

        if self.config.action_space.has_continuous() and not self.config.action_space.has_discrete():
            mean, std = self.actor(states)
            mean, std = torch.squeeze(mean), torch.squeeze(std)
            normal = Normal(mean, std) 

            if eval:
                action = mean
            else:
                action = normal.sample()
            
            return action, normal.log_prob(action).sum(dim=-1)
        elif self.config.action_space.has_discrete() and not self.config.action_space.has_continuous():
            branch_probs = self.actor(states)  # List of tensors for each branch
            if eval:
                # Choose the action with the highest probability in each branch
                actions = [torch.argmax(probs, dim=1) for probs in branch_probs]
                action_log_probs = [Categorical(probs).log_prob(action) for probs, action in zip(branch_probs, actions)]
                return torch.stack(actions, dim=1), torch.stack(action_log_probs, dim=1).sum(dim=1)
            else:
                # Sample actions from the distribution for each branch
                action_probs = [Categorical(probs) for probs in branch_probs]
                actions = [prob.sample() for prob in action_probs]
                log_probs = [prob.log_prob(action) for prob, action in zip(action_probs, actions)]
                return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1).sum(dim=1)
            
    def update(self, 
            states: torch.Tensor, 
            next_states: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            dones: torch.Tensor, 
            truncs: torch.Tensor
        )->dict[str, Tensor]:

        indices = torch.randperm(states.size(0))
        states = states[indices]
        next_states = next_states[indices]
        actions = actions[indices]
        rewards = rewards[indices]
        dones = dones[indices]
        truncs = truncs[indices]
        
        with torch.no_grad():
            batch_rewards = rewards.clone()
            values = self.critic(states)
            next_values = self.target_critic(next_states) # self.critic(next_states) 
            target, advantages = self.compute_gae_and_targets(rewards.unsqueeze(-1).clone(), dones.unsqueeze(-1).clone(), truncs.unsqueeze(-1).clone(), values.clone(), next_values.clone())
            # advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            
        if self.config.action_space.has_continuous():
            dist = Normal(*self.actor(states))
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            pg_loss = -log_probs * advantages
            critic_loss = F.mse_loss(self.critic(states.detach()), target)
        else:
            critic_loss = F.mse_loss(self.critic(states), torch.unsqueeze(target, dim=-1))
            branch_probs = self.actor(states.detach())
            dists = [Categorical(probs) for probs in branch_probs]
            actions = actions.to(dtype=torch.int64).T  # Transpose to match the shape of branch_probs
            log_probs = torch.stack([dist.log_prob(action) for dist, action in zip(dists, actions)], dim=1).sum(dim=1)
            entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
            pg_loss = -log_probs * advantages

        # Update the policy network
        self.optimizers['actor'].zero_grad()
        (pg_loss - (self.config.entropy_coefficient * entropy)).mean().backward()
        clip_grad_norm_(self.optimizers['actor'].param_groups[0]['params'], self.config.max_grad_norm)
        self.optimizers['actor'].step()

        # Update the value network
        self.optimizers['critic'].zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.optimizers['critic'].param_groups[0]['params'], self.config.max_grad_norm)
        self.optimizers['critic'].step()

        # Update target network
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {
            "Actor loss": pg_loss.mean(),
            "Critic loss": critic_loss.mean(),
            "Entropy": entropy.mean(),
            "Rewards": batch_rewards.mean()
        }
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        return {
            "actor": AdamW(
                self.actor.parameters(), 
                lr=self.config.policy_learning_rate
            ),
            "critic": AdamW(
                self.critic.parameters(), 
                lr=self.config.value_learning_rate
            )
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "actor": self.actor.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.actor.load_state_dict(state_dicts["actor"])