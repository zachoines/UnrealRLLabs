import torch
from torch.optim import Optimizer, AdamW
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import Tensor
from typing import Dict, List
from Networks import *
from Config import *

class Agent:
    def __init__(
            self,
            config: Config,
            policy_learning_rate: float = 1e-4,
            value_learning_rate: float = 1e-3,
            gamma: float = 0.99, 
            entropy_coefficient: float = 1e-2,
            max_grad_norm: float = 0.5
        ):

        self.config: Config = config

        """
        Base class representing a reinforcement learning agent.

        Attributes:
            config (Config): Contains various network, training, and environment parameters
            policy_learning_rate (float): The learning rate for the policy network.
            value_learning_rate (float): The learning rate for the value network.
            gamma (float): The discount factor for future rewards.
            entropy_coefficient (float): The coefficient for the entropy bonus, encouraging exploration.
            max_grad_norm (float): The maximum allowed norm for the gradient (for gradient clipping).
        """
        self.optimizers = {}
        self.eps = 1e-8
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
    
    def save(self, location: str)->None:
        torch.save(self.state_dict(), location)

    def load(self, location: str)->None:
        raise NotImplementedError
    
    def state_dict(self)-> Dict[str,Dict]:
        raise NotImplementedError

    def get_actions(self, state: torch.Tensor, eval: bool = False, **kwargs)->tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    def rescaleAction(self, action : Tensor, min : float, max: float) -> torch.Tensor:
        return min + (0.5 * (action + 1.0) * (max - min))
    
    def calc_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        num_envs, num_steps = rewards.shape
        running_returns = torch.zeros(num_envs, dtype=torch.float32)
        returns = torch.zeros_like(rewards)
        
        for i in range(num_steps - 1, -1, -1):
            running_returns = rewards[:, i] + (1 - dones[:, i]) * self.gamma * running_returns
            returns[:, i] = running_returns

        return returns
    
    def save_train_state(self):
        return None

    def restore_train_state(self, state):
        pass
    
    def compute_gae_and_targets(self, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, gamma: float=0.99, lambda_: float=0.95):
        """
        Compute GAE and bootstrapped targets for PPO.

        :param rewards: (torch.Tensor) Rewards.
        :param dones: (torch.Tensor) Done flags.
        :param truncs: (torch.Tensor) Truncation flags.
        :param values: (torch.Tensor) State values.
        :param next_values: (torch.Tensor) Next state values.
        :param gamma: (float) Discount factor.
        :param lambda_: (float) GAE smoothing factor.
        :return: Bootstrapped targets and advantages.

        The λ parameter in the Generalized Advantage Estimation (GAE) algorithm acts as a trade-off between bias and variance in the advantage estimation.
        When λ is close to 0, the advantage estimate is more biased, but it has less variance. It would rely more on the current reward and less on future rewards. This could be useful when your reward signal is very noisy because it reduces the impact of that noise on the advantage estimate.
        On the other hand, when λ is close to 1, the advantage estimate has more variance but is less biased. It will take into account a longer sequence of future rewards.
        """
        batch_size = rewards.shape[0]  # Determine batch size from the rewards tensor
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(batch_size)):
            non_terminal = 1.0 - torch.clamp(dones[t, :] - truncs[t, :], 0.0, 1.0)
            delta = (rewards[t, :] + (gamma * next_values[t, :] * non_terminal)) - values[t, :]
            last_gae_lam = delta + (gamma * lambda_ * non_terminal * last_gae_lam)
            advantages[t, :] = last_gae_lam * non_terminal

        # Compute bootstrapped targets by adding unnormalized advantages to values 
        targets = values + advantages
        return targets, advantages

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor )->dict[str, Tensor]:
        raise NotImplementedError
    
class A2C(Agent):
    def __init__(
            self,
            config: Config,
            tau: float = 0.01
        ):
        super.__init__(self, config)
        
        """
        A class that represents an advantage actor critic agent

        Attributes:
            config (Config): Contains various network, training, and environment parameters
            tau (float): The factor for soft update of the target networks.
        """

        self.tau = tau

        if self.config.envParams.action_space.is_continuous():
            self.actor = ContinousPolicyNetwork(
                self.config.envParams.state_size, 
                self.config.envParams.num_actions, 
                self.config.networkParams.hidden_size, 
                device=self.config.trainParams.device
            )    
        elif self.config.envParams.action_space.is_discrete():
            self.actor = DiscretePolicyNetwork(
                self.config.envParams.state_size, 
                self.config.envParams.num_actions, 
                self.config.networkParams.hidden_size, 
                device=self.config.trainParams.device
            )
        else:
            raise NotImplementedError

        self.critic = ValueNetwork(
            self.config.envParams.state_size,
            self.config.networkParams.hidden_size, 
            device=self.config.trainParams.device
        )

        self.target_critic = ValueNetwork(
            self.config.envParams.state_size,
            self.config.networkParams.hidden_size, 
            device=self.config.trainParams.device
        )

        self.target_critic.load_state_dict(self.critic.state_dict())  
        self.optimizers = self.get_optimizers()
        
    def get_actions(self, states: torch.Tensor, eval=False, **kwargs)->tuple[Tensor, Tensor]:
        states.to(device=self.config.trainParams.device)

        if self.config.envParams.action_space.is_continuous():
            mean, std = self.actor(states)
            mean, std = torch.squeeze(mean), torch.squeeze(std)
            normal = torch.distributions.Normal(mean, std) 

            if eval:
                action = mean
            else:
                action = normal.sample()
            
            return action, normal.log_prob(action).sum(dim=-1)
        else:
            probs = self.actor(states)
            if eval:
                action = torch.argmax(probs)
                action_probs = torch.distributions.Categorical(probs)
                log_prob = action_probs.log_prob(action)
                return action, log_prob
            else:
               action_probs = torch.distributions.Categorical(probs)
               action = action_probs.sample()
               log_prob = action_probs.log_prob(action)
               return action, log_prob
          
    def update(self, 
            states: torch.Tensor, 
            next_states: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            dones: torch.Tensor, 
            truncs: torch.Tensor
        )->dict[str, Tensor]:
        
        # Reshape data
        with torch.no_grad():
            batch_rewards = rewards.clone()
            values: torch.Tensor = self.critic(states)
            next_values: torch.Tensor = self.target_critic(next_states)

            target, advantages = self.compute_gae_and_targets(
                rewards.unsqueeze(-1).clone(), 
                dones.unsqueeze(-1).clone(), 
                truncs.unsqueeze(-1).clone(), 
                values.clone(), 
                next_values.clone(), 
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            
        if self.config.envParams.action_space.is_continuous():
            
            # Calculate our losses
            dist = Normal(*self.actor(states))
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            pg_loss = -log_probs * advantages
            critic_loss = F.smooth_l1_loss(self.critic(states.detach()), target)

            # Update the policy network
            self.optimizers['actor'].zero_grad()
            (pg_loss - (self.entropy_coefficient * entropy)).mean().backward()
            clip_grad_norm_(self.optimizers['actor'].param_groups[0]['params'], self.max_grad_norm)
            self.optimizers['actor'].step()

            # Update the value network
            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.optimizers['critic'].param_groups[0]['params'], self.max_grad_norm)
            self.optimizers['critic'].step()

            # Update target network
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return {
                "Actor loss": pg_loss.mean(),
                "Critic loss": critic_loss.mean(),
                "Entropy": entropy.mean(),
                "Rewards": batch_rewards.mean()
            }
        else:
            actions = actions.to(dtype=torch.int64)

            # Critic loss
            critic_loss = F.mse_loss(self.critic(states), torch.unsqueeze(target, dim=-1))

            # Policy loss with entropy bonus
            action_probs = self.actor(states.clone().detach())
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy: torch.Tensor = dist.entropy()
            pg_loss: torch.Tensor = -log_probs * advantages
            loss: torch.Tensor = (pg_loss - self.entropy_coefficient * entropy).mean()

            # Optimize the models
            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            self.optimizers['critic'].step()
            
            self.optimizers['actor'].zero_grad()
            loss.backward()
            self.optimizers['actor'].step()

            # Update target network
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return {
                "Actor loss": pg_loss.mean(),
                "Critic loss": critic_loss,
                "Entropy": entropy.mean(),
                "Rewards": rewards.mean()
            }
    
    def get_optimizers(self) -> Dict[str, Optimizer]:
        return {
            "actor": AdamW(
                self.actor.parameters(), 
                lr=self.policy_learning_rate
            ),
            "critic": AdamW(
                self.critic.parameters(), 
                lr=self.value_learning_rate
            )
        }
    
    def state_dict(self)-> Dict[str,Dict]:
        return {
            "actor": self.actor.state_dict()
        }
    
    def load(self, location: str)->None:
        state_dicts = torch.load(location)
        self.actor.load_state_dict(state_dicts["actor"])