from typing import Dict, List, Tuple

class ActionSpace:
    def __init__(self, ContinuousActions: List[Tuple[float, float]], DiscreteActions: List[int]):
        self.continuous_actions = ContinuousActions
        self.discrete_actions = DiscreteActions

    def has_continuous(self) -> bool:
        return bool(self.continuous_actions)

    def has_discrete(self) -> bool:
        return bool(self.discrete_actions)

    def get_continuous_space(self) -> List[Tuple]:
        return self.continuous_actions

    def get_discrete_space(self) -> List[int]:
        return self.discrete_actions

class TrainInfo:
    def __init__(
            self,  
            BufferSize: int,
            BatchSize: int,
            NumEnvironments: int,
            MaxAgents: int
        ):
        self.BufferSize = BufferSize
        self.BatchSize = BatchSize 
        self.NumEnvironments = NumEnvironments 
        self.MaxAgents = MaxAgents

class ObservationSpace:
    def __init__(
            self,  
            StateSize: int,
            IsMultiAgent: bool,
            SingleAgentObsSize: int,
            NumAgents: int,
        ):
        self.state_size = StateSize
        self.is_multi_agent = IsMultiAgent
        self.single_agent_obs_size = SingleAgentObsSize
        self.max_agents = NumAgents

class BaseConfig:
    def __init__(
            self, 
            obs_space: ObservationSpace,
            action_space: ActionSpace,
            policy_learning_rate: float = 1e-4, 
            value_learning_rate: float = 1e-3, 
            gamma: float = 0.99, 
            entropy_coefficient: float = 1e-2, 
            max_grad_norm: float = 2.0, 
            num_environments: int = 1000, 
            eps: float = 1e-8,
            networks: Dict[str, Dict] = {},
            **kwargs
        ):
        self.action_space = action_space
        self.state_size = obs_space.state_size
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.num_environments = num_environments
        self.eps = eps
        self.networks = networks

class A2CConfig(BaseConfig):
    def __init__(
            self, 
            obs_space: ObservationSpace,
            action_space: ActionSpace,
            tau: float = 0.01,
            **kwargs
        ):
        super().__init__(
            obs_space,
            action_space,
            networks = {
                "policy" : 
                    {
                        "in_features": obs_space.state_size,
                        "out_features": len(action_space.continuous_actions),
                        "hidden_size": 128,
                    } 
                    if action_space.has_continuous() else
                    {
                        "in_features": obs_space.state_size,
                        "out_features": obs_space.action_space.discrete_actions,
                        "hidden_size": 256,
                    },
                "value" : {
                    "in_features": obs_space.state_size,
                    "hidden_size": 128,
                }
            },
            **kwargs
        )
        self.tau = tau

class MAPOCAConfig(BaseConfig):
    def __init__(
            self, 
            obs_space: ObservationSpace,
            action_space: ActionSpace,
            embed_size: int = 256,
            heads: int = 8,
            max_agents: int = 10,
            lambda_: float = 0.95,
            policy_clip: float = 0.2,
            value_clip: float = 0.1,
            hidden_size = 256,
            entropy_coefficient: float = 0.05,
            policy_learning_rate: float = 2.5e-4,
            value_learning_rate: float = 2.5e-4,
            max_grad_norm: float = 1.0,
            dropout_rate: float = 0.1,
            num_epocs: int = 3,
            num_mini_batches: int = 4,
            normalize_rewards: bool = False,
            normalize_advantages: bool = False,
            anneal_steps: int = 60000,
            icm_enabled: bool = False,
            **kwargs
        ):
        super().__init__(
            obs_space,
            action_space,
            entropy_coefficient = entropy_coefficient,
            policy_learning_rate = policy_learning_rate,
            value_learning_rate = value_learning_rate,
            max_grad_norm = max_grad_norm,
            networks={
                "RSA": {
                    "embed_size": embed_size, 
                    "heads": heads,
                    "dropout_rate": dropout_rate
                },
                "state_encoder": {
                    "state_dim": obs_space.single_agent_obs_size, 
                    "embed_size": embed_size
                },
                "state_action_encoder": {
                    "state_dim": obs_space.single_agent_obs_size, 
                    "action_dim": 
                        len(action_space.continuous_actions) 
                        if action_space.has_continuous() 
                        else len(action_space.discrete_actions), 
                    "embed_size": embed_size
                },
                "state_action_encoder2d": {
                    "state_size": obs_space.single_agent_obs_size, 
                    "action_dim": 
                        len(action_space.continuous_actions) 
                        if action_space.has_continuous() 
                        else len(action_space.discrete_actions), 
                    "embed_size": embed_size,
                    "dropout_rate": dropout_rate
                },
                "state_encoder2d": {
                    "state_size": obs_space.single_agent_obs_size, 
                    "embed_size": embed_size,
                    "dropout_rate": dropout_rate
                },
                "value_network": {
                    "in_features": embed_size + 1,
                    "hidden_size" : hidden_size,
                    "dropout_rate": dropout_rate
                },
                "policy" : {
                    "in_features": embed_size,
                    "out_features": 
                        len(action_space.continuous_actions) 
                        if action_space.has_continuous() 
                        else action_space.discrete_actions[0],
                    "hidden_size": hidden_size,
                    "dropout_rate": dropout_rate
                },
                "ICM": {
                    "icm_gain": .25,
                    "icm_beta": 0.5,
                    "embed_size": 64,
                    "state_encoder": {
                        "state_dim": obs_space.single_agent_obs_size, 
                        "embed_size": 64,
                    },
                    "state_encoder2d": {
                        "state_size": obs_space.single_agent_obs_size, 
                        "embed_size": 64,
                        "dropout_rate": dropout_rate
                    },
                    "rsa" : {
                        "embed_size": 64, 
                        "heads": 4,
                        "dropout_rate": dropout_rate
                    },
                    "inverse_head": {
                        "in_features": 2 * (64), 
                        "hidden_size": 128,
                        "out_features": len(action_space.continuous_actions) 
                            if action_space.has_continuous() 
                            else action_space.discrete_actions[0],
                        "dropout_rate": dropout_rate
                    },
                    "forward_head": {
                        "in_features": (64) + len(action_space.discrete_actions), 
                        "hidden_size": 128,
                        "out_features": 64,
                        "dropout_rate": dropout_rate
                    },
                },
            },
            **kwargs
        )

        self.embed_size = embed_size
        self.heads = heads
        self.lambda_ = lambda_
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.max_agents = max_agents
        self.num_epocs = num_epocs
        self.num_mini_batches = num_mini_batches
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        self.anneal_steps = anneal_steps
        self.icm_enabled = icm_enabled

class MAPOCALSTMConfig(BaseConfig):
    def __init__(
            self, 
            obs_space: ObservationSpace,
            action_space: ActionSpace,
            max_agents: int = 10,
            lambda_: float = 0.95,
            policy_clip: float = 0.1,
            value_clip: float = 0.1,
            entropy_coefficient: float = 0.1,
            policy_learning_rate: float = 1e-4,
            value_learning_rate: float = 5e-4,
            max_grad_norm: float = .5,
            dropout_rate: float = 0.00,
            num_epocs: int = 4,
            num_mini_batches: int = 8,
            normalize_rewards: bool = False,
            normalize_advantages: bool = True,
            anneal_steps: int = 30000,
            **kwargs
        ):
        super().__init__(
            obs_space,
            action_space,
            entropy_coefficient = entropy_coefficient,
            policy_learning_rate = policy_learning_rate,
            value_learning_rate = value_learning_rate,
            max_grad_norm = max_grad_norm,
            networks={
                "policy_network" : {
                    "state_encoder": {
                        "input_size": obs_space.single_agent_obs_size, 
                        "output_size": 64,
                        "dropout_rate": dropout_rate
                    },
                    "RSA": {
                        "embed_size": 64,
                        "heads": 4,
                        "dropout_rate": dropout_rate
                    },
                    "LSTM" : {
                        "in_features": 64, 
                        "output_size": 128, 
                        "num_layers": 2,
                        "dropout": dropout_rate
                    },
                    "policy_head" : {
                        "in_features": 128,
                        "out_features": 
                            len(action_space.continuous_actions) 
                            if action_space.has_continuous() 
                            else action_space.discrete_actions[0],
                        "hidden_size": 256,
                        "dropout_rate": dropout_rate
                    },
                },
                "value_network" : {
                    "state_encoder": {
                        "input_size": obs_space.single_agent_obs_size, 
                        "output_size": 64,
                        "dropout_rate": dropout_rate
                    },
                    "state_action_encoder": {
                        "state_dim": obs_space.single_agent_obs_size, 
                        "action_dim": 
                            len(action_space.continuous_actions) 
                            if action_space.has_continuous() 
                            else len(action_space.discrete_actions), 
                        "output_size": 64,
                        "dropout_rate": dropout_rate
                    },
                    "RSA": {
                        "embed_size": 64,
                        "heads": 4,
                        "dropout_rate": dropout_rate
                    },
                    "LSTM" : {
                        "in_features": 64, 
                        "output_size": 128, 
                        "num_layers": 2,
                        "dropout": dropout_rate
                    },
                    "value_head": {
                        "in_features": 128,
                        "hidden_size": 256,
                        "dropout_rate": dropout_rate
                    }
                }
            },
            **kwargs
        )

        self.lambda_ = lambda_
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.max_agents = max_agents
        self.num_epocs = num_epocs
        self.num_mini_batches = num_mini_batches
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        self.anneal_steps = anneal_steps

class MAPOCA_LSTM_Light_Config(BaseConfig):
    def __init__(
            self, 
            obs_space: ObservationSpace,
            action_space: ActionSpace,
            max_agents: int = 10,
            lambda_: float = 0.95,
            policy_clip: float = 0.1,
            value_clip: float = 0.1,
            entropy_coefficient: float = 0.1,
            learning_rate: float = 3e-4,
            max_grad_norm: float = .5,
            dropout_rate: float = 0.0,
            num_epocs: int = 8,
            num_mini_batches: int = 4,
            normalize_rewards: bool = False,
            normalize_advantages: bool = True,
            anneal_steps: int = 50000,
            **kwargs
        ):
        super().__init__(
            obs_space,
            action_space,
            entropy_coefficient = entropy_coefficient,
            max_grad_norm = max_grad_norm,
            networks={
                "MultiAgentEmbeddingNetwork" : {
                    "agent_obs_encoder": {
                        "input_size": obs_space.single_agent_obs_size, 
                        "output_size":  128,
                        "dropout_rate": dropout_rate,
                        "activation": True
                    },
                    "agent_embedding_encoder": {
                        "input_size": 256,
                        "output_size": 128,
                        "dropout_rate": dropout_rate,
                        "activation": True
                    },
                    "obs_actions_encoder": {
                        "state_dim": 128, 
                        "action_dim": 
                            len(action_space.continuous_actions) 
                            if action_space.has_continuous() 
                            else len(action_space.discrete_actions), 
                        "output_size":  128, # needs to match output of agent_embedding_encoder
                        "dropout_rate": dropout_rate,
                        "activation": True
                    },    
                },
                "LSTM" : {
                    "in_features": 128, 
                    "output_size": 256, 
                    "num_layers": 2,
                    "dropout": dropout_rate
                },
                "RSA": {
                    "embed_size": 256,
                    "heads": 16,
                    "dropout_rate": dropout_rate
                },
                "policy_network" : {               
                    "policy_head" : {
                        "in_features": 256,
                        "out_features": 
                            len(action_space.continuous_actions) 
                            if action_space.has_continuous() 
                            else action_space.discrete_actions[0],
                        "hidden_size": 512,
                        "dropout_rate": dropout_rate
                    },
                },
                "critic_network" : {
                    "baseline_head" : {
                        "in_features": 128,
                        "hidden_size": 256,
                        "dropout_rate": dropout_rate
                    },
                    "value_head": {
                        "in_features": 256,
                        "hidden_size": 512,
                        "dropout_rate": dropout_rate
                    },
                    "value_RSA" : {
                        "embed_size": 128,
                        "heads": 8,
                        "dropout_rate": dropout_rate
                    }
                }
            },
            **kwargs
        )
        self.lambda_ = lambda_
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.learning_rate = learning_rate
        self.max_agents = max_agents
        self.num_epocs = num_epocs
        self.num_mini_batches = num_mini_batches
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        self.anneal_steps = anneal_steps
        self.schedulers = {
            "learning_rate" : {
                "max_lr": self.learning_rate,
                "total_steps": self.anneal_steps,
                "anneal_strategy": 'cos',
                "pct_start": 0.10,
                "div_factor": 25,
                "final_div_factor": 500,
            },
            "entropy" : {
                "max_entropy": self.entropy_coefficient, 
                "anneal_steps" : self.anneal_steps,
                "pct_start": 0.10, 
                "div_factor": 1.0,
                "final_div_factor": 10
            },
            "policy_clip" : {
                "max_clip": self.policy_clip, 
                "anneal_steps" : self.anneal_steps,
                "pct_start": 0.10, 
                "div_factor": 1.0,
                "final_div_factor": 4
            },
            "value_clip" : {
                "max_clip": self.value_clip, 
                "anneal_steps" : self.anneal_steps,
                "pct_start": 0.10, 
                "div_factor": 1.0,
                "final_div_factor": 4
            }
        }