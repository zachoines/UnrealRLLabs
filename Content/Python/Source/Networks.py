import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Dict, Union, List, Any
from torch.distributions import Normal
from abc import ABC, abstractmethod
from .Utility import init_weights_leaky_relu, init_weights_leaky_relu_conv

class BasePolicyNetwork(nn.Module, ABC):
    """
    Abstract base class for policy networks, either continuous or discrete.
    Defines the standard interface:
      - get_actions(emb, eval=False) => (actions, log_probs, entropies)
      - recompute_log_probs(emb, actions) => (log_probs, entropies)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_actions(self, emb: torch.Tensor, eval: bool = False):
        """
        emb => (Batch, Features).
        Return => (actions, log_probs, entropies) => each => shape (Batch,...)
        """
        raise NotImplementedError

    @abstractmethod
    def recompute_log_probs(self, emb: torch.Tensor, actions: torch.Tensor):
        """
        emb => (Batch, Features).
        actions => shape (Batch, #action_dims).
        Return => (log_probs, entropies) => each => shape (Batch,).
        """
        raise NotImplementedError


class DiscretePolicyNetwork(BasePolicyNetwork):
    """
    For discrete or multi-discrete action spaces.
    1) forward(x) => single-cat logits or list of multi-cat branch logits
    2) get_actions => sample from Categorical => (actions, log_probs, entropies)
    3) recompute_log_probs => re-calc log_probs & ent. sum across branches if multi-discrete
    """

    def __init__(self, in_features: int, out_features, hidden_size: int):
        super().__init__()
        self.in_features  = in_features
        self.hidden_size  = hidden_size
        # if out_features is int => single-cat
        # if out_features is list => multi-discrete
        self.out_is_multi = isinstance(out_features, list)

        # Shared MLP for hidden layers
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
        )

        if not self.out_is_multi:
            # single-cat
            self.head = nn.Linear(hidden_size, out_features)
        else:
            # multi-discrete => multiple heads
            self.branch_heads = nn.ModuleList([
                nn.Linear(hidden_size, branch_sz)
                for branch_sz in out_features
            ])

        self.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x: torch.Tensor):
        """
        Return either a single-cat logits => shape (B, outDim),
        or a list of branch logits => each => shape (B, branch_sz).
        """
        feats = self.shared(x)
        if not self.out_is_multi:
            return self.head(feats)  # => (B, outDim)
        else:
            return [bh(feats) for bh in self.branch_heads]

    def get_actions(self, emb: torch.Tensor, eval: bool = False):
        """
        (actions, log_probs, entropies).
        If multi-discrete, sum log_probs & ent across branches.
        """
        logits_list = self.forward(emb)
        B = emb.shape[0]

        if not self.out_is_multi:
            # single-cat
            dist = torch.distributions.Categorical(logits=logits_list)
            if eval:
                actions = torch.argmax(logits_list, dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)  # => (B,)
            entropies = dist.entropy()          # => (B,)
            return actions, log_probs, entropies
        else:
            # multi-discrete => list of Categorical
            actions_list, lp_list, ent_list = [], [], []
            for branch_logits in logits_list:
                dist_b = torch.distributions.Categorical(logits=branch_logits)
                if eval:
                    a_b = torch.argmax(branch_logits, dim=-1)
                else:
                    a_b = dist_b.sample()
                lp_b  = dist_b.log_prob(a_b)
                ent_b = dist_b.entropy()
                actions_list.append(a_b)
                lp_list.append(lp_b)
                ent_list.append(ent_b)

            # combine
            # shape => (B, #branches)
            acts_stacked = torch.stack(actions_list, dim=-1)
            lp_stacked   = torch.stack(lp_list, dim=-1).sum(dim=-1)
            ent_stacked  = torch.stack(ent_list, dim=-1).sum(dim=-1)
            return acts_stacked, lp_stacked, ent_stacked

    def recompute_log_probs(self, emb: torch.Tensor, actions: torch.Tensor):
        """
        For multi-discrete, 'actions' => shape (B, #branches).
        For single-cat, => shape (B,).
        Return => (log_probs, entropies)
        """
        logits_list = self.forward(emb)
        B = emb.shape[0]

        if not self.out_is_multi:
            dist = torch.distributions.Categorical(logits=logits_list)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            return log_probs, entropies
        else:
            # multi-discrete
            # actions => (B, num_branches)
            actions = actions.long()
            lp_list, ent_list = [], []
            for i, branch_logits in enumerate(logits_list):
                dist_b = torch.distributions.Categorical(logits=branch_logits)
                a_b    = actions[:, i]
                lp_b   = dist_b.log_prob(a_b)
                ent_b  = dist_b.entropy()
                lp_list.append(lp_b)
                ent_list.append(ent_b)

            log_probs = torch.stack(lp_list, dim=-1).sum(dim=-1)
            entropies = torch.stack(ent_list, dim=-1).sum(dim=-1)
            return log_probs, entropies


class RSA(nn.Module):
    """
    Residual Self-Attention (RSA):
    Produces relational embeddings among agents.

    Input:  (Batch,Agents,H)
    Output: (Batch,Agents,H)
    """
    def __init__(self, embed_size: int, heads: int, dropout_rate=0.0):
        super(RSA, self).__init__()
        self.query_embed = nn.Linear(embed_size, embed_size)
        self.key_embed   = nn.Linear(embed_size, embed_size)
        self.value_embed = nn.Linear(embed_size, embed_size)
        
        self.input_norm  = nn.LayerNorm(embed_size)
        self.output_norm = nn.LayerNorm(embed_size)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_size, heads, batch_first=True, dropout=dropout_rate
        )

        self.query_embed.apply(lambda m: init_weights_leaky_relu(m, 0.01))
        self.key_embed.apply(lambda m: init_weights_leaky_relu(m, 0.01))
        self.value_embed.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => shape (Batch, Agents, H)
        """
        x = self.input_norm(x)
        q = self.query_embed(x)
        k = self.key_embed(x)
        v = self.value_embed(x)
        out, _ = self.multihead_attn(q, k, v)
        out = x + out
        return self.output_norm(out)


class LinearNetwork(nn.Module):
    """
    A simple MLP block that includes:
      - Linear(in_features, out_features)
      - Optional Dropout
      - Optional LeakyReLU(0.01)
      - Optional LayerNorm

    Args:
        in_features (int):  input feature dimension
        out_features (int): output feature dimension
        dropout_rate (float): if >0, adds Dropout(dropout_rate)
        activation (bool): if True, adds LeakyReLU(0.01)
        layer_norm (bool): if True, adds LayerNorm(out_features) at the end

    Returns:
        A sequential block => [Linear, (Dropout?), (LeakyReLU?), (LayerNorm?)]
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout_rate: float = 0.0,
                 activation: bool = True,
                 layer_norm: bool = False):
        super().__init__()
        layers = []

        # 1) mandatory linear
        layers.append(nn.Linear(in_features, out_features))

        # 2) optional dropout
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        # 3) optional leaky relu
        if activation:
            layers.append(nn.LeakyReLU(0.01))

        # 4) optional layer norm
        if layer_norm:
            layers.append(nn.LayerNorm(out_features))

        self.model = nn.Sequential(*layers)

        # apply Kaiming init for all linear submodules
        self.model.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StatesEncoder(nn.Module):
    """
    Encodes a state vector of size (B, state_dim) into (B, output_size).
    Optionally includes dropout and a LeakyReLU activation.
    """
    def __init__(self, state_dim, hidden_size, output_size, dropout_rate=0.0, activation=False, layer_norm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            LinearNetwork(state_dim, hidden_size, dropout_rate, activation, layer_norm),
            LinearNetwork(hidden_size, output_size, dropout_rate, activation, layer_norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatesActionsEncoder(nn.Module):
    """
    Concatenates state + action => feeds into a small MLP => (B, output_size).
    Useful for baseline or Q-value networks that need (s,a) pairs.
    """
    def __init__(self, state_dim, action_dim, hidden_size, output_size,
                 dropout_rate=0.0, activation=False, layer_norm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            LinearNetwork(state_dim + action_dim, hidden_size, dropout_rate, activation, layer_norm),
            LinearNetwork(hidden_size, output_size, dropout_rate, activation, layer_norm)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    Produces a scalar value from an embedding of shape (B, in_features).
    Typically used for value functions or baselines in RL.
    """
    def __init__(self, in_features: int, hidden_size: int, dropout_rate=0.0):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features, 1),
        )
        # Apply Kaiming init to these layers
        self.value_net.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x)


class SharedCritic(nn.Module):
    """
    Shared critic that outputs:
     1) A **global value function** `V(s)`, shared across agents in an environment.
     2) **Per-agent counterfactual baselines** `Q_ψ`, used for advantage calculations.
    
    Uses **attention-based aggregation** to determine **which agents contribute most**.
    """

    def __init__(self, config):
        super(SharedCritic, self).__init__()
        net_cfg = config['agent']['params']['networks']['critic_network']

        self.value_head = ValueNetwork(**net_cfg['value_head'])
        self.baseline_head = ValueNetwork(**net_cfg['baseline_head'])

        # **Learnable Attention Weights for Value Aggregation** (Per-Environment)
        self.value_attention = nn.Linear(net_cfg['value_head']['in_features'], 1)  # (H → 1)
        
        # **Learnable Attention Weights for Baseline Aggregation** (Per-Agent)
        self.baseline_attention = nn.Linear(net_cfg['baseline_head']['in_features'], 1)  # (H → 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute **per-environment** value:
        - Input: x => (S,E,A,H)
        - We perform **attention-based weighted sum across agents (dim=2)**.
        - Output: (S,E,1)
        """
        S, E, A, H = x.shape

        # Compute per-agent importance weights
        attn_weights = self.value_attention(x).squeeze(-1)  # Shape: (S,E,A)
        attn_weights = torch.softmax(attn_weights, dim=2)  # Normalize across agents

        # Weighted sum across agents
        weighted_emb = (attn_weights.unsqueeze(-1) * x).sum(dim=2)  # (S,E,H)

        # Compute final value estimate
        flat = weighted_emb.reshape(S * E, H)
        vals = self.value_head(flat).reshape(S, E, 1)

        return vals

    def baselines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute **per-agent counterfactual baselines**:
        - Input: x => (S,E,A,A,H) (groupmate-aware embeddings)
        - We perform **attention-based weighted sum across groupmates (dim=3)**.
        - Output: (S,E,A,1)
        """
        S, E, A, A2, H = x.shape
        assert A == A2, "x must be shape (S,E,A,A,H)."

        # Compute per-groupmate importance weights
        attn_weights = self.baseline_attention(x).squeeze(-1)  # Shape: (S,E,A,A)
        attn_weights = torch.softmax(attn_weights, dim=3)  # Normalize across groupmates

        # Weighted sum across groupmates
        weighted_emb = (attn_weights.unsqueeze(-1) * x).sum(dim=3)  # (S,E,A,H)

        # Compute per-agent baselines
        flat = weighted_emb.reshape(S * E * A, H)
        base = self.baseline_head(flat).reshape(S, E, A, 1)

        return base


class TanhContinuousPolicyNetwork(BasePolicyNetwork):
    """
    A Tanh-squashed Normal distribution. Typical usage:
      1) 'forward(x)' => produce unbounded mean + clamped log_std in [log_std_min, log_std_max].
      2) 'get_actions(emb)' => 
         - sample raw_action ~ Normal(mean, std)
         - a = tanh(raw_action)
         - compute log_probs (with tanh correction) and approximate entropies
      3) 'recompute_log_probs(emb, actions)':
         - invert actions => raw_action = atanh(actions)
         - re-calc log_probs & entropies with factorized Normal + correction
      4) For entropies, we support:
         - 'analytic' => single-sample correction
         - 'mc' => sample-based approach for more accurate integral
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        log_std_min: float,
        log_std_max: float,
        entropy_method: str = "analytic",
        n_entropy_samples: int = 5
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.entropy_method = entropy_method
        self.n_entropy_samples = n_entropy_samples

        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
        )

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, out_features),
        )

        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, out_features),
        )

        self.shared.apply(lambda m: init_weights_leaky_relu(m, 0.01))
        self.mean_head.apply(lambda m: init_weights_leaky_relu(m, 0.01))
        self.log_std_head.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x):
        feats = self.shared(x)
        mean  = self.mean_head(feats)
        raw_log_std = self.log_std_head(feats)
        clamped_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )
        return mean, clamped_log_std

    def get_actions(self, emb: torch.Tensor, eval: bool=False):
        """
        1) sample or return mean from Normal(mean, std)
        2) tanh-squash => a
        3) compute log_probs & approximate entropies
        Return => (actions, log_probs, entropies) => shape => (B, outDim), (B,), (B,)
        """
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if eval:
            raw_action = mean
        else:
            raw_action = dist.rsample()  # reparameterized sample

        actions = torch.tanh(raw_action)

        # log_probs with tanh correction
        logp_raw = dist.log_prob(raw_action).sum(dim=-1)
        eps = 1e-6
        correction = torch.sum(torch.log(1 - actions.pow(2) + eps), dim=-1)
        log_probs = logp_raw - correction

        # approximate ent
        entropies = self._approx_entropy(dist, actions, raw_action)
        return actions, log_probs, entropies

    def recompute_log_probs(self, emb: torch.Tensor, actions: torch.Tensor):
        """
        For already-squashed 'actions' in [-1,1], invert => raw_action=atanh(a_clamped),
        then compute log_probs & ent. Return => (log_probs, entropies) => shape => (B,).
        """

        # 1) Forward pass => (mean, log_std) => dist=Normal(mean, std)
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # 2) clamp final actions inside safe [-1..1]
        eps = 1e-8
        a_clamped = torch.clamp(actions, -1+eps, 1-eps)

        # 3) stable atanh => raw_action = 0.5 * log( (1+a)/(1-a) )
        # clamp ratio => max
        ratio = (1 + a_clamped).clamp_min(eps) / (1 - a_clamped).clamp_min(eps)
        raw_action = 0.5 * torch.log(ratio)

        # 4) Normal log-prob => sum across action dims
        logp_raw = dist.log_prob(raw_action).sum(dim=-1)

        # 5) Tanh correction => sum(log(1 - a^2 + eps)) across dims
        correction = torch.sum(torch.log(1 - a_clamped.pow(2) + eps), dim=-1)
        log_probs = logp_raw - correction

        # 6) approximate ent => same as in get_actions
        entropies = self._approx_entropy(dist, a_clamped, raw_action)
        return log_probs, entropies


    def _approx_entropy(self, dist: Normal, actions: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        """
        Approx Tanh(Normal) entropy with either 'mc' or 'analytic' approach.
        :return: (B,) shaped tensor
        """
        if self.entropy_method == "mc":
            return self._mc_entropy(dist)
        else:
            return self._analytic_entropy(dist, actions)

    def _mc_entropy(self, dist: Normal) -> torch.Tensor:
        """
        MC-based approach:
         1) draw n samples
         2) transform => a = tanh(...)
         3) approximate => H ~ -E[log p(a_samps)]
        """
        n = self.n_entropy_samples
        raw_samps = dist.rsample([n])  # => (n,B,outDim)
        a_samps   = torch.tanh(raw_samps)
        logp_raw  = dist.log_prob(raw_samps).sum(dim=-1) # => (n,B)
        eps = 1e-6
        corr = torch.sum(torch.log(1 - a_samps.pow(2) + eps), dim=-1)
        logp_all = logp_raw - corr
        return - logp_all.mean(dim=0)  # => (B,)

    def _analytic_entropy(self, dist: Normal, actions: torch.Tensor) -> torch.Tensor:
        """
        'analytic'/naive approach:
        ent_raw = factorized normal entropy => dist.entropy().sum(dim=-1)
        correction = sum(log(1 - actions^2))
        => ent = ent_raw - correction
        """
        ent_raw = dist.entropy().sum(dim=-1)
        eps = 1e-6
        corr = torch.sum(torch.log(1 - actions.pow(2) + eps), dim=-1)
        return ent_raw - corr


class MultiAgentEmbeddingNetwork(nn.Module):
    def __init__(self, net_cfg: Dict[str, Any]):
        super(MultiAgentEmbeddingNetwork, self).__init__()

        # 1) The dictionary based obs embedder
        self.base_encoder = CrossAttentionFeatureExtractor(**net_cfg["cross_attention_feature_extractor"])

        # 2) For obs and obs+actions encoding
        self.f = StatesActionsEncoder(**net_cfg["obs_actions_encoder"])
        self.g = StatesEncoder(**net_cfg["obs_encoder"])

    # --------------------------------------------------------------------
    #  Produce "base embedding" from raw dictionary of observarion
    # --------------------------------------------------------------------
    def get_base_embedding(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict => { "central":(S,E,cDim), "agent":(S,E,NA,aDim) }
        => shape => (S,E,NA, aggregator_out_dim)
        """
        return self.base_encoder(obs_dict)


    # --------------------------------------------------------------------
    #  Step 2c: embed_for_baseline
    # --------------------------------------------------------------------
    def get_baseline_embeddings(self,
                           common_emb: torch.Tensor,
                           actions: torch.Tensor=None) -> torch.Tensor:
        groupmates_emb, groupmates_actions = \
            self._split_agent_and_groupmates(common_emb, actions)


        agent_obs_emb = self.g(common_emb) # => shape (S,E,A,1,H)
        groupmates_obs_actions_emb= self.f(groupmates_emb, groupmates_actions) # => (S,E,A,(A-1),H)
        
        # (S,E,A,H) and (S,E,A,A,H2)
        return torch.cat([agent_obs_emb.unsqueeze(dim=3), groupmates_obs_actions_emb], dim=3)

    # --------------------------------------------------------------------
    #  Helper: _split_agent_and_groupmates
    # --------------------------------------------------------------------
    def _split_agent_and_groupmates(self,
                                    obs_4d: torch.Tensor,
                                    actions: torch.Tensor):
        """
        obs_4d => (S,E,A,H)
        actions=> (S,E,A, act_dim)  [assuming single vector per agent]
                  or (S,E,A, # branches) for discrete
        We want:
          agent_obs => (S,E,A,1,H)
          groupmates_obs => (S,E,A,A-1,H)
          groupmates_actions => (S,E,A,A-1, act_dim)
        """
        S,E,A,H = obs_4d.shape

        group_obs_list = []
        group_act_list = []

        for i in range(A):
            # groupmates => everything but agent i
            gm_obs = torch.cat([obs_4d[..., :i, :],
                                obs_4d[..., i+1:, :]], dim=2).unsqueeze(2)
            gm_act = torch.cat([actions[..., :i, :],
                                actions[..., i+1:, :]], dim=2).unsqueeze(2)

            group_obs_list.append(gm_obs)
            group_act_list.append(gm_act)

        # combine
        group_obs = torch.cat(group_obs_list, dim=2)               # => (S,E,A,A-1,H)
        group_act = torch.cat(group_act_list, dim=2)               # => (S,E,A,A-1,act_dim)

        return group_obs.contiguous(), group_act.contiguous()


class CrossAttentionFeatureExtractor(nn.Module):
    def __init__(
        self, 
        obs_size: int, 
        num_agents: int, 
        h: int = 50, 
        w: int = 50, 
        embed_dim: int = 64, 
        num_heads: int = 4, 
        cnn_channels: list = [32, 64], 
        cnn_kernel_sizes: list = [5, 3], 
        cnn_strides: list = [2, 2], 
        transformer_layers: int = 2, 
        scale_ratios: list = [1/4, 1/8, 1/16],
        top_k_scales: int = 2
    ):
        super().__init__()

        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.h, self.w = h, w
        self.scale_ratios = scale_ratios
        self.num_scales = len(scale_ratios)
        self.scale_sizes = [math.floor(h * r) for r in scale_ratios]
        self.top_k_scales = top_k_scales  

        # Agent MLP encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Positional embedding for each agent
        self.agent_pos_embedding = nn.Embedding(num_agents, embed_dim)

        # CNNs for adaptive multi-scale feature extraction
        self.central_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, cnn_channels[0], kernel_size=cnn_kernel_sizes[0], stride=cnn_strides[0], padding=cnn_kernel_sizes[0] // 2),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=cnn_kernel_sizes[1], stride=cnn_strides[1], padding=cnn_kernel_sizes[1] // 2),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[1], embed_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((s, s))
            )
            for s in self.scale_sizes
        ])

        # Transformer for each scale
        self.terrain_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
                num_layers=transformer_layers
            )
            for _ in range(self.num_scales)
        ])

        # Positional embeddings for adaptive patches
        self.patch_position_embeddings = nn.ModuleList([
            nn.Embedding(s * s, embed_dim)
            for s in self.scale_sizes
        ])

        # Cross-Attention for Agent-Terrain interaction at each scale
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(self.num_scales)
        ])

        # Agent-to-Agent Self-Attention
        self.agent_self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Hierarchical Transformer for Merging Multi-Scale Representations
        self.scale_merging_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )

        self.apply(lambda m: init_weights_leaky_relu_conv(m, 0.01))
        self.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, state):
        """
        state: Dict containing:
            - "central": Tensor of shape [num_steps, num_envs, h * w]
            - "agent":   Tensor of shape [num_steps, num_envs, num_agents, obs_size]
        """
        central_state = state["central"]
        agent_state = state["agent"]

        num_steps, num_envs, num_agents, obs_size = agent_state.shape

        assert num_agents == self.num_agents, "Mismatch in agent count and model config"

        # Process agent states (MLP)
        agent_state_flat = agent_state.reshape(-1, obs_size)
        agent_embed = self.agent_encoder(agent_state_flat)
        agent_embed = agent_embed.view(num_steps, num_envs, num_agents, self.embed_dim)

        # Add unique agent positional embeddings
        agent_indices = torch.arange(num_agents, device=agent_state.device).expand(num_steps, num_envs, num_agents)
        agent_pos_embeds = self.agent_pos_embedding(agent_indices)
        agent_embed = agent_embed + agent_pos_embeds  

        agent_embed = agent_embed.view(-1, num_agents, self.embed_dim)

        multi_scale_outputs = []
        for scale_idx, scale_size in enumerate(self.scale_sizes):

            # Process central state with CNN
            central_scaled = central_state.reshape(num_steps, num_envs, 1, self.h, self.w)
            central_embed = self.central_cnns[scale_idx](central_scaled.view(-1, 1, self.h, self.w))

            # Flatten CNN output into patches
            num_patches = scale_size * scale_size
            central_embed = central_embed.view(-1, self.embed_dim, num_patches).permute(0, 2, 1)

            # Generate Keys (`K`) as Positional Embeddings
            patch_indices = torch.arange(num_patches, device=agent_state.device).expand(agent_embed.shape[0], num_patches)
            key_embeds = self.patch_position_embeddings[scale_idx](patch_indices)

            # Transformer for global terrain understanding
            central_embed = self.terrain_transformers[scale_idx](central_embed)

            # **Cross-Attention: Query (Agents) → Key (Positions) → Value (Processed Terrain Features)**
            attn_output, _ = self.cross_attentions[scale_idx](agent_embed, key_embeds, central_embed)
            multi_scale_outputs.append(attn_output)

        # Aggregate multi-scale outputs
        agent_embed = torch.stack(multi_scale_outputs, dim=0).mean(dim=0)

        # **NEW: Agent-to-Agent Communication**
        agent_embed, _ = self.agent_self_attention(agent_embed, agent_embed, agent_embed)

        return agent_embed.view(num_steps, num_envs, num_agents, self.embed_dim)


class GN2d(nn.Module):
    """
    Applies GroupNorm to a (B, C, H, W) tensor.
    By default, num_groups=1 => 'InstanceNorm'-like behavior,
    normalizing each channel across HxW for each sample.
    """
    def __init__(self, num_channels: int, num_groups: int = 1):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => shape (B, C, H, W)
        return self.gn(x)


class ResidualBlock(nn.Module):
    """
    A simple residual block with channel-only GroupNorm:
      conv1 -> GN2d -> LeakyReLU
      conv2 -> GN2d
      skip + final LeakyReLU
    """
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.gn1   = GN2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.gn2   = GN2d(channels)

        # Apply Kaiming init to the conv layers
        self.apply(lambda m: init_weights_leaky_relu_conv(m, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.conv2(out)
        out = self.gn2(out)

        out += residual
        out = F.leaky_relu(out, negative_slope=0.01)
        return out


class SpatialNetwork2D(nn.Module):
    """
    2D convolutional network with ResidualBlocks + GroupNorm.
    Steps:
      1) Reshape => (B,1,h,w)
      2) entry conv => GN2d => LeakyReLU
      3) residual blocks (MaxPool2d every 2 blocks)
      4) flatten => final fc => LayerNorm => (B, out_features)
    """
    def __init__(
        self,
        h: int = 50,
        w: int = 50,
        in_channels: int = 1,
        base_channels: int = 16,
        num_blocks: int = 4,
        out_features: int = 128
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.out_features = out_features

        # 1) Entry conv => GN2d => LeakyReLU
        self.entry_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.entry_gn   = GN2d(base_channels)

        # 2) Build residual blocks + optional pooling
        blocks = []
        current_channels = base_channels
        for i in range(num_blocks):
            blocks.append(ResidualBlock(current_channels, kernel_size=3, padding=1))
            if (i + 1) % 2 == 0:
                blocks.append(nn.MaxPool2d(kernel_size=2))
        self.res_blocks = nn.Sequential(*blocks)

        # 3) Final Linear => out_features
        pool_count = num_blocks // 2
        final_h = h // (2 ** pool_count)
        final_w = w // (2 ** pool_count)
        in_features = current_channels * final_h * final_w
        self.final_fc = nn.Linear(in_features, out_features)

        # 4) Optionally a layer norm on the final embedding
        self.final_ln = nn.LayerNorm(out_features)

        # Now apply Kaiming init
        # - conv layers => init_weights_leaky_relu_conv
        # - linear layers => init_weights_leaky_relu
        self.entry_conv.apply(lambda m: init_weights_leaky_relu_conv(m, 0.01))
        self.final_fc.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (B, N) with N = h*w
        1) reshape => (B,1,h,w)
        2) conv => GN2d => LeakyReLU
        3) residual blocks => flatten => final fc => LN
        """
        B, N = x.shape
        if N != self.h * self.w:
            raise ValueError(f"Expected input size {self.h*self.w}, got {N}.")

        # reshape => (B,1,h,w)
        x_img = x.reshape(B, 1, self.h, self.w)

        # entry conv => groupnorm => leaky relu
        out = self.entry_conv(x_img)
        out = self.entry_gn(out)
        out = F.leaky_relu(out, 0.01)

        # residual blocks (w/ occasional MaxPool2d)
        out = self.res_blocks(out)

        # flatten => final fc => final LN
        B2, C2, H2, W2 = out.shape
        out_flat = out.reshape(B2, C2 * H2 * W2)
        emb = self.final_fc(out_flat)
        emb = self.final_ln(emb)
        return emb


class CoordConv2D(nn.Module):
    """
    A simple 'coord conv' approach to inject absolute (x,y) positions as additional channels.
    Input shape: (B, 1, H, W) for the wave, or (B, in_channels, H, W).
    Output shape: (B, in_channels+2, H, W).
      - The 2 extra channels are X_ij, Y_ij in [-1, 1].
    """
    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

        # Pre-compute normalized coordinate grids in [-1..1]
        # shape => (H, W)
        xs = torch.linspace(-1.0, 1.0, w).reshape(1, w).expand(h, w)
        ys = torch.linspace(-1.0, 1.0, h).reshape(h, 1).expand(h, w)

        # so xs[i, j] is x-coord, ys[i, j] is y-coord
        # => final shape => (2, H, W)
        coords = torch.stack([ys, xs], dim=0)  
        # register as buffer so it's on correct device
        self.register_buffer("coords", coords, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (B, C, H, W). We'll append 2 coordinate channels => (B, C+2, H, W).
        """
        B, C, H, W = x.shape
        if H != self.h or W != self.w:
            raise ValueError(f"CoordConv2D expects H,W=({self.h},{self.w}). Got {H,W}.")

        # coords => shape (2,H,W), unsqueeze => (1,2,H,W), expand => (B,2,H,W)
        coords_expanded = self.coords.unsqueeze(0).expand(B, -1, -1, -1)
        # cat => (B, C+2, H, W)
        out = torch.cat([x, coords_expanded], dim=1)
        return out


class LightweightSpatialNetwork2D(nn.Module):
    """
    A smaller CNN for 2D wave heightmaps with coordinate injection for absolute position.
    Steps:
      1) Reshape => (B, 1, H, W)   # single channel wave
      2) Append 2 channels => (x,y) => total 3 channels
      3) A few small conv layers => GN => LeakyReLU
      4) Minimal pooling (1 or 2 times) to reduce dimension
      5) flatten => final fc => LN => out_features
    Typically for wave features => 50x50 input
    """

    def __init__(
        self,
        h: int = 50,
        w: int = 50,
        in_channels: int = 1,
        base_channels: int = 8,
        num_conv_layers: int = 2,
        out_features: int = 64,
        pooling_layers: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        """
        :param h,w: input height/width
        :param in_channels: typically 1 for the wave
        :param base_channels: #channels in the first conv
        :param num_conv_layers: how many conv->norm->relu blocks
        :param out_features: final output dimension
        :param pooling_layers: how many times we do a MaxPool2d(kernel_size=2)
        :param kernel_size: kernel for convs (3 is typical)
        :param dropout: if >0, we do nn.Dropout after conv layers
        """
        super().__init__()
        self.h = h
        self.w = w
        self.out_features = out_features

        # 1) We'll do a small 'coordconv' to add x,y channels => total in_channels + 2
        self.coordconv = CoordConv2D(h, w)
        in_ch = in_channels + 2

        # 2) Build conv layers
        layers = []
        current_channels = in_ch
        for i in range(num_conv_layers):
            conv = nn.Conv2d(
                in_channels=current_channels, 
                out_channels=base_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size//2
            )
            layers.append(conv)
            # groupnorm or layernorm can be used. We'll do groupnorm with 1 group => instance-norm style
            layers.append(nn.GroupNorm(num_groups=1, num_channels=base_channels))
            layers.append(nn.LeakyReLU(0.01))

            current_channels = base_channels
        
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        # 3) optional pooling
        #  do pooling_layers times => each => reduce spatial by half
        pool_list = []
        for _ in range(pooling_layers):
            pool_list.append(nn.MaxPool2d(kernel_size=2))

        self.conv_stack = nn.Sequential(*layers, *pool_list)

        # 4) figure out final shape after these pools
        # for each pool => h,w => h/2, w/2
        final_h = h >> pooling_layers  # h // 2**pooling_layers
        final_w = w >> pooling_layers
        in_feat = current_channels * final_h * final_w

        # 5) final fc => out_features
        self.final_fc = nn.Linear(in_feat, out_features)
        self.final_ln = nn.LayerNorm(out_features)

        # apply kaiming init to all conv layers
        self.apply(lambda m: init_weights_leaky_relu_conv(m, 0.01))
        self.apply(lambda m: init_weights_leaky_relu(m, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => shape (B, N) with N = h*w
        => we reshape => (B,1,h,w), then add x,y coords => (B,3,h,w)
        => pass conv stack => flatten => final fc => LN => (B,out_features)
        """
        B, N = x.shape
        if N != self.h * self.w:
            raise ValueError(f"Expected input size {self.h*self.w}, got {N}.")

        # reshape => (B, 1, h, w)
        x_img = x.reshape(B, 1, self.h, self.w)

        # coord conv => (B, 3, h, w)
        out = self.coordconv(x_img)

        # pass conv stack
        out = self.conv_stack(out)
        # flatten
        B2, C2, H2, W2 = out.shape
        out_flat = out.reshape(B2, C2 * H2 * W2)

        # final fc => layer norm
        emb = self.final_fc(out_flat)
        emb = self.final_ln(emb)
        return emb
