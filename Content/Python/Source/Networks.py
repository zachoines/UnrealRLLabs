import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Dict, Union, List, Any
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from abc import ABC, abstractmethod
from .Utility import init_weights_leaky_relu, init_weights_gelu_linear, init_weights_gelu_conv, create_2d_sin_cos_pos_emb

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
            nn.GELU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(0.01),
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


class LinearNetwork(nn.Module):
    """
    A simple MLP block that includes:
      - Linear(in_features, out_features)
      - Optional Dropout
      - Optional GELU(0.01)
      - Optional LayerNorm

    Args:
        in_features (int):  input feature dimension
        out_features (int): output feature dimension
        dropout_rate (float): if >0, adds Dropout(dropout_rate)
        activation (bool): if True, adds GELU(0.01)
        layer_norm (bool): if True, adds LayerNorm(out_features) at the end

    Returns:
        A sequential block => [Linear, (Dropout?), (GELU?), (LayerNorm?)]
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout_rate: float = 0.0,
                 activation: bool = True,
                 layer_norm: bool = False,
                 linear_init_scale: float = 1.0):
        super().__init__()
        layers = []

        # 1) mandatory linear
        layers.append(nn.Linear(in_features, out_features))

        # 2) optional dropout
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        # 3) optional leaky relu
        if activation:
            layers.append(nn.GELU())

        # 4) optional layer norm
        if layer_norm:
            layers.append(nn.LayerNorm(out_features))

        self.model = nn.Sequential(*layers)

        # apply Kaiming init for all linear submodules
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StatesEncoder(nn.Module):
    """
    Encodes a state vector of size (B, state_dim) into (B, output_size).
    Optionally includes dropout and a LeakyReLU activation.
    """
    def __init__(self, state_dim, hidden_size, output_size, dropout_rate=0.0, activation=False, layer_norm: bool = False, linear_init_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            LinearNetwork(state_dim, hidden_size, dropout_rate, activation, layer_norm, linear_init_scale),
            LinearNetwork(hidden_size, output_size, dropout_rate, activation, layer_norm, linear_init_scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatesActionsEncoder(nn.Module):
    """
    Concatenates state + action => feeds into a small MLP => (B, output_size).
    Useful for baseline or Q-value networks that need (s,a) pairs.
    """
    def __init__(self, state_dim, action_dim, hidden_size, output_size,
                 dropout_rate=0.0, activation=False, layer_norm: bool = False, linear_init_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            LinearNetwork(state_dim + action_dim, hidden_size, dropout_rate, activation, layer_norm, linear_init_scale),
            LinearNetwork(hidden_size, output_size, dropout_rate, activation, layer_norm, linear_init_scale)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
    

class ValueNetwork(nn.Module):
    """
    Produces a scalar value from an embedding (B, in_features).
    Now uses:
      - hidden_size for two hidden layers
      - GELU activation
      - final output = (B,1)
    """

    def __init__(self, in_features: int, hidden_size: int, dropout_rate=0.0, linear_init_scale: float = 1.0):
        super().__init__()
        # If you want dropout, you can place nn.Dropout(...) between layers.
        # For simplicity, we'll just keep it out, or you can insert it after the linear layers.

        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (B, in_features)
        returns => (B,1)
        """
        return self.value_net(x)


class SharedCritic(nn.Module):
    """
    A "global" critic that outputs:
      1) A single value V(s) per environment (averaging agent embeddings)
      2) A per-agent baseline for each agent (averaging over groupmates or neighbors).
    """

    def __init__(self, net_cfg):
        super().__init__()

        self.value_head = ValueNetwork(**net_cfg["value_head"])
        self.baseline_head = ValueNetwork(**net_cfg["baseline_head"])
        self.baseline_attention = ResidualAttention(
            **net_cfg['baseline_attention']
        )

        self.value_attention = ResidualAttention(
            **net_cfg['baseline_attention']
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(lambda m: init_weights_gelu_linear(m, scale=net_cfg["linear_init_scale"]))

    def values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-environment value function V(s).
        """

        S, E, A, H = x.shape
        
        # Self-Attention Over agents
        x_flat = x.view(S * E, A, H)
        x_attn, _ = self.value_attention(x_flat, x_flat, x_flat)
        env_emb = x_attn.mean(dim=1)  # (S,E,H)

        # Feed value_head
        flat = env_emb.reshape(S * E, H)  # (S*E,H)
        vals = self.value_head(flat)      # (S*E,1)

        # Reshape back
        vals = vals.view(S, E, 1)
        return vals

    def baselines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-agent counterfactual baselines
        """

        S, E, A, A2, H = x.shape
        
        # Flatten
        x_flat = x.view(S * E * A, A, H)

        # Self-Attention Over Groupmates
        x_attn, _ = self.baseline_attention(x_flat, x_flat, x_flat)

        # Mean over groupmates dimension
        agent_emb = x_attn.mean(dim=1)  # (S*E*A, A, H) => (S*E*A, H)
        base = self.baseline_head(agent_emb) # => (S*E*A,1)

        # 3) Reshape => (S,E,A,1)
        base = base.view(S, E, A, 1)
        return base


class TanhContinuousPolicyNetwork(nn.Module):
    """
    A Tanh-squashed Normal distribution using PyTorch's TransformedDistribution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        mean_scale: float,
        log_std_min: float,
        log_std_max: float,
        entropy_method: str = "analytic",
        n_entropy_samples: int = 5,
        linear_init_scale: float = 1.0,
    ):
        super().__init__()
        self.mean_scale = mean_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.entropy_method = entropy_method
        self.n_entropy_samples = n_entropy_samples

        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features),
        )

        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features),
        )

        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale))


    def forward(self, x: torch.Tensor):
        feats = self.shared(x)

        # Mean in [-mean_scale, +mean_scale]
        raw_mean = self.mean_head(feats)
        mean = self.mean_scale * torch.tanh(raw_mean)

        # log_std in [log_std_min, log_std_max]
        raw_log_std = self.log_std_head(feats)
        clamped_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )

        return mean, clamped_log_std

    def get_actions(self, emb: torch.Tensor, eval: bool=False):
        """
        Return final_action in [-1,1], plus log_probs & entropies
        """
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Base Normal
        base_dist = Normal(mean, std)
        transform = TanhTransform(cache_size=1)
        tanh_dist = TransformedDistribution(base_dist, transform)

        if eval:
            # deterministic => we can do transform(mean), or use tanh_dist.mean
            actions = tanh_dist.mean  # shape => (B, out_features)
        else:
            # sample in [-1,1]
            actions = tanh_dist.rsample()  # => (B, out_features)

        action_clamped = actions.clamp(-1+1e-6, 1-1e-6) # To avoid NAN's caused by tahn saturation
        log_probs = tanh_dist.log_prob(action_clamped).sum(dim=-1)  # sum across action dims
        entropies = self._compute_entropy(tanh_dist)

        return action_clamped, log_probs, entropies

    def recompute_log_probs(self, emb: torch.Tensor, stored_actions: torch.Tensor):
        """
        'stored_actions' are the final actions in [-1,1].
        We invert them internally (via transform.inv) for stable log_prob calculation.
        """
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        base_dist = Normal(mean, std)
        transform = TanhTransform(cache_size=1)
        tanh_dist = TransformedDistribution(base_dist, transform)

        # We do *not* have raw actions. Instead, we have [-1,1] actions => invert via TanhTransform
        log_probs = tanh_dist.log_prob(stored_actions).sum(dim=-1)
        entropies = self._compute_entropy(tanh_dist)
        return log_probs, entropies

    def _compute_entropy(self, tanh_dist: TransformedDistribution) -> torch.Tensor:
        """
        We rely on .entropy() or do an MC fallback for TanhDist.
        """
        if self.entropy_method == "mc":
            return self._mc_entropy(tanh_dist)
        else:
            try:
                ent = tanh_dist.entropy()  # shape => (B, out_features)
                return ent.sum(dim=-1)
            except NotImplementedError:
                return self._mc_entropy(tanh_dist)

    def _mc_entropy(self, tanh_dist: TransformedDistribution) -> torch.Tensor:
        n = self.n_entropy_samples
        samps = tanh_dist.rsample([n])    # => (n, B, out_features)
        lp = tanh_dist.log_prob(samps).sum(dim=-1)  # => (n,B)
        return -lp.mean(dim=0)  # => (B,)

   
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


    def get_baseline_embeddings(self,
                           common_emb: torch.Tensor,
                           actions: torch.Tensor=None) -> torch.Tensor:
        groupmates_emb, groupmates_actions = \
            self._split_agent_and_groupmates(common_emb, actions)


        agent_obs_emb = self.get_state_embeddings(common_emb) # => shape (S,E,A,H)
        groupmates_obs_actions_emb= self.get_state_action_embeddings(groupmates_emb, groupmates_actions) # => (S,E,A,(A-1),H)
        
        # (S,E,A,1, H) and (S,E,A,A-1,H)
        return torch.cat([agent_obs_emb.unsqueeze(dim=3), groupmates_obs_actions_emb], dim=3), agent_obs_emb
    
    def get_state_embeddings(self, common_emb: torch.Tensor) -> torch.Tensor:
        return self.g(common_emb)

    def get_state_action_embeddings(self, common_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.f(common_emb, actions)

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


class AgentIDPosEnc(nn.Module):
    """
    Sinusoidal positional encoder for integer agent IDs.
    """
    def __init__(self, num_freqs: int = 8, id_embed_dim: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.id_embed_dim = id_embed_dim
        self.linear = nn.Linear(2 * num_freqs, id_embed_dim)

        init.kaiming_normal_(self.linear.weight, a=0.01)
        nn.init.constant_(self.linear.bias, 0.0)

    def sinusoidal_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => shape (...), typically integer IDs
        returns => shape (..., 2*num_freqs) for sin/cos expansions
        """
        x_float = x.float()
        freq_exponents = torch.arange(self.num_freqs, device=x.device, dtype=torch.float32)
        scales = torch.pow(2.0, freq_exponents)  # (num_freqs,)
        x_float = x_float.unsqueeze(-1)  # => (..., 1)
        multiplied = x_float * scales   # => (..., num_freqs)

        sines = torch.sin(multiplied)
        cosines = torch.cos(multiplied)
        return torch.cat([sines, cosines], dim=-1)  # => (..., 2*num_freqs)

    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        agent_ids => shape (any...), e.g. (S,E,A)
        returns => shape (same..., id_embed_dim)
        """
        shape_in = agent_ids.shape
        flat = agent_ids.reshape(-1)
        feats = self.sinusoidal_features(flat)
        out_lin = self.linear(feats)
        out_lin = out_lin.view(*shape_in, self.id_embed_dim)
        return out_lin


class ResidualAttention(nn.Module):
    """
    A flexible residual multi-head attention block with layernorm.
    Supports:
      - Self-attention (Q=K=V)
      - Cross-attention (Q != K,V) if self_attention=False
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = self_attention

        # Q, K, V transforms
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorms before attention
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        # Output LN
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, x_v=None):
        if self.self_attention or x_k is None:
            x_k = x_q
        if self.self_attention or x_v is None:
            x_v = x_q

        # LN
        qn = self.norm_q(x_q)
        kn = self.norm_k(x_k)
        vn = self.norm_v(x_v)

        # Q, K, V
        q_proj = self.query_proj(qn)
        k_proj = self.key_proj(kn)
        v_proj = self.value_proj(vn)

        # Multi-head attention
        attn_out, attn_weights = self.mha(q_proj, k_proj, v_proj)

        # Residual
        out = x_q + attn_out
        out = self.norm_out(out)

        return out, attn_weights


class FeedForwardBlock(nn.Module):
    """
    Transformer-style position-wise feed-forward sublayer, using GELU:
      x -> LN -> Linear -> GELU -> Linear -> Dropout -> x + residual
    """
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # a common choice
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return x + out


class CrossAttentionFeatureExtractor(nn.Module):
    """
    Multi-scale architecture with:
      - CNN feature extraction using GELU
      - 2D sin-cos positional encodings for map patches
      - Multiple cross-attn blocks per scale
      - Interleaved agent self-attn
      - Final LN + MLP on agent embeddings
      - Discrete category embedding + sinusoidal ID encoding for each agent
      - Weight initialization with manual scaling
    """

    def __init__(
        self,
        obs_size: int = 26,
        num_agents: int = 5,
        h: int = 50,
        w: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        cnn_channels: list = [32, 64, 128],
        cnn_kernel_sizes: list = [5, 3, 3],
        cnn_strides: list = [2, 2, 2],
        group_norms: list = [16, 8, 4],
        block_scales: list = [12, 6, 3],
        transformer_layers: int = 1,
        dropout_rate: float = 0.0,
        use_agent_id: bool = True,
        ff_hidden_factor: int = 4,
        # Extra: For ID pos enc
        id_num_freqs: int = 8,
        # scale factors for init
        conv_init_scale: float = 0.1,
        linear_init_scale: float = 1.0
    ):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.h = h
        self.w = w
        self.block_scales = block_scales
        self.use_agent_id = use_agent_id
        self.transformer_layers = transformer_layers
        self.ff_hidden_factor = ff_hidden_factor
        self.num_scales = len(block_scales)

        ####################################################
        # (A) Agent MLP (GELU)
        ####################################################
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        ####################################################
        # (B) Agent ID Embeddings + sinusoidal ID pos enc
        ####################################################
        if use_agent_id:
            self.agent_id_embedding = nn.Embedding(num_agents, embed_dim)
            nn.init.normal_(self.agent_id_embedding.weight, mean=0.0, std=0.01)
            
            self.agent_id_pos_enc = AgentIDPosEnc(num_freqs=id_num_freqs, id_embed_dim=embed_dim)
            self.agent_id_sum_ln = nn.LayerNorm(embed_dim)
        else:
            self.agent_id_embedding = None
            self.agent_id_pos_enc = None

        ####################################################
        # (C) CNN blocks (CoordConv) + 1x1 conv => embed_dim
        #     with GELU
        ####################################################
        self.blocks = nn.ModuleList()
        in_channels = 3  # 1 for heightmap + 2 for coords
        prev_c = in_channels
        for i, out_c in enumerate(cnn_channels):
            block = nn.Sequential(
                nn.Conv2d(prev_c, out_c, kernel_size=cnn_kernel_sizes[i],
                          stride=cnn_strides[i], padding=cnn_kernel_sizes[i] // 2),
                nn.GroupNorm(group_norms[i], out_c),
                nn.GELU(),

                nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(group_norms[i], out_c),
                nn.GELU()
            )
            self.blocks.append(block)
            prev_c = out_c

        # 1x1 conv => embed_dim
        self.block_embeds = nn.ModuleList()
        self.block_lns = nn.ModuleList()
        for c_out in cnn_channels:
            emb_conv = nn.Conv2d(c_out, embed_dim, kernel_size=1, stride=1, padding=0)
            ln = nn.LayerNorm(embed_dim)
            self.block_embeds.append(emb_conv)
            self.block_lns.append(ln)

        ####################################################
        # (D) Cross-Attn blocks per scale, repeated for each layer
        ####################################################
        self.cross_attn_blocks = nn.ModuleList()
        for layer_idx in range(transformer_layers):
            blocks_for_this_layer = nn.ModuleList()
            for scale_idx in range(self.num_scales):
                cross_attn = ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=False)
                ff_cross   = FeedForwardBlock(embed_dim, hidden_dim=embed_dim*self.ff_hidden_factor, dropout=dropout_rate)
                pair = nn.ModuleDict({
                    "attn": cross_attn,
                    "ffn": ff_cross
                })
                blocks_for_this_layer.append(pair)
            self.cross_attn_blocks.append(blocks_for_this_layer)

        ####################################################
        # (E) Agent Self-Attn (1 per transformer layer)
        ####################################################
        self.agent_self_attn = nn.ModuleList()
        self.agent_self_ffn  = nn.ModuleList()
        for layer_idx in range(transformer_layers):
            self_attn = ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=True)
            ff_self   = FeedForwardBlock(embed_dim, hidden_dim=embed_dim*self.ff_hidden_factor, dropout=dropout_rate)
            self.agent_self_attn.append(self_attn)
            self.agent_self_ffn.append(ff_self)

        ####################################################
        # (F) Final LN + MLP (GELU)
        ####################################################
        self.final_agent_ln = nn.LayerNorm(embed_dim)
        self.post_cross_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        ####################################################
        # Initialization with manual scaling
        ####################################################
        # 1) For CNN conv layers, we use 'conv_init_scale' (e.g. 0.1)
        self.blocks.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale))
        # Also for the block_embeds 1x1 conv:
        self.block_embeds.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale))

        # 2) For all linear layers (agent MLP, attention Q/K/V, feed-forward, etc.)
        #    we use 'linear_init_scale' (default 1.0)
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale))


    def forward(self, state):
        """
        Inputs:
          state["agent"]   => (S,E,A, obs_size)
          state["central"] => (S,E, h*w)

        Returns:
          agent_embed => (S,E,A, embed_dim)
          attn_weights => (S,E,A,A) from last self-attn
        """
        agent_state   = state["agent"]
        central_state = state["central"]
        device = agent_state.device
        S, E, A, obs_dim = agent_state.shape
        batch = S * E

        ####################################
        # 1) Agent MLP
        ####################################
        flat = agent_state.view(-1, obs_dim)
        agent_embed = self.agent_encoder(flat)  # => (S*E*A, embed_dim)
        agent_embed = agent_embed.view(batch, A, self.embed_dim)  # (B,A,d)

        ####################################
        # 2) Discrete ID embed + sinusoidal ID pos enc
        ####################################
        if self.use_agent_id:
            agent_ids = torch.arange(A, device=device).view(1,1,A).expand(S,E,A)
            # discrete embedding => (S,E,A, d)
            discrete_emb = self.agent_id_embedding(agent_ids)
            # sinusoidal ID pos => (S,E,A, d)
            pos_enc = self.agent_id_pos_enc(agent_ids)

            agent_embed_4d = agent_embed.view(S, E, A, self.embed_dim)
            agent_embed_4d = agent_embed_4d + discrete_emb + pos_enc
            agent_embed_4d = self.agent_id_sum_ln(agent_embed_4d)

            agent_embed = agent_embed_4d.view(batch, A, self.embed_dim)

        ####################################
        # 3) Reshape central_state => CNN
        ####################################
        central_state = central_state.view(S, E, self.h, self.w)
        central_state = central_state.view(batch, 1, self.h, self.w)  # => (B,1,H,W)

        # Add coordinate channels
        B, _, H, W = central_state.shape
        row_coords = torch.linspace(-1, 1, steps=H, device=device).view(1,1,H,1).expand(B,1,H,W)
        col_coords = torch.linspace(-1, 1, steps=W, device=device).view(1,1,1,W).expand(B,1,H,W)
        cnn_input = torch.cat([central_state, row_coords, col_coords], dim=1)  # (B,3,H,W)

        # Pass through each CNN block => multi-scale feats
        feats = []
        out = cnn_input
        for block in self.blocks:
            out = block(out)
            feats.append(out)

        ####################################
        # 4) Flatten CNN outputs => patch sequences + 2D pos enc
        ####################################
        patch_seqs = []
        for i, f in enumerate(feats):
            sH = sW = self.block_scales[i]
            pooled = F.adaptive_avg_pool2d(f, (sH, sW))  # (B, c_out, sH, sW)
            f_emb = self.block_embeds[i](pooled)         # => (B, d, sH, sW)

            B_, D_, HH_, WW_ = f_emb.shape
            patches = f_emb.view(B_, D_, HH_*WW_).transpose(1,2)  # => (B, #patches, d)
            patches = self.block_lns[i](patches)

            pos_2d = create_2d_sin_cos_pos_emb(HH_, WW_, self.embed_dim, device)
            patches = patches + pos_2d.unsqueeze(0)

            patch_seqs.append(patches)

        ####################################
        # 5) Transformer Layers:
        #    (Cross-attn blocks per scale) => agent self-attn
        ####################################
        attn_weights_final = None
        for layer_idx in range(self.transformer_layers):
            blocks_for_this_layer = self.cross_attn_blocks[layer_idx]

            for scale_idx in range(self.num_scales):
                scale_block = blocks_for_this_layer[scale_idx]
                cross_attn = scale_block["attn"]
                ff_cross   = scale_block["ffn"]

                scale_patches = patch_seqs[scale_idx]  # (B, #patches, d)
                agent_embed, _ = cross_attn(agent_embed, scale_patches, scale_patches)
                agent_embed = ff_cross(agent_embed)

            self_attn_block = self.agent_self_attn[layer_idx]
            self_ff = self.agent_self_ffn[layer_idx]

            agent_embed, attn_weights_final = self_attn_block(agent_embed)
            agent_embed = self_ff(agent_embed)

        ####################################
        # 6) Final LN + MLP
        ####################################
        agent_embed = self.final_agent_ln(agent_embed)
        agent_embed = self.post_cross_mlp(agent_embed) + agent_embed

        # (B,A,d) => (S,E,A,d)
        agent_embed = agent_embed.view(S, E, A, self.embed_dim)
        attn_weights_final = attn_weights_final.view(S, E, A, A)

        return agent_embed, attn_weights_final
