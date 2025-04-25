import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Dict, Union, List, Any
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from abc import ABC, abstractmethod
# Ensure Utility is importable, adjust path if necessary
from .Utility import init_weights_leaky_relu, init_weights_gelu_linear, init_weights_gelu_conv, create_2d_sin_cos_pos_emb

# --- Existing BasePolicyNetwork, DiscretePolicyNetwork, LinearNetwork ---
# --- StatesEncoder, StatesActionsEncoder, ValueNetwork remain the same ---
# --- TanhContinuousPolicyNetwork remains the same ---
# --- MultiAgentEmbeddingNetwork remains the same ---
# --- AgentIDPosEnc remains the same ---
# --- ResidualAttention remains the same ---
# --- FeedForwardBlock remains the same ---
# --- CrossAttentionFeatureExtractor remains the same ---

# Keep the existing helper classes above this point...


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
        # Ensure dimensions are compatible for concatenation
        # obs shape: (*, state_dim)
        # act shape: (*, action_dim)
        # Need to handle cases where action_dim might be a list (multi-discrete)
        # For now, assume act is already flattened or suitable for cat
        if act.shape[:-1] != obs.shape[:-1]:
             # Attempt to broadcast action shape to observation shape if necessary
             # Example: obs=(B, E, A, H), act=(B, E, A, D) -> cat result=(B, E, A, H+D)
             # Example: obs=(B, E, A, A-1, H), act=(B, E, A, A-1, D) -> cat result=(B, E, A, A-1, H+D)
             try:
                 act_expanded = act.expand(*obs.shape[:-1], act.shape[-1])
                 x = torch.cat([obs, act_expanded], dim=-1)
             except RuntimeError as e:
                 print(f"Error concatenating obs shape {obs.shape} and action shape {act.shape}: {e}")
                 # Handle error appropriately, maybe raise it again or return an error tensor
                 raise e
        else:
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
        # Handle self-attention case where K and V are derived from Q
        if self.self_attention:
            if x_k is not None or x_v is not None:
                 print("Warning: x_k and x_v provided to ResidualAttention with self_attention=True. They will be ignored.")
            x_k = x_q
            x_v = x_q
        # Handle cross-attention case where K and V might be missing (use x_q as default)
        else:
             if x_k is None:
                 print("Warning: x_k not provided for cross-attention. Using x_q as key.")
                 x_k = x_q
             if x_v is None:
                 print("Warning: x_v not provided for cross-attention. Using x_k as value.")
                 # Usually, value source is the same as key source in cross-attention
                 x_v = x_k 

        # LN before projections
        qn = self.norm_q(x_q)
        kn = self.norm_k(x_k)
        vn = self.norm_v(x_v)

        # Project Q, K, V
        q_proj = self.query_proj(qn)
        k_proj = self.key_proj(kn)
        v_proj = self.value_proj(vn)

        # Multi-head attention
        # Note: MHA expects query, key, value
        attn_out, attn_weights = self.mha(q_proj, k_proj, v_proj, need_weights=True) # Ensure weights are returned

        # Residual connection (add output of attention to the original query input x_q)
        out = x_q + attn_out
        out = self.norm_out(out) # LayerNorm after residual

        return out, attn_weights


class TanhContinuousPolicyNetwork(nn.Module):
    """
    Continuous policy network outputting a Tanh-squashed Normal distribution.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        mean_scale: float,
        std_init_bias: float = 0.0, # Configurable initial bias for the std dev head's final layer
        min_std: float = 1e-4, # Minimum standard deviation to prevent collapse
        entropy_method: str = "analytic", # "analytic" or "mc"
        n_entropy_samples: int = 5, #  only for mc
        linear_init_scale: float = 1.0,
    ):
        """
        Args:
            in_features: Dimension of the input embedding.
            out_features: Dimension of the continuous action space.
            hidden_size: Size of the hidden layers.
            mean_scale: Scaling factor for the tanh activation on the mean.
            std_init_bias: Initial bias value for the final layer of the std dev head.
                           Controls the average initial standard deviation.
            min_std: Minimum allowed standard deviation.
            entropy_method: Method to compute entropy ('analytic' or 'mc').
            n_entropy_samples: Number of samples for Monte Carlo entropy estimation.
            linear_init_scale: Scaling factor for Kaiming initialization of linear layers.
        """
        super().__init__()
        self.mean_scale = mean_scale
        self.min_std = min_std
        self.entropy_method = entropy_method
        self.n_entropy_samples = n_entropy_samples
        self.std_init_bias = std_init_bias

        # Shared layers processing the input embedding
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        # Head for predicting the mean of the Normal distribution
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features),
        )

        # Head for predicting parameters used to compute standard deviation
        self.std_param_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features)
        )

        # Apply standard Kaiming initialization with GELU assumption
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale))

        # Override the bias initialization for the final layer of std_param_head
        if hasattr(self.std_param_head[-1], 'bias'):
             nn.init.constant_(self.std_param_head[-1].bias, self.std_init_bias)

    def forward(self, x: torch.Tensor):
        """
        Computes the mean and log_std of the underlying Normal distribution
        based on the input embedding.

        Args:
            x: Input embedding tensor (shape: ..., in_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log_std tensors.
        """
        feats = self.shared(x)

        # Calculate mean (squashed to [-mean_scale, mean_scale])
        raw_mean = self.mean_head(feats)
        mean = self.mean_scale * torch.tanh(raw_mean)

        # Calculate standard deviation using softplus and add minimum floor
        std_params = self.std_param_head(feats)
        std = F.softplus(std_params) + self.min_std
        # Calculate log_std needed for Normal distribution
        log_std = torch.log(std)

        return mean, log_std

    def get_actions(self, emb: torch.Tensor, eval: bool=False):
        """
        Samples an action from the Tanh-squashed distribution, or returns the mean
        for evaluation. Also returns log probabilities and entropy.

        Args:
            emb: Input embedding tensor.
            eval: If True, return the deterministic mean of the distribution.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - actions: Sampled or mean action (in [-1, 1]).
                - log_probs: Log probability of the action.
                - entropies: Entropy of the distribution.
        """
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Define the base Normal distribution and the Tanh transformation
        base_dist = Normal(mean, std)
        transform = TanhTransform(cache_size=1) # cache_size=1 for efficiency
        tanh_dist = TransformedDistribution(base_dist, transform)

        # Sample or get mean based on eval flag
        if eval:
            # Use the distribution's mean for deterministic evaluation
            actions = tanh_dist.mean
        else:
            # Sample from the distribution using reparameterization trick
            actions = tanh_dist.rsample()

        # Clamp actions slightly away from -1 and 1 for numerical stability of log_prob
        action_clamped = actions.clamp(-1+1e-6, 1-1e-6)

        # Calculate log probability (summed across action dimensions)
        log_probs = tanh_dist.log_prob(action_clamped).sum(dim=-1)

        # Calculate entropy
        entropies = self._compute_entropy(tanh_dist)

        return action_clamped, log_probs, entropies

    def recompute_log_probs(self, emb: torch.Tensor, stored_actions: torch.Tensor):
        """
        Recomputes log probabilities and entropy for previously sampled actions
        given the current policy parameters and state embedding.

        Args:
            emb: Input embedding tensor.
            stored_actions: Actions previously sampled (in [-1, 1]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Recomputed log_probs and entropies.
        """
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Recreate the distribution with current parameters
        base_dist = Normal(mean, std)
        transform = TanhTransform(cache_size=1)
        tanh_dist = TransformedDistribution(base_dist, transform)

        # Calculate log probability of the stored actions under the current distribution
        # Clamping stored_actions might be necessary if they were very close to +/- 1
        stored_actions_clamped = stored_actions.clamp(-1+1e-6, 1-1e-6)
        log_probs = tanh_dist.log_prob(stored_actions_clamped).sum(dim=-1)

        # Calculate entropy of the current distribution
        entropies = self._compute_entropy(tanh_dist)

        return log_probs, entropies

    def _compute_entropy(self, tanh_dist: TransformedDistribution) -> torch.Tensor:
        """Helper to compute entropy using the configured method."""
        if self.entropy_method == "analytic":
            try:
                # Try analytic entropy calculation
                ent = tanh_dist.entropy()
                ent = ent.sum(dim=-1)
                return ent
            except NotImplementedError:
                # Fallback to MC estimate if analytic is not implemented
                # print("Warning: Analytic entropy failed, falling back to MC estimate.")
                return self._mc_entropy(tanh_dist)
        elif self.entropy_method == "mc":
            return self._mc_entropy(tanh_dist)
        else:
            raise ValueError(f"Unknown entropy_method: {self.entropy_method}")


    def _mc_entropy(self, tanh_dist: TransformedDistribution) -> torch.Tensor:
        """Estimates entropy using Monte Carlo sampling."""
        with torch.no_grad(): # No gradients needed for MC sampling
            # Sample multiple actions from the distribution
            action_samples = tanh_dist.rsample((self.n_entropy_samples,)) # Shape: (n_samples, ..., action_dim)
        # Calculate log probs for the samples and average
        log_probs = tanh_dist.log_prob(action_samples).sum(dim=-1) # Shape: (n_samples, ...)
        # Average log probs across samples and negate for entropy estimate
        entropy_estimate = -log_probs.mean(dim=0)
        return entropy_estimate
    

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
        

# ==============================================================================
# SharedCritic
# ==============================================================================
class SharedCritic(nn.Module):
    def __init__(self, net_cfg):
        super().__init__()
        self.value_head = ValueNetwork(**net_cfg["value_head"])
        self.baseline_head = ValueNetwork(**net_cfg["baseline_head"])
        self.value_attention = ResidualAttention(**net_cfg["value_attention"])
        self.baseline_attention = ResidualAttention(**net_cfg['baseline_attention'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_scale = net_cfg.get("linear_init_scale", 0.1)
        self.apply(lambda m: init_weights_gelu_linear(m, scale=init_scale))

    def values(self, x: torch.Tensor) -> torch.Tensor:
        """ Handles 3D (MB, A, H) or 4D (S, E, A, H) input """
        input_shape = x.shape
        H = input_shape[-1]
        A = input_shape[-2]
        leading_dims = input_shape[:-2]
        B = int(np.prod(leading_dims)) # S*E or MB

        x_flat = x.reshape(B, A, H)
        attn_out, _ = self.value_attention(x_flat) # -> (B, A, H)
        aggregated_emb = attn_out.mean(dim=1) # -> (B, H)
        vals = self.value_head(aggregated_emb) # -> (B, 1)
        vals = vals.view(*leading_dims, 1) # Reshape back, e.g. (S, E, 1) or (MB, 1)
        return vals

    def baselines(self, x: torch.Tensor) -> torch.Tensor:
        """ Handles 4D (MB, A, A2, H) or 5D (S, E, A, A2, H) input """
        input_shape = x.shape
        H = input_shape[-1]
        A2 = input_shape[-2] # Sequence length for attention (should be A)
        A = input_shape[-3]  # Agent index dimension
        leading_dims = input_shape[:-3] # (MB,) or (S, E)
        B = int(np.prod(leading_dims)) # S*E or MB

        # Flatten for batch processing by attention: (B*A, A2, H)
        x_flat = x.reshape(B * A, A2, H)
        x_attn, _ = self.baseline_attention(x_flat) # -> (B*A, A2, H)
        # Aggregate attention output - mean over the sequence dim (A2)
        agent_emb = x_attn.mean(dim=1)  # -> (B*A, H)
        base = self.baseline_head(agent_emb) # -> (B*A, 1)
        # Reshape back to original leading dims + agent dim + 1
        base = base.view(*leading_dims, A, 1) # e.g. (S, E, A, 1) or (MB, A, 1)
        return base
    

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


class MultiAgentEmbeddingNetwork(nn.Module):
    def __init__(self, net_cfg: Dict[str, Any]):
        super(MultiAgentEmbeddingNetwork, self).__init__()
        self.base_encoder = CrossAttentionFeatureExtractor(**net_cfg["cross_attention_feature_extractor"])
        self.f = StatesActionsEncoder(**net_cfg["obs_actions_encoder"])
        self.g = StatesEncoder(**net_cfg["obs_encoder"])

    def get_base_embedding(self, obs_dict: Dict[str, torch.Tensor]):
        # base_encoder now handles variable input dimensions and returns matching output shape
        agent_embeddings, attn_weights = self.base_encoder(obs_dict)
        return agent_embeddings, attn_weights

    def get_baseline_embeddings(self, common_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Prepares the input for the baseline critic head. Handles 3D/4D common_emb. """
        input_shape = common_emb.shape
        H = input_shape[-1]
        A = input_shape[-2] # Num Agents
        leading_dims = input_shape[:-2] # (MB,) or (S, E)
        B = int(np.prod(leading_dims)) # MB or S*E

        act_shape = actions.shape
        act_dim = act_shape[-1]
        act_leading_dims = act_shape[:-2]

        # Ensure actions have matching leading dimensions
        if leading_dims != act_leading_dims:
             raise ValueError(f"Leading dims mismatch: common_emb {leading_dims} vs actions {act_leading_dims}")

        baseline_input_list = []

        # Reshape inputs for encoders: (B*A, Dim)
        common_emb_flat = common_emb.reshape(B*A, H)
        actions_flat = actions.reshape(B*A, act_dim)

        # Encode all agent observations using self.g -> (B*A, H)
        all_agent_obs_emb_flat = self.g(common_emb_flat)
        # Encode all agent obs-action pairs using self.f -> (B*A, H)
        all_agent_obs_act_emb_flat = self.f(common_emb_flat, actions_flat)

        # Reshape back to (B, A, H)
        all_agent_obs_emb = all_agent_obs_emb_flat.view(B, A, H)
        all_agent_obs_act_emb = all_agent_obs_act_emb_flat.view(B, A, H)

        # Construct the sequence [agent_j_obs, gm_1_obs_act, ..., gm_A-1_obs_act] for each agent j
        for j in range(A):
            agent_j_obs_emb = all_agent_obs_emb[:, j:j+1, :]  # (B, 1, H)
            groupmates_obs_act_emb_list = []
            for i in range(A):
                if i != j:
                    groupmates_obs_act_emb_list.append(all_agent_obs_act_emb[:, i:i+1, :]) # (B, 1, H)

            if A > 1:
                 groupmates_obs_act_emb = torch.cat(groupmates_obs_act_emb_list, dim=1) # (B, A-1, H)
                 baseline_j_input = torch.cat([agent_j_obs_emb, groupmates_obs_act_emb], dim=1) # (B, A, H)
            else: # Handle single agent case
                 baseline_j_input = agent_j_obs_emb # (B, 1, H) - seq len is 1

            # Append results for agent j, add agent dimension 'A' back later
            baseline_input_list.append(baseline_j_input.unsqueeze(1)) # (B, 1, A, H) or (B, 1, 1, H)

        # Concatenate across the agent dimension 'j' which is dim=1 now
        # Resulting shape: (B, A, A, H) or (B, A, 1, H) if A=1
        final_baseline_input = torch.cat(baseline_input_list, dim=1)

        # Reshape back to original leading dimensions
        output_shape = leading_dims + (A, A, H) # e.g., (S, E, A, A, H) or (MB, A, A, H)
        return final_baseline_input.view(output_shape).contiguous()

    def get_state_embeddings(self, common_emb: torch.Tensor) -> torch.Tensor:
        input_shape = common_emb.shape
        H = input_shape[-1]; A = input_shape[-2]; leading_dims = input_shape[:-2]
        B = int(np.prod(leading_dims))
        emb_flat = self.g(common_emb.reshape(B*A, H))
        return emb_flat.view(*leading_dims, A, H) # Use H from output if different

    def get_state_action_embeddings(self, common_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
         emb_shape = common_emb.shape; act_shape = actions.shape
         H = emb_shape[-1]; D = act_shape[-1]; A = emb_shape[-2]; leading_dims = emb_shape[:-2]
         B = int(np.prod(leading_dims))
         common_emb_flat = common_emb.reshape(B*A, H); actions_flat = actions.reshape(B*A, D)
         encoded_flat = self.f(common_emb_flat, actions_flat); OutputH = encoded_flat.shape[-1]
         output_shape = leading_dims + (A, OutputH,)
         return encoded_flat.view(output_shape)

    def _split_agent_and_groupmates(self, obs_4d: torch.Tensor, actions: torch.Tensor): pass


class CrossAttentionFeatureExtractor(nn.Module):
    """
    Multi-scale architecture with a SINGLE sequential CNN to handle all input channels,
    but we capture intermediate outputs at each stage to form multiple scales.

    Steps:
      - CNN feature extraction from multi-channel input, capturing the output after each block
      - 2D sin-cos positional encodings for CNN patches
      - Multiple cross-attn blocks per scale, repeated for each transformer layer
      - Interleaved agent self-attn
      - Final LN + MLP on agent embeddings
      - (Optional) Discrete category embedding + sinusoidal ID encoding for each agent
    """

    def __init__(
        self,
        agent_obs_size: int = 9,
        num_agents: int = 10, # Max number of agents for ID embedding if used
        in_channels: int = 5,
        h: int = 50,
        w: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        cnn_channels: list = [32, 64, 128],
        cnn_kernel_sizes: list = [5, 3, 3],
        cnn_strides: list = [2, 2, 2],
        group_norms: list = [16, 8, 4], # Ensure group_norms[i] divides cnn_channels[i]
        block_scales: list = [12, 6, 3], # Target H/W for pooling after each CNN block
        transformer_layers: int = 2,
        dropout_rate: float = 0.0,
        use_agent_id: bool = True,
        ff_hidden_factor: int = 4,
        id_num_freqs: int = 8,
        conv_init_scale: float = 0.1,
        linear_init_scale: float = 1.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.h = h
        self.w = w
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.ff_hidden_factor = ff_hidden_factor
        self.use_agent_id = use_agent_id
        self.block_scales = block_scales
        self.num_scales = len(block_scales)
        if len(cnn_channels) != len(block_scales) or len(cnn_channels) != len(cnn_kernel_sizes) or \
           len(cnn_channels) != len(cnn_strides) or len(cnn_channels) != len(group_norms):
             raise ValueError("CNN config lists (channels, kernels, strides, group_norms, block_scales) must have the same length.")


        # ---------------------------------------------------------
        # A) Simple Agent MLP to embed agent states
        # ---------------------------------------------------------
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_obs_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # ---------------------------------------------------------
        # B) (Optional) Agent ID Embedding + Sinusoidal
        # ---------------------------------------------------------
        if use_agent_id:
            self.agent_id_embedding = nn.Embedding(num_agents, embed_dim)
            nn.init.normal_(self.agent_id_embedding.weight, mean=0.0, std=0.01)
            self.agent_id_pos_enc = AgentIDPosEnc(num_freqs=id_num_freqs, id_embed_dim=embed_dim)
            self.agent_id_sum_ln = nn.LayerNorm(embed_dim)
        else:
            self.agent_id_embedding = None
            self.agent_id_pos_enc = None

        # ---------------------------------------------------------
        # C) SINGLE CNN with multiple blocks
        #    We have in_channels + 2 coordinate channels at input
        # ---------------------------------------------------------
        total_in_c = self.in_channels + 2
        self.cnn_blocks = nn.ModuleList()
        prev_c = total_in_c

        for i, out_c in enumerate(cnn_channels):
             # Ensure group_norms[i] divides out_c
             if out_c % group_norms[i] != 0:
                 raise ValueError(f"group_norms[{i}] ({group_norms[i]}) must divide cnn_channels[{i}] ({out_c})")

             block = nn.Sequential(
                 nn.Conv2d(prev_c, out_c, kernel_size=cnn_kernel_sizes[i],
                           stride=cnn_strides[i], padding=cnn_kernel_sizes[i]//2),
                 nn.GroupNorm(group_norms[i], out_c),
                 nn.GELU(),

                 # Added another conv layer per block as suggested by some architectures
                 nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
                 nn.GroupNorm(group_norms[i], out_c),
                 nn.GELU()
            )
             self.cnn_blocks.append(block)
             prev_c = out_c


        # For each scale, we do a 1x1 conv => embed_dim
        self.block_embeds = nn.ModuleList()
        self.block_lns = nn.ModuleList()
        for out_c in cnn_channels:
            emb_conv = nn.Conv2d(out_c, embed_dim, kernel_size=1, stride=1)
            ln = nn.LayerNorm(embed_dim) # LayerNorm applied on the embedding dimension
            self.block_embeds.append(emb_conv)
            self.block_lns.append(ln)


        # Precompute positional embeddings for each scale to avoid recomputation
        self.positional_embeddings = nn.ParameterList()
        for scale in self.block_scales:
             pos_emb = create_2d_sin_cos_pos_emb(scale, scale, embed_dim, device=torch.device("cpu")) # Create on CPU first
             self.positional_embeddings.append(nn.Parameter(pos_emb, requires_grad=False))

        # ---------------------------------------------------------
        # D) Cross-Attn blocks (per scale) repeated for each layer
        # ---------------------------------------------------------
        self.cross_attn_blocks = nn.ModuleList()
        for _ in range(transformer_layers):
            # For each layer, we have a set of cross-attn blocks (one per scale).
            blocks_per_scale = nn.ModuleList()
            for scale_idx in range(self.num_scales):
                cross_attn = ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=False)
                ff_cross   = FeedForwardBlock(embed_dim, hidden_dim=embed_dim * ff_hidden_factor, dropout=dropout_rate)
                scale_mod  = nn.ModuleDict({
                    "attn": cross_attn,
                    "ffn": ff_cross
                })
                blocks_per_scale.append(scale_mod)
            self.cross_attn_blocks.append(blocks_per_scale)

        # ---------------------------------------------------------
        # E) Agent Self-Attn for each transformer layer
        # ---------------------------------------------------------
        self.agent_self_attn = nn.ModuleList()
        self.agent_self_ffn  = nn.ModuleList()
        for _ in range(transformer_layers):
            self_attn = ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=True)
            ff_self   = FeedForwardBlock(embed_dim, hidden_dim=embed_dim * ff_hidden_factor, dropout=dropout_rate)
            self.agent_self_attn.append(self_attn)
            self.agent_self_ffn.append(ff_self)

        # ---------------------------------------------------------
        # F) Final LN + MLP
        # ---------------------------------------------------------
        self.final_agent_ln = nn.LayerNorm(embed_dim)
        self.post_cross_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # ---------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------
        # CNN conv layers => conv_init_scale
        self.cnn_blocks.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale))
        self.block_embeds.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale))

        # All linear layers => linear_init_scale
        # Exclude LayerNorm/GroupNorm biases/weights from scaling if desired
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)


    def forward(self, state_dict: Dict[str, torch.Tensor]):
        """
        Inputs:
          state_dict["agent"]   => shape (..., A, agent_obs_size) e.g., (S,E,A,D) or (MB,A,D)
          state_dict["central"] => shape (..., in_channels*h*w) e.g., (S,E, flat) or (MB, flat)
        Returns:
          agent_embed => (..., A, embed_dim) matching leading dims of input
          attn_weights => (..., A, A) from the last agent self-attn
        """
        agent_state = state_dict["agent"]
        central_data = state_dict["central"]
        device = agent_state.device

        # --- Determine Input Shape ---
        agent_shape = agent_state.shape
        leading_dims = agent_shape[:-2] # e.g., (S, E) or (MB,)
        A = agent_shape[-2]          # Number of agents
        obs_dim = agent_shape[-1]    # Agent observation dim
        B = int(np.prod(leading_dims)) # Total batch size (S*E or MB)


        # --- Move precomputed positional embeddings to the correct device ---
        if self.positional_embeddings[0].device != device:
            for i in range(len(self.positional_embeddings)):
                self.positional_embeddings[i] = nn.Parameter(self.positional_embeddings[i].to(device), requires_grad=False)

        # ------------------------------------------
        # 1) Encode agent-state with agent_encoder
        # ------------------------------------------
        # Reshape to (B*A, obs_dim)
        flat_agents = agent_state.view(B * A, obs_dim)
        agent_embed = self.agent_encoder(flat_agents) # (B*A, embed_dim)
        agent_embed = agent_embed.view(B, A, self.embed_dim)  # (B, A, embed_dim)

        # ------------------------------------------
        # 2) Optionally add agent ID embeddings
        # ------------------------------------------
        if self.use_agent_id:
            agent_ids = torch.arange(A, device=device).view(1, A).expand(B, A) # (B, A)
            discrete_emb = self.agent_id_embedding(agent_ids)    # (B, A, embed_dim)
            pos_enc      = self.agent_id_pos_enc(agent_ids)      # (B, A, embed_dim)
            agent_embed = agent_embed + discrete_emb + pos_enc # Add embeddings
            agent_embed = self.agent_id_sum_ln(agent_embed)      # Apply LayerNorm

        # ------------------------------------------
        # 3) Reshape central => (B, in_channels, H, W) then add coordinate channels
        # ------------------------------------------
        expected_central_flat_dim = self.in_channels * self.h * self.w
        # Reshape central data, assuming leading dims match agent_state
        central_data = central_data.view(B, expected_central_flat_dim)
        central_data = central_data.view(B, self.in_channels, self.h, self.w)

        row_coords = torch.linspace(-1, 1, steps=self.h, device=device).view(1, 1, self.h, 1).expand(B, 1, self.h, self.w)
        col_coords = torch.linspace(-1, 1, steps=self.w, device=device).view(1, 1, 1, self.w).expand(B, 1, self.h, self.w)
        cnn_input = torch.cat([central_data, row_coords, col_coords], dim=1) # (B, in_channels+2, H, W)

        # ------------------------------------------
        # 4) Pass through CNN blocks => multi-scale feats
        # ------------------------------------------
        feats = []
        out_cnn = cnn_input
        for block in self.cnn_blocks:
            out_cnn = block(out_cnn)
            feats.append(out_cnn)

        # ------------------------------------------
        # 5) Process each scale: Pool, Embed, Flatten, Add Positional Encoding
        # ------------------------------------------
        patch_seqs = []
        for i, f in enumerate(feats):
            sH = sW = self.block_scales[i]
            pooled = F.adaptive_avg_pool2d(f, (sH, sW)) # (B, c_out, sH, sW)
            emb_2d = self.block_embeds[i](pooled) # (B, embed_dim, sH, sW)
            B_, D_, HH_, WW_ = emb_2d.shape
            patches = emb_2d.view(B_, D_, HH_*WW_).transpose(1, 2).contiguous() # (B, N_patches, embed_dim)
            patches = self.block_lns[i](patches)
            pos_2d = self.positional_embeddings[i].unsqueeze(0) # (1, N_patches, embed_dim)
            patches = patches + pos_2d
            patch_seqs.append(patches)

        # ------------------------------------------
        # 6) Transformer Layers: Cross-Attention and Self-Attention
        # ------------------------------------------
        attn_weights_final = None
        for layer_idx in range(self.transformer_layers):
            scale_blocks = self.cross_attn_blocks[layer_idx]
            for scale_idx, block_dict in enumerate(scale_blocks):
                cross_attn = block_dict["attn"]
                ff_cross   = block_dict["ffn"]
                scale_patches = patch_seqs[scale_idx]
                agent_embed, _ = cross_attn(agent_embed, scale_patches, scale_patches)
                agent_embed = ff_cross(agent_embed)

            self_attn_block = self.agent_self_attn[layer_idx]
            self_ff         = self.agent_self_ffn[layer_idx]
            agent_embed, attn_weights = self_attn_block(agent_embed)
            if layer_idx == self.transformer_layers - 1:
                 attn_weights_final = attn_weights # (B, A, A)
            agent_embed = self_ff(agent_embed)

        # ------------------------------------------
        # 7) Final LN + MLP
        # ------------------------------------------
        agent_embed = self.final_agent_ln(agent_embed)
        residual = agent_embed
        agent_embed = self.post_cross_mlp(agent_embed)
        agent_embed = agent_embed + residual

        # --- Reshape Output to Match Input's Leading Dimensions ---
        # Final agent_embed shape is (B, A, embed_dim)
        # Reshape back using the original leading_dims, A, and embed_dim
        output_shape = leading_dims + (A, self.embed_dim)
        agent_embed = agent_embed.view(output_shape) # e.g., (S, E, A, H) or (MB, A, H)

        # Reshape final attention weights if they exist
        if attn_weights_final is not None:
             attn_output_shape = leading_dims + (A, A)
             attn_weights_final = attn_weights_final.view(attn_output_shape) # e.g., (S, E, A, A) or (MB, A, A)

        return agent_embed, attn_weights_final
        """
        Inputs:
          state_dict["agent"]   => shape (S,E,A, agent_obs_size)
          state_dict["central"] => shape (S,E, in_channels*h*w)
        Returns:
          agent_embed => (S,E,A, embed_dim)
          attn_weights => (S,E,A,A) from the last agent self-attn (for visualization/analysis)
        """
        agent_state = state_dict["agent"]
        central_data= state_dict["central"]
        device = agent_state.device

        # Move precomputed positional embeddings to the correct device if needed
        if self.positional_embeddings[0].device != device:
            for i in range(len(self.positional_embeddings)):
                self.positional_embeddings[i] = nn.Parameter(self.positional_embeddings[i].to(device), requires_grad=False)


        S, E, A, obs_dim = agent_state.shape
        B = S * E # Combined batch dimension

        # ------------------------------------------
        # 1) Encode agent-state with agent_encoder
        # ------------------------------------------
        flat_agents = agent_state.view(B * A, obs_dim) # (B*A, obs_dim)
        agent_embed = self.agent_encoder(flat_agents) # (B*A, embed_dim)
        agent_embed = agent_embed.view(B, A, self.embed_dim)  # (B, A, embed_dim)


        # 2) Optionally add agent ID embeddings
        if self.use_agent_id:
            # Assuming agent IDs are implicitly 0 to A-1 for each environment in the batch
            agent_ids = torch.arange(A, device=device).view(1, A).expand(B, A) # (B, A)
            discrete_emb = self.agent_id_embedding(agent_ids)  # (B, A, embed_dim)
            pos_enc      = self.agent_id_pos_enc(agent_ids)    # (B, A, embed_dim)

            agent_embed = agent_embed + discrete_emb + pos_enc # Add embeddings
            agent_embed = self.agent_id_sum_ln(agent_embed)    # Apply LayerNorm

        # ------------------------------------------
        # 3) Reshape central => (B, in_channels, H, W) then add coordinate channels
        # ------------------------------------------
        # Ensure central_data has the correct total size
        expected_central_flat_dim = self.in_channels * self.h * self.w
        if central_data.shape[-1] != expected_central_flat_dim:
            raise ValueError(f"Expected central data flat dim {expected_central_flat_dim}, got {central_data.shape[-1]}")
        
        central_data = central_data.view(B, self.in_channels, self.h, self.w)

        # Create coordinate channels dynamically on the correct device
        row_coords = torch.linspace(-1, 1, steps=self.h, device=device).view(1, 1, self.h, 1).expand(B, 1, self.h, self.w)
        col_coords = torch.linspace(-1, 1, steps=self.w, device=device).view(1, 1, 1, self.w).expand(B, 1, self.h, self.w)
        cnn_input = torch.cat([central_data, row_coords, col_coords], dim=1) # (B, in_channels+2, H, W)

        # ------------------------------------------
        # 4) Pass through CNN blocks => multi-scale feats
        # ------------------------------------------
        feats = []
        out_cnn = cnn_input
        for block in self.cnn_blocks:
            out_cnn = block(out_cnn)
            feats.append(out_cnn) # Store output of each block

        # ------------------------------------------
        # 5) Process each scale: Pool, Embed, Flatten, Add Positional Encoding
        # ------------------------------------------
        patch_seqs = []
        for i, f in enumerate(feats):
            sH = sW = self.block_scales[i] # Target size for this scale

            # Adaptive average pooling to target size (sH, sW)
            pooled = F.adaptive_avg_pool2d(f, (sH, sW)) # (B, c_out, sH, sW)

            # Project feature channels to embed_dim using 1x1 Conv
            emb_2d = self.block_embeds[i](pooled) # (B, embed_dim, sH, sW)

            # Flatten spatial dimensions: (B, embed_dim, sH, sW) -> (B, embed_dim, sH*sW) -> (B, sH*sW, embed_dim)
            B_, D_, HH_, WW_ = emb_2d.shape
            patches = emb_2d.view(B_, D_, HH_*WW_).transpose(1, 2).contiguous() # (B, N_patches, embed_dim)

            # Apply LayerNorm
            patches = self.block_lns[i](patches)

            # Add precomputed 2D sinusoidal positional embedding for this scale
            # Positional embedding shape is (sH*sW, embed_dim)
            pos_2d = self.positional_embeddings[i].unsqueeze(0) # Add batch dim -> (1, N_patches, embed_dim)
            patches = patches + pos_2d # Add to patch embeddings

            patch_seqs.append(patches) # Store sequence of patches for this scale

        # ------------------------------------------
        # 6) Transformer Layers: Cross-Attention and Self-Attention
        # ------------------------------------------
        attn_weights_final = None # To store weights from the last self-attention layer
        for layer_idx in range(self.transformer_layers):
            scale_blocks = self.cross_attn_blocks[layer_idx] # ModuleList of ModuleDicts for this layer

            # CROSS-ATTENTION: Agent embeddings attend to each scale's patches
            for scale_idx, block_dict in enumerate(scale_blocks):
                cross_attn = block_dict["attn"]
                ff_cross   = block_dict["ffn"]

                scale_patches = patch_seqs[scale_idx] # (B, N_patches_scale, embed_dim)

                # Agent embeddings (queries) attend to patch embeddings (keys, values)
                agent_embed, _ = cross_attn(
                    x_q=agent_embed,    # (B, A, embed_dim)
                    x_k=scale_patches,  # (B, N_patches_scale, embed_dim)
                    x_v=scale_patches   # (B, N_patches_scale, embed_dim)
                )
                # Apply feed-forward block after cross-attention
                agent_embed = ff_cross(agent_embed)

            # AGENT SELF-ATTENTION: Agents attend to each other
            self_attn_block = self.agent_self_attn[layer_idx]
            self_ff         = self.agent_self_ffn[layer_idx]

            # Agents attend to themselves
            agent_embed, attn_weights = self_attn_block(agent_embed) # Get weights here
             # Store weights from the *last* self-attention layer
            if layer_idx == self.transformer_layers - 1:
                 attn_weights_final = attn_weights # (B, A, A)

            # Apply feed-forward block after self-attention
            agent_embed = self_ff(agent_embed)

        # ------------------------------------------
        # 7) Final LN + MLP
        # ------------------------------------------
        agent_embed = self.final_agent_ln(agent_embed) # Apply LayerNorm
        residual = agent_embed # Store pre-MLP value for residual connection
        agent_embed = self.post_cross_mlp(agent_embed) # Apply MLP
        agent_embed = agent_embed + residual # Add residual connection

        # Reshape back to (S, E, A, embed_dim)
        agent_embed = agent_embed.view(S, E, A, self.embed_dim)

        # Reshape final attention weights if they exist
        if attn_weights_final is not None:
             # attn_weights_final shape was (B, A, A), where B = S * E
             attn_weights_final = attn_weights_final.view(S, E, A, A)


        return agent_embed, attn_weights_final