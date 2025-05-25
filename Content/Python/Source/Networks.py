# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Dict, Union, List, Any, Optional, Tuple
from torch.distributions import Beta, AffineTransform, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from abc import ABC, abstractmethod

# Assuming Utility functions are correctly imported from your project structure
from .Utility import init_weights_leaky_relu, init_weights_gelu_linear, init_weights_gelu_conv, create_2d_sin_cos_pos_emb

class BasePolicyNetwork(nn.Module, ABC):
    """Abstract base class for policy networks (discrete or continuous)."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_actions(self, emb: torch.Tensor, eval: bool = False):
        """
        Given embeddings, compute actions, log probabilities, and entropies.

        Args:
            emb (torch.Tensor): Input embeddings (Batch, ..., Features).
            eval (bool): If True, return deterministic actions (e.g., mean or mode).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: actions, log_probs, entropies.
        """
        raise NotImplementedError

    @abstractmethod
    def recompute_log_probs(self, emb: torch.Tensor, actions: torch.Tensor):
        """
        Recompute log probabilities and entropies for given actions and embeddings,
        using the current network parameters. Useful for PPO updates.

        Args:
            emb (torch.Tensor): Input embeddings (Batch, ..., Features).
            actions (torch.Tensor): Actions previously taken (Batch, ..., ActionDim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: log_probs, entropies.
        """
        raise NotImplementedError


class DiscretePolicyNetwork(BasePolicyNetwork):
    """
    Policy network for discrete or multi-discrete action spaces using Categorical distributions.
    Accepts dropout_rate via **kwargs for shared layers.
    """
    def __init__(self, in_features: int, out_features: Union[int, List[int]], hidden_size: int, linear_init_scale: float = 1.0, **kwargs):
        super().__init__()
        self.in_features  = in_features
        self.hidden_size  = hidden_size
        self.out_is_multi = isinstance(out_features, list)
        dropout_rate = kwargs.get('dropout_rate', 0.0) # Extract dropout rate

        # Shared MLP layers with optional dropout
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

        # Output head(s) for logits
        if not self.out_is_multi:
            # Single discrete action space
            self.head = nn.Linear(hidden_size, out_features)
        else:
            # Multi-discrete action space (one head per branch)
            self.branch_heads = nn.ModuleList([
                nn.Linear(hidden_size, branch_sz)
                for branch_sz in out_features
            ])

        # Initialize linear layers using GELU assumption
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor):
        """Computes logits for the Categorical distribution(s)."""
        feats = self.shared(x)
        if not self.out_is_multi:
            return self.head(feats) # Returns tensor (Batch, num_actions)
        else:
            # Returns list of tensors [(Batch, branch1_size), (Batch, branch2_size), ...]
            return [bh(feats) for bh in self.branch_heads]

    def get_actions(self, emb: torch.Tensor, eval: bool = False):
        """Samples actions or takes argmax, returns actions, log_probs, entropies."""
        logits_output = self.forward(emb)
        B = emb.shape[0] # Assumes Batch dim is the first dim

        if not self.out_is_multi:
            logits_tensor = logits_output
            dist = torch.distributions.Categorical(logits=logits_tensor)
            actions = torch.argmax(logits_tensor, dim=-1) if eval else dist.sample()
            log_probs = dist.log_prob(actions) # Shape (B,)
            entropies = dist.entropy() # Shape (B,)
            return actions, log_probs, entropies
        else:
            logits_list = logits_output
            actions_list, lp_list, ent_list = [], [], []
            for branch_logits in logits_list:
                dist_b = torch.distributions.Categorical(logits=branch_logits)
                a_b = torch.argmax(branch_logits, dim=-1) if eval else dist_b.sample()
                lp_b  = dist_b.log_prob(a_b)
                ent_b = dist_b.entropy()
                actions_list.append(a_b)
                lp_list.append(lp_b)
                ent_list.append(ent_b)

            # Stack actions, sum log_probs and entropies across branches
            acts_stacked = torch.stack(actions_list, dim=-1) # Shape (B, num_branches)
            lp_stacked   = torch.stack(lp_list, dim=-1).sum(dim=-1) # Shape (B,)
            ent_stacked  = torch.stack(ent_list, dim=-1).sum(dim=-1) # Shape (B,)
            return acts_stacked, lp_stacked, ent_stacked

    def recompute_log_probs(self, emb: torch.Tensor, actions: torch.Tensor):
        """Recomputes log_probs and entropies for given actions."""
        logits_output = self.forward(emb)
        B = emb.shape[0]

        if not self.out_is_multi:
            logits_tensor = logits_output
            dist = torch.distributions.Categorical(logits=logits_tensor)
            # Ensure actions are Long type and remove trailing dim if present (e.g., from (B, 1))
            log_probs = dist.log_prob(actions.long().squeeze(-1))
            entropies = dist.entropy()
            return log_probs, entropies
        else:
            logits_list = logits_output
            actions = actions.long() # Shape should be (B, num_branches)
            lp_list, ent_list = [], []
            for i, branch_logits in enumerate(logits_list):
                dist_b = torch.distributions.Categorical(logits=branch_logits)
                a_b    = actions[:, i] # Actions for the current branch
                lp_b   = dist_b.log_prob(a_b)
                ent_b  = dist_b.entropy()
                lp_list.append(lp_b)
                ent_list.append(ent_b)

            log_probs = torch.stack(lp_list, dim=-1).sum(dim=-1)
            entropies = torch.stack(ent_list, dim=-1).sum(dim=-1)
            return log_probs, entropies


class LinearNetwork(nn.Module):
    """Simple MLP block: Linear -> [Dropout] -> [GELU] -> [LayerNorm]"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout_rate: float = 0.0,
                 activation: bool = True,
                 layer_norm: bool = False,
                 linear_init_scale: float = 1.0):
        super().__init__()
        layers = []
        linear_layer = nn.Linear(in_features, out_features)
        layers.append(linear_layer)
        # Dropout is applied *after* the linear layer, before activation/norm
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        if activation:
            layers.append(nn.GELU())
        if layer_norm:
            # LayerNorm is applied on the output features
            layers.append(nn.LayerNorm(out_features))
        self.model = nn.Sequential(*layers)
        # Apply initialization only to the Linear layer within this block
        init_weights_gelu_linear(linear_layer, scale=linear_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StatesEncoder(nn.Module):
    """Encodes state vector using two LinearNetwork blocks."""
    def __init__(self, state_dim: int, hidden_size: int, output_size: int, **kwargs):
        super().__init__()
        # Filter kwargs to pass to each LinearNetwork
        # First block usually has activation & norm based on kwargs
        kwargs_hidden = kwargs.copy()
        # Second block (output) typically doesn't have activation/norm
        kwargs_output = {k:v for k,v in kwargs.items() if k not in ['activation', 'layer_norm']}
        kwargs_output['activation'] = False
        kwargs_output['layer_norm'] = False

        self.net = nn.Sequential(
            LinearNetwork(state_dim, hidden_size, **kwargs_hidden),
            LinearNetwork(hidden_size, output_size, **kwargs_output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatesActionsEncoder(nn.Module):
    """Encodes concatenated state-action pair using two LinearNetwork blocks."""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, output_size: int, **kwargs):
        super().__init__()
        # Filter kwargs similarly to StatesEncoder
        kwargs_hidden = kwargs.copy()
        kwargs_output = {k:v for k,v in kwargs.items() if k not in ['activation', 'layer_norm']}
        kwargs_output['activation'] = False
        kwargs_output['layer_norm'] = False

        self.net = nn.Sequential(
            LinearNetwork(state_dim + action_dim, hidden_size, **kwargs_hidden),
            LinearNetwork(hidden_size, output_size, **kwargs_output)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # Ensure correct broadcasting/expansion for concatenation
        if act.shape[:-1] != obs.shape[:-1]:
             try:
                 # Expand action dimensions (except the last) to match observation
                 act_expanded = act.expand(*obs.shape[:-1], act.shape[-1])
                 x = torch.cat([obs, act_expanded], dim=-1)
             except RuntimeError as e:
                 print(f"Error concatenating obs shape {obs.shape} and action shape {act.shape}: {e}")
                 raise e
        else:
            # Shapes already match for concatenation
            x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    """MLP predicting a scalar value from input features, with optional dropout."""
    def __init__(self, in_features: int, hidden_size: int, dropout_rate: float = 0.0,
                 linear_init_scale: float = 1.0, # General scale for hidden layers
                 output_layer_init_gain: Optional[float] = None): # Specific gain for the output layer
        super().__init__()
        
        # Define layers individually for more control over initialization
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        
        self.output_layer = nn.Linear(hidden_size, 1) # Outputs a single scalar value

        # Initialize hidden layers using the general linear_init_scale
        # Assuming init_weights_gelu_linear is available in the scope
        init_weights_gelu_linear(self.fc1, scale=linear_init_scale)
        init_weights_gelu_linear(self.fc2, scale=linear_init_scale)

        # Specific initialization for the output layer
        if output_layer_init_gain is not None:
            init.orthogonal_(self.output_layer.weight, gain=output_layer_init_gain)
            if self.output_layer.bias is not None:
                init.zeros_(self.output_layer.bias)
        else:
            # Fallback to the general init scale if specific gain is not provided for the output layer
            init_weights_gelu_linear(self.output_layer, scale=linear_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        x = self.output_layer(x)
        return x


class ResidualAttention(nn.Module):
    """
    Multi-Head Attention block with residual connection and pre-LayerNorm.
    Supports both self-attention and cross-attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, self_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = self_attention

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Layer normalization applied *before* projections (Pre-LN)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

        # Multi-Head Attention layer (dropout applied to attention weights)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        # Layer normalization applied *after* the residual connection (Post-LN)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, x_v=None):
        """
        Forward pass for the attention block.

        Args:
            x_q (torch.Tensor): Query tensor.
            x_k (torch.Tensor, optional): Key tensor. If None and self_attention=True, uses x_q.
                                         If None and self_attention=False, defaults to x_q.
            x_v (torch.Tensor, optional): Value tensor. If None and self_attention=True, uses x_q.
                                         If None and self_attention=False, defaults to x_k.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor after attention and residual connection,
                                             and attention weights.
        """
        # Determine K, V based on attention type and provided inputs
        if self.self_attention:
            # In self-attention, K and V are typically derived from Q
            key_input = x_q if x_k is None else x_k
            value_input = x_q if x_v is None else x_v
        else: # Cross-attention
            key_input = x_q if x_k is None else x_k # Default K to Q if not provided
            value_input = key_input if x_v is None else x_v # Default V to K if not provided

        # Apply pre-LayerNorm
        qn = self.norm_q(x_q)
        kn = self.norm_k(key_input)
        vn = self.norm_v(value_input)

        # Linear projections
        q_proj = self.query_proj(qn)
        k_proj = self.key_proj(kn)
        v_proj = self.value_proj(vn)

        # Multi-Head Attention computation
        # MHA expects query, key, value arguments
        attn_out, attn_weights = self.mha(q_proj, k_proj, v_proj, need_weights=True)

        # Residual connection: Add attention output to original query input
        out = x_q + attn_out
        # Apply post-LayerNorm
        out = self.norm_out(out)

        return out, attn_weights


class TanhContinuousPolicyNetwork(BasePolicyNetwork):
    """
    Continuous policy network using a Tanh-squashed Normal distribution.
    Features state-dependent variance computed via softplus activation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        mean_scale: float,
        std_init_bias: float = 0.0,
        min_std: float = 1e-4,
        entropy_method: str = "analytic",
        n_entropy_samples: int = 5,
        linear_init_scale: float = 1.0,
        dropout_rate: float = 0.0 # Added dropout_rate argument
    ):
        super().__init__()
        self.mean_scale = mean_scale
        self.min_std = min_std
        self.entropy_method = entropy_method
        self.n_entropy_samples = n_entropy_samples
        self.std_init_bias = std_init_bias

        # Shared MLP backbone with optional dropout
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

        # Head for predicting the mean, squashed by tanh
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

        # Initialize linear layers
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

        # Override the bias initialization for the final layer of std_param_head
        if hasattr(self.std_param_head[-1], 'bias'):
             nn.init.constant_(self.std_param_head[-1].bias, self.std_init_bias)

    def forward(self, x: torch.Tensor):
        """Computes mean and log_std for the underlying Normal distribution."""
        feats = self.shared(x)

        # Mean calculation with tanh squashing and scaling
        raw_mean = self.mean_head(feats)
        mean = self.mean_scale * torch.tanh(raw_mean)

        # Standard deviation calculation using softplus activation + minimum floor
        std_params = self.std_param_head(feats)
        # softplus ensures positivity, min_std prevents collapse
        std = F.softplus(std_params) + self.min_std
        # log_std is required for the Normal distribution
        log_std = torch.log(std)

        return mean, log_std

    def get_actions(self, emb: torch.Tensor, eval: bool=False):
        """Samples actions or returns mean, computes log_probs and entropy."""
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Define the base Normal and the Tanh transformation
        base_dist = Normal(mean, std)
        # Caching the inverse Tanh transformation can speed up log_prob calculation
        transform = TanhTransform(cache_size=1)
        tanh_dist = TransformedDistribution(base_dist, transform)

        # Get action: mean for eval, sample for training
        actions = tanh_dist.mean if eval else tanh_dist.rsample()

        # Clamp actions slightly inside [-1, 1] for numerical stability of log_prob
        action_clamped = actions.clamp(-1+1e-6, 1-1e-6)

        # Calculate log probability (sum across action dimensions)
        # log_prob automatically accounts for the Tanh transformation Jacobian
        log_probs = tanh_dist.log_prob(action_clamped).sum(dim=-1)

        # Calculate entropy
        entropies = self._compute_entropy(tanh_dist)

        return action_clamped, log_probs, entropies

    def recompute_log_probs(self, emb: torch.Tensor, stored_actions: torch.Tensor):
        """Recomputes log_probs and entropy for given actions and current policy."""
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Recreate the distribution with current parameters
        base_dist = Normal(mean, std)
        transform = TanhTransform(cache_size=1)
        tanh_dist = TransformedDistribution(base_dist, transform)

        # Clamp stored actions for stability before calculating log_prob
        stored_actions_clamped = stored_actions.clamp(-1+1e-6, 1-1e-6)
        log_probs = tanh_dist.log_prob(stored_actions_clamped).sum(dim=-1)

        # Calculate entropy of the current distribution
        entropies = self._compute_entropy(tanh_dist)

        return log_probs, entropies

    def _compute_entropy(self, tanh_dist: TransformedDistribution) -> torch.Tensor:
        """Computes entropy using the specified method (analytic approx or MC)."""
        if self.entropy_method == "analytic":
            # Analytic entropy for TanhNormal is complex.
            # A common approximation is to use the entropy of the base Normal distribution.
            # Note: This is an approximation and doesn't account for the volume change from Tanh.
            try:
                # Attempt to use the base distribution's entropy
                ent = tanh_dist.base_dist.entropy()
                # Sum across action dimensions if entropy is per-dimension
                if ent.shape == tanh_dist.base_dist.mean.shape:
                    ent = ent.sum(dim=-1)
                return ent
            except NotImplementedError:
                 # If base entropy fails or isn't implemented, fall back to MC
                 return self._mc_entropy(tanh_dist)
        elif self.entropy_method == "mc":
            # Use Monte Carlo estimation
            return self._mc_entropy(tanh_dist)
        else:
            raise ValueError(f"Unknown entropy_method: {self.entropy_method}")

    def _mc_entropy(self, tanh_dist: TransformedDistribution) -> torch.Tensor:
        """Estimates entropy using Monte Carlo sampling."""
        with torch.no_grad(): # Disable gradients for sampling
            # Sample actions from the Tanh-squashed distribution
            action_samples = tanh_dist.rsample((self.n_entropy_samples,)) # Shape: (n_samples, ..., action_dim)
        # Calculate log probabilities of the samples
        log_probs = tanh_dist.log_prob(action_samples).sum(dim=-1) # Shape: (n_samples, ...)
        # Estimate entropy as the negative mean log probability
        entropy_estimate = -log_probs.mean(dim=0)
        return entropy_estimate


class GaussianPolicyNetwork(BasePolicyNetwork):
    """
    Continuous policy network using an unbounded Normal (Gaussian) distribution.
    Predicts state-dependent mean and standard deviation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int, # This is the action dimension
        hidden_size: int,
        std_init_bias: float = 0.0, # Controls initial standard deviation
        min_std: float = 1e-4,      # Minimum standard deviation to prevent collapse
        # Optional: Clipping log_std can help stabilize variance
        log_std_min: float = -20.0, # Lower bound for log_std output
        log_std_max: float = 2.0,   # Upper bound for log_std output
        linear_init_scale: float = 1.0,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.min_std = min_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.std_init_bias = std_init_bias

        # Shared MLP backbone
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

        # Head for predicting the mean of the Gaussian
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features), # Outputs action_dim means
        )

        # Head for predicting parameters used to compute standard deviation
        self.std_param_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features) # Outputs action_dim std parameters
        )

        # Initialize linear layers
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

        # Override the bias initialization for the final layer of std_param_head
        if hasattr(self.std_param_head[-1], 'bias'):
             nn.init.constant_(self.std_param_head[-1].bias, self.std_init_bias)

    def forward(self, x: torch.Tensor):
        """Computes mean and log_std for the Normal distribution."""
        feats = self.shared(x)

        # --- Mean Calculation ---
        # Direct output, no tanh squash or scaling needed for unbounded Gaussian
        mean = self.mean_head(feats)

        # --- Standard Deviation Calculation ---
        std_params = self.std_param_head(feats)

        # Optional: Clip the raw log_std parameters before applying softplus
        # This can prevent extreme variance values if the network outputs large numbers
        std_params_clipped = torch.clamp(std_params, self.log_std_min, self.log_std_max)

        # Use softplus to ensure positivity, add min_std floor
        std = F.softplus(std_params_clipped) + self.min_std
        log_std = torch.log(std) # Normal distribution uses log_std

        return mean, log_std

    def get_actions(self, emb: torch.Tensor, eval: bool = False):
        """Samples actions or returns mean, computes log_probs and entropy."""
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Define the base Normal distribution
        dist = Normal(mean, std)

        # Get action: mean for eval, sample for training
        # Use rsample for differentiable sampling during training
        actions = dist.mean if eval else dist.rsample()

        # --- IMPORTANT: Do NOT clamp actions here for a pure Gaussian policy ---
        # The environment (C++ code) will handle the clamping/bounding
        # action_clamped = actions # No clamping

        # Calculate log probability (sum across action dimensions)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Calculate entropy (sum across action dimensions)
        entropies = dist.entropy().sum(dim=-1)

        return actions, log_probs, entropies # Return unclamped actions

    def recompute_log_probs(self, emb: torch.Tensor, stored_actions: torch.Tensor):
        """Recomputes log_probs and entropy for given actions and current policy."""
        mean, log_std = self.forward(emb)
        std = torch.exp(log_std)

        # Recreate the distribution with current parameters
        dist = Normal(mean, std)

        # Calculate log_prob of the potentially unbounded actions that were stored
        log_probs = dist.log_prob(stored_actions).sum(dim=-1)

        # Calculate entropy of the current distribution
        entropies = dist.entropy().sum(dim=-1)

        return log_probs, entropies


class BetaPolicyNetwork(BasePolicyNetwork):
    """
    Continuous policy network using a Beta distribution, scaled to [-1, 1].
    Outputs alpha and beta parameters for the Beta distribution.
    Handles entropy calculation for the transformed distribution manually.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        param_init_bias: float = 0.0,
        min_concentration: float = 1.001,
        linear_init_scale: float = 1.0,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.action_dim = out_features
        self.min_concentration = min_concentration

        # Shared MLP backbone
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

        # Head for predicting raw alpha parameters
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features)
        )

        # Head for predicting raw beta parameters
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features)
        )

        # Initialize linear layers
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

        # Apply specific bias initialization to the output layers for alpha/beta params
        if hasattr(self.alpha_head[-1], 'bias'):
             nn.init.constant_(self.alpha_head[-1].bias, param_init_bias)
        if hasattr(self.beta_head[-1], 'bias'):
             nn.init.constant_(self.beta_head[-1].bias, param_init_bias)

        # Precompute log(scale) for entropy calculation
        self.log_scale = torch.log(torch.tensor(2.0)) # From AffineTransform(loc=-1, scale=2)

    def _get_transformed_distribution(self, emb: torch.Tensor):
        """ Helper to create the Beta distribution scaled to [-1, 1]. """
        feats = self.shared(emb)
        raw_alpha = self.alpha_head(feats)
        raw_beta = self.beta_head(feats)

        alpha = self.min_concentration + F.softplus(raw_alpha)
        beta = self.min_concentration + F.softplus(raw_beta)

        base_dist = Beta(alpha, beta)
        # loc=-1, scale=2 maps (0,1) to (-1,1)
        transform = AffineTransform(loc=-1.0, scale=2.0, cache_size=1)
        scaled_beta_dist = TransformedDistribution(base_dist, transform)
        return scaled_beta_dist

    def forward(self, x: torch.Tensor):
        """ Computes alpha and beta parameters. """
        feats = self.shared(x)
        raw_alpha = self.alpha_head(feats)
        raw_beta = self.beta_head(feats)
        alpha = self.min_concentration + F.softplus(raw_alpha)
        beta = self.min_concentration + F.softplus(raw_beta)
        return alpha, beta

    def get_actions(self, emb: torch.Tensor, eval: bool = False):
        """ Samples actions or returns mean, computes log_probs and entropy. """
        dist = self._get_transformed_distribution(emb)
        actions = dist.mean if eval else dist.rsample()

        # --- Calculate Log Probability ---
        # TransformedDistribution correctly handles the Jacobian for log_prob
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # --- Calculate Entropy (Manual Workaround) ---
        # H[Transformed] = H[Base] + log|scale|
        base_entropy = dist.base_dist.entropy() # Entropy of Beta(alpha, beta)
        # Ensure log_scale is on the same device as base_entropy
        log_scale_term = self.log_scale.to(base_entropy.device)
        # Add log|scale| for each action dimension
        entropies = (base_entropy + log_scale_term).sum(dim=-1)

        return actions, log_probs, entropies

    def recompute_log_probs(self, emb: torch.Tensor, stored_actions: torch.Tensor):
        """ Recomputes log_probs and entropy for given actions and current policy. """
        dist = self._get_transformed_distribution(emb)

        # Clamp stored actions slightly for log_prob stability as Beta is undefined at boundaries 0, 1
        # (corresponding to -1, 1 in the transformed space)
        stored_actions_clamped = stored_actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # --- Calculate Log Probability ---
        log_probs = dist.log_prob(stored_actions_clamped).sum(dim=-1)

        # --- Calculate Entropy (Manual Workaround) ---
        # H[Transformed] = H[Base] + log|scale|
        base_entropy = dist.base_dist.entropy()
        log_scale_term = self.log_scale.to(base_entropy.device)
        entropies = (base_entropy + log_scale_term).sum(dim=-1)

        return log_probs, entropies

class FeedForwardBlock(nn.Module):
    """Transformer FFN sublayer: LN -> Linear -> GELU -> Dropout -> Linear -> Dropout + Residual"""
    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            # Default hidden dimension is often 4x embedding dimension
            hidden_dim = 4 * embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        # Dropout applied after activation and after second linear layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.norm(x)      # Pre-LN
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out) # Dropout after activation
        out = self.linear2(out)
        out = self.dropout2(out) # Dropout after second linear layer
        return residual + out    # Add residual connection


class SharedCritic(nn.Module):
    """
    Shared critic architecture.
    MODIFIED to integrate IQN and Distillation networks for value and baseline.
    It now expects its main config (net_cfg) to contain sub-configs for
    iqn_params and distillation_params, and feature_dim for IQNs.
    """
    def __init__(self, net_cfg: Dict[str, Any]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract IQN and Distillation specific configurations
        # It's good practice to provide default empty dicts if keys might be missing
        iqn_config = net_cfg.get("iqn_params", {})
        distillation_config = net_cfg.get("distillation_params", {})

        # The feature_dim for IQN should be the dimension of embeddings
        # that are fed into it (e.g., output of attention or input to ValueNetwork).
        # This should be specified in the 'critic_network' part of the main JSON config.
        feature_dim = net_cfg.get("feature_dim_for_iqn", 128) # Default if not specified

        # Original PPO-style value and baseline heads
        self.value_head_ppo = ValueNetwork(**net_cfg["value_head"])
        self.baseline_head_ppo = ValueNetwork(**net_cfg["baseline_head"])

        # Attention mechanisms
        self.value_attention = ResidualAttention(**net_cfg["value_attention"])
        self.baseline_attention = ResidualAttention(**net_cfg['baseline_attention'])

        # IQN Networks
        self.value_iqn_net = ImplicitQuantileNetwork(
            feature_dim=feature_dim,
            iqn_config=iqn_config,
            linear_init_scale=net_cfg.get("linear_init_scale", 0.1), # Pass general init scale
            dropout_rate=iqn_config.get("dropout_rate", 0.0) # Pass dropout if specified for IQN
        )
        self.baseline_iqn_net = ImplicitQuantileNetwork(
            feature_dim=feature_dim,
            iqn_config=iqn_config,
            linear_init_scale=net_cfg.get("linear_init_scale", 0.1),
            dropout_rate=iqn_config.get("dropout_rate", 0.0)
        )

        # Distillation Networks
        num_quantiles = iqn_config.get("num_quantiles", 32)
        self.value_distill_net = DistillationNetwork(
            num_quantiles=num_quantiles,
            distillation_config=distillation_config,
            linear_init_scale=net_cfg.get("linear_init_scale", 0.1),
            dropout_rate=distillation_config.get("dropout_rate", 0.0)
        )
        self.baseline_distill_net = DistillationNetwork(
            num_quantiles=num_quantiles,
            distillation_config=distillation_config,
            linear_init_scale=net_cfg.get("linear_init_scale", 0.1),
            dropout_rate=distillation_config.get("dropout_rate", 0.0)
        )

        init_scale = net_cfg.get("linear_init_scale", 0.1)
        # Apply initialization to attention layers if they don't do it internally
        # Assuming ValueNetwork, ImplicitQuantileNetwork, DistillationNetwork handle their own init.
        self.value_attention.apply(lambda m: init_weights_gelu_linear(m, scale=init_scale) if isinstance(m, nn.Linear) and not isinstance(m, nn.LayerNorm) else None)
        self.baseline_attention.apply(lambda m: init_weights_gelu_linear(m, scale=init_scale) if isinstance(m, nn.Linear) and not isinstance(m, nn.LayerNorm) else None)

    def values(self, x: torch.Tensor, taus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the centralized state value V(s) using IQN and Distillation.
        Input x shape: (..., A, H_input_to_attention)
        taus shape: (B_flat, num_quantiles_for_IQN_input, 1) or (B_flat * num_quantiles, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - v_distilled: Distilled scalar state value. Shape: (..., 1)
                - value_quantiles: Quantile values from IQN. Shape: (B_flat, num_quantiles)
        """
        input_shape = x.shape
        # H_attention_out is the output dim of attention, which is feature_dim for IQN
        H_attention_out = self.value_attention.embed_dim
        A = input_shape[-2]
        leading_dims = input_shape[:-2]
        B_flat = int(np.prod(leading_dims))

        x_flat = x.reshape(B_flat, A, x.shape[-1]) # x.shape[-1] is input dim to attention
        attn_out, _ = self.value_attention(x_flat)  # (B_flat, A, H_attention_out)
        aggregated_emb = attn_out.mean(dim=1)      # (B_flat, H_attention_out)

        v_ppo = self.value_head_ppo(aggregated_emb) # (B_flat, 1)
        value_quantiles = self.value_iqn_net(aggregated_emb, taus) # (B_flat, num_quantiles)
        v_distilled = self.value_distill_net(v_ppo, value_quantiles) # (B_flat, 1)

        v_distilled_reshaped = v_distilled.view(*leading_dims, 1)
        return v_distilled_reshaped, value_quantiles

    def baselines(self, x: torch.Tensor, taus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the per-agent counterfactual baseline B(s, a_j) using IQN and Distillation.
        Input x shape: (..., A, SeqLen, H_input_to_attention)
        taus shape: (B_flat*A, num_quantiles_for_IQN_input, 1) or (B_flat*A * num_quantiles, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - b_distilled: Distilled scalar baseline values. Shape: (..., A, 1)
                - baseline_quantiles: Quantile values from IQN. Shape: (B_flat*A, num_quantiles)
        """
        input_shape = x.shape
        H_attention_out = self.baseline_attention.embed_dim
        SeqLen = input_shape[-2]
        A = input_shape[-3]
        leading_dims = input_shape[:-3]
        B_flat = int(np.prod(leading_dims))

        x_flat_for_attn = x.reshape(B_flat * A, SeqLen, x.shape[-1])
        x_attn, _ = self.baseline_attention(x_flat_for_attn) # (B_flat*A, SeqLen, H_attention_out)
        agent_emb_for_baseline = x_attn.mean(dim=1)          # (B_flat*A, H_attention_out)

        b_ppo = self.baseline_head_ppo(agent_emb_for_baseline) # (B_flat*A, 1)
        baseline_quantiles = self.baseline_iqn_net(agent_emb_for_baseline, taus) # (B_flat*A, num_quantiles)
        b_distilled = self.baseline_distill_net(b_ppo, baseline_quantiles) # (B_flat*A, 1)

        b_distilled_reshaped = b_distilled.view(*leading_dims, A, 1)
        return b_distilled_reshaped, baseline_quantiles

class AgentIDPosEnc(nn.Module):
    """Sinusoidal positional encoder for integer agent IDs."""
    def __init__(self, num_freqs: int = 8, id_embed_dim: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.id_embed_dim = id_embed_dim
        # Linear layer projects sinusoidal features to the target embedding dimension
        self.linear = nn.Linear(2 * num_freqs, id_embed_dim)
        # Initialize linear layer weights and biases
        init.kaiming_normal_(self.linear.weight, a=0.01) # Kaiming init often used with ReLU/Leaky ReLU
        if self.linear.bias is not None: nn.init.constant_(self.linear.bias, 0.0)

    def sinusoidal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Generates sinusoidal features for input tensor x."""
        x_float = x.float().unsqueeze(-1) # Ensure float and add feature dim: (..., 1)
        # Create frequency bands (powers of 2)
        freq_exponents = torch.arange(self.num_freqs, device=x.device, dtype=torch.float32)
        scales = torch.pow(2.0, freq_exponents).unsqueeze(0) # Shape: (1, num_freqs)
        # Apply frequencies: shape (..., num_freqs)
        multiplied = x_float * scales
        # Compute sine and cosine features
        sines = torch.sin(multiplied)
        cosines = torch.cos(multiplied)
        # Concatenate sine and cosine features: shape (..., 2 * num_freqs)
        return torch.cat([sines, cosines], dim=-1)

    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """Encodes agent IDs into embeddings."""
        shape_in = agent_ids.shape
        # Generate sinusoidal features for flattened IDs
        feats = self.sinusoidal_features(agent_ids.reshape(-1)) # Shape: (Batch*..., 2*num_freqs)
        # Project features to target dimension
        out_lin = self.linear(feats) # Shape: (Batch*..., id_embed_dim)
        # Reshape back to original shape + embedding dimension
        return out_lin.view(*shape_in, self.id_embed_dim)


class MultiAgentEmbeddingNetwork(nn.Module):
    """
    Top-level network combining the CrossAttentionFeatureExtractor
    with specific encoders for state-only and state-action embeddings,
    primarily used for baseline calculation.
    """
    def __init__(self, net_cfg: Dict[str, Any]):
        super(MultiAgentEmbeddingNetwork, self).__init__()
        # Initialize sub-modules using configurations passed via spread operator
        # base_encoder handles the main processing of agent and central states
        self.base_encoder = CrossAttentionFeatureExtractor(**net_cfg["cross_attention_feature_extractor"])
        # f encodes state-action pairs (used for groupmates in baseline)
        self.f = StatesActionsEncoder(**net_cfg["obs_actions_encoder"])
        # g encodes state-only (used for the agent itself in baseline)
        self.g = StatesEncoder(**net_cfg["obs_encoder"])

    def get_base_embedding(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Computes the primary agent embeddings using the base_encoder.
        These embeddings are used for the policy and state value calculation.
        """
        return self.base_encoder(obs_dict)

    def get_baseline_embeddings(self, common_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Constructs the specific input required for the counterfactual baseline network.
        For each agent 'j', it combines its state embedding (from self.g) with the
        state-action embeddings of all other agents (from self.f).

        Args:
            common_emb: Base agent embeddings (output of get_base_embedding),
                        shape (..., A, H).
            actions: Actions taken by agents, shape (..., A, ActionDim).

        Returns:
            Tensor: Structured embeddings for baseline input,
                    shape (..., A, SeqLen, H), where SeqLen is typically A.
        """
        input_shape = common_emb.shape
        H = input_shape[-1]; A = input_shape[-2]; leading_dims = input_shape[:-2]
        B = int(np.prod(leading_dims))

        # Handle action shape and dimension determination
        act_dim = actions.shape[-1] if actions.dim() > len(leading_dims) + 1 else 1
        # Add trailing dimension if actions tensor is missing it (e.g., discrete actions)
        if actions.dim() == len(leading_dims) + 1: actions = actions.unsqueeze(-1)

        # Reshape for batch processing by encoders f and g
        common_emb_flat = common_emb.reshape(B * A, H)
        actions_flat = actions.reshape(B * A, act_dim)

        # Encode all agent observations (state-only) using encoder g
        all_agent_obs_emb_flat = self.g(common_emb_flat)
        # Encode all agent state-action pairs using encoder f
        all_agent_obs_act_emb_flat = self.f(common_emb_flat, actions_flat)

        OutputH = all_agent_obs_emb_flat.shape[-1] # Assume g and f output same dim H

        # Reshape back to (B, A, H)
        all_agent_obs_emb = all_agent_obs_emb_flat.view(B, A, OutputH)
        all_agent_obs_act_emb = all_agent_obs_act_emb_flat.view(B, A, OutputH)

        # Construct the input sequence for each agent j's baseline calculation
        baseline_input_list = []
        for j in range(A):
            # Agent j's own state embedding (shape: B, 1, H)
            agent_j_obs_emb = all_agent_obs_emb[:, j:j+1, :]

            # Groupmates' state-action embeddings (shape: B, A-1, H or B, 0, H)
            groupmates_obs_act_emb = torch.cat([
                all_agent_obs_act_emb[:, i:i+1, :] for i in range(A) if i != j
            ], dim=1) if A > 1 else torch.empty(B, 0, OutputH, device=common_emb.device)

            # Create the sequence: [agent_j_obs, gm_1_obs_act, ..., gm_A-1_obs_act]
            # Shape: (B, A, H) if A > 1, or (B, 1, H) if A = 1
            baseline_j_input = torch.cat([agent_j_obs_emb, groupmates_obs_act_emb], dim=1)

            # Add an agent dimension (dim=1) for later concatenation
            baseline_input_list.append(baseline_j_input.unsqueeze(1)) # -> (B, 1, SeqLen, H)

        # Concatenate along the agent dimension (dim=1)
        final_baseline_input = torch.cat(baseline_input_list, dim=1) # -> (B, A, SeqLen, H)

        # Reshape back to original leading dimensions
        output_shape = leading_dims + final_baseline_input.shape[1:]
        return final_baseline_input.view(output_shape).contiguous()

    def get_state_embeddings(self, common_emb: torch.Tensor) -> torch.Tensor:
        """Helper to get state-only embeddings using encoder g."""
        input_shape = common_emb.shape
        H = input_shape[-1]; A = input_shape[-2]; leading_dims = input_shape[:-2]
        B = int(np.prod(leading_dims))
        emb_flat = self.g(common_emb.reshape(B*A, H))
        OutputH = emb_flat.shape[-1]
        return emb_flat.view(*leading_dims, A, OutputH)

    def get_state_action_embeddings(self, common_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Helper to get state-action embeddings using encoder f."""
        emb_shape = common_emb.shape; act_shape = actions.shape
        H = emb_shape[-1]; A = emb_shape[-2]; leading_dims = emb_shape[:-2]
        act_dim = actions.shape[-1] if actions.dim() > len(leading_dims) + 1 else 1
        if actions.dim() == len(leading_dims) + 1: actions = actions.unsqueeze(-1)

        B = int(np.prod(leading_dims))
        common_emb_flat = common_emb.reshape(B*A, H)
        actions_flat = actions.reshape(B*A, act_dim)
        encoded_flat = self.f(common_emb_flat, actions_flat)
        OutputH = encoded_flat.shape[-1]
        output_shape = leading_dims + (A, OutputH,)
        return encoded_flat.view(output_shape)

    def _split_agent_and_groupmates(self, obs_4d: torch.Tensor, actions: torch.Tensor):
        # This method seems unused and can likely be removed or implemented if needed later.
        pass


class CrossAttentionFeatureExtractor(nn.Module):
    """
    Processes central state (CNN) and agent states (MLP), then integrates them
    using multi-scale cross-attention followed by agent self-attention.
    """
    def __init__(
        self,
        agent_obs_size: int,
        num_agents: int, # Max agents for ID embedding
        in_channels: int, # Central state channels (e.g., height, delta, R, G, B)
        h: int,           # Central state height
        w: int,           # Central state width
        embed_dim: int,   # Dimension for embeddings and attention
        num_heads: int,   # Number of attention heads
        cnn_channels: list, # List of output channels for each CNN block
        cnn_kernel_sizes: list, # List of kernel sizes for each CNN block
        cnn_strides: list,      # List of strides for each CNN block
        group_norms: list,      # List of group numbers for GroupNorm in each CNN block
        block_scales: list,     # Target spatial dimensions (H/W) after pooling for each CNN block's output
        transformer_layers: int,# Number of transformer layers (CrossAttn + SelfAttn)
        dropout_rate: float,    # Dropout rate for attention and FFN blocks
        use_agent_id: bool,     # Whether to use agent ID embeddings
        ff_hidden_factor: int,  # Factor to scale embed_dim for FFN hidden size
        id_num_freqs: int,      # Number of frequencies for sinusoidal agent ID encoding
        conv_init_scale: float, # Initialization scale for CNN layers
        linear_init_scale: float # Initialization scale for Linear layers
    ):
        super().__init__()
        # Store configuration parameters
        self.in_channels = in_channels; self.h = h; self.w = w; self.embed_dim = embed_dim
        self.num_heads = num_heads; self.transformer_layers = transformer_layers
        self.ff_hidden_factor = ff_hidden_factor; self.use_agent_id = use_agent_id
        self.block_scales = block_scales; self.num_scales = len(block_scales)

        # Validate CNN configuration list lengths
        if not (len(cnn_channels) == self.num_scales == len(cnn_kernel_sizes) == len(cnn_strides) == len(group_norms)):
             raise ValueError("CNN config lists (channels, kernels, strides, group_norms) must have the same length as block_scales.")

        # --- Sub-module Definitions ---

        # A) Agent MLP Encoder: Encodes individual agent observations
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_obs_size, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )

        # B) Optional Agent ID Embedding: Adds unique ID info if enabled
        if use_agent_id:
            self.agent_id_embedding = nn.Embedding(num_agents, embed_dim)
            nn.init.normal_(self.agent_id_embedding.weight, mean=0.0, std=0.01) # Small random init
            self.agent_id_pos_enc = AgentIDPosEnc(num_freqs=id_num_freqs, id_embed_dim=embed_dim)
            self.agent_id_sum_ln = nn.LayerNorm(embed_dim) # Normalize after adding ID embeddings
        else:
            self.agent_id_embedding = None; self.agent_id_pos_enc = None

        # C) CNN Backbone: Extracts features from the central state (image-like)
        total_in_c = self.in_channels + 2 # Add 2 channels for spatial coordinates
        self.cnn_blocks = nn.ModuleList()
        prev_c = total_in_c
        for i, out_c in enumerate(cnn_channels):
             # Ensure GroupNorm groups divide channels
             if out_c % group_norms[i] != 0: raise ValueError(f"group_norms[{i}] must divide cnn_channels[{i}]")
             # Define a CNN block: Conv -> GroupNorm -> GELU -> Conv -> GroupNorm -> GELU
             block = nn.Sequential(
                 nn.Conv2d(prev_c, out_c, kernel_size=cnn_kernel_sizes[i], stride=cnn_strides[i], padding=cnn_kernel_sizes[i]//2),
                 nn.GroupNorm(group_norms[i], out_c), nn.GELU(),
                 nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1), # Additional conv layer
                 nn.GroupNorm(group_norms[i], out_c), nn.GELU()
             )
             self.cnn_blocks.append(block); prev_c = out_c # Update input channels for next block

        # D) CNN Output Embeddings & LayerNorms: Project CNN features to embed_dim for each scale
        self.block_embeds = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in cnn_channels])
        self.block_lns = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in cnn_channels]) # LN applied on feature dim

        # E) Positional Embeddings: Precomputed 2D sinusoidal embeddings for patch positions
        self.positional_embeddings = nn.ParameterList(
            [nn.Parameter(create_2d_sin_cos_pos_emb(s, s, embed_dim, torch.device("cpu")), requires_grad=False) for s in block_scales]
        )

        # F) Transformer Layers: Stacked layers of cross-attention and self-attention
        self.cross_attn_blocks = nn.ModuleList() # Holds cross-attention modules for each layer/scale
        self.agent_self_attn = nn.ModuleList()   # Holds self-attention modules for each layer
        self.agent_self_ffn  = nn.ModuleList()   # Holds FFN modules for self-attention path
        ff_hidden_dim = embed_dim * ff_hidden_factor # Calculate hidden dim for FFNs
        for _ in range(transformer_layers):
            # Create cross-attention blocks for each scale within this transformer layer
            blocks_per_scale = nn.ModuleList()
            for _ in range(self.num_scales):
                 blocks_per_scale.append(nn.ModuleDict({
                     "attn": ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=False),
                     "ffn": FeedForwardBlock(embed_dim, hidden_dim=ff_hidden_dim, dropout=dropout_rate)
                 }))
            self.cross_attn_blocks.append(blocks_per_scale)
            # Create self-attention block for agents within this transformer layer
            self.agent_self_attn.append(ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=True))
            self.agent_self_ffn.append(FeedForwardBlock(embed_dim, hidden_dim=ff_hidden_dim, dropout=dropout_rate))

        # G) Final Layers: Apply LayerNorm and an MLP after the transformer stack
        self.post_cross_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # H) Initialization: Apply specific initializations to CNN and Linear layers
        self.cnn_blocks.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale))
        self.block_embeds.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale))
        # Initialize Linear layers, skipping Norm layers
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale)
                   if isinstance(m, nn.Linear) and not isinstance(m, (nn.LayerNorm, nn.GroupNorm))
                   else None)


    def forward(self, state_dict: Dict[str, torch.Tensor]):
        """
        Forward pass through the embedding network.

        Args:
            state_dict (Dict[str, torch.Tensor]): Dictionary containing:
                'agent': Agent observations (..., A, agent_obs_size).
                'central': Central state observations (..., flat_central_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - agent_embed: Final agent embeddings (..., A, embed_dim).
                - attn_weights_final: Attention weights from the last self-attention layer (..., A, A).
        """
        agent_state = state_dict["agent"]
        central_data = state_dict["central"]
        device = agent_state.device

        # Ensure positional embeddings are on the correct device
        if self.positional_embeddings[0].device != device:
            for i in range(len(self.positional_embeddings)):
                self.positional_embeddings[i] = nn.Parameter(self.positional_embeddings[i].to(device), requires_grad=False)

        # --- Shape Determination ---
        agent_shape = agent_state.shape
        leading_dims = agent_shape[:-2]; A = agent_shape[-2]; obs_dim = agent_shape[-1]
        B = int(np.prod(leading_dims)) # Flatten batch dimensions

        # --- 1. Agent Encoding ---
        flat_agents = agent_state.view(B * A, obs_dim)
        agent_embed = self.agent_encoder(flat_agents)
        agent_embed = agent_embed.view(B, A, self.embed_dim) # Reshape back: (B, A, H)

        # --- 2. Add Agent ID Embeddings (Optional) ---
        if self.use_agent_id:
            # Create agent IDs [0, 1, ..., A-1] for each batch element
            agent_ids = torch.arange(A, device=device).view(1, A).expand(B, A)
            agent_embed = agent_embed + self.agent_id_embedding(agent_ids) + self.agent_id_pos_enc(agent_ids)
            agent_embed = self.agent_id_sum_ln(agent_embed) # Normalize sum

        # --- 3. Central State Preparation ---
        expected_central_flat_dim = self.in_channels * self.h * self.w
        central_data = central_data.view(B, expected_central_flat_dim) # Flatten batch dims
        central_data = central_data.view(B, self.in_channels, self.h, self.w) # Reshape to CxHxW
        # Create and concatenate coordinate channels
        row_coords = torch.linspace(-1, 1, steps=self.h, device=device).view(1, 1, self.h, 1).expand(B, 1, self.h, self.w)
        col_coords = torch.linspace(-1, 1, steps=self.w, device=device).view(1, 1, 1, self.w).expand(B, 1, self.h, self.w)
        cnn_input = torch.cat([central_data, row_coords, col_coords], dim=1) # Shape: (B, C+2, H, W)

        # --- 4. CNN Feature Extraction ---
        cnn_feature_maps = []; current_map = cnn_input
        for block in self.cnn_blocks:
            current_map = block(current_map)
            cnn_feature_maps.append(current_map) # Store output of each block

        # --- 5. Multi-Scale Patch Processing ---
        patch_sequences = []
        for i, feature_map in enumerate(cnn_feature_maps):
            target_h, target_w = self.block_scales[i], self.block_scales[i]
            # Pool features to target scale
            pooled_map = F.adaptive_avg_pool2d(feature_map, (target_h, target_w)) # (B, C_out, target_h, target_w)
            # Project features to embedding dimension
            embedded_map = self.block_embeds[i](pooled_map) # (B, embed_dim, target_h, target_w)
            # Flatten spatial dims and transpose: (B, N_patches, embed_dim)
            B_, D_, H_, W_ = embedded_map.shape
            patches = embedded_map.view(B_, D_, H_*W_).transpose(1, 2).contiguous()
            # Apply LayerNorm
            patches = self.block_lns[i](patches)
            # Add positional embeddings
            pos_embedding = self.positional_embeddings[i].unsqueeze(0) # (1, N_patches, embed_dim)
            patches = patches + pos_embedding
            patch_sequences.append(patches)

        # --- 6. Transformer Layers ---
        attn_weights_final = None
        for layer_idx in range(self.transformer_layers):
            # Cross-Attention: Agents attend to each scale's patches
            scale_attn_blocks = self.cross_attn_blocks[layer_idx]
            for scale_idx, block_module_dict in enumerate(scale_attn_blocks):
                agent_embed, _ = block_module_dict["attn"](agent_embed, patch_sequences[scale_idx], patch_sequences[scale_idx])
                agent_embed = block_module_dict["ffn"](agent_embed)

            # Self-Attention: Agents attend to each other
            agent_embed, attn_weights = self.agent_self_attn[layer_idx](agent_embed)
            # Store attention weights from the last layer
            if layer_idx == self.transformer_layers - 1:
                attn_weights_final = attn_weights # Shape: (B, A, A)
            # Apply FFN after self-attention
            agent_embed = self.agent_self_ffn[layer_idx](agent_embed)

        # --- 7. Final Processing ---
        residual = agent_embed # If you want a residual around the post_cross_mlp
        agent_embed = self.post_cross_mlp(agent_embed)
        agent_embed = agent_embed + residual

        # --- Reshape Output ---
        # Reshape agent embeddings back to original leading dimensions
        output_shape = leading_dims + (A, self.embed_dim)
        agent_embed = agent_embed.view(output_shape)
        # Reshape attention weights if available
        if attn_weights_final is not None:
             attn_output_shape = leading_dims + (A, A)
             attn_weights_final = attn_weights_final.view(attn_output_shape)

        return agent_embed, attn_weights_final
    

class RNDTargetNetwork(nn.Module):
    """
    Random Network Distillation (RND) Target Network.
    Its weights are fixed after random initialization.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        # Initialize weights (e.g., LeakyReLU-appropriate Kaiming normal)
        # The specific initialization can be tuned.
        self.apply(lambda m: init_weights_leaky_relu(m, negative_slope=0.01) if isinstance(m, nn.Linear) else None)

        # Target network weights are fixed after random initialization
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RND target network.
        Args:
            x (torch.Tensor): Input tensor (e.g., state/feature embeddings).
        Returns:
            torch.Tensor: Output random features.
        """
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """
    Random Network Distillation (RND) Predictor Network.
    This network is trained to predict the output of the RNDTargetNetwork.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        # Initialize weights (e.g., LeakyReLU-appropriate Kaiming normal)
        # The specific initialization can be tuned.
        self.apply(lambda m: init_weights_leaky_relu(m, negative_slope=0.01) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RND predictor network.
        Args:
            x (torch.Tensor): Input tensor (e.g., state/feature embeddings).
        Returns:
            torch.Tensor: Predicted random features.
        """
        return self.net(x)
    

class ImplicitQuantileNetwork(nn.Module):
    """
    Implicit Quantile Network (IQN) for distributional reinforcement learning.
    It predicts the quantile values of a state (or state-action) value distribution.
    """
    def __init__(self,
                 feature_dim: int,        # Dimension of the input state/feature embeddings
                 iqn_config: Dict[str, Any], # Dictionary containing IQN specific parameters
                 linear_init_scale: float = 1.0,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.iqn_config = iqn_config

        self.num_quantiles = iqn_config.get("num_quantiles", 32)
        self.quantile_embedding_dim = iqn_config.get("quantile_embedding_dim", 64)
        self.hidden_size = iqn_config.get("hidden_size", 128) # Hidden size for IQN's internal MLP

        # Cosine embedding layer for quantile fractions (τ)
        # This projects τ into a higher dimensional space using cosine basis functions.
        # See equation (4) in the IQN paper (Dabney et al., 2018).
        self.cosine_embedding_layer = nn.Linear(self.quantile_embedding_dim, self.feature_dim)
        # Pre-calculate i * pi for cosine embeddings for efficiency
        # self.i_pi = nn.Parameter(torch.arange(1, self.quantile_embedding_dim + 1, dtype=torch.float32) * math.pi, requires_grad=False)
        # A common way is to pre-calculate a range of i:
        self.register_buffer('i_pi_values', torch.arange(0, self.quantile_embedding_dim, dtype=torch.float32) * math.pi)


        # MLP to process the merged state and quantile embeddings
        self.merge_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_size),
            nn.GELU(), # Or GELU, consistent with your other networks
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

        # Output layer that predicts the quantile values
        self.output_layer = nn.Linear(self.hidden_size, 1) # Outputs a single value per quantile sample

        # Initialize weights
        # Cosine embedding layer might have specific init, but Linear with GELU-like init is a reasonable start
        init_weights_gelu_linear(self.cosine_embedding_layer, scale=linear_init_scale)
        self.merge_mlp.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)
        init_weights_gelu_linear(self.output_layer, scale=linear_init_scale)


    def embed_quantiles(self, taus: torch.Tensor) -> torch.Tensor:
        """
        Embeds quantile fractions τ using cosine basis functions.
        Args:
            taus (torch.Tensor): Tensor of quantile fractions τ, shape (BatchSize, NumQuantilesToEmbed, 1)
                                 or (BatchSize * NumQuantilesToEmbed, 1).
                                 The τ values should be in [0, 1].
        Returns:
            torch.Tensor: Embedded quantile fractions, shape (BatchSize, NumQuantilesToEmbed, feature_dim)
                          or (BatchSize * NumQuantilesToEmbed, feature_dim).
        """
        # Ensure taus has a trailing dimension if it's flat
        if taus.dim() == 1: # (BatchSize * NumQuantilesToEmbed)
            taus_unsqueezed = taus.unsqueeze(-1)
        elif taus.dim() == 2: # (BatchSize, NumQuantilesToEmbed)
            taus_unsqueezed = taus.unsqueeze(-1) # -> (BatchSize, NumQuantilesToEmbed, 1)
        else:
            taus_unsqueezed = taus

        # Cosine embedding: cos(i * pi * τ)
        # taus_unsqueezed shape: (..., 1)
        # self.i_pi_values shape: (quantile_embedding_dim,)
        # Broadcasting taus_unsqueezed with self.i_pi_values
        # (..., 1) * (quantile_embedding_dim,) -> (..., quantile_embedding_dim)
        cosine_features = torch.cos(taus_unsqueezed * self.i_pi_values.view(1, -1).to(taus_unsqueezed.device))
        
        # Project cosine features to feature_dim
        quantile_embeddings = F.gelu(self.cosine_embedding_layer(cosine_features)) # (..., feature_dim)
        return quantile_embeddings

    def forward(self, state_features: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """
        Predicts quantile values for the given state features and quantile fractions.

        Args:
            state_features (torch.Tensor): State or feature embeddings from the main network.
                                           Shape: (BatchSize, feature_dim).
            taus (torch.Tensor): Quantile fractions τ to sample.
                                 Shape: (BatchSize, num_quantiles_to_sample_for_this_forward_pass, 1)
                                 or (BatchSize * num_quantiles_to_sample_for_this_forward_pass, 1).
                                 The `num_quantiles_to_sample_for_this_forward_pass` is typically `self.num_quantiles`
                                 during training for loss calculation, or a different number (e.g., for evaluation if needed).

        Returns:
            torch.Tensor: Predicted quantile values. Shape: (BatchSize, num_quantiles_to_sample_for_this_forward_pass).
        """
        batch_size = state_features.shape[0]
        
        # Embed quantile fractions
        # taus might come in as (B, N_taus, 1) or (B * N_taus, 1)
        # If taus is (B * N_taus, 1), state_features needs to be repeated
        # If taus is (B, N_taus, 1), state_features needs to be unsqueezed and repeated

        if taus.dim() == 2 and taus.shape[0] == batch_size * self.num_quantiles: # (B * N_taus, 1)
            num_sampled_taus_per_state = self.num_quantiles
            state_features_expanded = state_features.repeat_interleave(num_sampled_taus_per_state, dim=0) # (B * N_taus, feature_dim)
            quantile_embeddings = self.embed_quantiles(taus) # (B * N_taus, feature_dim)
        elif taus.dim() == 3 and taus.shape[0] == batch_size: # (B, N_taus, 1)
            num_sampled_taus_per_state = taus.shape[1]
            state_features_expanded = state_features.unsqueeze(1).expand(-1, num_sampled_taus_per_state, -1) # (B, N_taus, feature_dim)
            quantile_embeddings = self.embed_quantiles(taus) # (B, N_taus, feature_dim)
            # Reshape for element-wise product and MLP
            state_features_expanded = state_features_expanded.reshape(-1, self.feature_dim)
            quantile_embeddings = quantile_embeddings.reshape(-1, self.feature_dim)
        else:
            raise ValueError(f"Unsupported taus shape: {taus.shape} for state_features shape: {state_features.shape}")

        # Element-wise product of state features and quantile embeddings
        merged_features = state_features_expanded * quantile_embeddings # (B * N_taus, feature_dim)

        # Pass through MLP
        x = self.merge_mlp(merged_features) # (B * N_taus, hidden_size)
        quantile_values = self.output_layer(x)   # (B * N_taus, 1)

        # Reshape to (BatchSize, num_sampled_taus_per_state)
        quantile_values = quantile_values.view(batch_size, num_sampled_taus_per_state)

        return quantile_values
    

class DistillationNetwork(nn.Module):
    """
    Distills information from a scalar PPO-style value/baseline (V_ppo)
    and a set of quantile values (Z_tau) from an IQN into a single scalar output.
    """
    def __init__(self,
                 num_quantiles: int,          # Number of quantile values from IQN (e.g., 32, 64)
                 distillation_config: Dict[str, Any], # Config for this network
                 linear_init_scale: float = 1.0,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.distillation_config = distillation_config
        self.hidden_size = distillation_config.get("hidden_size", 128)
        self.combination_method = distillation_config.get("combination_method", "hadamard_product") # 'hadamard_product' or 'concatenate'

        # The input dimension to the first linear layer depends on the combination method
        if self.combination_method == "hadamard_product":
            # V_ppo (scalar) is expanded and multiplied element-wise with IQN quantiles (vector)
            # So, the input features to the MLP will be num_quantiles
            self.input_dim_to_mlp = self.num_quantiles
        elif self.combination_method == "concatenate":
            # V_ppo (scalar, becomes 1 feature) is concatenated with IQN quantiles (num_quantiles features)
            self.input_dim_to_mlp = 1 + self.num_quantiles
        elif self.combination_method == "average_iqn_concatenate":
            # V_ppo (scalar) is concatenated with the mean of IQN quantiles (scalar)
            self.input_dim_to_mlp = 1 + 1
        else:
            raise ValueError(f"Unsupported combination_method: {self.combination_method}")

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim_to_mlp, self.hidden_size),
            nn.GELU(), # Or GELU
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(), # Or GELU
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, 1) # Outputs a single distilled scalar value
        )

        # Initialize weights
        self.mlp.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

    def forward(self, v_ppo: torch.Tensor, quantile_values: torch.Tensor) -> torch.Tensor:
        """
        Combines V_ppo and quantile_values and processes them to output a distilled scalar value.

        Args:
            v_ppo (torch.Tensor): Scalar value output from the original PPO-style head.
                                  Shape: (BatchSize, 1).
            quantile_values (torch.Tensor): Quantile values from the IQN.
                                            Shape: (BatchSize, num_quantiles).

        Returns:
            torch.Tensor: The distilled scalar value. Shape: (BatchSize, 1).
        """
        batch_size = v_ppo.shape[0]
        if v_ppo.shape != (batch_size, 1):
            raise ValueError(f"Expected v_ppo shape ({batch_size}, 1), got {v_ppo.shape}")
        if quantile_values.shape != (batch_size, self.num_quantiles):
            raise ValueError(f"Expected quantile_values shape ({batch_size}, {self.num_quantiles}), got {quantile_values.shape}")

        if self.combination_method == "hadamard_product":
            # Expand v_ppo to match the shape of quantile_values for element-wise product
            v_ppo_expanded = v_ppo.expand_as(quantile_values) # (BatchSize, num_quantiles)
            combined_input = v_ppo_expanded * quantile_values  # (BatchSize, num_quantiles)
        elif self.combination_method == "concatenate":
            combined_input = torch.cat([v_ppo, quantile_values], dim=-1) # (BatchSize, 1 + num_quantiles)
        elif self.combination_method == "average_iqn_concatenate":
            mean_quantile_value = quantile_values.mean(dim=1, keepdim=True) # (BatchSize, 1)
            combined_input = torch.cat([v_ppo, mean_quantile_value], dim=-1) # (BatchSize, 2)
        else:
            # This case should have been caught in __init__
            raise ValueError(f"Unsupported combination_method: {self.combination_method}")

        distilled_value = self.mlp(combined_input) # (BatchSize, 1)
        return distilled_value
