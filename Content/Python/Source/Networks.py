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
from .Utility import (
    init_weights_leaky_relu, 
    init_weights_gelu_linear, 
    init_weights_gelu_conv, 
    create_2d_sin_cos_pos_emb
)

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

        # Shared MLP backbone with LayerNorm and Dropout
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

        # Head for predicting raw alpha parameters
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features)
        )

        # Head for predicting raw beta parameters
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_features)
        )

        # Initialize linear layers
        self.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale))

        # Apply specific bias initialization to the output layers for alpha/beta params
        if hasattr(self.alpha_head[-1], 'bias'):
             nn.init.constant_(self.alpha_head[-1].bias, param_init_bias)
        if hasattr(self.beta_head[-1], 'bias'):
             nn.init.constant_(self.beta_head[-1].bias, param_init_bias)

        # Precompute log(scale) for entropy calculation
        self.register_buffer('log_scale', torch.log(torch.tensor(2.0))) # From AffineTransform(loc=-1, scale=2)

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
        """Samples actions or returns mean, computes log_probs and entropy."""
        dist = self._get_transformed_distribution(emb)
        if eval:
            base_mean = dist.base_dist.mean
            actions = dist.transforms[0](base_mean)
        else:
            # Beta distribution may lack rsample; use non-reparameterized sample when needed
            actions = dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()

        # --- Calculate Log Probability ---
        # TransformedDistribution correctly handles the Jacobian for log_prob
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # --- Calculate Entropy (Manual Workaround) ---
        # H[Transformed] = H[Base] + log|scale|
        base_entropy = dist.base_dist.entropy() # Entropy of Beta(alpha, beta)
        # Add log|scale| for each action dimension
        entropies = (base_entropy + self.log_scale).sum(dim=-1)

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
        entropies = (base_entropy + self.log_scale).sum(dim=-1)

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


class ContextualImportanceAggregator(nn.Module):
    """Aggregates agent features with learned importance weights.

    The module operates over the agent dimension (dim=1) and supports
    optional masking where ``True`` values denote padded agents that should
    be ignored during aggregation.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        # Initialize linear layers using project helper
        self.apply(lambda m: init_weights_gelu_linear(m) if isinstance(m, nn.Linear) else None)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Apply importance weighting over agents.

        Args:
            x: Tensor of shape ``(B, A, embed_dim)``.
            mask: Optional boolean tensor of shape ``(B, A)`` where ``True``
                marks padded agents to ignore.
            reduce: If ``True`` (default), aggregate across agents and return
                ``(B, embed_dim)``. If ``False``, return the weighted features
                per agent with shape ``(B, A, embed_dim)``.

        Returns:
            Tensor of shape ``(B, embed_dim)`` if ``reduce`` else
            ``(B, A, embed_dim)``.
        """
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        valid = (~mask).unsqueeze(-1)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1)
        global_ctx = (x * valid).sum(dim=1, keepdim=True) / denom
        global_ctx = self.context_encoder(global_ctx).expand(-1, x.shape[1], -1)

        combined = torch.cat([x, global_ctx], dim=-1)
        importance_logits = self.importance_scorer(combined).squeeze(-1)
        importance_logits = importance_logits.masked_fill(mask, float('-inf'))
        importance = F.softmax(importance_logits, dim=1).unsqueeze(-1)

        gate_values = self.gate(combined)
        gate_values = gate_values.masked_fill(mask.unsqueeze(-1), 0.0)
        gated_features = x * gate_values

        weighted = gated_features * importance
        if reduce:
            return weighted.sum(dim=1)
        else:
            return weighted


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
        iqn_config = net_cfg.get("iqn_params", {})

        # Original PPO-style value and baseline heads
        self.value_head_ppo = ValueNetwork(**net_cfg["value_head"])
        self.baseline_head_ppo = ValueNetwork(**net_cfg["baseline_head"])

        # Attention mechanisms
        self.value_attention = ResidualAttention(**net_cfg["value_attention"])
        self.baseline_attention = ResidualAttention(**net_cfg['baseline_attention'])

        # Agent-wise aggregators
        self.value_aggregator = ContextualImportanceAggregator(
            embed_dim=net_cfg["value_attention"]["embed_dim"]
        )
        self.baseline_aggregator = ContextualImportanceAggregator(
            embed_dim=net_cfg["baseline_attention"]["embed_dim"]
        )

        # IQN Networks
        self.value_iqn_net = ImplicitQuantileNetwork(
            **iqn_config.get("iqn_network", {})
        )
        self.baseline_iqn_net = ImplicitQuantileNetwork(
            **iqn_config.get("iqn_network", {})
        )

        # Distillation Networks
        self.value_distill_net = DistillationNetwork(
            **iqn_config.get("distillation_network", {})
        )
        self.baseline_distill_net = DistillationNetwork(
            **iqn_config.get("distillation_network", {})
        )

        init_scale = net_cfg.get("linear_init_scale", 0.1)
        # Apply initialization to attention layers if they don't do it internally
        # Assuming ValueNetwork, ImplicitQuantileNetwork, DistillationNetwork handle their own init.
        self.value_attention.apply(lambda m: init_weights_gelu_linear(m, scale=init_scale) if isinstance(m, nn.Linear) and not isinstance(m, nn.LayerNorm) else None)
        self.baseline_attention.apply(lambda m: init_weights_gelu_linear(m, scale=init_scale) if isinstance(m, nn.Linear) and not isinstance(m, nn.LayerNorm) else None)

    def values(
        self,
        x: torch.Tensor,
        taus: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the centralized state value V(s) using IQN and Distillation.

        Args:
            x: Input tensor of shape ``(..., A, H_input_to_attention)``.
            taus: Quantile fractions for IQN.
            agent_mask: Optional boolean tensor ``(..., A)`` with ``True`` for
                padded agents.

        Returns:
            Tuple containing the distilled scalar value ``(..., 1)`` and
            quantile values ``(B_flat, num_quantiles)``.
        """
        input_shape = x.shape
        H_attention_out = self.value_attention.embed_dim
        A = input_shape[-2]
        leading_dims = input_shape[:-2]
        B_flat = int(np.prod(leading_dims))

        x_flat = x.reshape(B_flat, A, x.shape[-1])
        attn_out, _ = self.value_attention(x_flat)  # (B_flat, A, H_attention_out)

        if agent_mask is not None:
            agent_mask = agent_mask.view(B_flat, A)
        aggregated_emb = self.value_aggregator(attn_out, mask=agent_mask)

        v_ppo = self.value_head_ppo(aggregated_emb)
        value_quantiles = self.value_iqn_net(aggregated_emb, taus)
        v_distilled = self.value_distill_net(v_ppo, value_quantiles)

        v_distilled_reshaped = v_distilled.view(*leading_dims, 1)
        return v_distilled_reshaped, value_quantiles

    def baselines(
        self,
        x: torch.Tensor,
        taus: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the counterfactual baseline B(s, a) using IQN and Distillation.

        Args:
            x: Input tensor of shape ``(..., A, SeqLen, H_input_to_attention)``.
            taus: Quantile fractions for IQN.
            agent_mask: Optional boolean tensor ``(..., A)`` with ``True`` for
                padded agents.

        Returns:
            Tuple containing distilled baseline values ``(..., A, 1)`` and
            quantile values ``(B_flat, A, num_quantiles)``.
        """
        input_shape = x.shape
        H_attention_out = self.baseline_attention.embed_dim
        SeqLen = input_shape[-2]
        A = input_shape[-3]
        leading_dims = input_shape[:-3]
        B_flat = int(np.prod(leading_dims))

        x_flat_for_attn = x.reshape(B_flat * A, SeqLen, x.shape[-1])
        x_attn, _ = self.baseline_attention(x_flat_for_attn)
        per_agent_embs = x_attn.mean(dim=1).view(B_flat, A, H_attention_out)

        if agent_mask is not None:
            agent_mask = agent_mask.view(B_flat, A)

        # Importance weighting over agents without collapsing the agent dimension
        weighted_embs = self.baseline_aggregator(
            per_agent_embs, mask=agent_mask, reduce=False
        )  # (B_flat, A, H_attention_out)

        # Flatten agent dimension for heads
        flat_weighted = weighted_embs.view(B_flat * A, H_attention_out)

        b_ppo = self.baseline_head_ppo(flat_weighted)

        taus_expanded = taus.unsqueeze(1).expand(-1, A, -1, -1)
        taus_expanded = taus_expanded.reshape(B_flat * A, taus.shape[-2], 1)
        baseline_quantiles = self.baseline_iqn_net(flat_weighted, taus_expanded)
        b_distilled = self.baseline_distill_net(b_ppo, baseline_quantiles)

        b_distilled = b_distilled.view(B_flat, A, 1)
        baseline_quantiles = baseline_quantiles.view(B_flat, A, -1)

        b_distilled_reshaped = b_distilled.view(*leading_dims, A, 1)
        baseline_quantiles = baseline_quantiles.view(*leading_dims, A, -1)
        return b_distilled_reshaped, baseline_quantiles


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
    def __init__(self, input_size, output_size, hidden_size=256, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
            feature_dim: int,
            hidden_size: int = 128,
            quantile_embedding_dim: int = 64,
            linear_init_scale: float = 1.0,
            num_quantiles: int = 32,
            dropout_rate: float = 0.0
        ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_quantiles = num_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        self.hidden_size = hidden_size

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
    """Distills a scalar value/baseline with quantile information."""

    def __init__(
        self,
        num_quantiles: Optional[int] = None,
        hidden_size: int = 128,
        combination_method: str = "hadamard_product",
        linear_init_scale: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.combination_method = combination_method
        self.linear_init_scale = linear_init_scale
        self.dropout_rate = dropout_rate

        self.mlp: Optional[nn.Sequential] = None
        self.input_dim_to_mlp: Optional[int] = None

        if num_quantiles is not None:
            self._build_mlp(num_quantiles)

    def _build_mlp(self, num_quantiles: int) -> None:
        if self.combination_method == "hadamard_product":
            input_dim = num_quantiles
        elif self.combination_method == "concatenate":
            input_dim = 1 + num_quantiles
        elif self.combination_method == "average_iqn_concatenate":
            input_dim = 2
        else:
            raise ValueError(f"Unsupported combination_method: {self.combination_method}")

        self.input_dim_to_mlp = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, 1),
        )
        self.mlp.apply(
            lambda m: init_weights_gelu_linear(m, scale=self.linear_init_scale)
            if isinstance(m, nn.Linear)
            else None
        )


    def forward(self, v_ppo: torch.Tensor, quantile_values: torch.Tensor) -> torch.Tensor:
        """Combine PPO scalar and quantile values into a distilled scalar."""

        batch_size = v_ppo.shape[0]
        if v_ppo.shape != (batch_size, 1):
            raise ValueError(f"Expected v_ppo shape ({batch_size}, 1), got {v_ppo.shape}")

        num_q = quantile_values.shape[1]
        expected_dim = (
            num_q
            if self.combination_method == "hadamard_product"
            else 1 + num_q
            if self.combination_method == "concatenate"
            else 2
        )
        if self.mlp is None or self.input_dim_to_mlp != expected_dim:
            self._build_mlp(num_q)

        if self.combination_method == "hadamard_product":
            v_ppo_expanded = v_ppo.expand_as(quantile_values)
            combined_input = v_ppo_expanded * quantile_values
        elif self.combination_method == "concatenate":
            combined_input = torch.cat([v_ppo, quantile_values], dim=-1)
        elif self.combination_method == "average_iqn_concatenate":
            mean_quantile_value = quantile_values.mean(dim=1, keepdim=True)
            combined_input = torch.cat([v_ppo, mean_quantile_value], dim=-1)
        else:
            raise ValueError(f"Unsupported combination_method: {self.combination_method}")

        return self.mlp(combined_input)
    

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
    """Encodes a state vector using a 2-layer MLP with Norm and Dropout."""
    def __init__(self, state_dim: int, hidden_size: int, output_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatesActionsEncoder(nn.Module):
    """Encodes a concatenated state-action pair using a 2-layer MLP with Norm and Dropout."""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, output_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if act.shape[:-1] != obs.shape[:-1]:
            try:
                act_expanded = act.expand(*obs.shape[:-1], act.shape[-1])
                x = torch.cat([obs, act_expanded], dim=-1)
            except RuntimeError as e:
                print(f"Error concatenating obs shape {obs.shape} and action shape {act.shape}: {e}")
                raise e
        else:
            x = torch.cat([obs, act], dim=-1)
        return self.net(x) 


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

        # --- Sub-layers ---
        # Projections for Query, Key, Value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Layer Normalization (Pre-LN architecture)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

        # The core attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        
        # Final Layer Normalization after residual connection
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, x_v=None, key_padding_mask: Optional[torch.Tensor] = None,
                return_attention_only: bool = False):
        """
        Forward pass for the attention block.
        Accepts an optional key_padding_mask to ignore padded elements in the key/value sequences.
        """
        # Determine Key and Value inputs. For self-attention, they are the same as the Query.
        if self.self_attention:
            key_input = x_q if x_k is None else x_k
            value_input = x_q if x_v is None else x_v
        else: # Cross-attention
            key_input = x_q if x_k is None else x_k
            value_input = key_input if x_v is None else x_v

        # --- Pre-LN: Normalize inputs before attention ---
        qn = self.norm_q(x_q)
        kn = self.norm_k(key_input)
        vn = self.norm_v(value_input)

        # Project Query, Key, Value
        q_proj = self.query_proj(qn)
        k_proj = self.key_proj(kn)
        v_proj = self.value_proj(vn)

        # --- Core Multi-Head Attention ---
        # The key_padding_mask (True for padded) tells MHA which keys to ignore.
        attn_out, attn_weights = self.mha(
            q_proj, k_proj, v_proj,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )

        # --- FIX FOR NaN PROPAGATION ---
        # If all keys are masked, the softmax in MHA results in NaN.
        # We replace any resulting NaNs with 0.0 to prevent network failure.
        # This means "attending to nothing" results in a zero-vector output.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        if return_attention_only:
            return attn_out, attn_weights

        out = x_q + attn_out
        out = self.norm_out(out)
        return out, attn_weights


class LinearSequenceEmbedder(nn.Module):
    def __init__(self, name: str, input_feature_dim: int, max_seq_len_for_pos_enc: int,
                 linear_stem_hidden_sizes: List[int], embed_dim: int, dropout_rate: float,
                 linear_init_scale: float, add_positional_encoding: bool = True):
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.add_positional_encoding = add_positional_encoding
        self.input_arity = 2 # For SeqLen, FeatDim

        layers = []
        prev_dim = input_feature_dim
        for h_dim in linear_stem_hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim), # LayerNorm on hidden features
                nn.GELU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.mlp_stem = nn.Sequential(*layers)
        
        if self.add_positional_encoding:
             self.pos_emb_1d = nn.Parameter(
                 create_2d_sin_cos_pos_emb(max_seq_len_for_pos_enc, 1, embed_dim, torch.device("cpu")).squeeze(1),
                 requires_grad=False
             )
        else:
            self.pos_emb_1d = None

        self.mlp_stem.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)


    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (B_eff, SeqLen, input_feature_dim)
        # padding_mask shape: (B_eff, SeqLen), True where padded
        B_eff, S, _ = x.shape

        # --- Process through MLP ---
        embedded_seq = self.mlp_stem(x)

        # --- Add Positional Encoding ---
        if self.add_positional_encoding and self.pos_emb_1d is not None:
            if self.pos_emb_1d.device != embedded_seq.device:
                 self.pos_emb_1d.data = self.pos_emb_1d.data.to(embedded_seq.device)
            
            pos_to_add = self.pos_emb_1d[:S, :].unsqueeze(0)
            embedded_seq = embedded_seq + pos_to_add
        
        # --- FIX FOR NaN GENERATION ---
        # Apply the padding mask AFTER all embedding and positional encoding steps.
        # This zeroes out the final embeddings for padded sequence elements,
        if padding_mask is not None:
            # `padding_mask` is True for PADDED entries. We want a mask where VALID entries are 1.
            valid_mask = ~padding_mask.unsqueeze(-1) # Shape: (B_eff, SeqLen, 1)
            embedded_seq = embedded_seq * valid_mask.float() # Zero out the padded embeddings

        return embedded_seq
    

class LinearVectorEmbedder(nn.Module):
    """Processes a single vector input and projects it to embed_dim, outputting (B, 1, embed_dim)."""
    def __init__(self, name: str, input_feature_dim: int, 
                 linear_stem_hidden_sizes: List[int], embed_dim: int, 
                 dropout_rate: float, linear_init_scale: float):
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.input_arity = 1 # For FeatDim

        layers = []
        prev_dim = input_feature_dim
        for h_dim in linear_stem_hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.mlp_stem = nn.Sequential(*layers)
        self.mlp_stem.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B_eff, input_feature_dim)
        embedded_vector = self.mlp_stem(x) # (B_eff, embed_dim)
        return embedded_vector.unsqueeze(1) # (B_eff, 1, embed_dim) - treat as sequence of length 1


class WindowAttention(nn.Module):
    """Self-attention over non-overlapping windows."""
    def __init__(self, embed_dim: int, num_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = self.window_size
        if H % ws != 0 or W % ws != 0:
            raise ValueError(f"WindowAttention requires H and W divisible by window_size {ws}")
        x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0,1,3,2,4,5).contiguous()
        windows = x.view(-1, ws * ws, C)
        q = self.norm_q(windows)
        k = self.norm_k(windows)
        v = self.norm_v(windows)
        attn_out, _ = self.attn(q, k, v)
        attn_out = windows + attn_out
        attn_out = self.norm_out(attn_out)
        attn_out = attn_out.view(B, H // ws, W // ws, ws, ws, C).permute(0,1,3,2,4,5).contiguous()
        return attn_out.view(B, H, W, C)


class PatchMerging(nn.Module):
    """Merge 2x2 neighboring patches."""
    def __init__(self, embed_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim if out_dim is not None else embed_dim
        self.norm = nn.LayerNorm(4 * embed_dim)
        self.reduction = nn.Linear(4 * embed_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        assert N == H * W
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("PatchMerging requires even H and W")
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        H_new, W_new = H // 2, W // 2
        x = x.view(B, H_new * W_new, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H_new, W_new


class SwinBlock(nn.Module):
    """Basic Swin block with optional patch merging."""
    def __init__(self, embed_dim: int, num_heads: int, window_size: int, ff_hidden_dim: int, dropout: float = 0.0, merge: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.merge = PatchMerging(embed_dim) if merge else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        assert N == H * W
        x_hw = x.view(B, H, W, C)
        attn_out = self.attn(self.norm1(x_hw))
        x_hw = x_hw + attn_out
        x_flat = x_hw.view(B, H * W, C)
        ffn_out = self.ffn(self.norm2(x_flat))
        x_flat = x_flat + ffn_out
        if self.merge is not None:
            x_flat, H, W = self.merge(x_flat, H, W)
        return x_flat, H, W


class AdaptiveSpatialTokenizer(nn.Module):
    """Selects a subset of spatial tokens based on learned importance."""

    def __init__(
        self,
        embed_dim: int,
        base_tokens: int = 16,
        min_tokens: int = 8,
        max_tokens: int = 24,
        importance_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.importance_net = nn.Sequential(
            nn.Linear(embed_dim, importance_hidden),
            nn.GELU(),
            nn.Linear(importance_hidden, 1)
        )
        self.base_tokens = base_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected tokens and their indices.

        Args:
            x: Tensor of shape (B, N, C) representing N spatial tokens.

        Returns:
            (selected_tokens, indices) where ``selected_tokens`` has shape
            (B, K, C) and ``indices`` has shape (B, K).
        """
        B, N, C = x.shape
        k = min(max(self.min_tokens, self.base_tokens), min(self.max_tokens, N))
        scores = self.importance_net(x).squeeze(-1)  # (B, N)
        topk_val, topk_idx = torch.topk(scores, k=k, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        selected = torch.gather(x, 1, gather_idx)
        return selected, topk_idx


class ParallelCrossAttention(nn.Module):
    """Apply cross-attention to multiple components in parallel with learnable fusion."""

    def __init__(self, embed_dim: int, num_heads: int, component_names: List[str], dropout: float = 0.0):
        super().__init__()
        self.cross_attns = nn.ModuleDict({
            name: ResidualAttention(embed_dim, num_heads, dropout=dropout, self_attention=False)
            for name in component_names
        })
        # Gating networks produce a single scalar per component, per batch and agent
        self.gating_nets = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1)
            )
            for name in component_names
        })
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        agent_embed: torch.Tensor,
        components: Dict[str, torch.Tensor],
        key_padding_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attn_outputs = []
        gate_scores = []
        for name, feats in components.items():
            if name not in self.cross_attns:
                continue
            mask = None
            if key_padding_masks and name in key_padding_masks:
                mask = key_padding_masks[name]
            attn_out, _ = self.cross_attns[name](
                agent_embed,
                feats,
                feats,
                key_padding_mask=mask,
                return_attention_only=True,
            )
            attn_outputs.append(attn_out)
            # Use the attended representation to compute gating so weights depend
            # on both the agent embedding and the corresponding component
            gate_scores.append(self.gating_nets[name](attn_out).squeeze(-1))

        if not attn_outputs:
            return agent_embed

        gates_stack = torch.stack(gate_scores, dim=0)  # (num_comp, B, A)
        gate_weights = F.softmax(gates_stack, dim=0).unsqueeze(-1)  # (num_comp, B, A, 1)
        attn_stack = torch.stack(attn_outputs, dim=0)  # (num_comp, B, A, H)
        combined = (attn_stack * gate_weights).sum(dim=0)
        out = agent_embed + combined
        return self.norm(out)


class HierarchicalCrossAttention(nn.Module):
    """Two-stage attention over spatial and object tokens."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        half_heads = max(1, num_heads // 2)
        self.spatial_attention = ResidualAttention(embed_dim, half_heads, dropout=dropout, self_attention=False)
        self.object_attention = ResidualAttention(embed_dim, half_heads, dropout=dropout, self_attention=False)
        self.fusion_attention = ResidualAttention(embed_dim, num_heads, dropout=dropout, self_attention=False)

    def forward(self, agent_embed: torch.Tensor, height_tokens: Optional[torch.Tensor],
                object_tokens: Optional[torch.Tensor], height_mask: Optional[torch.Tensor] = None,
                object_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_inputs = []
        if height_tokens is not None:
            spatial_info, _ = self.spatial_attention(agent_embed, height_tokens, height_tokens,
                                                     key_padding_mask=height_mask, return_attention_only=True)
            attn_inputs.append(spatial_info)
        if object_tokens is not None:
            object_info, _ = self.object_attention(agent_embed, object_tokens, object_tokens,
                                                   key_padding_mask=object_mask, return_attention_only=True)
            attn_inputs.append(object_info)

        if not attn_inputs:
            return agent_embed

        combined = torch.stack(attn_inputs, dim=1).mean(dim=1)
        out, _ = self.fusion_attention(agent_embed, combined, combined)
        return out


class CNNSpatialEmbedder(nn.Module):
    """
    Processes a single image-like central input (e.g., height map, grayscale image)
    through a CNN stem, flattens it into a patch sequence, adds positional embeddings,
    and projects to the main embedding dimension.
    """
    def __init__(self, name: str,
                 input_channels_data: int,
                 initial_h: int, initial_w: int,
                 cnn_stem_channels: List[int], cnn_stem_kernels: List[int],
                 cnn_stem_strides: List[int], cnn_stem_group_norms: List[int],
                 target_pool_scale: int,
                 embed_dim: int, dropout_rate: float, conv_init_scale: float,
                 add_spatial_coords: bool = True,
                 swin_stages: Optional[List[Dict[str, Any]]] = None,
                 adaptive_tokenizer_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_pool_scale = target_pool_scale
        self.embed_dim = embed_dim
        self.add_spatial_coords = add_spatial_coords
        self.data_input_channels = input_channels_data

        self.swin_stages_cfg = swin_stages or []
        self.swin_blocks = nn.ModuleList()
        for stg in self.swin_stages_cfg:
            self.swin_blocks.append(
                SwinBlock(
                    embed_dim=embed_dim,
                    num_heads=stg.get("num_heads", 4),
                    window_size=stg.get("window_size", 2),
                    ff_hidden_dim=embed_dim * 4,
                    dropout=dropout_rate,
                    merge=stg.get("merge", False),
                )
            )

        self.effective_cnn_input_channels = self.data_input_channels + 2 if self.add_spatial_coords else self.data_input_channels
        
        if self.effective_cnn_input_channels <= 0:
            raise ValueError(f"CNNSpatialEmbedder '{name}': effective_cnn_input_channels must be > 0, got {self.effective_cnn_input_channels} (data_channels: {self.data_input_channels}, add_spatial_coords: {self.add_spatial_coords})")

        cnn_layers = []
        prev_c = self.effective_cnn_input_channels
        for i, out_c in enumerate(cnn_stem_channels):
            if out_c <= 0 or cnn_stem_group_norms[i] <= 0 or out_c % cnn_stem_group_norms[i] != 0 :
                raise ValueError(f"CNNSpatialEmbedder '{name}': group_norms[{i}] ({cnn_stem_group_norms[i]}) must be > 0 and divide cnn_channels[{i}] ({out_c}) which also must be > 0.")
            cnn_layers.extend([
                nn.Conv2d(prev_c, out_c, kernel_size=cnn_stem_kernels[i], stride=cnn_stem_strides[i], padding=cnn_stem_kernels[i]//2),
                nn.GroupNorm(cnn_stem_group_norms[i], out_c),
                nn.GELU(),
                nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_c = out_c
        self.cnn_stem = nn.Sequential(*cnn_layers)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_pool_scale, target_pool_scale))
        self.projection = nn.Conv2d(prev_c, embed_dim, kernel_size=1) 
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.pos_emb = nn.Parameter(
            create_2d_sin_cos_pos_emb(target_pool_scale, target_pool_scale, embed_dim, self.device),
            requires_grad=False
        )

        if adaptive_tokenizer_cfg is not None:
            self.tokenizer = AdaptiveSpatialTokenizer(
                embed_dim=embed_dim,
                base_tokens=adaptive_tokenizer_cfg.get("base_tokens", 16),
                min_tokens=adaptive_tokenizer_cfg.get("min_tokens", 8),
                max_tokens=adaptive_tokenizer_cfg.get("max_tokens", 24),
                importance_hidden=adaptive_tokenizer_cfg.get("importance_hidden", 64),
            )
        else:
            self.tokenizer = None

        self.cnn_stem.apply(lambda m: init_weights_gelu_conv(m, scale=conv_init_scale) if isinstance(m, nn.Conv2d) else None)
        init_weights_gelu_conv(self.projection, scale=conv_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x expected by CNNSpatialEmbedder should be (EffectiveBatch, self.data_input_channels, H, W)
        # OR (EffectiveBatch, H, W) if self.data_input_channels == 1.
        
        # Handle input tensor that might be 3D (B_eff, H, W) for single-channel data
        if x.ndim == 3: # Implicit C=1
            if self.data_input_channels != 1:
                raise ValueError(
                    f"CNNSpatialEmbedder '{self.name}': Received 3D input (B,H,W), implying C=1, "
                    f"but module configured for data_input_channels={self.data_input_channels}."
                )
            x = x.unsqueeze(1)  # Add channel dimension: (EffectiveBatch, 1, H, W)
        elif x.ndim != 4:
            raise ValueError(
                f"CNNSpatialEmbedder '{self.name}': Expected 3D or 4D input, got {x.ndim}D tensor with shape {x.shape}."
            )
        
        B_eff, C_tensor, H, W = x.shape
        
        if C_tensor != self.data_input_channels:
            raise ValueError(
                f"CNNSpatialEmbedder '{self.name}': Input tensor has {C_tensor} channels, "
                f"but module configured for data_input_channels={self.data_input_channels}."
            )

        if self.add_spatial_coords:
            row_coords = torch.linspace(-1, 1, steps=H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B_eff, 1, H, W)
            col_coords = torch.linspace(-1, 1, steps=W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B_eff, 1, H, W)
            x_processed = torch.cat([x, row_coords, col_coords], dim=1) 
        else:
            x_processed = x
            
        if x_processed.shape[1] != self.effective_cnn_input_channels:
             raise ValueError(
                 f"CNNSpatialEmbedder '{self.name}': Effective input channels to CNN mismatch after spatial coord handling. "
                 f"Got {x_processed.shape[1]}, expected {self.effective_cnn_input_channels}"
            )

        features = self.cnn_stem(x_processed)         
        pooled = self.adaptive_pool(features)         
        projected = self.projection(pooled)           
        
        patches = projected.view(B_eff, self.embed_dim, -1).transpose(1, 2).contiguous()
        patches = self.layer_norm(patches)

        patches = patches + self.pos_emb.unsqueeze(0)

        H_curr = self.target_pool_scale
        W_curr = self.target_pool_scale
        x_hw = patches.view(B_eff, H_curr, W_curr, self.embed_dim)
        x_seq = x_hw.view(B_eff, -1, self.embed_dim)
        for blk in self.swin_blocks:
            x_seq, H_curr, W_curr = blk(x_seq, H_curr, W_curr)

        if self.tokenizer is not None:
            x_seq, _ = self.tokenizer(x_seq)

        return x_seq


class CrossAttentionFeatureExtractor(nn.Module):
    """Embeds agent observations and a set of central inputs for transformer
    processing. Image-like components are converted into patch sequences via
    :class:`CNNSpatialEmbedder` so they can participate in cross-attention."""
    def __init__(
        self,
        agent_obs_size: int,
        num_agents: int,
        embed_dim: int,
        num_heads: int,
        transformer_layers: int,
        dropout_rate: float,
        use_agent_id: bool,
        ff_hidden_factor: int,
        id_num_freqs: int,
        conv_init_scale: float,
        linear_init_scale: float,
        central_processing_configs: List[Dict[str, Any]],
        environment_shape_config: Dict[str, Any],
        fusion_type: str = "sequential"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        self.use_agent_id = use_agent_id
        self.transformer_layers = transformer_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_type = fusion_type

        # --- Agent Observation Encoder ---
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_obs_size, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.agent_encoder.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

        # --- Optional Agent ID Encoding ---
        if use_agent_id:
            self.agent_id_pos_enc = AgentIDPosEnc(num_freqs=id_num_freqs, id_embed_dim=embed_dim)
            self.agent_id_sum_ln = nn.LayerNorm(embed_dim)
        else:
            self.agent_id_pos_enc = None
            self.agent_id_sum_ln = None
            
        # --- Central State Embedders ---
        self.central_embedders = nn.ModuleDict()
        self.enabled_central_component_names: List[str] = []

        # Parse shape info from the environment config first
        central_comp_shapes_from_env_config: Dict[str, Dict[str, int]] = {}
        if "state" in environment_shape_config and "central" in environment_shape_config["state"]:
            for comp_def_env_shape in environment_shape_config["state"]["central"]:
                if comp_def_env_shape.get("enabled", False):
                    comp_name_from_env_shape = comp_def_env_shape.get("name")
                    if comp_name_from_env_shape: 
                        central_comp_shapes_from_env_config[comp_name_from_env_shape] = comp_def_env_shape.get("shape", {})

        # Create embedder modules based on the processing config, using the shapes parsed above
        for comp_cfg_proc in central_processing_configs: 
            name = comp_cfg_proc["name"]
            if not comp_cfg_proc.get("enabled", False): continue
            if name not in central_comp_shapes_from_env_config: continue

            self.enabled_central_component_names.append(name)
            comp_type = comp_cfg_proc["type"]
            env_shape_for_comp = central_comp_shapes_from_env_config[name]

            if comp_type == "cnn":
                h = env_shape_for_comp.get("h")
                w = env_shape_for_comp.get("w")
                c = env_shape_for_comp.get("c")

                self.central_embedders[name] = CNNSpatialEmbedder(
                    name=name,
                    input_channels_data=c,
                    initial_h=h,
                    initial_w=w,
                    cnn_stem_channels=comp_cfg_proc.get("cnn_stem_channels", []),
                    cnn_stem_kernels=comp_cfg_proc.get("cnn_stem_kernels", []),
                    cnn_stem_strides=comp_cfg_proc.get("cnn_stem_strides", []),
                    cnn_stem_group_norms=comp_cfg_proc.get("cnn_stem_group_norms", []),
                    target_pool_scale=comp_cfg_proc.get("target_pool_scale", 1),
                    embed_dim=embed_dim,
                    dropout_rate=dropout_rate,
                    conv_init_scale=conv_init_scale,
                    add_spatial_coords=comp_cfg_proc.get("add_spatial_coords_to_cnn_input", True),
                    swin_stages=comp_cfg_proc.get("swin_stages", []),
                    adaptive_tokenizer_cfg=comp_cfg_proc.get("adaptive_tokenizer"),
                )
            elif comp_type == "linear_sequence": 
                self.central_embedders[name] = LinearSequenceEmbedder(
                    name=name,
                    input_feature_dim=env_shape_for_comp.get("feature_dim", 0), 
                    linear_stem_hidden_sizes=comp_cfg_proc.get("linear_stem_hidden_sizes", []),
                    embed_dim=embed_dim,
                    dropout_rate=dropout_rate,
                    linear_init_scale=linear_init_scale,
                    add_positional_encoding=comp_cfg_proc.get("add_positional_encoding", True),
                    max_seq_len_for_pos_enc=env_shape_for_comp.get("max_length", 0)
                )
            else:
                raise ValueError(f"Unsupported central component type: {comp_type} for component '{name}'")

        # --- Transformer Blocks ---
        self.cross_attention_blocks = nn.ModuleList() 
        self.agent_self_attention_blocks = nn.ModuleList()
        self.agent_feed_forward_blocks = nn.ModuleList()
        ff_hidden_dim = embed_dim * ff_hidden_factor

        for _ in range(transformer_layers):
            if self.fusion_type == "parallel":
                self.cross_attention_blocks.append(
                    ParallelCrossAttention(embed_dim, num_heads, self.enabled_central_component_names, dropout=dropout_rate)
                )
            elif self.fusion_type == "hierarchical":
                self.cross_attention_blocks.append(
                    HierarchicalCrossAttention(embed_dim, num_heads, dropout=dropout_rate)
                )
            else:
                layer_cross_attns = nn.ModuleDict()
                for comp_name in self.enabled_central_component_names:
                    layer_cross_attns[comp_name] = ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=False)
                self.cross_attention_blocks.append(layer_cross_attns)
            
            self.agent_self_attention_blocks.append(ResidualAttention(embed_dim, num_heads, dropout=dropout_rate, self_attention=True))
            self.agent_feed_forward_blocks.append(FeedForwardBlock(embed_dim, hidden_dim=ff_hidden_dim, dropout=dropout_rate))
        
        # --- Final Output Layers ---
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_hidden_factor // 2), nn.GELU(),
            nn.Linear(embed_dim * ff_hidden_factor // 2, embed_dim)
        )
        self.output_mlp.apply(lambda m: init_weights_gelu_linear(m, scale=linear_init_scale) if isinstance(m, nn.Linear) else None)

    def forward(self, obs_dict: Dict[str, Any], 
                central_component_padding_masks: Optional[Dict[str, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        agent_state = obs_dict.get("agent")
        central_states_components = obs_dict.get("central", {})

        # --- Determine Batch Shape ---
        is_batched_sequence = agent_state.ndim == 4
        if is_batched_sequence:
            B_orig, S_orig, A_runtime, _ = agent_state.shape
        else: 
            B_orig, A_runtime, _ = agent_state.shape
            S_orig = 1
        effective_batch_for_embedders = B_orig * S_orig

        # --- Encode Agent Observations ---
        agent_input_flat = agent_state.reshape(effective_batch_for_embedders * A_runtime, -1)
        agent_embed = self.agent_encoder(agent_input_flat)
        if self.use_agent_id and self.agent_id_pos_enc:
            agent_ids = torch.arange(A_runtime, device=agent_embed.device).repeat(effective_batch_for_embedders)
            id_enc = self.agent_id_pos_enc(agent_ids)
            agent_embed = self.agent_id_sum_ln(agent_embed + id_enc)
        agent_embed = agent_embed.view(effective_batch_for_embedders, A_runtime, self.embed_dim)

        # --- Encode Central State Components ---
        processed_central_features: Dict[str, torch.Tensor] = {}
        for comp_name, embedder_module in self.central_embedders.items():
            if comp_name in central_states_components:
                central_comp_tensor = central_states_components[comp_name]
                content_shape = central_comp_tensor.shape[(2 if is_batched_sequence else 1):]
                comp_data_for_embedder = central_comp_tensor.reshape(effective_batch_for_embedders, *content_shape)

                padding_mask_for_comp = None
                if central_component_padding_masks and (comp_name + "_mask") in central_component_padding_masks:
                    mask_tensor = central_component_padding_masks[comp_name + "_mask"]
                    if mask_tensor is not None:
                        padding_mask_for_comp = mask_tensor.reshape(effective_batch_for_embedders, -1)

                if isinstance(embedder_module, LinearSequenceEmbedder):
                    processed_central_features[comp_name] = embedder_module(comp_data_for_embedder, padding_mask=padding_mask_for_comp)
                else:  # CNNSpatialEmbedder or other types without masks
                    processed_central_features[comp_name] = embedder_module(comp_data_for_embedder)

        # --- Transformer Layers ---
        final_self_attn_weights = None
        for i in range(self.transformer_layers):
            if self.fusion_type == "parallel":
                mask_dict = {}
                if central_component_padding_masks:
                    for comp_name in self.enabled_central_component_names:
                        mask = central_component_padding_masks.get(comp_name + "_mask")
                        if mask is not None:
                            mask_dict[comp_name] = mask.reshape(effective_batch_for_embedders, -1).to(agent_embed.device)
                agent_embed = self.cross_attention_blocks[i](agent_embed, processed_central_features, mask_dict)
            elif self.fusion_type == "hierarchical":
                h_tokens = processed_central_features.get("height_map")
                obj_tokens = processed_central_features.get("gridobject_sequence")
                h_mask = None
                obj_mask = None
                if central_component_padding_masks:
                    if "height_map_mask" in central_component_padding_masks:
                        m = central_component_padding_masks["height_map_mask"]
                        if m is not None:
                            h_mask = m.reshape(effective_batch_for_embedders, -1).to(agent_embed.device)
                    if "gridobject_sequence_mask" in central_component_padding_masks:
                        m = central_component_padding_masks["gridobject_sequence_mask"]
                        if m is not None:
                            obj_mask = m.reshape(effective_batch_for_embedders, -1).to(agent_embed.device)
                agent_embed = self.cross_attention_blocks[i](agent_embed, h_tokens, obj_tokens, height_mask=h_mask, object_mask=obj_mask)
            else:
                for comp_name in self.enabled_central_component_names:
                    if comp_name in processed_central_features:
                        central_feature_sequence = processed_central_features[comp_name]

                        key_padding_mask = None
                        if central_component_padding_masks and (comp_name + "_mask") in central_component_padding_masks:
                            mask_tensor = central_component_padding_masks[comp_name + "_mask"]
                            if mask_tensor is not None:
                                key_padding_mask = mask_tensor.reshape(effective_batch_for_embedders, -1).to(agent_embed.device)

                        agent_embed, _ = self.cross_attention_blocks[i][comp_name](
                            x_q=agent_embed,
                            x_k=central_feature_sequence,
                            x_v=central_feature_sequence,
                            key_padding_mask=key_padding_mask
                        )
            
            # Self-Attention: Agents attend to each other
            agent_embed, attn_weights = self.agent_self_attention_blocks[i](agent_embed)
            if i == self.transformer_layers - 1:
                final_self_attn_weights = attn_weights

            # Feed-Forward Network
            agent_embed = self.agent_feed_forward_blocks[i](agent_embed)

        # --- Final Processing ---
        agent_embed = self.final_norm(agent_embed)
        agent_embed = self.output_mlp(agent_embed)

        # Reshape back to original batch dimensions (if sequence)
        if is_batched_sequence: 
            agent_embed_final = agent_embed.view(B_orig, S_orig, A_runtime, self.embed_dim)
            if final_self_attn_weights is not None:
                 final_self_attn_weights = final_self_attn_weights.view(B_orig, S_orig, A_runtime, A_runtime)
        else: 
            agent_embed_final = agent_embed
            
        return agent_embed_final, final_self_attn_weights


class MultiAgentEmbeddingNetwork(nn.Module):
    def __init__(self, net_cfg: Dict[str, Any], environment_config_for_shapes: Dict[str, Any]):
        """
        Args:
            net_cfg (Dict[str, Any]): Configuration specific to this network and its submodules,
                                      typically from agent_config["networks"]["MultiAgentEmbeddingNetwork"].
            environment_config_for_shapes (Dict[str, Any]): The top-level "environment" block from the main JSON config,
                                                            used to pass global shape information.
        """
        super(MultiAgentEmbeddingNetwork, self).__init__()
        
        ca_config = net_cfg["cross_attention_feature_extractor"]
        
        # Correctly pass the "shape" dictionary from the "environment" config block
        self.base_encoder = CrossAttentionFeatureExtractor(
            **ca_config, 
            environment_shape_config=environment_config_for_shapes.get("shape", {}) 
        )
        
        emb_out_dim = ca_config["embed_dim"]
        
        obs_enc_cfg = net_cfg["obs_encoder"].copy() 
        obs_enc_cfg["state_dim"] = emb_out_dim 
        self.g = StatesEncoder(**obs_enc_cfg) 

        obs_act_enc_cfg = net_cfg["obs_actions_encoder"].copy() 
        obs_act_enc_cfg["state_dim"] = emb_out_dim 
        
        # Determine action_dim for obs_actions_encoder ('f') from the environment_config_for_shapes
        # environment_config_for_shapes is cfg["environment"]
        action_shape_root = environment_config_for_shapes.get("shape", {}).get("action", {})
        action_spec_dict = action_shape_root.get("agent", action_shape_root.get("central", {})) # Prioritize agent

        if "discrete" in action_spec_dict: 
            obs_act_enc_cfg["action_dim"] = len(action_spec_dict["discrete"]) # Number of discrete branches
        elif "continuous" in action_spec_dict: 
            obs_act_enc_cfg["action_dim"] = len(action_spec_dict["continuous"]) # Number of continuous dimensions
        else: 
            print(f"Warning: MultiAgentEmbeddingNetwork could not determine action_dim for obs_actions_encoder ('f') from environment_config_for_shapes. Using value from net_cfg: {obs_act_enc_cfg.get('action_dim', 'not found, will default to 1 in StatesActionsEncoder if not set')}")
            # StatesActionsEncoder might have its own default or require action_dim
            if "action_dim" not in obs_act_enc_cfg:
                 # This ensures StatesActionsEncoder doesn't fail if action_dim is missing entirely
                 obs_act_enc_cfg["action_dim"] = obs_act_enc_cfg.get("action_dim", 1) 


        self.f = StatesActionsEncoder(**obs_act_enc_cfg)

    def get_base_embedding(self, obs_dict: Dict[str, Any], 
                           central_component_padding_masks: Optional[Dict[str, torch.Tensor]] = None
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.base_encoder(obs_dict, central_component_padding_masks=central_component_padding_masks)

    def get_baseline_embeddings(self, common_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        is_batched_sequence = common_emb.ndim == 4
        if is_batched_sequence:
            B_orig, S_orig, A, H_embed = common_emb.shape
            effective_batch_size = B_orig * S_orig
            common_emb_flat_for_gf = common_emb.reshape(effective_batch_size, A, H_embed)
            
            # Ensure actions are reshaped correctly based on their original dimensions
            # common_emb is (B,S,A,H), actions from runner is (B,S,A,ActDim) or (B,S,A,Branches)
            if actions.ndim == common_emb.ndim: # (B,S,A,ActDim/Branches)
                 actions_flat_for_gf = actions.reshape(effective_batch_size, A, -1)
            elif actions.ndim == 3 and common_emb.ndim == 4 and actions.shape[0] == effective_batch_size : # (B*S, A, ActDim/Branches)
                 actions_flat_for_gf = actions # Already flattened along B*S
            else:
                raise ValueError(f"Action shape {actions.shape} incompatible with common_emb shape {common_emb.shape} for baseline.")

        else: # common_emb.ndim == 3 (B_orig_eff, A, H_embed)
            B_orig, A, H_embed = common_emb.shape
            S_orig = 1 
            effective_batch_size = B_orig
            common_emb_flat_for_gf = common_emb
            # actions should be (B_orig_eff, A, ActDim/Branches)
            if actions.shape[0] != effective_batch_size or actions.ndim !=3:
                raise ValueError(f"Action shape {actions.shape} incompatible with common_emb shape {common_emb.shape} for baseline (non-sequence).")
            actions_flat_for_gf = actions

        g_input = common_emb_flat_for_gf.reshape(effective_batch_size * A, H_embed)
        agent_j_features_flat = self.g(g_input) 
        H_baseline_feat = agent_j_features_flat.shape[-1]
        agent_j_features = agent_j_features_flat.view(effective_batch_size, A, H_baseline_feat)

        actions_reshaped_for_f = actions_flat_for_gf.reshape(effective_batch_size * A, -1)
        groupmate_i_features_flat = self.f(g_input, actions_reshaped_for_f) 
        groupmate_i_features = groupmate_i_features_flat.view(effective_batch_size, A, H_baseline_feat)
        
        baseline_input_sequences_list = []
        for j_agent_idx in range(A): 
            seq_for_j_list = []
            seq_for_j_list.append(agent_j_features[:, j_agent_idx, :]) 
            for i_groupmate_idx in range(A):
                if i_groupmate_idx == j_agent_idx: continue
                seq_for_j_list.append(groupmate_i_features[:, i_groupmate_idx, :]) 
            
            seq_for_j_tensor = torch.stack(seq_for_j_list, dim=1) 
            baseline_input_sequences_list.append(seq_for_j_tensor)
        
        baseline_input_stacked_agents = torch.stack(baseline_input_sequences_list, dim=0)
        baseline_input_final_flat = baseline_input_stacked_agents.permute(1, 0, 2, 3).contiguous()

        if is_batched_sequence:
            return baseline_input_final_flat.view(B_orig, S_orig, A, A, H_baseline_feat)
        else: 
            return baseline_input_final_flat.view(B_orig, A, A, H_baseline_feat)


class AgentIDPosEnc(nn.Module):  # Included for completeness
    """Sinusoidal positional encoder for integer agent IDs."""

    def __init__(self, num_freqs: int = 8, id_embed_dim: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.id_embed_dim = id_embed_dim
        self.linear = nn.Linear(2 * num_freqs, id_embed_dim)
        init.kaiming_normal_(self.linear.weight, a=0.01)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

        # Precompute frequency scales and register as a buffer so it moves with
        # the module when ``to(device)`` is called.
        freq_indices = torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_scales", torch.pow(2.0, freq_indices))

    def sinusoidal_features(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float().unsqueeze(-1)
        scales = self.freq_scales
        if scales.device != x.device:
            scales = scales.to(x.device)
        multiplied = x_float * scales.unsqueeze(0)
        sines = torch.sin(multiplied)
        cosines = torch.cos(multiplied)
        return torch.cat([sines, cosines], dim=-1)

    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        shape_in = agent_ids.shape
        feats = self.sinusoidal_features(agent_ids.reshape(-1)) 
        out_lin = self.linear(feats) 
        return out_lin.view(*shape_in, self.id_embed_dim)
    

class ForwardDynamicsModel(nn.Module):
    """A simple MLP to predict the next state's feature embedding."""
    def __init__(self, embed_dim: int, action_dim: int, hidden_size: int = 256, dropout_rate=0.1):
        """
        Initializes the forward dynamics model.

        Args:
            embed_dim (int): The dimension of the input state and output predicted state embeddings.
            action_dim (int): The dimension of the action vector.
            hidden_size (int): The size of the hidden layers in the MLP.
        """
        super().__init__()
        # The network takes a concatenated state embedding and action vector as input.
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, embed_dim) # Predicts an embedding of the same dimension as the input state.
        )
        # Apply a standard weight initialization.
        self.apply(lambda m: init_weights_leaky_relu(m) if isinstance(m, nn.Linear) else None)

    def forward(self, state_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predicts the embedding of the next state.

        Args:
            state_embed (torch.Tensor): The feature embedding of the current state.
            action (torch.Tensor): The action taken in the current state.

        Returns:
            torch.Tensor: The predicted feature embedding of the next state.
        """
        # Concatenate the state embedding and action to form the input.
        x = torch.cat([state_embed, action], dim=-1)
        return self.net(x)
