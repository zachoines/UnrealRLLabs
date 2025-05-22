import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict, Tuple

def create_2d_sin_cos_pos_emb(h: int, w: int, embed_dim: int, device: torch.device):
    """
    Create a 2D sine-cosine positional encoding => shape (h*w, embed_dim),
    splitting embed_dim in half for row, half for col.
    """
    half_dim = embed_dim // 2  # We'll use half for row, half for col

    def position_encoding_1d(n: int, half_d: int):
        """
        Returns shape (n, half_d), standard Transformer approach for 1D pos encoding.
        """
        pos = torch.arange(n, device=device, dtype=torch.float).unsqueeze(1)  # (n,1)
        i = torch.arange(half_d, device=device, dtype=torch.float).unsqueeze(0)  # (1, half_d)
        exponent = 2.0 * i / float(half_d)
        div_term = torch.pow(10000.0, -exponent)  # (1, half_d)

        angles = pos * div_term  # => (n, half_d)
        pe = torch.zeros_like(angles, device=device)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    row_emb = position_encoding_1d(h, half_dim)  # (h, half_dim)
    col_emb = position_encoding_1d(w, half_dim)  # (w, half_dim)

    # Combine into (h, w, embed_dim)
    row_emb = row_emb.unsqueeze(1).expand(h, w, half_dim)  # (h,w,half_dim)
    col_emb = col_emb.unsqueeze(0).expand(h, w, half_dim)  # (h,w,half_dim)

    pos_2d = torch.cat([row_emb, col_emb], dim=2)  # (h, w, embed_dim)
    pos_2d = pos_2d.reshape(h*w, embed_dim)        # => (h*w, embed_dim)
    return pos_2d

def init_weights_gelu_conv(m: nn.Module, scale: float = 1.0):
    """
    Kaiming init for Conv2d, treating GELU similar to ReLU (nonlinearity='relu'),
    then manually scaling by `scale`.
    """
    if isinstance(m, nn.Conv2d):
        # Use a=0.0 => ReLU approximation
        nn.init.kaiming_normal_(
            m.weight,
            a=0.0,  # ~ ReLU
            mode='fan_in',
            nonlinearity='relu'
        )
        if scale != 1.0:
            m.weight.data.mul_(scale)

        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_gelu_linear(m: nn.Module, scale: float = 1.0):
    """
    Kaiming init for Linear, also treating GELU as ReLU,
    then manually scaling by `scale`.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.0,  # ~ ReLU
            mode='fan_in',
            nonlinearity='relu'
        )
        if scale != 1.0:
            m.weight.data.mul_(scale)

        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_leaky_relu_conv(m: nn.Module, negative_slope: float = 0.01):
    """
    Applies Kaiming initialization to nn.Conv2d layers,
    assuming a LeakyReLU with the given negative_slope.
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(
            m.weight,
            a=negative_slope,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def init_weights_leaky_relu(m: nn.Module, negative_slope: float = 0.01):
    """
    Applies Kaiming initialization to nn.Linear layers,
    assuming a LeakyReLU with the given negative_slope.
    """
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(
            m.weight,
            a=negative_slope,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

class OneCycleCosineScheduler:
    def __init__(self, max_value, total_steps, pct_start=0.3, div_factor=25., final_div_factor=1e4):
        self.max_value = max_value
        self.min_value = max_value / div_factor
        self.final_value = max_value / final_div_factor
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.t = 0
        self.current_value = self.min_value
        self.increase_steps = int(self.pct_start * self.total_steps)
        self.decrease_steps = self.total_steps - self.increase_steps

    def step(self):
        if self.t < self.increase_steps:  # Cosine increase
            self.current_value = self.min_value + (self.max_value - self.min_value) * \
                                 (1 - np.cos(np.pi * self.t / self.increase_steps)) / 2
        elif self.t < self.total_steps:  # Cosine decrease
            adjusted_t = self.t - self.increase_steps
            self.current_value = self.max_value - (self.max_value - self.final_value) * \
                                 (1 - np.cos(np.pi * adjusted_t / self.decrease_steps)) / 2
        else:  # Final value phase
            self.current_value = self.final_value
        self.t += 1

        return self.current_value

    def value(self):
        return self.current_value

    def reset(self):
        self.t = 0

class OneCycleLRWithMin(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, optimizer, max_lr, total_steps=None, anneal_strategy='cos', pct_start=0.3, div_factor=25., final_div_factor=1e4, **kwargs):
        super().__init__(optimizer, max_lr=max_lr, total_steps=total_steps, anneal_strategy=anneal_strategy, pct_start=pct_start,div_factor=div_factor, final_div_factor=final_div_factor, **kwargs)
        self.min_lr = max_lr / final_div_factor

    def step(self, epoch=None):
        if self._step_count >= self.total_steps:
            # Set all param groups to the minimum lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.min_lr
        else:
            super().step(epoch)

class LinearValueScheduler:
    """
    A scheduler that linearly transitions a scalar value from `start_value`
    to `end_value` over `total_iters` steps.

    Example:
      scheduler = LinearValueScheduler(
          start_value=0.01,
          end_value=0.0,
          total_iters=100000
      )
      ...
      # each update step:
      current_entropy_coeff = scheduler.step()
      # use current_entropy_coeff in your loss calculation
      ...
    """

    def __init__(self, start_value: float, end_value: float, total_iters: int):
        self.start_value = start_value
        self.end_value   = end_value
        self.total_iters = max(1, total_iters)  # avoid zero division
        self.counter     = 0

    def step(self) -> float:
        """
        Moves the scheduler forward by 1 step and returns
        the newly computed value (linearly interpolated).

        If we exceed total_iters, it remains at end_value.
        """
        # fraction in [0..1]
        fraction = min(self.counter / float(self.total_iters), 1.0)
        new_value = (1.0 - fraction)*self.start_value + fraction*self.end_value

        # increment counter
        if self.counter < self.total_iters:
            self.counter += 1

        return new_value

    def current_value(self) -> float:
        """
        Returns the scheduler's current value without incrementing.
        Useful if you need to peek at it before stepping.
        """
        fraction = min(self.counter / float(self.total_iters), 1.0)
        return (1.0 - fraction)*self.start_value + fraction*self.end_value

class RunningMeanStdNormalizer:
    """
    Dictionary-capable normalizer that tracks running mean/variance for each key's
    last-dimension 'features'. Also works on a single Tensor.

    Warmup behavior (Statistics Gathering Cutoff Interpretation):
    - If 'warmup_steps' > 0:
        - During the first 'warmup_steps' calls to 'update()', statistics are accumulated.
        - After 'warmup_steps' calls to 'update()', statistics become frozen and are no longer updated.
    - If 'warmup_steps' == 0:
        - Statistics are updated indefinitely with every call to 'update()'.
    - The 'normalize()' method always uses the current (potentially frozen) statistics.
    """

    def __init__(self, warmup_steps: int = 0, epsilon: float = 1e-4, device: torch.device = torch.device("cpu")):
        self.device = device
        self.epsilon = epsilon 
        self.warmup_steps = warmup_steps
        self.update_calls_count = 0 # Tracks total number of calls to update()
        self.stats: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {} # mean, var, count

    def update(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Incorporate new data into the running mean+var.
        Updates statistics if warmup_steps == 0 OR if warmup_steps > 0 and update_calls_count < warmup_steps.
        """
        
        should_update_this_call = False
        if self.warmup_steps == 0: # No cutoff, always update
            should_update_this_call = True
        elif self.update_calls_count < self.warmup_steps: # Within warmup/gathering period
            should_update_this_call = True
        
        # Increment total calls regardless, to track progression against warmup_steps
        # This ensures that even if should_update_this_call becomes false,
        # we know we've passed the warmup_steps threshold.
        self.update_calls_count +=1 

        if not should_update_this_call:
            return # Statistics are frozen

        # Proceed to update stats
        if isinstance(x, torch.Tensor):
            if x.numel() > 0:
                self._update_single(x, key="__single__")
        elif isinstance(x, dict):
            for k, v in x.items():
                if v is not None and v.numel() > 0:
                    self._update_single(v, key=k)
        else:
            raise ValueError("Unsupported type for update: must be Tensor or dict of Tensors.")

    def normalize(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
                 ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns a normalized version of 'x' using the current stored stats for each key.
        Normalization is always applied based on the latest statistics (which might be frozen).
        """
        if isinstance(x, torch.Tensor):
            if x.numel() == 0: return x
            return self._normalize_single(x, key="__single__")
        elif isinstance(x, dict):
            out = {}
            for k,v in x.items():
                if v is not None:
                    if v.numel() == 0: out[k] = v
                    else: out[k] = self._normalize_single(v, key=k)
                else:
                    out[k] = None
            return out
        else:
            raise ValueError("Unsupported type for normalize: must be Tensor or dict of Tensors.")

    def _update_single(self, tensor: torch.Tensor, key: str):
        tensor = tensor.to(self.device)

        if tensor.ndim == 0: 
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1: 
            tensor = tensor.unsqueeze(-1)
        
        num_features = tensor.shape[-1]

        if key not in self.stats:
            mean_ = torch.zeros(num_features, device=self.device, dtype=tensor.dtype)
            var_  = torch.ones(num_features, device=self.device, dtype=tensor.dtype)
            # Initialize count with epsilon. Note: count is a scalar tensor.
            count_= torch.full((1,), self.epsilon, device=self.device, dtype=torch.float64)
            self.stats[key] = (mean_, var_, count_)

        mean, var, count = self.stats[key]
        
        if mean.shape[-1] != num_features:
            raise RuntimeError(f"Feature dimension mismatch for key '{key}' during update. "
                               f"Stored normalizer fdim: {mean.shape[-1]}, input tensor fdim: {num_features}")

        flat_tensor = tensor.reshape(-1, num_features)
        batch_n = flat_tensor.shape[0]
        if batch_n == 0: return

        batch_mean = flat_tensor.mean(dim=0)
        batch_var = flat_tensor.var(dim=0, unbiased=True) if batch_n > 1 else torch.zeros_like(batch_mean)

        if torch.isnan(batch_mean).any() or torch.isnan(batch_var).any():
            print(f"RunningMeanStdNormalizer Warning: NaN in batch stats for key '{key}'. Skipping update.")
            return

        # Welford's algorithm for updating mean and variance
        total_count_prev = count # This is a scalar tensor, e.g., tensor(100.001)
        # batch_n is an int. For tensor operations, ensure consistent types or explicit casting.
        total_count_new = total_count_prev + float(batch_n) 
        
        delta = batch_mean - mean
        new_mean = mean + delta * (float(batch_n) / total_count_new.clamp(min=1e-8)) # Use 1e-8 for count clamp too
        
        # M2 = var * count (approximately, for running sum of squares of differences)
        # More stable update for variance:
        m_a = var * total_count_prev 
        m_b = batch_var * float(batch_n)
        
        # Ensure terms in division are not zero
        clamped_total_count_new = total_count_new.clamp(min=1e-8)
        new_var = (m_a + m_b + torch.square(delta) * (total_count_prev * float(batch_n)) / clamped_total_count_new) / clamped_total_count_new
        new_var = torch.clamp(new_var, min=1e-8) # Clamp variance to a small positive to avoid sqrt(0) in normalize

        self.stats[key] = (new_mean, new_var, total_count_new)


    def _normalize_single(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.stats: 
            # This can happen if normalize is called before any updates for this key,
            # or if warmup_steps=0 and update was never called.
            # Initialize with default stats if not present, then normalize.
            # This makes normalize more robust if called on a "cold" normalizer.
            print(f"RunningMeanStdNormalizer Info: normalize called for key '{key}' but no stats found. Initializing default stats.")
            if tensor.ndim == 0: temp_fdim = 1
            elif tensor.ndim == 1: temp_fdim = 1 # Assuming (M,) is M samples of 1 feature
            else: temp_fdim = tensor.shape[-1]
            
            temp_mean = torch.zeros(temp_fdim, device=self.device, dtype=tensor.dtype)
            temp_var = torch.ones(temp_fdim, device=self.device, dtype=tensor.dtype)
            # Epsilon is for the count, not directly for var here, but using 1e-8 for sqrt is good
            return (tensor.to(self.device) - temp_mean.unsqueeze(0)) / torch.sqrt(temp_var.unsqueeze(0) + 1e-8)

        mean, var, count = self.stats[key]
        tensor_on_device = tensor.to(self.device)

        if tensor_on_device.ndim == 0:
            tensor_on_device = tensor_on_device.view(1, 1)
        elif tensor_on_device.ndim == 1:
            tensor_on_device = tensor_on_device.unsqueeze(-1)
        
        if tensor_on_device.shape[-1] != mean.shape[-1]:
            raise RuntimeError(
                f"Feature dimension mismatch in _normalize_single for key '{key}'. "
                f"Input tensor features: {tensor_on_device.shape[-1]}, "
                f"Normalizer features: {mean.shape[-1]}. Input shape: {tensor.shape}"
            )
        
        mean_reshaped = mean.unsqueeze(0) 
        var_reshaped  = var.unsqueeze(0)

        # Use the stored variance, which should be clamped >= 1e-8 from _update_single
        return (tensor_on_device - mean_reshaped) / torch.sqrt(var_reshaped + 1e-8)
    

class PopArtNormalizer(nn.Module):
    def __init__(self, output_layer: nn.Linear, beta: float = 0.999, epsilon: float = 1e-5, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.output_layer = output_layer # The final linear layer of the value head
        self.beta = beta         # For EMA of mean and second moment
        self.epsilon = epsilon   # For numerical stability
        self.device = device

        # Running statistics for the raw returns (targets)
        self.register_buffer('mu', torch.zeros(1, device=self.device))
        self.register_buffer('nu', torch.ones(1, device=self.device)) # Second moment E[X^2]
        self.register_buffer('sigma', torch.ones(1, device=self.device))
        self.register_buffer('count', torch.tensor(0.0, device=self.device)) # Using float for potentially large counts with EMA

    @torch.no_grad()
    def update_stats(self, raw_targets_batch: torch.Tensor):
        # raw_targets_batch should be a flat tensor of raw (unnormalized) returns
        if raw_targets_batch.numel() == 0:
            return

        batch_mu = raw_targets_batch.mean()
        batch_nu = (raw_targets_batch ** 2).mean()

        # Exponential Moving Average (EMA) update for stability
        # If count is small, weigh new batch more heavily
        # For very first batch, directly set stats, or use a scheme to rapidly adapt.
        if self.count < 10: # Initialize with first few batches directly or with high weight
             current_beta = 0.1 # Or some other aggressive initial beta
        else:
             current_beta = self.beta

        new_mu = current_beta * self.mu + (1 - current_beta) * batch_mu
        new_nu = current_beta * self.nu + (1 - current_beta) * batch_nu
        
        self.count += raw_targets_batch.numel() # Can track total elements seen

        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)
        self.sigma.copy_(torch.sqrt(self.nu - self.mu**2).clamp(min=self.epsilon))

        # Adapt the output layer of the associated value network
        if torch.is_tensor(self.output_layer.weight) and torch.is_tensor(self.output_layer.bias): # Check if parameters exist
            self.output_layer.weight.data.mul_(old_sigma / self.sigma)
            self.output_layer.bias.data.mul_(old_sigma / self.sigma)
            self.output_layer.bias.data.add_((old_mu - self.mu) / self.sigma)

    def normalize_targets(self, raw_targets: torch.Tensor) -> torch.Tensor:
        return (raw_targets - self.mu) / self.sigma

    def denormalize_outputs(self, normalized_outputs: torch.Tensor) -> torch.Tensor:
        return normalized_outputs * self.sigma + self.mu