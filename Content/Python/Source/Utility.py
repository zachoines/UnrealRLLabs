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

    - If you pass a single Tensor x => shape (..., features), it stores stats under "__single__".
    - If you pass a dictionary of Tensors => each value shape (..., features),
      it tracks each dictionary key separately.

    Each update() call does a standard running mean/variance calculation.
    It expects inputs to have features as the last dimension, e.g. (N, num_features).
    For scalar values like rewards, input should be (N, 1).

    Warmup behavior:
    - During 'warmup_steps', 'update()' will accumulate statistics.
    - During 'warmup_steps', 'normalize()' will return the original, unnormalized data.
    - After 'warmup_steps', 'normalize()' will return normalized data.
    """

    def __init__(self, warmup_steps: int = 0, epsilon: float = 1e-4, device: torch.device = torch.device("cpu")):
        self.device = device
        self.epsilon = epsilon # Used for initial count to prevent division by zero
        self.warmup_steps = warmup_steps
        self.current_warmup_step_count = 0 # Tracks number of update calls during warmup phase
        self.stats: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {} # mean, var, count

    def update(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Incorporate new data into the running mean+var.
        Input tensors are expected to have features as the last dimension, e.g. (N, num_features).
        For scalar values (like rewards), input should be (N, 1).
        This method ALWAYS updates the statistics.
        It also increments current_warmup_step_count if still in warmup phase.
        """
        if isinstance(x, torch.Tensor):
            if x.numel() > 0:
                self._update_single(x, key="__single__")
        elif isinstance(x, dict):
            for k, v in x.items():
                if v is not None and v.numel() > 0:
                    self._update_single(v, key=k)
        else:
            raise ValueError("Unsupported type for update: must be Tensor or dict of Tensors.")

        # Increment warmup step counter if we are in the warmup phase
        if self.warmup_steps > 0 and self.current_warmup_step_count < self.warmup_steps:
            self.current_warmup_step_count += 1

    def normalize(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
                 ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns a normalized version of 'x' using the stored stats for each key,
        OR the original 'x' if still in the warmup phase.
        """
        # If still in warmup phase, return original data
        if self.warmup_steps > 0 and self.current_warmup_step_count < self.warmup_steps:
            return x

        # Otherwise, proceed with normalization
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

        # Ensure tensor is at least 2D (num_samples, num_features)
        if tensor.ndim == 0: 
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1: 
            tensor = tensor.unsqueeze(-1)
        
        num_features = tensor.shape[-1]

        if key not in self.stats:
            mean_ = torch.zeros(num_features, device=self.device, dtype=tensor.dtype)
            var_  = torch.ones(num_features, device=self.device, dtype=tensor.dtype)
            count_= torch.full((1,), self.epsilon, device=self.device, dtype=torch.float64) # Use float for count
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

        total_count_prev = count
        total_count_new = total_count_prev + batch_n # batch_n is an int, count is tensor
        
        delta = batch_mean - mean
        # Ensure batch_n is float for division if total_count_new is float
        new_mean = mean + delta * (float(batch_n) / total_count_new.clamp(min=1e-6))
        
        delta2 = batch_mean - new_mean 
        
        # Welford's M2 update: M2_new = M2_old + delta * delta_prime * batch_n
        # M2 = var * (count - 1) for unbiased, or var * count for biased.
        # Let's use the direct update for combined variance:
        # var_new = (count_old * var_old + count_batch * var_batch + delta^2 * count_old * count_batch / count_new) / count_new
        m_a = var * total_count_prev # Using total_count_prev (which is a tensor)
        m_b = batch_var * float(batch_n)
        
        new_var = (m_a + m_b + torch.square(delta) * (total_count_prev * float(batch_n)) / total_count_new.clamp(min=1e-6)) / total_count_new.clamp(min=1e-6)
        new_var = torch.clamp(new_var, min=0.0)

        self.stats[key] = (new_mean, new_var, total_count_new)


    def _normalize_single(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.stats: # Should not happen if normalize is called after warmup
            print(f"RunningMeanStdNormalizer Warning: normalize called for key '{key}' but no stats found. Returning original tensor.")
            return tensor 

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

        return (tensor_on_device - mean_reshaped) / torch.sqrt(var_reshaped + 1e-8)
