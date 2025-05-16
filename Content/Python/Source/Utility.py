import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict

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

    Each update() call does a standard running mean/variance calculation on the *flattened*
    data except for the last dimension. So for shape (B,T,..., features), we gather
    B*T*... samples across each feature dimension.
    """

    def __init__(self, warmup_steps: int = 0, epsilon: float = 1e-4, device: torch.device = torch.device("cpu")):
        """
        :param epsilon: initial count to avoid divide-by-zero
        :param device: store mean/var on this device
        """
        self.device = device
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # stats[key] = (mean: Tensor, var: Tensor, count: float)
        # single Tensors store under key="__single__"
        self.stats = {}

    def update(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Incorporate new data into the running mean+var.
        Usually, you'd pass single states from get_states() (which might be shape (1,NumEnv,features) or so).
        
        If 'x' is a dict => we handle each key's last dimension separately.
        If 'x' is a single Tensor => store in self.stats["__single__"].
        """
        # Stop calling update if warmup steps have been exceeded
        if self.warmup_steps > 0:
            self.current_step += 1
            skip_update = self.warmup_steps < self.current_step
            if skip_update:
                return

        if isinstance(x, torch.Tensor):
            self._update_single(x, key="__single__")
        elif isinstance(x, dict):
            for k, v in x.items():
                self._update_single(v, key=k)
        else:
            raise ValueError("Unsupported type for update: must be Tensor or dict of Tensors.")

    def normalize(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
                 ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns a normalized version of 'x' using the stored stats for each key.
        - If 'x' is single => use stats["__single__"] if it exists.
        - If 'x' is dict => for each k => stats[k].
        We flatten all but last dim, subtract mean, divide sqrt(var+1e-8).
        """
        if isinstance(x, torch.Tensor):
            return self._normalize_single(x, key="__single__")
        elif isinstance(x, dict):
            out = {}
            for k,v in x.items():
                out[k] = self._normalize_single(v, key=k)
            return out
        else:
            raise ValueError("Unsupported type for normalize: must be Tensor or dict of Tensors.")

    # ---------------------------------------------------------------------
    #                          Internal Methods
    # ---------------------------------------------------------------------

    def _update_single(self, tensor: torch.Tensor, key: str):
        """
        Flatten all but last dimension => shape(N, features).
        Update running mean & var for this 'key'.
        Made robust to batch_count <= 1 for variance calculation and
        various input tensor dimensionalities.
        """
        tensor = tensor.to(self.device) # Ensure tensor is on the correct device

        # Determine feature dimension and initialize stats if key not found
        if key not in self.stats:
            if tensor.ndim == 0: # Scalar input
                fdim = 1
            elif tensor.ndim == 1: # 1D input (e.g., M samples of a single feature if not unsqueezed yet)
                # If it's meant to be M samples of 1 feature, fdim should be 1.
                # The caller should ideally .unsqueeze(-1) it to (M,1) before passing.
                # However, if it arrives as (M,), we'll assume it's M samples of 1 feature.
                fdim = 1
            else: # N-D input, last dimension is features
                fdim = tensor.shape[-1]
            
            mean_ = torch.zeros(fdim, device=self.device, dtype=tensor.dtype)
            var_  = torch.ones(fdim, device=self.device, dtype=tensor.dtype) # Initialize variance to 1
            count_= torch.full((1,), self.epsilon, device=self.device, dtype=torch.float64) # Use float64 for count
            self.stats[key] = (mean_, var_, count_)

        mean, var, cnt = self.stats[key]

        # Prepare 'flat' tensor with shape (batch_count_val, features)
        if tensor.ndim == 0: # Scalar input
            flat = tensor.view(1, 1) # Treat as 1 sample, 1 feature
        elif tensor.ndim == 1: # 1D input (M,), e.g. from boolean indexing
            # This case is hit if the calling code (e.g. MAPOCAAgent) passes a 1D tensor.
            # Your fix .unsqueeze(-1) in MAPOCAAgent makes the input 2D (M,1), so this branch might be less common now for that specific call.
            flat = tensor.unsqueeze(-1) # Treat as M samples, 1 feature. Shape: (M, 1)
        else: # 2D or higher-D input, e.g. (M, 1) after your unsqueeze, or (N, F)
            # Ensures last dimension is features, all others are flattened into samples.
            flat = tensor.view(-1, tensor.shape[-1]) 
        
        batch_count_val = flat.size(0)
        if batch_count_val == 0:
            return # No new data to update with

        batch_mean = flat.mean(dim=0) # Shape: (features,)
        
        # Calculate batch variance robustly
        if batch_count_val > 1:
            batch_var  = flat.var(dim=0, unbiased=True) # unbiased=True is fine for N > 1
        else: # batch_count_val == 1
            # Variance of a single data point is 0.
            # Using unbiased=False would also give 0. torch.var(unbiased=True) gives NaN for N=1.
            batch_var = torch.zeros_like(batch_mean) # Shape: (features,)

        # Ensure no NaNs from mean/var calculation if flat had NaNs initially (though should be checked before)
        if torch.isnan(batch_mean).any() or torch.isnan(batch_var).any():
            print(f"Warning: NaN detected in batch_mean or batch_var for key '{key}'. Skipping update for this batch.")
            # This can happen if 'flat' itself contained NaNs.
            return

        # Update running statistics using Welford's algorithm or similar stable method
        batch_count_tensor = torch.full_like(cnt, float(batch_count_val)) # Match dtype of cnt (float64)

        delta = batch_mean - mean
        new_count = cnt + batch_count_tensor

        # Update mean
        new_mean = mean + delta * (batch_count_tensor / new_count.clamp(min=1e-6)) # Clamp new_count to avoid div by zero if it became 0
        
        # Update variance: Var_new = (count_old * Var_old + count_batch * Var_batch) / count_new + count_old * count_batch * (Mean_old - Mean_batch)^2 / (count_new^2)
        # This is a more stable way to combine variances:
        # new_var = (cnt * var + batch_count_tensor * batch_var) / new_count + \
        #           (cnt * batch_count_tensor * torch.square(delta)) / (new_count * new_count.clamp(min=1e-6))
        # Simpler variant (Welford's M2 update style, adapted):
        m_a = var * cnt
        m_b = batch_var * batch_count_tensor
        new_var = (m_a + m_b + torch.square(delta) * (cnt * batch_count_tensor) / new_count.clamp(min=1e-6)) / new_count.clamp(min=1e-6)
        
        # Ensure variance is not negative due to floating point issues
        new_var = torch.clamp(new_var, min=0.0)

        self.stats[key] = (new_mean, new_var, new_count)

    def _normalize_single(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalizes 'tensor' (shape => (..., features)) using self.stats[key].
        If stats not exist for 'key', returns tensor as is.
        """
        if key not in self.stats:
            # no stats => return unmodified
            return tensor

        mean, var, cnt = self.stats[key]
        # shape => (features,). broadcast to tensor's last dimension
        bshape = [1]*(tensor.dim()-1) + [tensor.shape[-1]]
        mean_ = mean.view(*bshape)
        var_  = var.view(*bshape)

        # (x - mean)/ sqrt(var + 1e-8)
        return (tensor.to(self.device) - mean_) / torch.sqrt(var_ + 1e-8)