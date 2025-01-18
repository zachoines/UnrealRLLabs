import torch
import numpy as np

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

    def value(self):
        return self.current_value

    def reset(self):
        self.t = 0

class ModifiedOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
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
    def __init__(self, epsilon: float = 1e-4, device: torch.device = torch.device("cpu")):
        self.mean = None
        self.var = None
        self.count = epsilon
        self.device = device

    def update(self, x: torch.Tensor):
        if self.mean is None or self.var is None:
            self.mean = torch.zeros(x.shape[-1]).to(self.device)
            self.var = torch.ones(x.shape[-1]).to(self.device)

        # Flatten the batch dimensions
        flat_x = x.view(-1, x.shape[-1])
        batch_mean = torch.mean(flat_x, dim=0)
        batch_var = torch.var(flat_x, dim=0)
        batch_count = flat_x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor):
        normalized_x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return normalized_x