import torch
import numpy as np

class CosineAnnealingScheduler:
    def __init__(self, T, H, L):
        self.T = T
        self.H = H
        self.L = L
        self.t = 0

    def step(self):
        self.t = min(self.t + 1, self.T)

    def value(self):
        """Calculate the current value based on the cosine annealing schedule.
        """
        return self.L + 0.5 * (self.H - self.L) * (1 + np.cos(np.pi * self.t / self.T))

    def reset(self):
        self.t = 0

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