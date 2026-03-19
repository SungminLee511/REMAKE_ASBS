"""Gaussian prior."""

import torch


class GaussianPrior:
    """X_0 ~ N(0, std^2 * I)."""

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, dim: int, device: str = "cpu") -> torch.Tensor:
        return torch.randn(batch_size, dim, device=device) * self.std
