"""2D Gaussian Mixture energy for demo/testing."""

import torch
from .base import EnergyFunction


class GaussianMixture2D(EnergyFunction):
    """
    Simple 2D Gaussian mixture for the demo.
    E(x) = -log sum_k exp(-||x - mu_k||^2 / (2 * std^2))
    """

    def __init__(self, device="cpu"):
        self.means = torch.tensor(
            [[-3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, -3.0]],
            dtype=torch.float32,
            device=device,
        )
        self.std = 0.5
        self.device = device

    @property
    def dim(self) -> int:
        return 2

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 2] -> [batch]"""
        # Move means to same device if needed
        means = self.means.to(x.device)
        diffs = x.unsqueeze(1) - means.unsqueeze(0)  # [batch, 4, 2]
        log_probs = -0.5 * (diffs / self.std).pow(2).sum(-1)  # [batch, 4]
        return -torch.logsumexp(log_probs, dim=-1)  # [batch]
