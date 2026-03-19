"""Double-Well (DW-4) energy: 4 particles in 2D."""

import torch
from .base import EnergyFunction


class DoubleWellEnergy(EnergyFunction):
    """
    DW-4: 4 particles in 2D (d=8).
    Pairwise double-well potential: V(r) = a*(r^2 - r0^2)^2 + b*r^2
    """

    def __init__(self, a: float = 0.9, b: float = -4.0, r0: float = 1.5, device="cpu"):
        self.a = a
        self.b = b
        self.r0 = r0
        self.n_particles = 4
        self.spatial_dim = 2
        self.device = device

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim  # 8

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 8] or [batch, 4, 2]
        returns: [batch]
        """
        if x.dim() == 2 and x.shape[-1] == self.dim:
            x = x.view(-1, self.n_particles, self.spatial_dim)

        # Pairwise distances
        # x: [batch, 4, 2]
        diffs = x.unsqueeze(2) - x.unsqueeze(1)  # [batch, 4, 4, 2]
        dists = diffs.pow(2).sum(-1).sqrt()  # [batch, 4, 4]

        # Upper triangular (unique pairs)
        mask = torch.triu(torch.ones(4, 4, device=x.device), diagonal=1).bool()
        r = dists[:, mask]  # [batch, 6]

        # Double-well potential
        V = self.a * (r.pow(2) - self.r0 ** 2).pow(2) + self.b * r.pow(2)

        return V.sum(-1)  # [batch]
