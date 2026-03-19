"""Lennard-Jones energy for n-particle systems."""

import torch
from .base import EnergyFunction


class LennardJonesEnergy(EnergyFunction):
    """
    Standard Lennard-Jones potential for n particles in 3D:
    E(x) = sum_{i<j} 4*eps * [(sigma/r_ij)^12 - (sigma/r_ij)^6]
    """

    def __init__(
        self,
        n_particles: int = 13,
        spatial_dim: int = 3,
        epsilon: float = 1.0,
        sigma_lj: float = 1.0,
        device: str = "cpu",
    ):
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.epsilon = epsilon
        self.sigma_lj = sigma_lj
        self.device = device

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n*3] or [batch, n, 3]
        returns: [batch]
        """
        if x.dim() == 2 and x.shape[-1] == self.dim:
            x = x.view(-1, self.n_particles, self.spatial_dim)

        # Pairwise distances
        diffs = x.unsqueeze(2) - x.unsqueeze(1)  # [batch, n, n, 3]
        r2 = diffs.pow(2).sum(-1)  # [batch, n, n]

        # Upper triangular mask
        n = self.n_particles
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()

        r2_pairs = r2[:, mask]  # [batch, n_pairs]
        r2_pairs = r2_pairs.clamp(min=1e-6)  # stability

        sr2 = (self.sigma_lj ** 2) / r2_pairs  # (sigma/r)^2
        sr6 = sr2.pow(3)
        sr12 = sr6.pow(2)

        V = 4.0 * self.epsilon * (sr12 - sr6)
        return V.sum(-1)  # [batch]
