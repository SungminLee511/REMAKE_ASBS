"""Many-Well (MW-5) energy: 5 particles in 1D."""

import torch
from .base import EnergyFunction


class ManyWellEnergy(EnergyFunction):
    """
    MW-5: 5-dimensional multi-modal energy function.
    Based on the SCLD setup (Chen et al., 2025).

    Each coordinate has a double-well potential, plus pairwise coupling.
    E(x) = sum_i V(x_i) + lambda * sum_{i<j} (x_i - x_j)^2

    V(x) = a*(x^2 - 1)^2  (double-well per coordinate)
    """

    def __init__(
        self,
        n_dim: int = 5,
        a: float = 1.0,
        coupling: float = 0.1,
        device: str = "cpu",
    ):
        self.n_dim = n_dim
        self.a = a
        self.coupling = coupling
        self.device = device

    @property
    def dim(self) -> int:
        return self.n_dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 5]
        returns: [batch]
        """
        # Single-particle double-well
        V = self.a * (x.pow(2) - 1.0).pow(2)  # [batch, 5]
        E = V.sum(dim=-1)  # [batch]

        # Pairwise coupling
        if self.coupling > 0:
            # sum_{i<j} (x_i - x_j)^2
            diffs = x.unsqueeze(-1) - x.unsqueeze(-2)  # [batch, 5, 5]
            mask = torch.triu(torch.ones(self.n_dim, self.n_dim, device=x.device), diagonal=1).bool()
            pair_dist_sq = diffs.pow(2)[:, mask]  # [batch, n_pairs]
            E = E + self.coupling * pair_dist_sq.sum(dim=-1)

        return E
