"""Dirac delta prior: X_0 = 0 (for AS special case)."""

import torch


class DiracPrior:
    """X_0 = 0. Used for the AS special case (memoryless)."""

    def sample(self, batch_size: int, dim: int, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, dim, device=device)
