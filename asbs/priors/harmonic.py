"""Harmonic oscillator prior for n-particle systems."""

import math
import torch


class HarmonicPrior:
    """
    Harmonic prior for n-particle systems:
    μ(x) ∝ exp(-α/2 * Σ_{i,j} ||x_i - x_j||^2)

    After centering, equivalent to isotropic Gaussian with
    variance 1/(2*alpha*n) per coordinate.
    """

    def __init__(self, n_particles: int, spatial_dim: int, alpha: float = 1.0):
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.alpha = alpha

    def sample(self, batch_size: int, dim: int = None, device: str = "cpu") -> torch.Tensor:
        std = 1.0 / math.sqrt(2.0 * self.alpha * self.n_particles)
        x = torch.randn(batch_size, self.n_particles, self.spatial_dim, device=device) * std
        # Remove center of mass
        x = x - x.mean(dim=1, keepdim=True)
        # Flatten to [batch, n*d]
        return x.view(batch_size, -1)
