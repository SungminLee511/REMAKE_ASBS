"""Geometric noise schedule for ASBS SDE."""

import math
import torch
import numpy as np
from scipy import integrate


class GeometricNoiseSchedule:
    """
    Geometric noise schedule: σ_t = σ_min^(1-t) · σ_max^t

    Key quantities:
    - sigma(t): diffusion coefficient at time t
    - cumulative_variance(t): σ̄²_t = ∫₀ᵗ σ_τ² dτ
    - bridge_params(t, x0, x1): (mean, std) of Brownian bridge at time t
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0, n_grid: int = 10000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Precompute cumulative variance on a fine grid for interpolation
        self._grid_t = np.linspace(0.0, 1.0, n_grid)
        self._grid_cv = np.zeros(n_grid)
        for i in range(1, n_grid):
            t_val = self._grid_t[i]
            val, _ = integrate.quad(
                lambda tau: (sigma_min ** (1 - tau) * sigma_max ** tau) ** 2,
                0.0, t_val,
            )
            self._grid_cv[i] = val

        # Store as tensors for fast interpolation
        self._grid_t_tensor = torch.tensor(self._grid_t, dtype=torch.float64)
        self._grid_cv_tensor = torch.tensor(self._grid_cv, dtype=torch.float64)

        # Cache bar_sigma_1^2
        self._cv_1 = float(self._grid_cv[-1])

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """σ_t = σ_min^(1-t) · σ_max^t"""
        return self.sigma_min ** (1.0 - t) * self.sigma_max ** t

    def cumulative_variance(self, t) -> torch.Tensor:
        """σ̄²_t = ∫₀ᵗ σ_τ² dτ, via linear interpolation on precomputed grid."""
        if isinstance(t, (int, float)):
            t = torch.tensor([t], dtype=torch.float64)
        was_single = t.dim() == 0
        if was_single:
            t = t.unsqueeze(0)

        t_cpu = t.double().cpu()
        # Linear interpolation
        cv = torch.zeros_like(t_cpu, dtype=torch.float64)
        idx = torch.searchsorted(self._grid_t_tensor, t_cpu.clamp(0.0, 1.0)) - 1
        idx = idx.clamp(0, len(self._grid_t_tensor) - 2)

        t0 = self._grid_t_tensor[idx]
        t1 = self._grid_t_tensor[idx + 1]
        cv0 = self._grid_cv_tensor[idx]
        cv1 = self._grid_cv_tensor[idx + 1]

        frac = (t_cpu - t0) / (t1 - t0 + 1e-30)
        cv = cv0 + frac * (cv1 - cv0)

        result = cv.float()
        if was_single:
            result = result.squeeze(0)
        return result

    def cumulative_variance_scalar(self, t_val: float) -> float:
        """Scalar version for convenience."""
        return float(self.cumulative_variance(torch.tensor(t_val)))

    @property
    def bar_sigma_1_sq(self) -> float:
        """σ̄²_1 = ∫₀¹ σ_τ² dτ"""
        return self._cv_1
