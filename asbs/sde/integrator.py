"""Euler-Maruyama SDE integrator."""

import torch
from typing import Optional, Tuple, Union


def euler_maruyama_forward(
    drift_fn,
    x0: torch.Tensor,
    noise_schedule,
    n_steps: int,
    return_trajectory: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Integrate: dX_t = σ_t · u_θ(t, X_t) dt + σ_t dW_t

    Base drift f_t = 0 (Brownian motion base).
    The learned drift u_θ is MULTIPLIED by σ_t in the SDE.

    Args:
        drift_fn: callable(t, x) -> drift [batch, dim]
        x0: initial positions [batch, dim]
        noise_schedule: GeometricNoiseSchedule instance
        n_steps: number of Euler-Maruyama steps
        return_trajectory: if True, returns full trajectory

    Returns:
        x1 (and optionally the full trajectory tensor [n_steps+1, batch, dim])
    """
    dt = 1.0 / n_steps
    x = x0.clone()
    device = x0.device

    if return_trajectory:
        trajectory = [x.clone()]

    for n in range(n_steps):
        t_n = n * dt
        t_tensor = torch.full((x.shape[0],), t_n, device=device)

        sigma_t = noise_schedule.sigma(t_tensor)  # [batch]

        # Drift from model
        u = drift_fn(t_tensor, x)  # [batch, dim]

        # Noise
        eps = torch.randn_like(x)

        # Euler-Maruyama step
        # dX = σ_t * u_θ * dt + σ_t * sqrt(dt) * ε
        sigma_t_expanded = sigma_t.unsqueeze(-1)  # [batch, 1]
        x = x + sigma_t_expanded * u * dt + sigma_t_expanded * (dt ** 0.5) * eps

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=0)
    return x
