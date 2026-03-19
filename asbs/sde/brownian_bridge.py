"""Brownian bridge sampling and drift target computation."""

import torch


def sample_brownian_bridge(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    noise_schedule,
) -> torch.Tensor:
    """
    Sample X_t from the Brownian bridge p^base(X_t | X_0, X_1).

    For f_t = 0 base (dX = σ_t dW_t):

    X_t | (X_0, X_1) ~ N(mean_t, var_t * I)

    where:
        mean_t = X_0 + (σ̄²_t / σ̄²_1) * (X_1 - X_0)
        var_t  = σ̄²_t * (1 - σ̄²_t / σ̄²_1)
             = σ̄²_t * (σ̄²_1 - σ̄²_t) / σ̄²_1

    Args:
        x0: [batch, dim] start points
        x1: [batch, dim] end points
        t: [batch] times in (0, 1)
        noise_schedule: GeometricNoiseSchedule instance

    Returns:
        x_t: [batch, dim]
    """
    device = x0.device

    # Cumulative variances
    cv_t = noise_schedule.cumulative_variance(t).to(device)  # [batch]
    cv_1 = noise_schedule.bar_sigma_1_sq  # scalar

    # Expand for broadcasting
    cv_t = cv_t.unsqueeze(-1)  # [batch, 1]

    # Bridge mean and variance
    ratio = cv_t / cv_1  # [batch, 1]
    mean = x0 + ratio * (x1 - x0)  # [batch, dim]
    var = cv_t * (1.0 - ratio)  # [batch, 1]
    std = var.clamp(min=1e-10).sqrt()  # [batch, 1]

    # Sample
    eps = torch.randn_like(x0)
    x_t = mean + std * eps

    return x_t


def bridge_drift_target(
    x_t: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    noise_schedule,
) -> torch.Tensor:
    """
    Compute ∇_{X_t} log p^base(X_1 | X_t) = (X_1 - X_t) / (σ̄²_1 - σ̄²_t)

    This is the "pointing toward X_1" term in the AM regression target.

    Args:
        x_t: [batch, dim]
        x1: [batch, dim]
        t: [batch]
        noise_schedule: GeometricNoiseSchedule instance

    Returns:
        drift_target: [batch, dim]
    """
    device = x_t.device
    cv_t = noise_schedule.cumulative_variance(t).to(device)  # [batch]
    cv_1 = noise_schedule.bar_sigma_1_sq  # scalar

    denom = (cv_1 - cv_t).unsqueeze(-1)  # [batch, 1]
    denom = denom.clamp(min=1e-10)

    return (x1 - x_t) / denom
