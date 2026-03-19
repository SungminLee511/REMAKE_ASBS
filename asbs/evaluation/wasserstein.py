"""Wasserstein-2 and Energy-Wasserstein evaluation metrics."""

import torch
import numpy as np
import ot


def wasserstein_2(samples_gen: torch.Tensor, samples_ref: torch.Tensor) -> float:
    """
    Compute W2 distance between generated and reference samples.
    Uses POT library.

    Args:
        samples_gen: [N, dim] generated samples
        samples_ref: [M, dim] reference samples

    Returns:
        W2 distance (scalar)
    """
    x = samples_gen.detach().cpu().numpy()
    y = samples_ref.detach().cpu().numpy()

    n, m = len(x), len(y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    M = ot.dist(x, y, metric="sqeuclidean")
    w2_sq = ot.emd2(a, b, M)

    return float(np.sqrt(max(w2_sq, 0.0)))


def energy_wasserstein_2(
    samples_gen: torch.Tensor,
    samples_ref: torch.Tensor,
    energy_fn,
) -> float:
    """
    Compute Energy-W2: W2 on energy values rather than positions.

    Args:
        samples_gen: [N, dim]
        samples_ref: [M, dim]
        energy_fn: energy function with .energy() method

    Returns:
        E-W2 distance (scalar)
    """
    with torch.no_grad():
        e_gen = energy_fn.energy(samples_gen).cpu().numpy().reshape(-1, 1)
        e_ref = energy_fn.energy(samples_ref).cpu().numpy().reshape(-1, 1)

    n, m = len(e_gen), len(e_ref)
    a = np.ones(n) / n
    b = np.ones(m) / m

    M = ot.dist(e_gen, e_ref, metric="sqeuclidean")
    w2_sq = ot.emd2(a, b, M)

    return float(np.sqrt(max(w2_sq, 0.0)))
