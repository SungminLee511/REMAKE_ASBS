"""KL divergence for 1D marginal distributions (torsion angles)."""

import torch
import numpy as np


def kl_divergence_1d(
    samples_gen: torch.Tensor,
    samples_ref: torch.Tensor,
    n_bins: int = 100,
    range_min: float = -np.pi,
    range_max: float = np.pi,
    eps: float = 1e-10,
) -> float:
    """
    Compute KL(gen || ref) for 1D distributions via histograms.

    Used for torsion angle marginals in alanine dipeptide evaluation.

    Args:
        samples_gen: [N] generated samples
        samples_ref: [M] reference samples
        n_bins: number of histogram bins
        range_min, range_max: histogram range
        eps: smoothing constant to avoid log(0)

    Returns:
        KL divergence (scalar)
    """
    gen = samples_gen.detach().cpu().numpy()
    ref = samples_ref.detach().cpu().numpy()

    bins = np.linspace(range_min, range_max, n_bins + 1)

    p, _ = np.histogram(gen, bins=bins, density=True)
    q, _ = np.histogram(ref, bins=bins, density=True)

    # Add smoothing
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    kl = np.sum(p * np.log(p / q))
    return float(kl)
