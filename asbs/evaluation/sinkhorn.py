"""Sinkhorn distance for evaluation."""

import torch
import numpy as np
import ot


def sinkhorn_distance(
    samples_gen: torch.Tensor,
    samples_ref: torch.Tensor,
    reg: float = 0.1,
) -> float:
    """
    Compute entropy-regularized Sinkhorn distance.

    Args:
        samples_gen: [N, dim]
        samples_ref: [M, dim]
        reg: regularization parameter

    Returns:
        Sinkhorn distance (scalar)
    """
    x = samples_gen.detach().cpu().numpy()
    y = samples_ref.detach().cpu().numpy()

    n, m = len(x), len(y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    M = ot.dist(x, y, metric="sqeuclidean")
    sk = ot.sinkhorn2(a, b, M, reg)

    return float(np.sqrt(max(sk, 0.0)))
