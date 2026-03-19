"""Conformer generation evaluation metrics: Coverage Recall and AMR."""

import torch
import numpy as np


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between two sets of points after Kabsch alignment.

    Args:
        P: [n_atoms, 3] generated conformer
        Q: [n_atoms, 3] reference conformer

    Returns:
        RMSD after optimal alignment
    """
    # Center
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    # Covariance matrix
    H = P.T @ Q

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation
    P_aligned = P @ R.T

    # RMSD
    rmsd = np.sqrt(((P_aligned - Q) ** 2).sum(axis=-1).mean())
    return float(rmsd)


def coverage_recall(
    generated: list,
    reference: list,
    threshold: float = 1.0,
) -> tuple:
    """
    Compute coverage recall and absolute mean RMSD.

    Args:
        generated: list of [n_atoms, 3] np arrays (generated conformers)
        reference: list of [n_atoms, 3] np arrays (reference conformers)
        threshold: RMSD threshold in Angstroms

    Returns:
        (recall, amr): coverage recall fraction, absolute mean RMSD
    """
    n_ref = len(reference)
    covered = 0
    rmsds = []

    for ref_conf in reference:
        min_rmsd = float("inf")
        for gen_conf in generated:
            rmsd = kabsch_rmsd(gen_conf, ref_conf)
            min_rmsd = min(min_rmsd, rmsd)
        rmsds.append(min_rmsd)
        if min_rmsd < threshold:
            covered += 1

    recall = covered / n_ref if n_ref > 0 else 0.0
    amr = np.mean(rmsds) if rmsds else 0.0

    return float(recall), float(amr)
