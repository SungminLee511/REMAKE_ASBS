"""
ASBS Evaluation Script.

Usage:
    python eval.py --experiment demo_2d --n_samples 2048
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import CONFIGS, build_energy, build_prior
from asbs.sde.noise_schedule import GeometricNoiseSchedule
from asbs.sde.integrator import euler_maruyama_forward
from asbs.models.mlp import TimeDependentMLP, CorrectorMLP
from asbs.models.egnn import EGNN, CorrectorEGNN
from asbs.evaluation.wasserstein import wasserstein_2, energy_wasserstein_2


def load_model(cfg, save_dir, device):
    """Load trained drift model."""
    x_dim = cfg["x_dim"]
    model_type = cfg.get("model", "mlp")

    if model_type == "egnn":
        drift_model = EGNN(
            n_particles=cfg["n_particles"],
            coord_dim=cfg["spatial_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )
    else:
        drift_model = TimeDependentMLP(
            x_dim=x_dim,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )

    drift_path = os.path.join(save_dir, "drift_model.pt")
    if os.path.exists(drift_path):
        drift_model.load_state_dict(torch.load(drift_path, map_location=device))
    else:
        print(f"WARNING: No checkpoint found at {drift_path}")

    return drift_model.to(device)


def generate_samples(drift_model, prior, energy_fn, noise_schedule, n_samples, n_steps, device):
    """Generate samples from trained model."""
    drift_model.eval()
    with torch.no_grad():
        x0 = prior.sample(n_samples, energy_fn.dim, device)
        x1 = euler_maruyama_forward(
            drift_fn=drift_model,
            x0=x0,
            noise_schedule=noise_schedule,
            n_steps=n_steps,
        )
    return x1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--eval_steps", type=int, default=500, help="SDE steps for eval (more=better)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference samples .pt")
    args = parser.parse_args()

    if args.experiment not in CONFIGS:
        print(f"Available: {list(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[args.experiment]
    device = args.device if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join("results", args.experiment)

    # Build components
    energy_fn = build_energy(cfg, device)
    prior = build_prior(cfg)
    noise_schedule = GeometricNoiseSchedule(
        sigma_min=cfg["sigma_min"],
        sigma_max=cfg["sigma_max"],
    )

    # Load model
    drift_model = load_model(cfg, save_dir, device)

    # Generate samples
    print(f"Generating {args.n_samples} samples with {args.eval_steps} SDE steps...")
    samples = generate_samples(
        drift_model, prior, energy_fn, noise_schedule,
        args.n_samples, args.eval_steps, device,
    )

    # Energy statistics
    with torch.no_grad():
        energies = energy_fn.energy(samples)
    print(f"\nEnergy statistics:")
    print(f"  Mean: {energies.mean().item():.4f}")
    print(f"  Std:  {energies.std().item():.4f}")
    print(f"  Min:  {energies.min().item():.4f}")
    print(f"  Max:  {energies.max().item():.4f}")

    # W2 against reference if provided
    if args.reference:
        ref_samples = torch.load(args.reference, map_location=device)
        w2 = wasserstein_2(samples, ref_samples)
        ew2 = energy_wasserstein_2(samples, ref_samples, energy_fn)
        print(f"\nW2 distance:  {w2:.4f}")
        print(f"E-W2 distance: {ew2:.4f}")

    # Save samples
    out_path = os.path.join(save_dir, "eval_samples.pt")
    torch.save(samples.cpu(), out_path)
    print(f"\nSamples saved to {out_path}")

    # Energy histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energies.cpu().numpy(), bins=100, density=True, alpha=0.7, label="Generated")
    if args.reference:
        with torch.no_grad():
            ref_e = energy_fn.energy(ref_samples).cpu().numpy()
        ax.hist(ref_e, bins=100, density=True, alpha=0.5, label="Reference")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title(f"{args.experiment} — Energy Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eval_energy_hist.png"), dpi=150)
    plt.close()
    print(f"Energy histogram saved to {save_dir}/eval_energy_hist.png")


if __name__ == "__main__":
    main()
