"""
ASBS Training Entry Point.
Simple config-dict based (no Hydra for now).

Usage:
    python train.py --experiment demo_2d
    python train.py --experiment dw4_asbs
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from asbs.sde.noise_schedule import GeometricNoiseSchedule
from asbs.models.mlp import TimeDependentMLP, CorrectorMLP
from asbs.priors.gaussian import GaussianPrior
from asbs.priors.dirac import DiracPrior
from asbs.priors.harmonic import HarmonicPrior
from asbs.energies.gaussian_mixture import GaussianMixture2D
from asbs.energies.double_well import DoubleWellEnergy
from asbs.energies.lennard_jones import LennardJonesEnergy
from asbs.models.egnn import EGNN, CorrectorEGNN
from asbs.training.trainer import ASBSTrainer


# ============================================================
# Experiment Configs
# ============================================================

CONFIGS = {
    "demo_2d": dict(
        energy="gaussian_mixture_2d",
        prior="gaussian",
        prior_std=3.0,
        model="mlp",
        x_dim=2,
        hidden_dim=128,
        n_layers=3,
        sigma_min=0.01,
        sigma_max=1.0,
        n_sde_steps=50,
        n_stages=20,
        am_steps=500,
        cm_steps=200,
        batch_size=512,
        lr=1e-3,
        buffer_max_size=10000,
        seed=0,
    ),
    "demo_2d_as": dict(
        energy="gaussian_mixture_2d",
        prior="dirac",
        model="mlp",
        x_dim=2,
        hidden_dim=128,
        n_layers=3,
        sigma_min=0.01,
        sigma_max=1.0,
        n_sde_steps=50,
        n_stages=20,
        am_steps=500,
        cm_steps=200,
        batch_size=512,
        lr=1e-3,
        buffer_max_size=10000,
        seed=0,
    ),
    "dw4_asbs": dict(
        energy="double_well",
        prior="harmonic",
        n_particles=4,
        spatial_dim=2,
        model="egnn",
        x_dim=8,
        hidden_dim=128,
        n_layers=4,
        sigma_min=0.01,
        sigma_max=1.0,
        n_sde_steps=100,
        n_stages=10,
        am_steps=5000,
        cm_steps=2000,
        batch_size=256,
        lr=1e-3,
        buffer_max_size=10000,
        seed=0,
    ),
    "dw4_as": dict(
        energy="double_well",
        prior="dirac",
        n_particles=4,
        spatial_dim=2,
        model="egnn",
        x_dim=8,
        hidden_dim=128,
        n_layers=4,
        sigma_min=0.01,
        sigma_max=1.0,
        n_sde_steps=100,
        n_stages=10,
        am_steps=5000,
        cm_steps=2000,
        batch_size=256,
        lr=1e-3,
        buffer_max_size=10000,
        seed=0,
    ),
    "lj13_asbs": dict(
        energy="lennard_jones",
        prior="harmonic",
        n_particles=13,
        spatial_dim=3,
        model="egnn",
        x_dim=39,
        hidden_dim=128,
        n_layers=4,
        sigma_min=0.01,
        sigma_max=1.0,
        n_sde_steps=100,
        n_stages=10,
        am_steps=5000,
        cm_steps=2000,
        batch_size=256,
        lr=1e-3,
        buffer_max_size=10000,
        seed=0,
    ),
    "lj55_asbs": dict(
        energy="lennard_jones",
        prior="harmonic",
        n_particles=55,
        spatial_dim=3,
        model="egnn",
        x_dim=165,
        hidden_dim=128,
        n_layers=4,
        sigma_min=0.01,
        sigma_max=1.0,
        n_sde_steps=100,
        n_stages=10,
        am_steps=5000,
        cm_steps=2000,
        batch_size=256,
        lr=1e-3,
        buffer_max_size=10000,
        seed=0,
    ),
}


def build_energy(cfg, device):
    name = cfg["energy"]
    if name == "gaussian_mixture_2d":
        return GaussianMixture2D(device=device)
    elif name == "double_well":
        return DoubleWellEnergy(device=device)
    elif name == "lennard_jones":
        n = cfg.get("n_particles", 13)
        return LennardJonesEnergy(n_particles=n, device=device)
    else:
        raise ValueError(f"Unknown energy: {name}")


def build_prior(cfg):
    name = cfg["prior"]
    if name == "gaussian":
        return GaussianPrior(std=cfg.get("prior_std", 1.0))
    elif name == "dirac":
        return DiracPrior()
    elif name == "harmonic":
        return HarmonicPrior(
            n_particles=cfg["n_particles"],
            spatial_dim=cfg["spatial_dim"],
        )
    else:
        raise ValueError(f"Unknown prior: {name}")


def plot_2d_callback(save_dir):
    """Create a plotting callback for 2D experiments."""
    os.makedirs(save_dir, exist_ok=True)

    def plot_fn(samples, stage):
        samples_np = samples.cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.5)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(f"Stage {stage}")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"stage_{stage:03d}.png"), dpi=150)
        plt.close()

    return plot_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="demo_2d")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.experiment not in CONFIGS:
        print(f"Available experiments: {list(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[args.experiment].copy()
    if args.seed is not None:
        cfg["seed"] = args.seed

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Seed
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build components
    energy_fn = build_energy(cfg, device)
    prior = build_prior(cfg)
    noise_schedule = GeometricNoiseSchedule(
        sigma_min=cfg["sigma_min"],
        sigma_max=cfg["sigma_max"],
    )

    x_dim = cfg["x_dim"]
    model_type = cfg.get("model", "mlp")

    if model_type == "egnn":
        n_particles = cfg["n_particles"]
        spatial_dim = cfg["spatial_dim"]
        drift_model = EGNN(
            n_particles=n_particles,
            coord_dim=spatial_dim,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )
        corrector_model = CorrectorEGNN(
            n_particles=n_particles,
            coord_dim=spatial_dim,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )
    else:
        drift_model = TimeDependentMLP(
            x_dim=x_dim,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )
        corrector_model = CorrectorMLP(
            x_dim=x_dim,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )

    # Plot callback for 2D
    plot_fn = None
    save_dir = os.path.join("results", args.experiment)
    if x_dim == 2:
        plot_fn = plot_2d_callback(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Trainer
    trainer = ASBSTrainer(
        energy_fn=energy_fn,
        drift_model=drift_model,
        corrector_model=corrector_model,
        prior=prior,
        noise_schedule=noise_schedule,
        device=device,
        n_sde_steps=cfg["n_sde_steps"],
        n_stages=cfg["n_stages"],
        am_steps=cfg["am_steps"],
        cm_steps=cfg["cm_steps"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        buffer_max_size=cfg["buffer_max_size"],
        plot_fn=plot_fn,
    )

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Energy: {cfg['energy']}, Prior: {cfg['prior']}")
    print(f"Dim: {x_dim}, Stages: {cfg['n_stages']}")
    print(f"AM steps/stage: {cfg['am_steps']}, CM steps/stage: {cfg['cm_steps']}")
    print(f"{'='*60}\n")

    # Train
    history = trainer.train()

    # Save final samples + model
    print("\nGenerating final samples...")
    final_samples = trainer.generate_samples(2048)
    torch.save(final_samples.cpu(), os.path.join(save_dir, "final_samples.pt"))
    torch.save(drift_model.state_dict(), os.path.join(save_dir, "drift_model.pt"))
    torch.save(corrector_model.state_dict(), os.path.join(save_dir, "corrector_model.pt"))

    # Final plot for 2D
    if x_dim == 2:
        samples_np = final_samples.cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Energy landscape + samples
        xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
        grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32, device=device)
        with torch.no_grad():
            E = energy_fn.energy(grid).cpu().numpy().reshape(200, 200)
        axes[0].contourf(xx, yy, E, levels=50, cmap="viridis")
        axes[0].scatter(samples_np[:, 0], samples_np[:, 1], s=1, c="red", alpha=0.3)
        axes[0].set_title("Energy + Samples")
        axes[0].set_aspect("equal")

        # 2. Sample density
        axes[1].hist2d(samples_np[:, 0], samples_np[:, 1], bins=100, range=[[-6, 6], [-6, 6]], cmap="hot")
        axes[1].set_title("Sample Density")
        axes[1].set_aspect("equal")

        # 3. Training curves
        stages = history["stage"]
        axes[2].plot(stages, history["am_loss"], "b-o", label="AM loss")
        axes[2].plot(stages, history["cm_loss"], "r-o", label="CM loss")
        axes[2].set_xlabel("Stage")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Training Curves")
        axes[2].legend()
        axes[2].set_yscale("log")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "final_summary.png"), dpi=200)
        plt.close()
        print(f"Final summary saved to {save_dir}/final_summary.png")

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
