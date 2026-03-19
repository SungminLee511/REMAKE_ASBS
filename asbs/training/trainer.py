"""Main ASBS training loop implementing Algorithm 1."""

import torch
import torch.nn as nn
from tqdm import tqdm

from ..sde.integrator import euler_maruyama_forward
from ..sde.brownian_bridge import sample_brownian_bridge, bridge_drift_target
from ..buffers.replay_buffer import ReplayBuffer


class ASBSTrainer:
    """
    Adjoint Schrödinger Bridge Sampler trainer.

    Implements Algorithm 1: alternating AM/CM optimization.
    """

    def __init__(
        self,
        energy_fn,
        drift_model: nn.Module,
        corrector_model: nn.Module,
        prior,
        noise_schedule,
        device: str = "cuda",
        # SDE params
        n_sde_steps: int = 100,
        # Training params
        n_stages: int = 20,
        am_steps: int = 500,
        cm_steps: int = 200,
        batch_size: int = 512,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
        # Buffer params
        buffer_max_size: int = 10000,
        buffer_min_size: int = 256,
        # Logging
        log_every: int = 100,
        plot_fn=None,
    ):
        self.energy_fn = energy_fn
        self.drift = drift_model.to(device)
        self.corrector = corrector_model.to(device)
        self.prior = prior
        self.ns = noise_schedule
        self.device = device

        self.n_sde_steps = n_sde_steps
        self.n_stages = n_stages
        self.am_steps = am_steps
        self.cm_steps = cm_steps
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.plot_fn = plot_fn

        self.replay_buffer = ReplayBuffer(buffer_max_size, energy_fn.dim, device)
        self.buffer_min_size = buffer_min_size

        self.drift_optimizer = torch.optim.Adam(self.drift.parameters(), lr=lr)
        self.corrector_optimizer = torch.optim.Adam(self.corrector.parameters(), lr=lr)

    @torch.no_grad()
    def generate_trajectories(self, batch_size: int):
        """Sample X_0 ~ μ, integrate SDE forward to get X_1. Stop-gradient."""
        x0 = self.prior.sample(batch_size, self.energy_fn.dim, self.device)
        self.drift.eval()
        x1 = euler_maruyama_forward(
            drift_fn=self.drift,
            x0=x0,
            noise_schedule=self.ns,
            n_steps=self.n_sde_steps,
        )
        self.drift.train()
        return x0, x1

    def am_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Adjoint Matching loss (Eq. 14).

        target = bridge_drift(x_t→x_1) - σ_t * (∇E(x_1) + h(x_1))
        pred = u_θ(t, x_t)
        loss = ||pred - target||²
        """
        batch_size = x0.shape[0]

        # Random time, avoid boundaries
        t = torch.rand(batch_size, device=self.device) * 0.998 + 0.001

        # Sample from Brownian bridge
        x_t = sample_brownian_bridge(x0, x1, t, self.ns)

        # Energy gradient at X_1 (detached — part of target, not prediction)
        grad_e = self.energy_fn.grad_energy(x1)

        # Corrector at X_1 (no grad through corrector during AM)
        with torch.no_grad():
            h = self.corrector(x1)

        # Bridge drift: points from X_t toward X_1
        bridge = bridge_drift_target(x_t, x1, t, self.ns)

        # σ_t for scaling
        sigma_t = self.ns.sigma(t).unsqueeze(-1)  # [batch, 1]

        # Full regression target
        target = bridge - sigma_t * (grad_e + h)

        # Model prediction
        pred = self.drift(t, x_t)

        loss = (pred - target).pow(2).mean()
        return loss

    def cm_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Corrector Matching loss (Eq. 15).

        target = -(X_1 - X_0) / σ̄²_1
        pred = h_ϕ(X_1)
        loss = ||pred - target||²
        """
        bar_sigma_1_sq = self.ns.bar_sigma_1_sq
        target = -(x1 - x0) / bar_sigma_1_sq
        pred = self.corrector(x1)
        loss = (pred - target).pow(2).mean()
        return loss

    def warmup_buffer(self):
        """Fill buffer to minimum size before training."""
        while len(self.replay_buffer) < self.buffer_min_size:
            x0, x1 = self.generate_trajectories(self.batch_size)
            self.replay_buffer.add(x0, x1)

    def train(self):
        """Main loop implementing Algorithm 1."""
        print("Warming up buffer...")
        self.warmup_buffer()

        history = {"am_loss": [], "cm_loss": [], "stage": []}

        for stage in range(1, self.n_stages + 1):
            print(f"\n{'='*60}")
            print(f"Stage {stage}/{self.n_stages}")
            print(f"{'='*60}")

            # === Generate fresh trajectories ===
            x0, x1 = self.generate_trajectories(self.batch_size)
            self.replay_buffer.add(x0, x1)

            # === ADJOINT MATCHING ===
            print(f"  AM phase ({self.am_steps} steps)...")
            self.drift.train()
            am_losses = []
            for step in range(1, self.am_steps + 1):
                x0_b, x1_b = self.replay_buffer.sample(self.batch_size)
                loss = self.am_loss(x0_b, x1_b)

                self.drift_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.drift.parameters(), self.grad_clip)
                self.drift_optimizer.step()

                am_losses.append(loss.item())
                if step % self.log_every == 0:
                    avg = sum(am_losses[-self.log_every:]) / self.log_every
                    print(f"    AM step {step}/{self.am_steps}, loss={avg:.6f}")

            avg_am = sum(am_losses) / len(am_losses)
            history["am_loss"].append(avg_am)

            # === Generate new trajectories with updated drift ===
            x0, x1 = self.generate_trajectories(self.batch_size)
            self.replay_buffer.add(x0, x1)

            # === CORRECTOR MATCHING ===
            print(f"  CM phase ({self.cm_steps} steps)...")
            self.corrector.train()
            cm_losses = []
            for step in range(1, self.cm_steps + 1):
                x0_b, x1_b = self.replay_buffer.sample(self.batch_size)
                loss = self.cm_loss(x0_b, x1_b)

                self.corrector_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.corrector.parameters(), self.grad_clip)
                self.corrector_optimizer.step()

                cm_losses.append(loss.item())
                if step % self.log_every == 0:
                    avg = sum(cm_losses[-self.log_every:]) / self.log_every
                    print(f"    CM step {step}/{self.cm_steps}, loss={avg:.6f}")

            avg_cm = sum(cm_losses) / len(cm_losses)
            history["cm_loss"].append(avg_cm)
            history["stage"].append(stage)

            print(f"  Stage {stage} summary: AM_loss={avg_am:.6f}, CM_loss={avg_cm:.6f}")

            # Plot if callback provided
            if self.plot_fn is not None:
                samples = self.generate_samples(1024)
                self.plot_fn(samples, stage)

        return history

    @torch.no_grad()
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the trained model."""
        self.drift.eval()
        x0 = self.prior.sample(n_samples, self.energy_fn.dim, self.device)
        x1 = euler_maruyama_forward(
            drift_fn=self.drift,
            x0=x0,
            noise_schedule=self.ns,
            n_steps=self.n_sde_steps,
        )
        self.drift.train()
        return x1
