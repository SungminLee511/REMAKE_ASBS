"""Comprehensive test of all ASBS components."""
import sys
sys.path.insert(0, "/home/RESEARCH/REMAKE_ASBS")

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ============================================================
print("=" * 60)
print("TEST 1: Noise Schedule")
print("=" * 60)
from asbs.sde import GeometricNoiseSchedule

ns = GeometricNoiseSchedule(sigma_min=0.01, sigma_max=1.0)
sigma_0 = ns.sigma(torch.tensor(0.0))
sigma_1 = ns.sigma(torch.tensor(1.0))
print(f"  sigma(0) = {sigma_0:.4f} (expect 0.01)")
print(f"  sigma(1) = {sigma_1:.4f} (expect 1.0)")
print(f"  bar_sigma_1^2 = {ns.bar_sigma_1_sq:.6f}")
assert abs(sigma_0 - 0.01) < 1e-6
assert abs(sigma_1 - 1.0) < 1e-6
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("TEST 2: Brownian Bridge")
print("=" * 60)
from asbs.sde import sample_brownian_bridge, bridge_drift_target

B = 1000
x0 = torch.zeros(B, 2, device=device)
x1 = torch.ones(B, 2, device=device)
t = torch.full((B,), 0.5, device=device)
xt = sample_brownian_bridge(x0, x1, t, ns)
# With geometric schedule, cv_0.5/cv_1 << 0.5 (variance concentrated near t=1)
cv_half = ns.cumulative_variance(torch.tensor(0.5)).item()
expected_mean = cv_half / ns.bar_sigma_1_sq  # ratio of cumulative variances
print(f"  Bridge mean at t=0.5: {xt.mean(0).cpu().numpy()} (expect ~{expected_mean:.4f})")
err = (xt.mean(0) - expected_mean).abs().max().item()
assert err < 0.1, f"Bridge mean error too large: {err}"
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("TEST 3: Euler-Maruyama (zero drift)")
print("=" * 60)
from asbs.sde import euler_maruyama_forward

def zero_drift(t, x):
    return torch.zeros_like(x)

x0 = torch.zeros(500, 2, device=device)
x1 = euler_maruyama_forward(zero_drift, x0, ns, n_steps=100)
var = x1.var(0).mean().item()
print(f"  Variance at t=1 (zero drift): {var:.4f} (expect ~{ns.bar_sigma_1_sq:.4f})")
assert abs(var - ns.bar_sigma_1_sq) < 0.05
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("TEST 4: Energy Functions")
print("=" * 60)
from asbs.energies import GaussianMixture2D, DoubleWellEnergy, LennardJonesEnergy, ManyWellEnergy

gm = GaussianMixture2D(device=device)
x = torch.randn(16, 2, device=device)
e = gm.energy(x)
g = gm.grad_energy(x)
print(f"  GMM2D: energy shape={e.shape}, grad shape={g.shape}")
assert e.shape == (16,) and g.shape == (16, 2)

dw = DoubleWellEnergy(device=device)
x = torch.randn(16, 8, device=device)
e = dw.energy(x)
g = dw.grad_energy(x)
print(f"  DW-4:  energy shape={e.shape}, grad shape={g.shape}")
assert e.shape == (16,) and g.shape == (16, 8)

lj = LennardJonesEnergy(n_particles=13, device=device)
x = torch.randn(8, 39, device=device) * 2
e = lj.energy(x)
g = lj.grad_energy(x)
print(f"  LJ-13: energy shape={e.shape}, grad shape={g.shape}")
assert e.shape == (8,) and g.shape == (8, 39)

mw = ManyWellEnergy(device=device)
x = torch.randn(16, 5, device=device)
e = mw.energy(x)
g = mw.grad_energy(x)
print(f"  MW-5:  energy shape={e.shape}, grad shape={g.shape}")
assert e.shape == (16,) and g.shape == (16, 5)
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("TEST 5: Priors")
print("=" * 60)
from asbs.priors import DiracPrior, GaussianPrior, HarmonicPrior

dp = DiracPrior()
x = dp.sample(100, 8, device)
assert x.abs().max() == 0
print(f"  Dirac: all zeros ✓")

gp = GaussianPrior(std=2.0)
x = gp.sample(1000, 8, device)
print(f"  Gaussian: std={x.std():.2f} (expect ~2.0)")
assert abs(x.std() - 2.0) < 0.3

hp = HarmonicPrior(n_particles=4, spatial_dim=3)
x = hp.sample(100, 12, device)
x_3d = x.view(100, 4, 3)
com = x_3d.mean(dim=1).abs().max().item()
print(f"  Harmonic: CoM max={com:.2e} (expect ~0)")
assert com < 1e-5
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("TEST 6: MLP Models")
print("=" * 60)
from asbs.models import TimeDependentMLP, CorrectorMLP

mlp = TimeDependentMLP(x_dim=8, hidden_dim=64, n_layers=2).to(device)
t = torch.rand(16, device=device)
x = torch.randn(16, 8, device=device)
out = mlp(t, x)
print(f"  Drift MLP: output shape={out.shape}")
assert out.shape == (16, 8)

cmlp = CorrectorMLP(x_dim=8, hidden_dim=64, n_layers=2).to(device)
out = cmlp(x)
print(f"  Corrector MLP: output shape={out.shape}")
print(f"  Corrector init output max: {out.abs().max():.6f} (expect ~0)")
assert out.abs().max() < 0.01  # zero-initialized
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("TEST 7: Replay Buffer")
print("=" * 60)
from asbs.buffers import ReplayBuffer

buf = ReplayBuffer(max_size=100, dim=8, device=device)
buf.add(torch.randn(50, 8), torch.randn(50, 8))
print(f"  Buffer size: {len(buf)} (expect 50)")
assert len(buf) == 50
x0, x1 = buf.sample(16)
print(f"  Sample shapes: x0={x0.shape}, x1={x1.shape}")
assert x0.shape == (16, 8)
buf.add(torch.randn(80, 8), torch.randn(80, 8))
print(f"  Buffer size after overflow: {len(buf)} (expect 100)")
assert len(buf) == 100
print("  PASSED\n")

# ============================================================
print("=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
