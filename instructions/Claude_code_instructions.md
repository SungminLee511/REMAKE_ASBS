# Claude Code Instructions: Implementing ASBS from Scratch

> **Objective:** Implement the Adjoint Schrödinger Bridge Sampler (ASBS) in PyTorch, reproducing all experiments from the paper. This guide is structured as a **spiral curriculum** — you build the simplest working version first, then revisit the same concepts at increasing complexity. Each phase has a **verification checkpoint** so you know it works before moving on.

-----

## Project Structure (Target)

```
asbs/
├── README.md                      # (File 1 — the theory document)
├── train.py                       # Main training entry point (Hydra-based)
├── eval.py                        # Evaluation script
├── environment.yml                # Conda environment
├── configs/                       # Hydra configs
│   ├── default.yaml
│   ├── experiment/
│   │   ├── demo_2d.yaml
│   │   ├── dw4_asbs.yaml
│   │   ├── dw4_as.yaml
│   │   ├── lj13_asbs.yaml
│   │   ├── lj13_as.yaml
│   │   ├── lj55_asbs.yaml
│   │   ├── lj55_as.yaml
│   │   ├── alanine_asbs.yaml
│   │   └── conformer_asbs.yaml
│   ├── model/
│   │   ├── mlp.yaml
│   │   └── egnn.yaml
│   └── energy/
│       ├── double_well.yaml
│       ├── many_well.yaml
│       ├── lennard_jones.yaml
│       ├── alanine.yaml
│       └── esen.yaml
├── asbs/
│   ├── __init__.py
│   ├── energies/                  # Energy function implementations
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract energy class
│   │   ├── double_well.py         # DW-4
│   │   ├── many_well.py           # MW-5
│   │   ├── lennard_jones.py       # LJ-13, LJ-55
│   │   ├── alanine.py             # Alanine dipeptide (OpenMM)
│   │   └── esen_wrapper.py        # eSEN energy model wrapper
│   ├── models/                    # Neural network architectures
│   │   ├── __init__.py
│   │   ├── mlp.py                 # Fully-connected networks
│   │   ├── egnn.py                # Equivariant Graph Neural Network
│   │   └── time_embedding.py      # Sinusoidal time embeddings
│   ├── sde/                       # SDE mechanics
│   │   ├── __init__.py
│   │   ├── noise_schedule.py      # Geometric noise schedule, cumulative variance
│   │   ├── integrator.py          # Euler-Maruyama integration
│   │   └── brownian_bridge.py     # Bridge sampling for AM loss
│   ├── losses/                    # Training objectives
│   │   ├── __init__.py
│   │   ├── adjoint_matching.py    # AM loss (Eq. 14)
│   │   └── corrector_matching.py  # CM loss (Eq. 15)
│   ├── priors/                    # Source distributions
│   │   ├── __init__.py
│   │   ├── gaussian.py
│   │   ├── dirac.py               # For AS special case
│   │   └── harmonic.py            # Harmonic oscillator prior
│   ├── buffers/                   # Replay buffer
│   │   ├── __init__.py
│   │   └── replay_buffer.py
│   ├── training/                  # Training loop
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop with alternating optimization
│   │   └── utils.py               # Logging, checkpointing
│   └── evaluation/                # Eval metrics
│       ├── __init__.py
│       ├── wasserstein.py         # W2, energy-W2
│       ├── sinkhorn.py            # Sinkhorn distance
│       ├── kl_divergence.py       # 1D KL for torsion marginals
│       └── conformer_metrics.py   # Coverage recall, AMR
├── data/                          # Reference samples, SMILES, etc.
├── scripts/
│   ├── demo.sh
│   ├── download_data.sh
│   └── run_experiments.sh
└── results/                       # Checkpoints, figures, logs
```

-----

## Phase 0: Environment & Scaffolding

### Goal

Set up the project structure and dependencies so everything is ready to build.

### Tasks

1. **Create `environment.yml`** with these core dependencies:
   
   ```yaml
   name: asbs
   channels:
     - pytorch
     - nvidia
     - conda-forge
   dependencies:
     - python=3.10
     - pytorch>=2.1
     - torchvision
     - torchaudio
     - pytorch-cuda=12.1
     - numpy
     - scipy
     - matplotlib
     - seaborn
     - hydra-core>=1.3
     - omegaconf
     - wandb
     - tqdm
     - pot          # Python Optimal Transport (for Sinkhorn/W2)
     - pip:
       - ase        # Atomic Simulation Environment (for molecular stuff)
   ```
1. **Create the directory structure** as shown above. Every `__init__.py` starts empty.
1. **Create `configs/default.yaml`** with a basic Hydra config:
   
   ```yaml
   seed: 0
   device: cuda
   
   # SDE parameters
   sde:
     sigma_min: 0.01
     sigma_max: 1.0
     n_steps: 100        # Euler-Maruyama steps
   
   # Training
   training:
     n_stages: 10         # Number of AM/CM alternating stages
     am_steps: 1000       # Gradient steps per AM stage
     cm_steps: 500        # Gradient steps per CM stage
     batch_size: 256
     lr: 1e-3
     optimizer: adam
   
   # Replay buffer
   buffer:
     max_size: 10000
     min_size: 256
   
   # Logging
   logging:
     log_every: 100
     save_every: 1000
     plot_every: 500
   ```

### Verification Checkpoint

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] Hydra config loads: `python -c "import hydra; print('ok')"`
- [ ] All directories exist

-----

## Phase 1: SDE Mechanics

### Goal

Build the fundamental SDE components: noise schedule, forward integration, and Brownian bridge sampling. These are the backbone of everything.

### Task 1.1: Noise Schedule (`asbs/sde/noise_schedule.py`)

Implement the **geometric noise schedule** and its cumulative variance.

```python
class GeometricNoiseSchedule:
    """
    σ_t = σ_min^(1-t) · σ_max^t
    
    Key quantities:
    - sigma(t): the diffusion coefficient at time t
    - cumulative_variance(t): σ̄²_t = ∫₀ᵗ σ_τ² dτ  (computed numerically)
    - bridge_params(t, X0, X1): returns (mean, std) of the Brownian bridge at time t
    """
```

**Implementation details:**

- `sigma(t)` — straightforward: `sigma_min ** (1-t) * sigma_max ** t`
- `cumulative_variance(t)` — compute the integral of sigma_squared from 0 to t using `scipy.integrate.quad` or a precomputed lookup table. For efficiency, precompute at 10000 grid points and interpolate.
- The analytic integral of the geometric schedule is available but it is safer to just numerically integrate.

### Task 1.2: Euler-Maruyama Integrator (`asbs/sde/integrator.py`)

```python
def euler_maruyama_forward(
    drift_fn,       # u_θ(t, x) -> drift vector
    x0,             # Initial positions [batch, dim]
    noise_schedule,  # GeometricNoiseSchedule instance
    n_steps,         # Number of discretization steps
    return_trajectory=False  # Whether to return all intermediate states
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Integrate: dX_t = σ_t · u_θ(t, X_t) dt + σ_t dW_t
    
    Note: base drift f_t = 0 (Brownian motion base).
    The learned drift u_θ is MULTIPLIED by σ_t in the SDE.
    
    Returns X_1 (and optionally the full trajectory).
    """
```

**Step-by-step for each timestep t_n -> t_{n+1}:**

1. Compute sigma at t_n from noise schedule
1. Compute drift: u = drift_fn(t_n, X_{t_n})
1. Sample noise: eps ~ N(0, I)
1. Update: X_{t_{n+1}} = X_{t_n} + sigma_{t_n} * u * dt + sigma_{t_n} * sqrt(dt) * eps

**Important:** Use `torch.no_grad()` when generating trajectories for the replay buffer (stop gradient).

### Task 1.3: Brownian Bridge (`asbs/sde/brownian_bridge.py`)

```python
def sample_brownian_bridge(
    x0,              # [batch, dim] - start points
    x1,              # [batch, dim] - end points
    t,               # [batch] or scalar - time to sample at
    noise_schedule,  # GeometricNoiseSchedule instance
) -> Tensor:
    """
    Sample X_t from the Brownian bridge p^base(X_t | X_0, X_1).
    
    For f_t = 0 base, dX = sigma_t dW_t:
    
    X_t | (X_0, X_1) ~ N( X_0 + (bar_sigma_t^2 / bar_sigma_1^2)(X_1 - X_0),
                           bar_sigma_t^2 (1 - bar_sigma_t^2 / bar_sigma_1^2) I )
    """
```

Also implement:

```python
def bridge_drift_target(x_t, x1, t, noise_schedule):
    """
    Compute nabla_{X_t} log p^base(X_1 | X_t) = (X_1 - X_t) / (bar_sigma_1^2 - bar_sigma_t^2)
    
    This is the "pointing toward X_1" term in the AM regression target.
    """
```

### Verification Checkpoint

- [ ] Generate 1000 samples from a Brownian bridge between X_0=0 and X_1=1 in 1D. Plot the density at t=0.5. It should be a Gaussian centered at ~0.5.
- [ ] Run Euler-Maruyama with zero drift from X_0=0. The distribution at t=1 should be N(0, bar_sigma_1^2).
- [ ] Verify that `bridge_drift_target` at t->1 gives a very large magnitude (pointing strongly at X_1), and at t->0 gives something proportional to (X_1 - X_0)/bar_sigma_1^2.

-----

## Phase 2: Energy Functions

### Goal

Implement all the energy functions. These are the “physics” of the problem.

### Task 2.1: Base Energy Class (`asbs/energies/base.py`)

```python
class EnergyFunction(ABC):
    @abstractmethod
    def energy(self, x: Tensor) -> Tensor:
        """Compute E(x). Input [batch, dim], output [batch]."""
        
    def grad_energy(self, x: Tensor) -> Tensor:
        """Compute nabla E(x) via autograd. Input/output [batch, dim]."""
        x = x.requires_grad_(True)
        e = self.energy(x)
        grad = torch.autograd.grad(e.sum(), x, create_graph=True)[0]
        return grad
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the sample space."""
```

### Task 2.2: Double-Well (DW-4) (`asbs/energies/double_well.py`)

4 particles in 2D. The energy is a sum of pairwise double-well potentials over all 6 pairs.

The pairwise potential is: V_DW(r) = a(r^2 - r_0^2)^2 + b*r^2 with typical parameters a=0.9, b=-4, r_0=1.5 (check the DEM/DDS repos for exact parameters).

**Implementation notes:**

- Input shape: [batch, 4, 2] or flattened [batch, 8]
- Compute all pairwise distances using `torch.cdist` or manual broadcasting
- The energy must be differentiable for `grad_energy`

### Task 2.3: Many-Well (MW-5) (`asbs/energies/many_well.py`)

5 particles in 1D, d=5. Use the SCLD setup from Chen et al. (2025). This is a multi-modal distribution that can be sampled analytically for ground-truth evaluation.

### Task 2.4: Lennard-Jones (`asbs/energies/lennard_jones.py`)

Standard Lennard-Jones potential for n particles in 3D:

```python
class LennardJonesEnergy(EnergyFunction):
    def __init__(self, n_particles, spatial_dim=3, epsilon=1.0, sigma_lj=1.0):
        self.n_particles = n_particles  # 13 or 55
        self.spatial_dim = spatial_dim
        self.epsilon = epsilon
        self.sigma_lj = sigma_lj
    
    @property
    def dim(self):
        return self.n_particles * self.spatial_dim
    
    def energy(self, x):
        # x: [batch, n_particles, 3]
        # Compute pairwise distances
        # Apply LJ formula: 4*eps * [(sigma/r)^12 - (sigma/r)^6]
        # Sum over all pairs
        # IMPORTANT: clamp r_ij to minimum value to avoid NaN/Inf
```

**Stability note:** When r_ij is very small, the r^{-12} term explodes. Clamp distances: `r_ij = r_ij.clamp(min=1e-6)`.

### Task 2.5: Alanine Dipeptide (`asbs/energies/alanine.py`)

This uses the OpenMM library to compute energies in internal coordinates.

```python
# Requires: pip install openmm openmmtools
class AlanineEnergy(EnergyFunction):
    """
    Alanine dipeptide in implicit solvent.
    - Input: internal coordinates (d=60)
    - Uses OpenMM to evaluate energy at 300K
    - Needs coordinate transformation: internal <-> Cartesian
    
    NOTE: This is the most complex energy to implement.
    Consider wrapping the bgflow library's alanine implementation.
    """
```

### Task 2.6: eSEN Wrapper (`asbs/energies/esen_wrapper.py`)

For conformer generation, wrap the eSEN foundation model:

```python
class ESENEnergy(EnergyFunction):
    """
    Wraps the eSEN model from HuggingFace.
    - Input: atomic positions [n_atoms, 3] + atomic numbers + bond info
    - Output: energy scalar
    - Requires: fairchem library
    """
```

### Verification Checkpoint

- [ ] DW-4: Compute energy for a random configuration. Verify gradient exists and has correct shape.
- [ ] LJ-13: Energy of a “crystal” configuration (particles on a grid) should be finite and negative.
- [ ] Plot 2D slices of the DW-4 energy landscape to visually verify double-well shape.
- [ ] All energies are differentiable: `torch.autograd.gradcheck` passes.

-----

## Phase 3: Source Distributions (Priors)

### Goal

Implement the source distributions that ASBS can use.

### Task 3.1: Dirac Delta Prior (`asbs/priors/dirac.py`)

```python
class DiracPrior:
    """X_0 = 0. Used for the AS special case."""
    def sample(self, batch_size, dim, device):
        return torch.zeros(batch_size, dim, device=device)
```

### Task 3.2: Gaussian Prior (`asbs/priors/gaussian.py`)

```python
class GaussianPrior:
    def __init__(self, dim, std=1.0):
        self.dim = dim
        self.std = std
    
    def sample(self, batch_size, device):
        return torch.randn(batch_size, self.dim, device=device) * self.std
```

### Task 3.3: Harmonic Prior (`asbs/priors/harmonic.py`)

For n-particle systems x = {x_i}:

The harmonic energy is: -(alpha/2) * sum_{i,j} ||x_i - x_j||^2

After centering (zero center of mass), this becomes isotropic Gaussian with variance 1/(2*alpha*n) per coordinate.

```python
class HarmonicPrior:
    def __init__(self, n_particles, spatial_dim, alpha=1.0):
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.alpha = alpha
    
    def sample(self, batch_size, device):
        x = torch.randn(batch_size, self.n_particles, self.spatial_dim, device=device)
        x = x / math.sqrt(2 * self.alpha * self.n_particles)
        x = x - x.mean(dim=1, keepdim=True)  # Remove center of mass
        return x
```

### Verification Checkpoint

- [ ] Harmonic prior samples have zero center of mass: `x.mean(dim=1).abs().max() < 1e-6`
- [ ] Gaussian prior samples have correct variance

-----

## Phase 4: Neural Network Models

### Goal

Build the drift and corrector networks.

### Task 4.1: Time Embedding (`asbs/models/time_embedding.py`)

Standard sinusoidal embedding for scalar time t in [0,1]:

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, t):
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

### Task 4.2: MLP (`asbs/models/mlp.py`)

For non-equivariant tasks (alanine dipeptide, 2D demos):

```python
class TimeDependentMLP(nn.Module):
    """
    MLP that takes (t, x) as input and outputs a vector of same dim as x.
    
    Architecture:
    - Sinusoidal time embedding
    - Concatenate [time_embed, x]
    - Multiple residual blocks with LayerNorm + SiLU
    - Output layer projecting to dim(x)
    """
    def __init__(self, x_dim, hidden_dim=256, n_layers=4, time_embed_dim=64):
        ...
    
    def forward(self, t, x):
        # t: [batch] or scalar
        # x: [batch, x_dim]
        # Returns: [batch, x_dim]
        ...
```

### Task 4.3: EGNN (`asbs/models/egnn.py`)

The **Equivariant Graph Neural Network** (Satorras et al., 2021) is critical for particle systems. It must preserve SE(3) equivariance.

```python
class EGNN(nn.Module):
    """
    Equivariant Graph Neural Network.
    
    Key properties:
    - Input: positions [batch, n_particles, 3] + features [batch, n_particles, feat_dim]
    - Output: position updates [batch, n_particles, 3] (equivariant)
    - Equivariance: if you rotate/translate the input, the output rotates/translates the same way
    
    Architecture (per layer):
    1. Compute pairwise messages:
       m_ij = MLP_edge(h_i, h_j, ||x_i - x_j||^2, edge_attr)
    2. Update positions (equivariant):
       x_i' = x_i + sum_j (x_i - x_j) * MLP_coord(m_ij)
    3. Update node features (invariant):
       h_i' = h_i + MLP_node(h_i, sum_j m_ij)
    
    Time conditioning: embed time t and add to node features.
    """
    def __init__(self, n_layers=4, hidden_dim=128, coord_dim=3, node_feat_dim=1):
        ...
    
    def forward(self, t, positions, node_features=None, edge_index=None):
        """
        t: [batch] scalar time
        positions: [batch, n_particles, 3]
        node_features: [batch, n_particles, feat_dim] (optional)
        edge_index: [2, n_edges] (optional, defaults to fully connected)
        
        Returns: position_updates [batch, n_particles, 3]
        """
```

**Critical implementation details for EGNN:**

1. **Fully connected graph** for LJ potentials (every particle interacts with every other)
1. **Molecular graph** for conformer generation (edges = bonds from SMILES)
1. **Time conditioning** via adding time embedding to node features
1. The position updates must be **equivariant**: output(Rx + t) = R * output(x) for rotation R and translation t
1. Only use **distances** (not positions) in MLPs to ensure invariance
1. Remove center-of-mass updates: `x_update = x_update - x_update.mean(dim=1, keepdim=True)`

### Task 4.4: Corrector Network

The corrector h_phi(x) does **not** depend on time t — it only acts at t=1.

For particle systems: use an EGNN without time conditioning.
For non-particle systems: use an MLP without time input.

### Verification Checkpoint

- [ ] **Equivariance test:** Generate random positions x and random rotation R. Verify `EGNN(t, Rx) ≈ R * EGNN(t, x)` up to numerical precision (~1e-5).
- [ ] **Translation invariance:** Verify `EGNN(t, x + c) ≈ EGNN(t, x)` for constant shift c.
- [ ] MLP outputs correct shape for alanine dimensions.

-----

## Phase 5: The 2D Demo (Your First Working ASBS)

### Goal

Get the full ASBS algorithm working on a **simple 2D problem** you can visualize. This is the most important phase — if this works, everything else is scaling up.

### Task 5.1: 2D Energy Function

Create a simple 2D Gaussian mixture energy:

```python
class GaussianMixture2D(EnergyFunction):
    def __init__(self):
        self.means = torch.tensor([[-3, 0], [3, 0], [0, 3], [0, -3]], dtype=torch.float32)
        self.std = 0.5
    
    @property
    def dim(self):
        return 2
    
    def energy(self, x):
        # x: [batch, 2]
        # Negative log of mixture of Gaussians
        diffs = x.unsqueeze(1) - self.means.unsqueeze(0)  # [batch, 4, 2]
        log_probs = -0.5 * (diffs / self.std).pow(2).sum(-1)  # [batch, 4]
        return -torch.logsumexp(log_probs, dim=-1)  # [batch]
```

### Task 5.2: Implement the Full Training Loop

This is where everything comes together. Build `asbs/training/trainer.py`:

```python
class ASBSTrainer:
    def __init__(self, config, energy_fn, drift_model, corrector_model, prior, noise_schedule):
        self.energy_fn = energy_fn
        self.drift = drift_model        # u_θ(t, x)
        self.corrector = corrector_model  # h_ϕ(x)
        self.prior = prior
        self.ns = noise_schedule
        self.replay_buffer = ReplayBuffer(config.buffer.max_size)
        
        self.drift_optimizer = Adam(self.drift.parameters(), lr=config.training.lr)
        self.corrector_optimizer = Adam(self.corrector.parameters(), lr=config.training.lr)
    
    def generate_trajectories(self, batch_size):
        """
        Sample X_0 ~ mu, then integrate SDE forward to get X_1.
        Store (X_0, X_1) in replay buffer.
        Uses torch.no_grad() — this is the stop-gradient.
        """
        with torch.no_grad():
            x0 = self.prior.sample(batch_size, self.energy_fn.dim, device)
            x1 = euler_maruyama_forward(
                drift_fn=self.drift,
                x0=x0,
                noise_schedule=self.ns,
                n_steps=config.sde.n_steps
            )
        return x0, x1
    
    def am_loss(self, x0, x1):
        """
        Adjoint Matching loss (Equation 14).
        
        1. Sample random time t ~ Uniform(0, 1)
        2. Sample X_t from Brownian bridge between X_0 and X_1
        3. Compute regression target:
           target = sigma_t * (X_1 - X_t)/(bar_sigma_1^2 - bar_sigma_t^2) 
                    - sigma_t * (grad_E(X_1) + h(X_1))
        4. Predict: pred = u_theta(t, X_t)
        5. Loss = ||pred - target||^2
        
        CRITICAL SIGN NOTE:
        The regression target for u_theta has TWO terms:
        a) The bridge drift: points from X_t toward X_1 (positive contribution)
        b) The terminal cost gradient: -sigma_t * (nabla_E + h) evaluated at X_1
        
        The final target is:  bridge_drift - sigma_t * (grad_E + h)
        
        But be careful: the exact form depends on how you parameterize the SDE.
        In the SDE: dX_t = sigma_t * u_theta * dt + sigma_t * dW_t
        The u_theta already has sigma_t factored in as a multiplier.
        
        Check the official repo to confirm the exact formula used.
        """
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=x0.device) * 0.998 + 0.001  # avoid t=0,1
        
        # Sample from Brownian bridge
        x_t = sample_brownian_bridge(x0, x1, t, self.ns)
        
        # Energy gradient at X_1
        grad_e = self.energy_fn.grad_energy(x1.detach())
        
        # Corrector at X_1 (no grad through corrector during AM)
        with torch.no_grad():
            h = self.corrector(x1)
        
        # Bridge drift target
        bridge = bridge_drift_target(x_t, x1, t, self.ns)
        
        # Full regression target
        sigma_t = self.ns.sigma(t).unsqueeze(-1)
        target = bridge - sigma_t * (grad_e + h)
        
        # Model prediction
        pred = self.drift(t, x_t)
        
        loss = (pred - target).pow(2).mean()
        return loss
    
    def cm_loss(self, x0, x1):
        """
        Corrector Matching loss (Equation 15).
        
        target = -(X_1 - X_0) / bar_sigma_1^2
        pred = h_phi(X_1)
        Loss = ||pred - target||^2
        """
        bar_sigma_1_sq = self.ns.cumulative_variance(1.0)
        target = -(x1 - x0) / bar_sigma_1_sq
        pred = self.corrector(x1)
        loss = (pred - target).pow(2).mean()
        return loss
    
    def train(self):
        """Main loop implementing Algorithm 1."""
        for stage in range(self.config.training.n_stages):
            # === Generate fresh trajectories ===
            x0, x1 = self.generate_trajectories(self.config.training.batch_size)
            self.replay_buffer.add(x0, x1)
            
            # === ADJOINT MATCHING STAGE ===
            for step in range(self.config.training.am_steps):
                x0_b, x1_b = self.replay_buffer.sample(self.config.training.batch_size)
                loss = self.am_loss(x0_b, x1_b)
                self.drift_optimizer.zero_grad()
                loss.backward()
                self.drift_optimizer.step()
            
            # === Generate new trajectories with updated drift ===
            x0, x1 = self.generate_trajectories(self.config.training.batch_size)
            self.replay_buffer.add(x0, x1)
            
            # === CORRECTOR MATCHING STAGE ===
            for step in range(self.config.training.cm_steps):
                x0_b, x1_b = self.replay_buffer.sample(self.config.training.batch_size)
                loss = self.cm_loss(x0_b, x1_b)
                self.corrector_optimizer.zero_grad()
                loss.backward()
                self.corrector_optimizer.step()
```

### Task 5.3: Replay Buffer (`asbs/buffers/replay_buffer.py`)

```python
class ReplayBuffer:
    """
    Stores (X_0, X_1) endpoint pairs from previous model evaluations.
    Fixed max size, FIFO eviction, random sampling for mini-batches.
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, x0, x1):
        """Add batch of (x0, x1) pairs. Evict oldest if over capacity."""
        for i in range(x0.shape[0]):
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
            self.buffer.append((x0[i].detach().cpu(), x1[i].detach().cpu()))
    
    def sample(self, batch_size):
        """Randomly sample a mini-batch."""
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        x0s = torch.stack([self.buffer[i][0] for i in indices])
        x1s = torch.stack([self.buffer[i][1] for i in indices])
        return x0s.to(device), x1s.to(device)
    
    def __len__(self):
        return len(self.buffer)
```

**Performance note:** The above is simple but slow. For production, store as contiguous tensors:

```python
class EfficientReplayBuffer:
    def __init__(self, max_size, dim):
        self.x0 = torch.zeros(max_size, dim)
        self.x1 = torch.zeros(max_size, dim)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
```

### Task 5.4: Visualization for 2D Demo

```python
def plot_2d_results(energy_fn, generated_samples, stage, save_path):
    """
    Create a multi-panel figure:
    1. Energy landscape contour plot with generated samples
    2. Histogram / KDE of generated sample density
    3. Energy histogram comparison (if ground truth available)
    """
```

### Task 5.5: 2D Demo Config (`configs/experiment/demo_2d.yaml`)

```yaml
defaults:
  - override /model: mlp

energy:
  type: gaussian_mixture_2d

prior:
  type: gaussian
  std: 3.0

sde:
  sigma_min: 0.01
  sigma_max: 1.0
  n_steps: 50

training:
  n_stages: 20
  am_steps: 500
  cm_steps: 200
  batch_size: 512
  lr: 1e-3

model:
  hidden_dim: 128
  n_layers: 3
```

### Verification Checkpoint (CRITICAL — most important checkpoint)

- [ ] Run the 2D demo. After ~5 stages, samples should visibly cluster around the 4 Gaussian modes.
- [ ] Plot the energy histogram of generated samples. It should roughly match the target.
- [ ] The AM loss should decrease within each stage.
- [ ] The CM loss should decrease within each stage.
- [ ] **Compare AS vs ASBS:** Run with DiracPrior (AS) and GaussianPrior (ASBS). Both should converge, but ASBS should produce straighter trajectories.

-----

## Phase 6: Synthetic Benchmarks (DW-4, MW-5, LJ-13, LJ-55)

### Goal

Scale up to the paper’s main benchmarks using EGNN and particle-system energies.

### Task 6.1: SE(3) Equivariance Utilities

```python
def remove_center_of_mass(x):
    """x: [batch, n_particles, 3] -> zero CoM"""
    return x - x.mean(dim=-2, keepdim=True)

def random_rotation_matrix(device):
    """Sample a random 3x3 rotation matrix (Haar measure on SO(3))"""
    # Use QR decomposition of random Gaussian matrix
    ...
```

### Task 6.2: Experiment Configs

Create configs for each experiment. Example for DW-4:

```yaml
# configs/experiment/dw4_asbs.yaml
energy:
  type: double_well
  n_particles: 4
  spatial_dim: 2

prior:
  type: harmonic
  n_particles: 4
  spatial_dim: 2
  alpha: 1.0

model:
  type: egnn
  n_layers: 4
  hidden_dim: 128

sde:
  sigma_min: 0.01
  sigma_max: 1.0
  n_steps: 100

training:
  n_stages: 10
  am_steps: 5000
  cm_steps: 2000
  batch_size: 256
  lr: 1e-3
```

**Expected results from paper (Table 2):**

- DW-4: W2 ~ 0.38, E-W2 ~ 0.19
- LJ-13: W2 ~ 1.59, E-W2 ~ 1.28
- LJ-55: W2 ~ 4.00, E-W2 ~ 27.69

### Task 6.3: Evaluation Pipeline

```python
def evaluate_synthetic(model, energy_fn, prior, noise_schedule, n_samples, reference_samples):
    """
    1. Generate n_samples from the model
    2. Compute W2 distance to reference_samples (using POT library)
    3. Compute energy-W2 distance
    4. Plot energy histogram comparison
    """
```

Use the **POT** (Python Optimal Transport) library for W2:

```python
import ot
M = ot.dist(generated, reference)
w2 = ot.emd2(a, b, M)  # or ot.sinkhorn2 for large samples
```

### Verification Checkpoint

- [ ] DW-4: Energy histogram has clear bimodal structure matching reference
- [ ] DW-4: W2 < 1.0 (paper achieves 0.38)
- [ ] LJ-13: Generated configs have no overlapping particles
- [ ] AS vs ASBS comparison: ASBS with harmonic prior outperforms AS with Dirac prior

-----

## Phase 7: Alanine Dipeptide

### Goal

Sample the Boltzmann distribution of a real molecule using internal coordinates.

### Task 7.1: Setup

Uses: internal coordinates (d=60), OpenMM for energy, bgflow for coordinate transforms, MLP model.

### Task 7.2: Coordinate Transformation

```python
class InternalCoordinateTransform:
    """
    Transforms between Cartesian [22, 3] and Internal [60].
    Uses bgflow or custom implementation.
    Must handle periodicity of torsion angles.
    """
```

### Task 7.3: Evaluation

- Compute 1D KL divergence for each torsion marginal (phi, psi, gamma1, gamma2, gamma3)
- Compute W2 on joint (phi, psi)
- Plot Ramachandran plots

### Verification Checkpoint

- [ ] Ramachandran plots show correct high-density regions
- [ ] KL on phi < 0.1, KL on psi < 0.1

-----

## Phase 8: Amortized Conformer Generation

### Goal

The largest experiment: one model generates conformers for many different molecules.

### Task 8.1: SPICE Data Pipeline

```python
class SPICEDataset(Dataset):
    """
    Loads molecular topologies from SPICE.
    Each item: (smiles, molecular_graph, n_atoms, atom_types, bond_types)
    NOTE: We do NOT use atomic positions — only topology!
    """
```

### Task 8.2: Graph-Conditional EGNN

```python
class ConditionalEGNN(EGNN):
    """
    EGNN conditioned on molecular graph:
    - Edge features include bond type
    - Node features include atomic number
    - Edge connectivity from molecular graph (not fully connected)
    - Must handle variable-size molecules
    """
```

### Task 8.3: eSEN Integration

Wrap the eSEN model from HuggingFace for energy evaluation. Requires the fairchem library and access to the checkpoint.

### Task 8.4: Conformer Evaluation

```python
def evaluate_conformers(generated, reference, threshold=1.0):
    """Coverage recall at threshold and Absolute Mean RMSD using Kabsch alignment."""
```

### Task 8.5: Distributed Training

Conformer generation requires 8 GPUs, ~72 hours. Use PyTorch DDP.

### Verification Checkpoint

- [ ] Coverage recall on SPICE test > 70% at 1.0 Angstrom without pretraining
- [ ] Generated conformers have reasonable bond lengths

-----

## Phase 9: Final Integration & Ablations

### Task 9.1: AS as Special Case

Implement Adjoint Sampling by setting prior=DiracPrior and skipping CM (or fixing h analytically).

### Task 9.2: Ablation Studies

1. Prior ablation: Dirac vs Gaussian vs Harmonic
1. Number of stages
1. Noise schedule comparison
1. Replay buffer size impact

### Task 9.3: Reproduce All Tables

```bash
python train.py experiment=dw4_asbs seed=0,1,2 -m
python train.py experiment=lj13_asbs seed=0,1,2 -m
python train.py experiment=lj55_asbs seed=0,1,2 -m
```

-----

## Common Pitfalls & Debugging Tips

### Sign Conventions

The AM loss has subtle sign issues. Double-check:

- The SDE drift convention: is it sigma_t * u_theta or just u_theta?
- The regression target: the energy gradient appears with a NEGATIVE sign
- The bridge drift: (X_1 - X_t)/(bar_sigma_1^2 - bar_sigma_t^2) is POSITIVE

### Numerical Stability

- Clamp LJ distances: `r.clamp(min=1e-6)`
- Avoid sigma_t = 0 or infinity at boundaries
- Use `torch.clamp` on bridge variance to avoid negatives near t=0 or t=1

### Stop Gradient

AM uses stopgrad(u) — trajectories generated with current weights but NO gradient flows through generation. Implement via `torch.no_grad()` during trajectory gen + detached tensors in buffer.

### Center of Mass

For particle systems, ALWAYS remove CoM after every SDE step and from model output:

```python
x = x - x.mean(dim=-2, keepdim=True)
```

### Time Sampling

Avoid exactly t=0 or t=1 where bridge quantities diverge: `t = torch.rand(...) * 0.998 + 0.001`

### Corrector Initialization

Initialize h^(0) = 0 by setting the corrector network’s last layer weights and biases to zero.

### Buffer Warm-up

Don’t train until buffer has at least min_size samples. Generate initial batch at startup.

-----

## Estimated Time per Phase

|Phase    |Content                         |Est. Time      |
|---------|--------------------------------|---------------|
|0        |Setup & scaffolding             |30 min         |
|1        |SDE mechanics                   |2-3 hours      |
|2        |Energy functions                |2-3 hours      |
|3        |Priors                          |1 hour         |
|4        |Neural networks (EGNN!)         |3-4 hours      |
|5        |**2D Demo (first working ASBS)**|**3-4 hours**  |
|6        |Synthetic benchmarks            |2-3 hours      |
|7        |Alanine dipeptide               |3-4 hours      |
|8        |Conformer generation            |4-6 hours      |
|9        |Ablations & tables              |2-3 hours      |
|**Total**|                                |**25-35 hours**|