# Official Repository Reference: Detailed Architecture & Code Guide

> This document covers two official repositories:
> 
> 1. **`facebookresearch/adjoint_samplers`** (plural) — ASBS paper code for **synthetic energies** (DW-4, LJ-13, LJ-55)
> 1. **`facebookresearch/adjoint_sampling`** (singular) — Original AS paper code for **amortized conformer generation** (SPICE, GEOM-DRUGS)

-----

## Part 1: `facebookresearch/adjoint_samplers` (ASBS — Synthetic Energies)

### 1.1 Top-Level Architecture

```
adjoint_samplers/              ← repo root
├── train.py                   ← Main entry point (Hydra @hydra.main)
├── environment.yml            ← micromamba/conda env
├── configs/
│   ├── train.yaml             ← Base config
│   └── experiment/            ← Per-experiment overrides
│       ├── dw4_asbs.yaml
│       ├── dw4_as.yaml
│       ├── lj13_asbs.yaml
│       ├── lj13_as.yaml
│       ├── lj55_asbs.yaml
│       └── lj55_as.yaml
├── scripts/
│   ├── demo.sh                ← 2D demo figure
│   └── download.sh            ← Downloads MCMC reference samples
├── adjoint_samplers/          ← Core library
│   ├── components/            ← SDE, models, matchers
│   ├── energies/              ← Energy function implementations
│   ├── evaluators/            ← Evaluation metrics
│   └── utils/                 ← Training utils, distributed mode
└── data/                      ← Reference samples (downloaded)
```

### 1.2 Config System (Hydra)

Entry point:

```python
@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):
```

Experiments selected via CLI:

```bash
python train.py experiment=dw4_asbs seed=0
python train.py experiment=lj13_asbs seed=0,1,2 -m  # multirun
```

Components instantiated from config using `hydra.utils.instantiate(cfg.XXX)`.

### 1.3 `train.py` — Main Entry Point (231 lines)

The exact flow extracted from source code:

```python
# 1. Setup & seeding
train_utils.setup(cfg)
seed = cfg.seed + distributed_mode.get_rank()
torch.manual_seed(seed); np.random.seed(seed)

# 2. Instantiate core components
energy = hydra.utils.instantiate(cfg.energy, device=device)
source = hydra.utils.instantiate(cfg.source, device=device)

# 3. Build the SDE stack
ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)       # base Brownian motion
controller = hydra.utils.instantiate(cfg.controller).to(device)  # learned drift u_θ
sde = ControlledSDE(ref_sde, controller).to(device)              # combined

# 4. ASBS-specific: corrector network + corrector matcher
if "corrector" in cfg:
    corrector = hydra.utils.instantiate(cfg.corrector).to(device)
    corrector_matcher = hydra.utils.instantiate(cfg.corrector_matcher, sde=sde)
else:
    corrector = corrector_matcher = None  # AS mode

# 5. Terminal cost gradient: computes ∇E(x) + h(x)
grad_term_cost = hydra.utils.instantiate(
    cfg.term_cost, corrector=corrector, energy=energy,
    ref_sde=ref_sde, source=source,
)

# 6. Adjoint matcher (AM loss)
adjoint_matcher = hydra.utils.instantiate(
    cfg.adjoint_matcher, grad_term_cost=grad_term_cost, sde=sde,
)

# 7. Optimizer — single Adam with param groups
if corrector is not None:
    optimizer = torch.optim.Adam([
        {'params': controller.parameters(), **cfg.adjoint_matcher.optim},
        {'params': corrector.parameters(), **cfg.corrector_matcher.optim},
    ])
else:
    optimizer = torch.optim.Adam(controller.parameters(), **cfg.adjoint_matcher.optim)

# 8. Main training loop — alternating AM/CM stages
for epoch in range(start_epoch, cfg.num_epochs):
    stage = train_utils.determine_stage(epoch, cfg)  # "adjoint" or "corrector"
    matcher, model = {
        "adjoint": (adjoint_matcher, controller),
        "corrector": (corrector_matcher, corrector),
    }.get(stage)
    
    loss = train_one_epoch(matcher, model, source, optimizer, lr_schedule, epoch, device, cfg)
    
    # Evaluation (runs only on main process, at configured frequency)
    if distributed_mode.is_main_process() and eval_this_epoch:
        if stage == "adjoint":
            samples = generate_eval_samples(sde, source, cfg)
            eval_dict = evaluator(samples)
```

**Key design:** Alternating optimization is controlled by `determine_stage(epoch, cfg)` which maps epoch numbers to “adjoint” or “corrector” based on the config schedule.

### 1.4 Core Components: SDE (`adjoint_samplers/components/sde.py`)

```python
class ReferenceSDE:
    """
    Base SDE: dX_t = f_t(X_t) dt + σ_t dW_t  (f_t = 0 for ASBS)
    
    Methods:
        sigma(t) → diffusion coefficient at time t
        cumulative_variance(t) → bar_sigma_t^2 = ∫₀ᵗ σ²_τ dτ
        transition_kernel(x_s, s, t) → (mean, std) of p^base(X_t | X_s)
        bridge(x_0, x_1, t) → (mean, std) of p^base(X_t | X_0, X_1)
    """

class Controller(nn.Module):
    """
    Learned drift u_θ(t, x). Wraps EGNN or MLP with time conditioning.
    forward(t, x) → drift [batch, dim]
    """

class ControlledSDE:
    """
    Combines ReferenceSDE + Controller:
    dX_t = σ_t · u_θ(t, X_t) dt + σ_t dW_t
    """

def sdeint(sde, x0, timesteps, only_boundary=False):
    """
    Euler-Maruyama integration.
    Returns (x0, x1) if only_boundary else full trajectory.
    """
```

### 1.5 Core Components: Matchers (`adjoint_samplers/components/matchers.py`)

```python
class AdjointMatcher:
    """
    AM loss (Eq. 14). Key attributes: grad_term_cost, sde, buffer.
    
    compute_loss(x0, x1):
        1. Sample t ~ U(eps, 1-eps)
        2. x_t = brownian_bridge_sample(x0, x1, t)
        3. target = bridge_drift(x_t, x1, t) - σ_t * grad_term_cost(x1)
        4. pred = controller(t, x_t)
        5. return MSE(pred, target)
    """

class CorrectorMatcher:
    """
    CM loss (Eq. 15).
    
    compute_loss(x0, x1):
        1. target = -(x1 - x0) / bar_sigma_1^2
        2. pred = corrector(x1)
        3. return MSE(pred, target)
    """
```

### 1.6 Terminal Cost (`GradTerminalCost`)

```python
class GradTerminalCost:
    """
    ASBS mode: returns ∇E(x) + h_ϕ(x)
    AS mode:   returns ∇E(x) + ∇log p^base_1(x)  [analytical]
    """
    def __call__(self, x1):
        grad_e = self.energy.grad(x1)
        if self.corrector is not None:
            return grad_e + self.corrector(x1)   # ASBS
        else:
            return grad_e + self.ref_sde.grad_log_marginal(x1, t=1.0)  # AS
```

### 1.7 Training Loop (`train_loop.py`)

```python
def train_one_epoch(matcher, model, source, optimizer, lr_schedule, epoch, device, cfg):
    """
    1. Generate trajectories → add to buffer
    2. For n_grad_steps: sample buffer → compute loss → backprop
    3. Return average loss
    """
```

### 1.8 Replay Buffer

```python
class ReplayBuffer:
    """
    Stores (x0, x1) pairs. Fixed capacity, FIFO eviction, random sampling.
    Efficient tensor storage. Enables many-to-one gradient ratio.
    """
```

### 1.9 Source Distributions

```python
class DiracSource:     # x0 = 0 (AS mode)
class GaussianSource:  # x0 ~ N(0, σ²I)
class HarmonicSource:  # x0 ~ harmonic prior, zero CoM
```

### 1.10 Energy Functions (`adjoint_samplers/energies/`)

```python
class Energy(ABC):
    def energy(self, x) -> Tensor    # E(x)
    def grad(self, x) -> Tensor      # ∇E(x) via autograd

class DoubleWellEnergy(Energy)      # 4 particles, 2D
class LennardJonesEnergy(Energy)    # n particles, 3D
class ManyWellEnergy(Energy)        # 5 particles, 1D
```

### 1.11 Models

```python
class EGNN(nn.Module):
    """
    Equivariant GNN for particle systems.
    Per layer: edge messages → coord update (equivariant) → node update (invariant)
    Time conditioning via sinusoidal embedding added to node features.
    Output: position updates with zero CoM.
    """

class TimeDependentMLP(nn.Module):
    """For non-equivariant tasks. Residual blocks + time conditioning."""
```

-----

## Part 2: `facebookresearch/adjoint_sampling` (AS — Conformer Generation)

### 2.1 Structure

```
adjoint_sampling/
├── train.py, eval.py, eval_distributed.sh
├── cache_dataset.py               ← Preprocess SPICE SMILES
├── download_models.py             ← Fetch checkpoints from HF
├── configs/experiment/
│   ├── spice_cartesian.yaml
│   ├── spice_cartesian_pretrain_for_bmam.yaml
│   ├── spice_cartesian_bmam.yaml
│   └── spice_torsion.yaml
├── adjoint_sampling/
│   ├── sampling/                  ← SDE, bridge, integration
│   ├── models/                    ← Graph-conditional EGNN, torsional
│   ├── training/                  ← Matchers, training loops
│   ├── energies/                  ← eSEN wrapper
│   └── data/                      ← SPICE loading, molecular graphs
└── data/
    ├── spice_train.txt            ← 24,477 SMILES
    ├── spice_test.txt, drugs_test.txt
    └── {spice,drugs}_test_conformers/
```

### 2.2 Key Differences from adjoint_samplers

|Aspect   |adjoint_samplers (ASBS)|adjoint_sampling (AS) |
|---------|-----------------------|----------------------|
|Focus    |Synthetic energies     |Conformer generation  |
|Energy   |Analytical LJ, DW      |eSEN neural model     |
|Prior    |Harmonic/Gaussian/Dirac|Dirac only            |
|Model    |EGNN (fixed graph)     |Graph-conditional EGNN|
|Corrector|Yes                    |No                    |
|Scale    |1 GPU, hours           |8 GPUs, 72 hours      |
|Coords   |Cartesian or internal  |Cartesian or torsional|

### 2.3 Graph-Conditional EGNN

```python
class MolecularEGNN(nn.Module):
    """
    Differences from standard EGNN:
    - Node features: atomic number embedding
    - Edge features: bond type embedding
    - Edge connectivity: from molecular graph (not fully connected)
    - Variable-size molecules via batched padding
    - Learns u_θ(t, x | g) conditioned on graph g
    """
```

### 2.4 Data & Evaluation

```python
# cache_dataset.py: SMILES → cached molecular graphs + torsion info
# eval.py: generate conformers → RMSD with Kabsch alignment → coverage recall + AMR
```

### 2.5 Training Commands

```bash
python cache_dataset.py                              # Preprocess
python train.py experiment=spice_cartesian           # Train AS
python train.py experiment=spice_cartesian_bmam \    # Train AS + pretrain
    init_model=path/to/pretrain_checkpoint.pt
python eval.py --checkpoint_path ... --max_n_refs 512
```

### 2.6 Pretrained Checkpoints (HuggingFace)

```
models/am/checkpoints/checkpoint_4999.pt         # Cartesian AS
models/bmam/checkpoints/checkpoint_4999.pt       # Cartesian AS + pretrain
models/torsion/checkpoints/checkpoint_3000.pt    # Torsional AS
```

-----

## Part 3: Shared Implementation Patterns

### 3.1 Stop-Gradient Pattern

```python
# Generate (NO GRAD) → Buffer (DETACHED) → Loss (GRAD through model only)
with torch.no_grad():
    x0 = source.sample([B]); x0, x1 = sdeint(sde, x0, timesteps)
buffer.add(x0.detach(), x1.detach())
# Later: x0, x1 = buffer.sample(B); loss = matcher.compute_loss(x0, x1)
```

### 3.2 Many-to-One Gradient Ratio

Generate trajectories ONCE → perform MANY gradient updates from buffer. Typical ratio: 1 generation → 10-100 updates. This is the key scalability trick.

### 3.3 Center-of-Mass Removal

Applied after source sampling, each SDE step, and model output:

```python
x = x - x.mean(dim=-2, keepdim=True)
```

### 3.4 DDP Strategy

Only controller/corrector networks are DDP-wrapped. SDE integration is NOT (no gradients flow through it).

-----

## Part 4: Exact Hyperparameters

### Synthetic Energies

|            |DW-4    |MW-5    |LJ-13   |LJ-55   |
|------------|--------|--------|--------|--------|
|d           |8       |5       |39      |165     |
|Model       |EGNN    |MLP     |EGNN    |EGNN    |
|Layers      |4       |4       |4       |4       |
|Hidden      |128     |128     |128     |128     |
|σ_min/max   |0.01/1.0|0.01/1.0|0.01/1.0|0.01/1.0|
|Steps       |100     |100     |100     |100     |
|Batch       |256     |256     |256     |256     |
|LR          |1e-3    |1e-3    |1e-3    |1e-3    |
|Prior (ASBS)|Harmonic|Gaussian|Harmonic|Harmonic|

### Conformer Generation

|Setting   |Value             |
|----------|------------------|
|Train mols|24,477 (SPICE)    |
|Test mols |80+80 (SPICE+GEOM)|
|Energy    |eSEN              |
|GPUs      |8                 |
|Time      |~72 hours         |

-----

## Part 5: Non-Obvious Tricks

1. **Time epsilon:** `t = rand * 0.998 + 0.001` — avoid bridge divergence at 0 and 1
1. **Corrector zero-init:** Last layer weights/bias = 0 so h^(0) outputs zero
1. **Gradient clipping:** `clip_grad_norm_(params, max_norm=1.0)`
1. **Energy grad detachment:** Energy gradient computed with `enable_grad` then `.detach()` — it’s part of the target, not the prediction
1. **Buffer freshness:** Stale samples hurt; periodically refresh substantial fraction
1. **Eval uses more SDE steps:** Training ~100, evaluation ~500-1000
1. **WT-ASBS extension** (facebookresearch/wt-asbs): Adds metadynamics bias for better mode coverage

-----

## Part 6: Paper Equation → Code Mapping

|Paper Eq.          |Code Location               |Class/Function                 |
|-------------------|----------------------------|-------------------------------|
|(2) SDE            |`components/sde.py`         |`ControlledSDE`, `sdeint`      |
|(14) AM loss       |`components/matchers.py`    |`AdjointMatcher.compute_loss`  |
|(15) CM loss       |`components/matchers.py`    |`CorrectorMatcher.compute_loss`|
|(19) Harmonic prior|source distributions        |`HarmonicSource.sample`        |
|Alg. 1             |`train.py` + `train_loop.py`|Stage scheduling loop          |
|Bridge sampling    |`components/sde.py`         |`ReferenceSDE.bridge`          |
|σ_t schedule       |`components/sde.py`         |`ReferenceSDE.sigma`           |

-----

## Part 7: Quick Start

### ASBS (synthetic):

```bash
git clone https://github.com/facebookresearch/adjoint_samplers.git && cd adjoint_samplers
micromamba env create -f environment.yml && micromamba activate adjoint_samplers
bash scripts/download.sh
bash scripts/demo.sh
python train.py experiment=dw4_asbs seed=0
```

### AS (conformers):

```bash
git clone https://github.com/facebookresearch/adjoint_sampling.git && cd adjoint_sampling
micromamba env create -f environment.yml && micromamba activate adjoint_sampling
huggingface-cli login
python cache_dataset.py
python train.py experiment=spice_cartesian
```