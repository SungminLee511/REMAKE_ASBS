# Adjoint Schrödinger Bridge Sampler (ASBS) — Complete Theory & Implementation Guide

> **Paper:** *Adjoint Schrödinger Bridge Sampler* (Liu, Choi, Chen, Miller, Chen — FAIR at Meta & Georgia Tech, NeurIPS 2025 Oral)  
> **Predecessors:** *Adjoint Sampling* (Havens et al., ICML 2025), *Adjoint Matching* (Domingo-Enrich et al., ICLR 2025)

-----

## Table of Contents

1. [Problem Statement](#1-problem-statement)
1. [Background: Diffusion Samplers](#2-background-diffusion-samplers)
1. [Stochastic Optimal Control (SOC)](#3-stochastic-optimal-control-soc)
1. [The Memoryless Condition](#4-the-memoryless-condition)
1. [The Schrödinger Bridge Formulation](#5-the-schrödinger-bridge-formulation)
1. [ASBS: Core Theory](#6-asbs-core-theory)
1. [The Two Matching Objectives](#7-the-two-matching-objectives)
1. [Alternating Optimization (Algorithm 1)](#8-alternating-optimization-algorithm-1)
1. [Convergence Theory](#9-convergence-theory)
1. [Practical Implementation Details](#10-practical-implementation-details)
1. [Energy Functions & Experiments](#11-energy-functions--experiments)
1. [Evaluation Metrics](#12-evaluation-metrics)

-----

## 1. Problem Statement

We aim to sample from a **Boltzmann distribution** $\nu(x)$ known only through an unnormalized, differentiable energy function $E(x) : \mathcal{X} \subseteq \mathbb{R}^d \to \mathbb{R}$:

$$\nu(x) := \frac{e^{-E(x)}}{Z}, \quad \text{where } Z := \int_{\mathcal{X}} e^{-E(x)} dx$$

The normalizing constant $Z$ is **intractable**. We can evaluate $E(x)$ and $\nabla_x E(x)$ at any point, but we have **no samples** from $\nu$.

**Physical intuition:** In molecular systems, $E(x)$ quantifies the stability of a 3D configuration $x$. Lower energy = more stable = higher probability under $\nu$. We want to generate these low-energy configurations without running expensive MCMC.

-----

## 2. Background: Diffusion Samplers

A **diffusion sampler** defines a stochastic differential equation (SDE) that transports samples from a simple source distribution $\mu$ to the target $\nu$:

$$dX_t = \big(f_t(X_t) + \sigma_t u_t^\theta(X_t)\big) dt + \sigma_t dW_t, \quad X_0 \sim \mu(X_0)$$

where:

- $f_t(x) : [0,1] \times \mathcal{X} \to \mathcal{X}$ is the **base drift** (often set to zero)
- $\sigma_t : [0,1] \to \mathbb{R}_{>0}$ is the **noise schedule** (diffusion coefficient)
- $\mu(x)$ is the **source distribution** (e.g., Gaussian, Dirac delta, harmonic prior)
- $u_t^\theta(x)$ is the **learned drift** (neural network, parameterized by $\theta$)
- $W_t$ is standard Brownian motion

**Goal:** Learn $u_t^\theta$ such that the marginal distribution at $t=1$ matches the target: $p^u(X_1) = \nu(X_1)$.

### Euler-Maruyama Discretization

In practice, we discretize $[0,1]$ into $N$ steps with step size $\Delta t = 1/N$:

$$X_{t+\Delta t} = X_t + \big(f_t(X_t) + \sigma_t u_t^\theta(X_t)\big) \Delta t + \sigma_t \sqrt{\Delta t}, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_d)$$

### Path Distribution

The SDE induces a **path distribution** $p^u$ over entire trajectories $(X_0, X_1, \ldots)$. The **base path distribution** when $u_t := 0$ is denoted $p^{\text{base}}$.

The **transition kernel** of the base process is:

$$p_{t|s}^{\text{base}}(y|x) := p^{\text{base}}(X_t = y | X_s = x)$$

For Brownian motion base ($f_t := 0$), this is a Gaussian:

$$p_{t|s}^{\text{base}}(y|x) = \mathcal{N}!\left(y;, x,, \left(\int_s^t \sigma_\tau^2 d\tau\right) I_d\right)$$

-----

## 3. Stochastic Optimal Control (SOC)

The SOC framework seeks an optimal drift $u_t^*$ by minimizing a cost functional:

$$\min_u ; \mathbb{E}_{X \sim p^u}!\left[\int_0^1 \frac{1}{2}|u_t(X_t)|^2 dt + g(X_1)\right] \quad \text{s.t. the SDE (2)}$$

where:

- $\frac{1}{2}|u_t|^2$ is the **running cost** (kinetic energy — penalizes large controls)
- $g(x) : \mathcal{X} \to \mathbb{R}$ is the **terminal cost** (pushes $X_1$ toward the target)

### Optimal Distribution

The optimal joint distribution has an analytical form:

$$p^*(X_0, X_1) = p^{\text{base}}(X_0, X_1) , e^{-g(X_1) + V_0(X_0)}$$

where $V_0(x)$ is the **initial value function**:

$$V_0(x) = -\log \int p_{1|0}^{\text{base}}(y|x) , e^{-g(y)} dy$$

This $V_0(x)$ is **intractable** in general and introduces a bias into the terminal marginal.

### Adjoint Matching (AM)

The key computational tool from Domingo-Enrich et al. (2025). Given an SOC problem, the optimal drift satisfies:

$$u_t^*(x) = \mathbb{E}*{p^{\text{base}}}!\left[\sigma_t \nabla*{X_t} \log p_{1|t}^{\text{base}}(X_1|X_t) ,\middle|, X_t = x\right] \cdot (\text{correction from } g)$$

More precisely, Adjoint Matching provides a tractable regression objective:

$$\mathcal{L}*{\text{AM}}(\theta) = \mathbb{E}*{p_{t|0,1}^{\text{base}} , p_{0,1}^{\bar{u}}}!\left[\left|u_t^\theta(X_t) - v_t(X_t, X_1)\right|^2\right], \quad \bar{u} = \text{stopgrad}(u)$$

where $v_t(X_t, X_1)$ is a tractable regression target derived from the base process transition kernel and the terminal cost.

-----

## 4. The Memoryless Condition

The **memoryless condition** assumes statistical independence between $X_0$ and $X_1$ under the base process:

$$p^{\text{base}}(X_0, X_1) \overset{\text{memoryless}}{:=} p^{\text{base}}(X_0) \cdot p^{\text{base}}(X_1)$$

### Why it is needed (for prior methods)

Under memorylessness, the initial value function $V_0(X_0)$ factors out of the terminal marginal:

$$p^*(X_1) \overset{\text{memoryless}}{=} \int p^{\text{base}}(X_0) , p^{\text{base}}(X_1) , e^{-g(X_1) + V_0(X_0)} dX_0 \propto p_1^{\text{base}}(X_1) , e^{-g(X_1)}$$

Setting $g(x) := \log \frac{p_1^{\text{base}}(x)}{\nu(x)}$ then gives $p^*(X_1) = \nu(X_1)$.

### How it is enforced

- **Dirac delta source:** $\mu(x) := \delta_0(x)$ with $f_t := 0$. The base process is pure Brownian motion starting at the origin. Used by PIS, DDS, and Adjoint Sampling.
- **Variance-Preserving SDE:** A linear drift $f_t(x) = -\frac{1}{2}\beta_t x$ with growing noise $\sigma_t = \sqrt{\beta_t}$ and Gaussian prior. The large noise drives $X_0$ and $X_1$ to independence.

### Why it is limiting

1. **No informative priors:** Forces $\mu = \delta$ (trivial source), preventing the use of domain-specific priors like the harmonic oscillator for molecules.
1. **Inefficient transport:** The noise required for memorylessness creates long, curved trajectories. Informative priors would enable shorter, straighter paths.
1. **No pretraining leverage:** Cannot initialize from a distribution that is already close to the target.

The VP-SDE satisfies memorylessness but injects so much noise that the process becomes extremely noisy (see Figure 1 in paper).

-----

## 5. The Schrödinger Bridge Formulation

The **Schrödinger Bridge (SB)** problem is a distributionally constrained optimization:

$$\min_u ; D_{\text{KL}}(p^u | p^{\text{base}}) = \mathbb{E}_{X \sim p^u}!\left[\int_0^1 \frac{1}{2}|u_t^\theta(X_t)|^2 dt\right]$$

$$\text{s.t.} \quad dX_t = \big(f_t(X_t) + \sigma_t u_t^\theta(X_t)\big) dt + \sigma_t dW_t, \quad X_0 \sim \mu(X_0), \quad X_1 \sim \nu(X_1)$$

**Key difference from SOC:** The SB problem has explicit **boundary constraints** on both $X_0$ and $X_1$, rather than a terminal cost $g(X_1)$.

### SB Optimality Conditions (Schrödinger System)

The kinetic-optimal drift $u_t^*$ satisfies:

$$u_t^*(x) = \sigma_t \nabla \log \varphi_t(x)$$

where $\varphi_t(x)$ and $\hat{\varphi}_t(x)$ are the **SB potentials** satisfying:

$$\varphi_t(x) = \int p_{1|t}^{\text{base}}(y|x) , \varphi_1(y) , dy, \qquad \varphi_0(x) \hat{\varphi}_0(x) = \mu(x)$$

$$\hat{\varphi}*t(x) = \int p*{t|0}^{\text{base}}(x|y) , \hat{\varphi}_0(y) , dy, \qquad \varphi_1(x) \hat{\varphi}_1(x) = \nu(x)$$

These are coupled integro-differential equations — hard to solve directly.

-----

## 6. ASBS: Core Theory

### Theorem 3.1: SOC Characteristics of SB

> **Every SB problem can be solved as an SOC problem** with a modified terminal cost.

Specifically, the kinetic-optimal drift $u_t^*$ from the SB problem solves:

$$\min_u ; \mathbb{E}_{X \sim p^u}!\left[\int_0^1 \frac{1}{2}|u_t(X_t)|^2 dt + \log \frac{\hat{\varphi}_1(X_1)}{\nu(X_1)}\right] \quad \text{s.t. the SDE (2)}$$

The terminal cost is $g(x) := \log \frac{\hat{\varphi}_1(x)}{\nu(x)}$.

**Comparison with Adjoint Sampling:** In AS, the terminal cost was $g(x) = \log \frac{p_1^{\text{base}}(x)}{\nu(x)}$. ASBS replaces $p_1^{\text{base}}$ with $\hat{\varphi}_1$ — the SB potential acts as a **corrector** that debiases the non-memoryless process.

### How $\hat{\varphi}_1$ debiases

The optimal joint distribution under this SOC problem is:

$$p^*(X_0, X_1) = p^{\text{base}}(X_0, X_1) \exp!\left(-\log \frac{\hat{\varphi}_1(X_1)}{\nu(X_1)} - \log \varphi_0(X_0)\right)$$

The terminal marginal at $t=1$ becomes:

$$p^*(X_1) = \int p^*(X_0, X_1) dX_0 = \frac{\nu(X_1)}{\hat{\varphi}_1(X_1)} \int p^{\text{base}}(X_1|X_0) \hat{\varphi}_0(X_0) dX_0 = \frac{\nu(X_1)}{\hat{\varphi}_1(X_1)} \cdot \hat{\varphi}_1(X_1) = \nu(X_1)$$

The cancellation is exact by construction of the SB potentials. This works for **any** source distribution $\mu$ — no memoryless condition needed.

-----

## 7. The Two Matching Objectives

Specializing to Boltzmann distributions where $\nu(x) \propto e^{-E(x)}$, the terminal cost becomes:

$$g(x) = \log \frac{\hat{\varphi}_1(x)}{\nu(x)} = E(x) + \log \hat{\varphi}_1(x) + \text{const}$$

### Adjoint Matching (AM) Objective — Equation (14)

$$\mathcal{L}*{\text{AM}}^{(k)}(\theta) = \mathbb{E}*{p_{t|0,1}^{\text{base}} , p_{0,1}^{\bar{u}}} \left[\left|u_t^\theta(X_t) + \sigma_t \big(\nabla E + h^{(k-1)}\big)(X_1)\right|^2\right], \quad \bar{u} = \text{stopgrad}(u)$$

**What this does:** The drift $u_t^\theta$ is regressed toward a target that depends on:

1. The **energy gradient** $\nabla E(X_1)$ — points toward lower energy
1. The **corrector** $h^{(k-1)}(X_1) \approx \nabla \log \hat{\varphi}_1(X_1)$ — debiases the non-memoryless process

**How the expectation is computed:**

- $X_0 \sim \mu$ (sample from source)
- $X_1 \sim p^{\bar{u}}(X_1|X_0)$ (run the current model forward with **stopped gradients**)
- $X_t \sim p_{t|0,1}^{\text{base}}(X_t|X_0, X_1)$ (sample the **Brownian bridge** between $X_0$ and $X_1$)

The Brownian bridge for the base process with $f_t := 0$ is:

$$p_{t|0,1}^{\text{base}}(X_t|X_0, X_1) = \mathcal{N}!\left(X_t;, \frac{(1-t) \bar{\sigma}_t^2}{\bar{\sigma}*1^2} X_0 + \frac{t \cdot \bar{\sigma}*{1-t}^2}{\bar{\sigma}_1^2} X_1,, \frac{\bar{\sigma}*t^2 \bar{\sigma}*{1-t}^2}{\bar{\sigma}_1^2} I_d\right)$$

where $\bar{\sigma}*t^2 := \int_0^t \sigma*\tau^2 d\tau$ is the cumulative variance.

The **regression target** for the AM loss at a given $(X_t, X_1)$ is:

$$v_t^{\text{AM}}(X_t, X_1) = -\sigma_t \big(\nabla E(X_1) + h^{(k-1)}(X_1)\big) + \sigma_t \nabla_{X_t} \log p_{1|t}^{\text{base}}(X_1|X_t)$$

For Brownian motion base, $\nabla_{X_t} \log p_{1|t}^{\text{base}}(X_1|X_t) = \frac{X_1 - X_t}{\bar{\sigma}_1^2 - \bar{\sigma}_t^2}$.

### Corrector Matching (CM) Objective — Equation (15)

$$\mathcal{L}*{\text{CM}}^{(k)}(\phi) = \mathbb{E}*{p_{0,1}^{u^{(k)}}} \left[\left|h_\phi(X_1) - \nabla_{X_1} \log p^{\text{base}}(X_1|X_0)\right|^2\right]$$

**What this does:** The corrector $h_\phi$ is regressed toward the score of the base transition kernel $p^{\text{base}}(X_1|X_0)$, evaluated at endpoints of on-policy trajectories from the current drift $u^{(k)}$.

For Brownian motion base:

$$\nabla_{X_1} \log p^{\text{base}}(X_1|X_0) = \nabla_{X_1} \log \mathcal{N}(X_1; X_0, \bar{\sigma}_1^2 I_d) = -\frac{X_1 - X_0}{\bar{\sigma}_1^2}$$

So the CM target is simply $-\frac{X_1 - X_0}{\bar{\sigma}_1^2}$, which is trivially computable.

**Crucial insight:** Neither objective requires samples from the target $\nu$. Both use only **on-policy samples** from the current model.

### Special Case: Memoryless ($\mu = \delta_0$)

When $X_0 = 0$ (Dirac delta):

- $\nabla_{X_1} \log p^{\text{base}}(X_1|X_0) = -X_1/\bar{\sigma}_1^2 = \nabla \log p_1^{\text{base}}(X_1)$
- The CM objective is minimized by $h^* = \nabla \log p_1^{\text{base}}$, which is known analytically
- The AM objective reduces to the original Adjoint Sampling objective
- No alternating optimization is needed — **AS is a special case of ASBS**

-----

## 8. Alternating Optimization (Algorithm 1)

```
Algorithm 1: Adjoint Schrödinger Bridge Sampler (ASBS)
─────────────────────────────────────────────────────
Require: Sample-able source X₀ ~ μ, differentiable energy E(x),
         parametrized u_θ(t,x) and h_ϕ(x)

1: Initialize h_ϕ^(0) := 0
2: for stage k = 1, 2, ... do
3:    Update drift u_θ^(k) by minimizing L_AM^(k)     ▷ Adjoint Matching
4:    Update corrector h_ϕ^(k) by minimizing L_CM^(k)  ▷ Corrector Matching
5: end for
```

### Stage-by-stage behavior

**Stage k=1 (initialization):**

- $h^{(0)} = 0$, so the AM target is just $-\sigma_t \nabla E(X_1)$
- The drift is learned to push samples toward low-energy regions
- This is essentially gradient descent on the energy landscape, transported through an SDE

**Stage k=2 onwards:**

- The corrector $h^{(1)}$ has been learned from stage 1 samples
- The AM target becomes $-\sigma_t(\nabla E + h^{(1)})(X_1)$
- The corrector corrects for the bias introduced by the non-memoryless source
- Samples progressively improve, which improves the corrector, which improves samples, etc.

### Inner loop structure

Within each stage $k$, both AM and CM are trained with standard gradient descent over many iterations. The paper uses **replay buffers** to reuse previously generated trajectories, enabling many gradient updates per energy evaluation (key scalability advantage).

-----

## 9. Convergence Theory

### Theorem 4.1: AM solves a Forward Half Bridge

The path distribution $p^{u^{(k)}}$ from stage $k$ solves:

$$p^{u^{(k)}} = \arg\min_p \left{D_{\text{KL}}(p | q^{\bar{h}^{(k-1)}}) : p_0 = \mu\right}$$

where $q^{\bar{h}^{(k-1)}}$ is a backward SDE defined by the corrector. This minimizes KL subject to the **source** constraint only (forward half bridge).

### Theorem 4.2: CM solves a Backward Half Bridge

The path distribution $q^{\bar{h}^{(k)}}$ from the CM update solves:

$$q^{\bar{h}^{(k)}} = \arg\min_q \left{D_{\text{KL}}(p^{u^{(k)}} | q) : q_1 = \nu\right}$$

This minimizes KL subject to the **target** constraint only (backward half bridge).

### Theorem 3.2: Global Convergence

> Algorithm 1 converges to the Schrödinger bridge solution, provided all matching stages achieve their critical points.

$$\lim_{k \to \infty} u^{(k)} = u^*$$

**Proof sketch:** Theorems 4.1 and 4.2 show that ASBS alternates between forward and backward projections, exactly implementing the **Iterative Proportional Fitting (IPF)** algorithm (also known as Sinkhorn iterations in optimal transport). IPF is known to converge to the SB solution.

-----

## 10. Practical Implementation Details

### Noise Schedule

The paper uses a **geometric noise schedule**:

$$\sigma_t = \sigma_{\min}^{1-t} \cdot \sigma_{\max}^t$$

The cumulative variance is $\bar{\sigma}*t^2 = \int_0^t \sigma*\tau^2 d\tau$, computed numerically.

### Time Discretization

The SDE is integrated with **Euler-Maruyama** using $N$ uniformly spaced steps:

$$X_{t_{n+1}} = X_{t_n} + \sigma_{t_n} u_\theta(t_n, X_{t_n}) \Delta t + \sigma_{t_n} \sqrt{\Delta t}, \epsilon_n$$

where $t_n = n/N$ and $\Delta t = 1/N$.

### Replay Buffer

A **replay buffer** stores tuples $(X_0, X_1, \text{energy}, \text{trajectory})$ from previous model evaluations. During training:

1. Sample a batch of new trajectories (requires energy evaluation)
1. Add to replay buffer
1. Sample multiple mini-batches from the buffer for gradient updates
1. This gives a **many-to-one ratio** of gradient updates to energy evaluations

### Brownian Bridge Sampling

Given endpoints $(X_0, X_1)$ from the replay buffer, intermediate points $X_t$ are sampled from the Brownian bridge:

$$X_t = \alpha_t X_0 + \beta_t X_1 + \gamma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_d)$$

where:

$$\alpha_t = \frac{(1-t)\bar{\sigma}_t^2}{\bar{\sigma}*1^2}, \quad \beta_t = \frac{t \cdot \bar{\sigma}*{1-t}^2}{\bar{\sigma}_1^2}, \quad \gamma_t^2 = \frac{\bar{\sigma}*t^2 \cdot \bar{\sigma}*{1-t}^2}{\bar{\sigma}_1^2}$$

(Note: for $f_t := 0$, the bridge depends on the specific form of $\sigma_t$.)

### Stop Gradient

The AM objective uses **stop gradient** on $u$: the trajectories are generated with the current model weights, but gradients do **not** flow through the trajectory generation. This makes the loss a simple regression.

### Source Distributions

**Gaussian prior:** $\mu(x) = \mathcal{N}(x; 0, \sigma_0^2 I_d)$

**Harmonic prior** (for $n$-particle systems): For particles $x = {x_i}_{i=1}^n$:

$$\mu_{\text{harmonic}}(x) \propto \exp!\left(-\frac{\alpha}{2} \sum_{i,j} |x_i - x_j|^2\right)$$

This is equivalent to sampling from an anisotropic Gaussian. The harmonic prior encodes the prior knowledge that particles in a molecule should be close to each other.

-----

## 11. Energy Functions & Experiments

### 11.1 Synthetic Energy Functions

All are based on **pairwise distances** of $n$-particle systems in $D$ spatial dimensions (total dimension $d = nD$).

#### Double-Well (DW-4)

- **Particles:** $n = 4$ in 2D → $d = 8$
- **Energy:** Sum of pairwise double-well potentials
- **Ground truth:** MCMC samples from Klein et al. (2023)

#### Many-Well (MW-5)

- **Particles:** $n = 5$ in 1D → $d = 5$
- **Energy:** Multi-modal potential with many wells
- **Ground truth:** Analytical sampling (possible for this system)
- **Metric:** Sinkhorn distance

#### Lennard-Jones 13 (LJ-13)

- **Particles:** $n = 13$ in 3D → $d = 39$
- **Energy:** Standard LJ potential: $E(x) = \sum_{i<j} 4\varepsilon\left[\left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6\right]$
- **Ground truth:** MCMC samples

#### Lennard-Jones 55 (LJ-55)

- **Particles:** $n = 55$ in 3D → $d = 165$
- **Energy:** Same LJ potential, much larger system
- **Ground truth:** MCMC samples

### 11.2 Alanine Dipeptide

- **System:** 22-atom molecule in implicit solvent
- **Dimension:** $d = 60$ (internal coordinates — bond lengths, angles, torsions)
- **Energy:** OpenMM library (Eastman et al., 2017) at 300K
- **Ground truth:** $10^7$ configurations from Molecular Dynamics
- **Evaluation:** KL divergence on 5 torsion angle marginals ($\phi, \psi, \gamma_1, \gamma_2, \gamma_3$) and Wasserstein-2 on joint $(\phi, \psi)$ Ramachandran plot

### 11.3 Amortized Conformer Generation

- **Task:** Conditional generation $\nu(x|g) \propto e^{-E(x|g)/\tau}$ at low temperature $\tau \ll 1$, conditioned on molecular topology $g$
- **Training molecules:** 24,477 from SPICE (SMILES strings only — **no atomic configurations used**)
- **Test molecules:** 80 from SPICE + 80 from GEOM-DRUGS
- **Energy function:** eSEN foundation model (Fu et al., 2025) — predicts DFT-accuracy energies cheaply
- **Ground truth:** CREST conformers
- **Model:** EGNN conditioned on molecular graph
- **Evaluation:** Coverage recall (%) and Absolute Mean RMSD at threshold 1.0Å

### 11.4 Model Architectures

**For particle systems (DW, LJ, conformers):**

- **Equivariant Graph Neural Network (EGNN)** by Satorras et al. (2021)
- Preserves SE(3) equivariance (rotation + translation invariance)
- Separate EGNNs for drift $u_\theta$ and corrector $h_\phi$

**For alanine dipeptide (internal coordinates):**

- Standard fully-connected neural networks (MLPs)
- Internal coordinates don’t require equivariance

-----

## 12. Evaluation Metrics

### Wasserstein-2 Distance ($W_2$)

Measures the optimal transport distance between generated and ground-truth samples:

$$W_2^2(\hat{p}, p) = \min_{\gamma \in \Gamma(\hat{p}, p)} \mathbb{E}_{(X,Y) \sim \gamma}!\left[|X - Y|^2\right]$$

### Energy Wasserstein ($E(\cdot)$-$W_2$)

Same as $W_2$ but computed on the **energy values** rather than the sample positions:

$$E(\cdot)\text{-}W_2^2 = W_2^2!\big({E(x_i)}*{i=1}^N, {E(y_j)}*{j=1}^M\big)$$

This captures whether the model generates samples at the correct energy levels.

### Sinkhorn Distance

An entropy-regularized approximation to optimal transport, used for MW-5.

### KL Divergence (for torsion marginals)

For alanine dipeptide, 1D histograms of each torsion angle are compared via:

$$D_{\text{KL}}(\hat{p}*\alpha | p*\alpha) = \sum_i \hat{p}*\alpha(i) \log \frac{\hat{p}*\alpha(i)}{p_\alpha(i)}$$

### Coverage Recall (for conformers)

Given a set of reference conformers $\mathcal{R}$ and generated conformers $\mathcal{G}$, **recall** at threshold $\delta$ is:

$$\text{Recall}(\delta) = \frac{|{r \in \mathcal{R} : \min_{g \in \mathcal{G}} \text{RMSD}(r, g) < \delta}|}{|\mathcal{R}|}$$

i.e., the fraction of reference conformers that are “covered” by at least one generated conformer within RMSD $\delta$.

### Absolute Mean RMSD (AMR)

For each matched pair, the average RMSD across all covered references.

-----

## Summary of Key Equations

|Equation                                                                                       |Role                         |Number in Paper             |
|-----------------------------------------------------------------------------------------------|-----------------------------|----------------------------|
|$\nu(x) = e^{-E(x)}/Z$                                                                         |Target Boltzmann distribution|(1)                         |
|$dX_t = (f_t + \sigma_t u_\theta) dt + \sigma_t dW_t$                                          |Diffusion sampler SDE        |(2)                         |
|$\min_u D_{\text{KL}}(p^u | p^{\text{base}})$ s.t. boundaries                                  |Schrödinger Bridge problem   |(3)                         |
|$\min_u \mathbb{E}[\int \frac{1}{2}|u_t|^2 dt + g(X_1)]$                                       |Stochastic Optimal Control   |(4)                         |
|$g(x) = \log \hat{\varphi}_1(x) / \nu(x)$                                                      |SOC terminal cost (ASBS)     |Thm 3.1                     |
|$\mathcal{L}*{\text{AM}}^{(k)} = \mathbb{E}|u*\theta + \sigma_t(\nabla E + h^{(k-1)})|^2$      |Adjoint Matching objective   |(14)                        |
|$\mathcal{L}*{\text{CM}}^{(k)} = \mathbb{E}|h*\phi(X_1) - \nabla_{X_1} \log p^{\text{base}}(X_1|X_0)|^2$                     |Corrector Matching objective|
|$\mu_{\text{harmonic}} \propto \exp(-\frac{\alpha}{2}\sum_{ij}|x_i - x_j|^2)$                  |Harmonic prior               |(19)                        |

-----

## Appendix: The Full Picture in One Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ASBS Training Loop                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Stage k                                               │   │
│  │                                                       │   │
│  │  1. ADJOINT MATCHING (update u_θ):                    │   │
│  │     X₀ ~ μ(source)                                   │   │
│  │     X₁ ~ SDE with u_θ (stop grad)                    │   │
│  │     X_t ~ BrownianBridge(X₀, X₁)                     │   │
│  │     target = -σ_t(∇E(X₁) + h_ϕ^(k-1)(X₁))          │   │
│  │            + σ_t ∇log p_base(X₁|X_t)                 │   │
│  │     loss = ‖u_θ(t, X_t) - target‖²                   │   │
│  │                                                       │   │
│  │  2. CORRECTOR MATCHING (update h_ϕ):                  │   │
│  │     X₀ ~ μ(source)                                   │   │
│  │     X₁ ~ SDE with u_θ^(k) (new drift)               │   │
│  │     target = -(X₁ - X₀) / σ̄₁²                       │   │
│  │     loss = ‖h_ϕ(X₁) - target‖²                       │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓ repeat                             │
│                   Converges to SB solution                   │
└─────────────────────────────────────────────────────────────┘
```