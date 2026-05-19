# SODE-Guard Methodology

This note expands the methodology sections of the TNNLS manuscript and indexes
each design decision to the corresponding code path.

## 1. Problem formulation

A network flow is represented by the 83-dimensional vector
`x ∈ ℝ⁸³` defined in [`src/data/feature_engineering.py`](../src/data/feature_engineering.py).
Given a labelled training set
`D = { (xᵢ, yᵢ) }`, the goal is to learn a classifier `F_θ` that minimises

> the cross-entropy loss on clean inputs **and** the *anti-concentration mass*
> of its margin under bounded perturbations `‖δ‖_∞ ≤ ε`.

## 2. SODE-Guard end-to-end pipeline

| Stage | File | Description |
|---|---|---|
| Encoding | [`src/models/egraphsage.py`](../src/models/egraphsage.py) | E-GraphSAGE attention-gated residual encoder produces `h₀ ∈ ℝ¹²⁸`. |
| SDE drift | [`src/models/drift_diffusion.py`](../src/models/drift_diffusion.py) | 3-layer GELU MLP with spectral norm (Lipschitz ≤ 1). |
| SDE diffusion | same | 3-layer MLP outputting `(d × m)` matrix; ellipticity floor `λ₀=10⁻³`. |
| Integration | [`src/sde/integrator.py`](../src/sde/integrator.py) | Euler–Maruyama with `dt=0.05` (20 steps), virtual Brownian tree. |
| Adjoint | [`src/sde/adjoint.py`](../src/sde/adjoint.py) | `torchsde.sdeint_adjoint` if installed, else discrete checkpointed autograd. |
| Inference | [`src/models/sode_guard.py`](../src/models/sode_guard.py) `forward_mc` | Averages softmax across `N_mc=8` paths. |
| Anti-concentration | [`src/regularizers/anti_concentration.py`](../src/regularizers/anti_concentration.py) | Carbery–Wright surrogate `L_AC`. |
| Loss | [`src/training/loss.py`](../src/training/loss.py) | `L = L_CE + λ · L_AC`, `λ = 0.10`. |

## 3. Algorithm 1 — Training one SODE-Guard step

```
Input:  batch (x_b, y_b),  paths P,  AC weight λ
1. h₀  ← Encoder(x_b)
2. for p in 1..P:
       sample W^(p) on [0, 1] via virtual-Brownian-tree
       h_T^(p) ← EulerMaruyama(h₀, f_θ, g_θ; W^(p))
       z^(p)   ← Head(h_T^(p))
3. z̄ ← mean_p z^(p)
4. L_CE ← cross_entropy(z̄, y_b)
5. L_AC ← AC(z^(1..P))          (Wiener-chaos surrogate of Prop. 1)
6. backprop through stochastic adjoint (Li et al. 2020)
7. update θ with Adam(lr=5e-4, wd=1e-5);  spectral norm projection in-place
```

## 4. Algorithm 2 — Anti-concentration certificate at inference

```
Input:  x,  N=256 paths,  chaos degree d* = 4,  β = 0.05,  conf = 0.95
1. for s in 1..N: h_T^(s) ← integrate SDE with seed s
2. probs ← mean_s softmax(Head(h_T^(s)))
3. margin ← probs[top1] − probs[top2]
4. L_g    ← local Lipschitz estimate via 4 random probes (eps_probe = 1e-3)
5. r*     ← (|margin| / L_g) · ( (1 − conf) / (C · d*) )^{d*}     # Carbery–Wright invert
6. return  prediction = argmax probs,  radius = r*
```

## 5. Why ellipticity + spectral norm together?

* The ellipticity floor `g g^T ⪰ λ₀ I` ensures the **Bismut–Elworthy–Li**
  formula gives a finite bound on `∇log p_T`, which is what links the
  Wiener-chaos truncation to a finite Lipschitz constant.
* Spectral normalisation on `f_θ` and `g_θ` keeps the **Lipschitz constant
  `L_g`** of the smoothed margin uniformly bounded, so the certificate
  radius `r*` does not collapse as `t → T`.

Empirically (ablation in Table 4), removing either ingredient (the `SDE_TGNN`
baseline) drops PGD-40 robustness from 93.1% to 87.2% on ICS3D at ε=0.03.

## 6. Connecting to RobustIDPS deployment

SODE-Guard is registered in `robustidps.ai v3` as model id
`sode_guard` (category `temporal`). It runs alongside the 14 detectors listed
in [`docs/ROBUSTIDPS_INTEGRATION.md`](ROBUSTIDPS_INTEGRATION.md). Latency
(P50 = 1.7 ms on A100) is fast enough for the platform's WebSocket Live
Monitor stream; we therefore do **not** enable RobustIDPS "fast mode" for
this model — the Monte-Carlo paths are required for the certificate.
