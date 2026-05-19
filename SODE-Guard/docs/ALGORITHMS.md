# Algorithms

Full pseudocode used by the manuscript, cross-referenced to the implementation.

## Algorithm 1 — SODE-Guard training step (`src/training/train.py`)

```
Inputs:
    batch (x_b, y_b),     paths P (= max(2, N_mc//2) = 4 by default),
    AC weight λ = 0.10,   chaos degree d* = 4, β-grid B,
    ellipticity floor λ₀ = 1e-3, dt = 0.05, horizon T = 1.0.

1.  h₀ ← Encoder(x_b)                           # E-GraphSAGE → ℝ^{B×128}
2.  for p = 1..P do
        seed_p ← p
        sample (W^p_t)_{t∈[0,T]} from virtual Brownian tree(seed_p)
        x_0 ← h₀
        for k = 0 .. T/dt − 1 do
            t_k ← k · dt
            f ← f_θ(x_k, t_k)
            g ← g_θ(x_k, t_k)
            g ← g + sqrt(λ₀) · I_{:m}            # ellipticity floor projection
            ΔW ← W^p_{t_{k+1}} − W^p_{t_k}
            x_{k+1} ← x_k + f · dt + g · ΔW
        z^p ← Head(x_T)
3.  z̄ ← mean_p z^p
4.  L_CE ← CE(z̄, y_b)
5.  m ← top1(z̄) − top2(z̄)                      # smoothed margin
6.  for β in B:
        L_AC ← L_AC + log(1 + C·d*·(β/‖m‖₂)^(1/d*) · σ(50·(β − |m|)).mean())
    L_AC ← L_AC / |B|
7.  L ← L_CE + λ · L_AC
8.  ∇θ L via stochastic adjoint (torchsde) or discrete autograd fallback
9.  Adam step (lr 5e-4, wd 1e-5); spectral-norm reprojected by parametrisation
10. cosine-anneal lr; clip gradient L2-norm to 1.0
```

## Algorithm 2 — Anti-concentration certificate (`src/evaluation/certificate.py`)

```
Inputs:
    test point x,  N = 256 MC paths,  d* = 4, β = 0.05, conf = 0.95.

1. compute probabilities p̄(x) = (1/N) Σ_s softmax(Head(SDE(Encoder(x); seed=s)))
2. ŷ ← argmax p̄(x)
3. m ← top1(p̄) − top2(p̄)
4. L_g ← max over 4 unit directions d of ‖log p̄(x+εd) − log p̄(x)‖ / ε   (ε=1e-3)
5. r* ← (|m| / L_g) · ((1 − conf) / (2 · d*))^{d*}
6. return (ŷ, r*)
```

## Algorithm 3 — PGD-40 with EOT for stochastic models (`src/attacks/pgd.py`)

```
Inputs:
    model M,   ε,   steps K = 40,   α = 2.5 ε / K,
    EOT samples E = 1  (raise to 4 for adaptive evaluation).

1. δ₀ ← Uniform(−ε, +ε)         (random start)
2. x_adv ← clip(x + δ₀, 0, 1)
3. for k = 1..K:
       grad ← 0
       for e = 1..E:
           ℓ ← CE(M(x_adv), y)
           grad ← grad + ∇_{x_adv} ℓ
       grad ← grad / E
       x_adv ← x_adv + α · sign(grad)
       x_adv ← x + clip(x_adv − x, −ε, +ε)
       x_adv ← clip(x_adv, 0, 1)
4. return x_adv
```

## Algorithm 4 — Carlini–Wagner ℓ2 (`src/attacks/cw.py`)

```
Inputs: model M, target labels y, c = 1.0, κ = 0, iter = 100, lr = 0.01.

Parametrise: x_adv = ½ (tanh w + 1) · (hi − lo) + lo.
Minimise:    L(w) = ‖x_adv − x‖² + c · max(real_y − max_{k≠y} z_k + κ, 0).
Optimiser:   Adam(w; lr).
Return:      x_adv after `iter` Adam steps.
```

## Algorithm 5 — Federated batched training fix (RobustIDPS v3 patch)

Documented for completeness: the v3 fix moves federated graph-temporal
training from full-batch adjacency matrix construction (160 GB peak) to a
mini-batched SGD with batch size 512, dropping memory by five orders of
magnitude. SODE-Guard does **not** use federated training directly but uses
the same mini-batched data pipeline (`src/data/registry.py`).
