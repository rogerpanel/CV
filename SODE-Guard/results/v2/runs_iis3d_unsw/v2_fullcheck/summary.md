# SODE-Guard v2 re-run: 1 seed(s) [0]

Data: 21074 flows after deduplication, 10 classes. Mean ± half-width of the 95% t-interval over seeds.

## Clean and adversarial (l-inf eps = 0.03, z-units)

| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |
|---|---|---|---|---|---|
| lgbm | 0.819 ± 0.000 | 0.858 ± 0.000 | 0.155 ± 0.000 | – | 0.160 ± 0.000 |
| mlp | 0.643 ± 0.000 | 0.759 ± 0.000 | 0.410 ± 0.000 | – | 0.390 ± 0.000 |
| mlp_at | 0.583 ± 0.000 | 0.715 ± 0.000 | 0.535 ± 0.000 | – | 0.529 ± 0.000 |
| ode_at | 0.598 ± 0.000 | 0.723 ± 0.000 | 0.528 ± 0.000 | – | 0.529 ± 0.000 |
| sode | 0.668 ± 0.000 | 0.783 ± 0.000 | 0.403 ± 0.000 | 0.400 ± 0.000 | 0.387 ± 0.000 |
| sode_at | 0.602 ± 0.000 | 0.725 ± 0.000 | 0.533 ± 0.000 | 0.530 ± 0.000 | 0.532 ± 0.000 |

## Per-class clean F1 (mean over seeds)

| model | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Normal | Reconnaissance | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.84 | 0.66 | 0.48 | 0.77 | 0.87 | 0.99 | 0.89 | 0.93 | 0.85 | 0.91 |
| mlp | 0.70 | 0.27 | 0.35 | 0.70 | 0.78 | 0.97 | 0.88 | 0.73 | 0.61 | 0.43 |
| mlp_at | 0.66 | 0.11 | 0.32 | 0.68 | 0.75 | 0.97 | 0.88 | 0.62 | 0.42 | 0.41 |
| ode_at | 0.71 | 0.11 | 0.26 | 0.70 | 0.76 | 0.97 | 0.89 | 0.62 | 0.43 | 0.54 |
| sode | 0.75 | 0.26 | 0.38 | 0.71 | 0.80 | 0.97 | 0.89 | 0.79 | 0.65 | 0.47 |
| sode_at | 0.71 | 0.11 | 0.31 | 0.70 | 0.76 | 0.98 | 0.89 | 0.61 | 0.43 | 0.52 |

## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)

| contrast | difference A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| sode vs mlp (clean) | +0.025 | [+0.009, +0.042] | 0.020 | 0.060 |
| sode vs lgbm (clean) | -0.151 | [-0.173, -0.129] | 0.010 | 0.050 |
| module 2: sode_at vs sode (worst case) | +0.144 | [+0.124, +0.163] | 0.010 | 0.050 |
| sode_at vs mlp_at (worst case) | +0.002 | [-0.011, +0.016] | 0.816 | 0.856 |
| sode_at vs ode_at (worst case) | +0.003 | [-0.004, +0.010] | 0.428 | 0.856 |
