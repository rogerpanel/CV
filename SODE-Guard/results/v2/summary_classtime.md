# SODE-Guard v2 re-run: 3 seed(s) [0, 1, 2]

Data: 68627 flows after deduplication, 9 classes. Mean ± half-width of the 95% t-interval over seeds.

## Clean and adversarial (l-inf eps = 0.03, z-units)

| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |
|---|---|---|---|---|---|
| lgbm | 0.602 ± 0.029 | 0.961 ± 0.006 | 0.040 ± 0.025 | – | 0.058 ± 0.027 |
| mlp | 0.492 ± 0.024 | 0.947 ± 0.009 | 0.330 ± 0.032 | – | 0.358 ± 0.039 |
| mlp_at | 0.484 ± 0.012 | 0.949 ± 0.004 | 0.435 ± 0.021 | – | 0.465 ± 0.004 |
| ode | 0.494 ± 0.038 | 0.951 ± 0.003 | 0.309 ± 0.060 | – | 0.340 ± 0.063 |
| sode | 0.494 ± 0.042 | 0.951 ± 0.004 | 0.309 ± 0.058 | 0.308 ± 0.057 | 0.342 ± 0.054 |
| sode_at | 0.491 ± 0.015 | 0.951 ± 0.001 | 0.434 ± 0.017 | 0.424 ± 0.006 | 0.462 ± 0.019 |

## Per-class clean F1 (mean over seeds)

| model | 0 | 1 | 11 | 2 | 3 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.99 | 0.95 | 0.94 | 0.99 | 0.05 | 0.00 | 0.70 | 0.31 | 0.50 |
| mlp | 0.99 | 0.93 | 0.93 | 0.98 | 0.03 | 0.07 | 0.17 | 0.00 | 0.33 |
| mlp_at | 0.99 | 0.93 | 0.93 | 0.99 | 0.06 | 0.02 | 0.14 | 0.00 | 0.31 |
| ode | 0.99 | 0.94 | 0.93 | 0.98 | 0.00 | 0.00 | 0.19 | 0.00 | 0.43 |
| sode | 0.99 | 0.94 | 0.93 | 0.98 | 0.00 | 0.00 | 0.20 | 0.00 | 0.42 |
| sode_at | 0.99 | 0.94 | 0.93 | 0.99 | 0.02 | 0.03 | 0.19 | 0.03 | 0.31 |

## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)

| contrast | difference A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| sode vs mlp (clean) | +0.002 | [-0.008, +0.011] | 0.747 | 1.000 |
| sode vs ode (clean) | -0.000 | [-0.005, +0.004] | 0.926 | 1.000 |
| sode vs lgbm (clean) | -0.109 | [-0.127, -0.092] | 0.004 | 0.020 |
| module 2: sode_at vs sode (worst case) | +0.120 | [+0.108, +0.138] | 0.004 | 0.020 |
| sode_at vs mlp_at (worst case) | -0.003 | [-0.012, +0.008] | 0.703 | 1.000 |
