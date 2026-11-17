# SODE-Guard v2 re-run: 1 seed(s) [0]

Data: 68627 flows after deduplication, 9 classes. Mean ± half-width of the 95% t-interval over seeds.

## Clean and adversarial (l-inf eps = 0.03, z-units)

| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |
|---|---|---|---|---|---|
| lgbm | 0.626 ± 0.000 | 0.953 ± 0.000 | 0.031 ± 0.000 | – | 0.035 ± 0.000 |
| mlp | 0.478 ± 0.000 | 0.920 ± 0.000 | 0.277 ± 0.000 | – | 0.277 ± 0.000 |
| mlp_at | 0.471 ± 0.000 | 0.900 ± 0.000 | 0.440 ± 0.000 | – | 0.424 ± 0.000 |
| ode_at | 0.465 ± 0.000 | 0.905 ± 0.000 | 0.442 ± 0.000 | – | 0.435 ± 0.000 |
| sode | 0.503 ± 0.000 | 0.938 ± 0.000 | 0.240 ± 0.000 | 0.239 ± 0.000 | 0.253 ± 0.000 |
| sode_at | 0.475 ± 0.000 | 0.906 ± 0.000 | 0.427 ± 0.000 | 0.408 ± 0.000 | 0.425 ± 0.000 |

## Certified accuracy (randomised smoothing, l2)

| model | r = 0.0 | r = 0.05 | r = 0.1 | r = 0.2 | r = 0.273 | r = 0.5 | ACR |
|---|---|---|---|---|---|---|---|
| mlp_rs | 0.586 ± 0.000 | 0.584 ± 0.000 | 0.578 ± 0.000 | 0.566 ± 0.000 | 0.554 ± 0.000 | 0.494 ± 0.000 | 0.391 ± 0.000 |
| sode_rs | 0.634 ± 0.000 | 0.616 ± 0.000 | 0.614 ± 0.000 | 0.606 ± 0.000 | 0.602 ± 0.000 | 0.570 ± 0.000 | 0.430 ± 0.000 |

## Per-class clean F1 (mean over seeds)

| model | 0 | 1 | 11 | 2 | 3 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.98 | 0.94 | 0.98 | 0.98 | 0.09 | 0.06 | 0.81 | 0.33 | 0.46 |
| mlp | 0.95 | 0.91 | 0.98 | 0.95 | 0.04 | 0.00 | 0.25 | 0.04 | 0.19 |
| mlp_at | 0.94 | 0.88 | 0.98 | 0.95 | 0.04 | 0.03 | 0.20 | 0.00 | 0.22 |
| ode_at | 0.94 | 0.89 | 0.98 | 0.95 | 0.05 | 0.00 | 0.21 | 0.00 | 0.16 |
| sode | 0.95 | 0.93 | 0.98 | 0.96 | 0.04 | 0.00 | 0.33 | 0.00 | 0.34 |
| sode_at | 0.94 | 0.89 | 0.98 | 0.95 | 0.05 | 0.04 | 0.27 | 0.00 | 0.16 |

## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)

| contrast | difference A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| sode vs mlp (clean) | +0.025 | [+0.008, +0.042] | 0.010 | 0.060 |
| sode vs lgbm (clean) | -0.123 | [-0.153, -0.090] | 0.010 | 0.060 |
| module 2: sode_at vs sode (worst case) | +0.172 | [+0.154, +0.193] | 0.010 | 0.060 |
| sode_at vs mlp_at (worst case) | +0.001 | [-0.009, +0.010] | 0.826 | 0.826 |
| sode_at vs ode_at (worst case) | -0.010 | [-0.016, -0.005] | 0.010 | 0.060 |
| module 3: sode_rs vs mlp_rs (certified @0.273) | +0.048 | [+0.030, +0.070] | 0.010 | 0.060 |
