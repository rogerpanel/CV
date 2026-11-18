# SODE-Guard v2 re-run: 1 seed(s) [0]

Data: 37996 flows after deduplication, 11 classes. Mean ± half-width of the 95% t-interval over seeds.

## Clean and adversarial (l-inf eps = 0.03, z-units)

| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |
|---|---|---|---|---|---|
| lgbm | 0.932 ± 0.000 | 0.903 ± 0.000 | 0.230 ± 0.000 | – | 0.230 ± 0.000 |
| mlp | 0.929 ± 0.000 | 0.903 ± 0.000 | 0.679 ± 0.000 | – | 0.675 ± 0.000 |
| mlp_at | 0.906 ± 0.000 | 0.884 ± 0.000 | 0.896 ± 0.000 | – | 0.879 ± 0.000 |
| ode_at | 0.903 ± 0.000 | 0.881 ± 0.000 | 0.883 ± 0.000 | – | 0.875 ± 0.000 |
| sode | 0.923 ± 0.000 | 0.900 ± 0.000 | 0.706 ± 0.000 | 0.704 ± 0.000 | 0.678 ± 0.000 |
| sode_at | 0.900 ± 0.000 | 0.883 ± 0.000 | 0.895 ± 0.000 | 0.879 ± 0.000 | 0.886 ± 0.000 |

## Per-class clean F1 (mean over seeds)

| model | Benign | Bot | Brute Force -Web | DDOS attack-HOIC | DDOS attack-LOIC-UDP | DDoS attacks-LOIC-HTTP | DoS attacks-GoldenEye | DoS attacks-Hulk | DoS attacks-Slowloris | Infilteration | SSH-Bruteforce |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.64 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.61 | 1.00 |
| mlp | 0.63 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 |
| mlp_at | 0.64 | 1.00 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.42 | 1.00 |
| ode_at | 0.63 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.41 | 1.00 |
| sode | 0.65 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 | 1.00 |
| sode_at | 0.63 | 1.00 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.45 | 1.00 |

## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)

| contrast | difference A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| sode vs mlp (clean) | -0.006 | [-0.013, +0.001] | 0.090 | 0.179 |
| sode vs lgbm (clean) | -0.008 | [-0.015, -0.001] | 0.020 | 0.060 |
| module 2: sode_at vs sode (worst case) | +0.208 | [+0.181, +0.244] | 0.010 | 0.050 |
| sode_at vs mlp_at (worst case) | +0.007 | [-0.008, +0.025] | 0.597 | 0.597 |
| sode_at vs ode_at (worst case) | +0.011 | [+0.003, +0.020] | 0.010 | 0.050 |
