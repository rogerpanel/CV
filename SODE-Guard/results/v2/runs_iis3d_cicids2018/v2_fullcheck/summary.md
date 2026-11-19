# SODE-Guard v2 re-run: 1 seed(s) [0]

Data: 37880 flows after deduplication, 11 classes. Mean ± half-width of the 95% t-interval over seeds.

## Clean and adversarial (l-inf eps = 0.03, z-units)

| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |
|---|---|---|---|---|---|
| lgbm | 0.907 ± 0.000 | 0.887 ± 0.000 | 0.088 ± 0.000 | – | 0.088 ± 0.000 |
| mlp | 0.883 ± 0.000 | 0.878 ± 0.000 | 0.714 ± 0.000 | – | 0.708 ± 0.000 |
| mlp_at | 0.880 ± 0.000 | 0.875 ± 0.000 | 0.875 ± 0.000 | – | 0.869 ± 0.000 |
| ode_at | 0.886 ± 0.000 | 0.879 ± 0.000 | 0.875 ± 0.000 | – | 0.846 ± 0.000 |
| sode | 0.899 ± 0.000 | 0.881 ± 0.000 | 0.706 ± 0.000 | 0.703 ± 0.000 | 0.677 ± 0.000 |
| sode_at | 0.879 ± 0.000 | 0.879 ± 0.000 | 0.873 ± 0.000 | 0.872 ± 0.000 | 0.867 ± 0.000 |

## Per-class clean F1 (mean over seeds)

| model | Benign | Bot | Brute Force -Web | DDOS attack-HOIC | DDOS attack-LOIC-UDP | DDoS attacks-LOIC-HTTP | DoS attacks-GoldenEye | DoS attacks-Hulk | DoS attacks-Slowloris | Infilteration | SSH-Bruteforce |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.57 | 1.00 | 0.87 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 0.56 | 1.00 |
| mlp | 0.63 | 1.00 | 0.72 | 1.00 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 0.40 | 1.00 |
| mlp_at | 0.61 | 1.00 | 0.72 | 1.00 | 0.97 | 0.99 | 0.99 | 1.00 | 1.00 | 0.41 | 1.00 |
| ode_at | 0.64 | 1.00 | 0.77 | 1.00 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 0.38 | 1.00 |
| sode | 0.60 | 1.00 | 0.85 | 1.00 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 0.49 | 1.00 |
| sode_at | 0.64 | 1.00 | 0.70 | 1.00 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 0.37 | 1.00 |

## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)

| contrast | difference A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| sode vs mlp (clean) | +0.016 | [+0.002, +0.030] | 0.040 | 0.119 |
| sode vs lgbm (clean) | -0.008 | [-0.017, +0.003] | 0.119 | 0.239 |
| module 2: sode_at vs sode (worst case) | +0.190 | [+0.170, +0.218] | 0.010 | 0.050 |
| sode_at vs mlp_at (worst case) | -0.002 | [-0.009, +0.003] | 0.507 | 0.507 |
| sode_at vs ode_at (worst case) | +0.021 | [+0.003, +0.040] | 0.010 | 0.050 |
