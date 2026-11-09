# v3 analysis: experts, hybrid, conformal

## In-distribution (mean ± sd over seeds; validation-tuned threshold)

| Benchmark | weights d/l/n | det F1 / AUROC | lex F1 / AUROC | nov F1 / AUROC | hybrid F1 / AUROC |
|---|---|---|---|---|---|
| a2a_sandbox | 0.66/0.16/0.18 | 0.962±0.005 / 0.997 | 0.955±0.000 / 0.995 | 0.631±0.001 / 0.687 | 0.960±0.003 / 0.996 |
| mcptox | 0.02/0.98/0.00 | 0.498±0.080 / 0.781 | 0.890±0.000 / 0.917 | 0.446±0.019 / 0.498 | 0.891±0.002 / 0.916 |
| mcp_safetybench | 0.44/0.48/0.08 | 0.651±0.135 / 0.761 | 0.822±0.000 / 0.935 | 0.568±0.016 / 0.626 | 0.846±0.064 / 0.954 |
| agentdojo | 0.34/0.66/0.00 | 0.882±0.053 / 0.987 | 1.000±0.000 / 1.000 | 0.458±0.000 / 0.523 | 0.961±0.054 / 1.000 |
| injecagent | 0.48/0.52/0.00 | 0.929±0.111 / 1.000 | 1.000±0.000 / 1.000 | 0.456±0.004 / 0.923 | 1.000±0.000 / 1.000 |

## Hybrid minus lexical (paired bootstrap over test episodes; Holm across benchmarks)

| Benchmark | ΔF1 [95% CI] | p (Holm) | ΔAUROC [95% CI] | p |
|---|---|---|---|---|
| a2a_sandbox | +0.006 [-0.005, +0.016] | 1.000 | +0.001 [-0.000, +0.003] | 0.169 |
| mcptox | +0.001 [+0.000, +0.004] | 1.000 | -0.001 [-0.002, +0.001] | 0.253 |
| mcp_safetybench | +0.025 [-0.038, +0.094] | 1.000 | +0.020 [-0.010, +0.061] | 0.257 |
| agentdojo | -0.040 [-0.077, -0.011] | 0.020 | +0.000 [+0.000, +0.000] | 1.000 |
| injecagent | +0.000 [+0.000, +0.000] | 1.000 | +0.000 [+0.000, +0.000] | 1.000 |

## Conformal false-alarm control (test FPR / recall; alpha = 0.05 split, and ACI)

| Benchmark | det split | lex split | nov split | hybrid split | det ACI | lex ACI | nov ACI | hybrid ACI |
|---|---|---|---|---|---|---|---|---|
| a2a_sandbox | 0.033 / 0.956 | 0.048 / 0.958 | 0.034 / 0.213 | 0.043 / 0.964 | 0.050 / 0.847 | 0.052 / 0.894 | 0.048 / 0.253 | 0.048 / 0.878 |
| mcptox | 0.171 / 0.506 | 0.186 / 0.590 | 0.029 / 0.035 | 0.186 / 0.596 | 0.057 / 0.233 | 0.071 / 0.221 | 0.060 / 0.086 | 0.071 / 0.223 |
| mcp_safetybench | n/a | n/a | n/a | n/a | 0.058 / 0.400 | 0.125 / 0.514 | 0.092 / 0.211 | 0.075 / 0.691 |
| agentdojo | n/a | n/a | n/a | n/a | 0.114 / 0.653 | 0.143 / 0.433 | 0.071 / 0.033 | 0.114 / 0.591 |
| injecagent | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Held-out templates (mean over folds of the per-fold seed means)

| Benchmark | folds | det F1 / AUROC | lex F1 / AUROC | nov F1 / AUROC | hybrid F1 / AUROC | hybrid>lex (F1) | hybrid>lex (AUROC) |
|---|---|---|---|---|---|---|---|
| a2a_sandbox | 11 | 0.374 / 0.518 | 0.467 / 0.606 | 0.423 / 0.482 | 0.399 / 0.469 | 2/11 | 3/11 |
| mcptox | 6 | 0.531 / 0.669 | 0.675 / 0.870 | 0.460 / 0.516 | 0.661 / 0.870 | 1/18 | 1/18 |
| mcp_safetybench | 4 | 0.464 / 0.596 | 0.528 / 0.670 | 0.465 / 0.623 | 0.568 / 0.658 | 8/12 | 5/12 |
| agentdojo | 6 | 0.798 / 0.915 | 0.902 / 0.981 | 0.523 / 0.557 | 0.902 / 0.981 | 0/18 | 0/18 |
| injecagent | 4 | 0.924 / 0.999 | 0.955 / 1.000 | 0.446 / 0.828 | 0.946 / 0.999 | 0/12 | 0/12 |

Sign test over all held-out (fold, seed) pairs: hybrid beats lexical in F1 11/71 (p=2.63e-09), in AUROC 9/71 (p=7.34e-11).

