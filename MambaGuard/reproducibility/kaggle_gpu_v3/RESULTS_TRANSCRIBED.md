# v3 GPU study: transcribed notebook output

**Provenance.** These tables are the text that the committed Kaggle version
printed in its final cell (`scripts/summarise_kaggle_results.py` and
`scripts/analyse_v3.py` run on `/kaggle/working/results_v3_all.zip`,
session 2026-11-08 11:39:51 to 14:26:37, Tesla T4, torch 2.11.0+cu130,
132 run files, 0 errors). The zip itself could not be downloaded (the
notebook's file-link proxy had expired), so the per-episode prediction files
of this run are not in the repository; the numbers below are copied verbatim
from the printed output. Re-running `kaggle/run_kaggle_study.py --study all`
regenerates them (Kaggle notebook cell 8).

Configuration: main detector = flat read-out MambaGuard with chunk scoring
(`chunk_mil`), seeds 0-4; held-out folds seeds 0-2 (A2A seed 0); validation
split by group into a selection half and a calibration half; lexical and
novelty experts scored on the same episodes; analysis pre-registered in
`scripts/analyse_v3.py` (Platt-calibrated experts, hybrid weights chosen on
the selection half, split conformal and ACI on the calibration half).

# MambaGuard Kaggle results

- session: study=all GPU=Tesla T4 torch=2.11.0+cu130 encoder=chunk_mil started=2026-11-08 11:39:51 finished=2026-11-08 14:26:37
- 132 run files, 5 certificates, 5 latency reports, 0 errors

## Main results (encoder: chunk_mil; threshold tuned on val; mean ± sd over seeds)

| Benchmark | n test (benign) | MambaGuard F1 | AUROC | recall | FPR | Lexical F1 | Lexical AUROC | Length |
|---|---|---|---|---|---|---|---|---|
| A2A sandbox | 660 (330) | 0.962±0.005 | 0.997±0.000 | 0.968±0.010 | 0.045±0.018 | 0.955 | 0.995 | 0.53 |
| MCPTox | 265 (70) | 0.498±0.080 | 0.781±0.052 | 0.335±0.127 | 0.034±0.053 | 0.899 | 0.917 | 0.5 |
| MCP-SafetyBench | 59 (24) | 0.651±0.135 | 0.761±0.155 | 0.646±0.160 | 0.300±0.305 | 0.822 | 0.935 | 0.5 |
| AgentDojo | 104 (14) | 0.880±0.055 | 0.987±0.009 | 0.924±0.051 | 0.029±0.064 | 1.000 | 1.000 | 0.5 |
| InjecAgent | 189 (3) | 0.949±0.113 | 1.000±0.001 | 1.000±0.000 | 0.133±0.298 | 1.000 | 1.000 | 0.5 |

## Held-out templates (test attacks use a wording/family absent from training)

| Benchmark | fold | test att/ben | MambaGuard F1 | AUROC | recall | FPR | Lexical F1 | Lexical AUROC | recall |
|---|---|---|---|---|---|---|---|---|---|
| A2A sandbox | finding=agent_card_jws_trust | 15/15 | 0.330 | 0.338 | 0.400 | 0.733 | 0.333 | 0.449 | 0.000 |
| A2A sandbox | finding=artifact_chunk_corruption | 15/15 | 0.333 | 0.453 | 1.000 | 1.000 | 0.333 | 0.489 | 1.000 |
| A2A sandbox | finding=cross_client_context_injection | 15/15 | 0.333 | 0.756 | 0.000 | 0.000 | 0.333 | 0.622 | 0.000 |
| A2A sandbox | finding=indefinite_interrupted_tasks | 15/15 | 0.333 | 0.556 | 0.000 | 0.000 | 0.333 | 1.000 | 0.000 |
| A2A sandbox | finding=multi_hop_identity_loss | 15/15 | 0.333 | 0.436 | 0.000 | 0.000 | 0.333 | 0.569 | 0.000 |
| A2A sandbox | finding=post_revocation_stream | 15/15 | 0.444 | 0.760 | 0.133 | 0.067 | 0.433 | 0.400 | 0.400 |
| A2A sandbox | finding=scope_amplification | 15/15 | 0.597 | 0.573 | 0.333 | 0.067 | 0.597 | 0.893 | 0.933 |
| A2A sandbox | finding=stale_task_state | 15/15 | 0.333 | 0.173 | 0.000 | 0.000 | 0.333 | 0.284 | 0.000 |
| A2A sandbox | finding=unattested_skill_claims | 15/15 | 0.333 | 0.587 | 0.000 | 0.000 | 0.466 | 0.347 | 0.133 |
| A2A sandbox | finding=unbounded_delegation_depth | 15/15 | 0.466 | 0.858 | 0.133 | 0.000 | 0.864 | 1.000 | 0.733 |
| A2A sandbox | finding=unverified_webhook_url | 15/15 | 0.280 | 0.209 | 0.467 | 0.867 | 0.641 | 0.609 | 0.400 |
| **A2A sandbox** | **mean over folds** | | 0.374±0.092 | 0.518±0.223 | 0.224±0.312 | 0.248±0.402 | 0.455±0.177 | 0.606±0.254 | 0.327±0.396 |
| MCPTox | paradigm=Template-1 | 28/70 | 0.534±0.106 | 0.646±0.047 | 0.131±0.115 | 0.005±0.008 | 0.832 | 0.950 | 0.964 |
| MCPTox | paradigm=Template-2 | 72/70 | 0.525±0.027 | 0.678±0.054 | 0.236±0.014 | 0.067±0.050 | 0.587 | 0.862 | 0.389 |
| MCPTox | paradigm=Template-3 | 95/70 | 0.524±0.064 | 0.641±0.025 | 0.375±0.152 | 0.243±0.151 | 0.589 | 0.851 | 0.326 |
| MCPTox | wrapper=ignore_previous | 60/70 | 0.581±0.046 | 0.662±0.048 | 0.289±0.038 | 0.062±0.050 | 0.854 | 0.849 | 0.917 |
| MCPTox | wrapper=important_tag | 66/70 | 0.524±0.149 | 0.704±0.018 | 0.232±0.192 | 0.043±0.043 | 0.574 | 0.843 | 0.288 |
| MCPTox | wrapper=plain | 69/70 | 0.500±0.067 | 0.682±0.023 | 0.193±0.080 | 0.048±0.059 | 0.616 | 0.866 | 0.319 |
| **MCPTox** | **mean over folds** | | 0.531±0.027 | 0.669±0.024 | 0.243±0.083 | 0.078±0.084 | 0.675±0.131 | 0.870±0.040 | 0.534±0.317 |
| MCP-SafetyBench | family=host_user | 6/24 | 0.420±0.160 | 0.736±0.014 | 0.833±0.289 | 0.639±0.339 | 0.498 | 0.646 | 0.667 |
| MCP-SafetyBench | family=name_competition | 9/24 | 0.440±0.034 | 0.460±0.065 | 0.370±0.231 | 0.431±0.265 | 0.492 | 0.671 | 0.556 |
| MCP-SafetyBench | family=poisoning | 15/24 | 0.482±0.089 | 0.486±0.003 | 0.178±0.168 | 0.111±0.158 | 0.494 | 0.661 | 0.133 |
| MCP-SafetyBench | family=result_injection | 5/24 | 0.485±0.130 | 0.703±0.126 | 0.800±0.000 | 0.542±0.191 | 0.628 | 0.700 | 0.200 |
| **MCP-SafetyBench** | **mean over folds** | | 0.457±0.032 | 0.596±0.143 | 0.545±0.323 | 0.431±0.229 | 0.528±0.067 | 0.670±0.023 | 0.389±0.262 |
| AgentDojo | family=direct | 40/14 | 0.618±0.092 | 0.782±0.033 | 0.575±0.132 | 0.190±0.109 | 0.871 | 0.938 | 0.850 |
| AgentDojo | family=dos | 40/14 | 0.806±0.066 | 0.952±0.028 | 0.900±0.025 | 0.286±0.124 | 0.949 | 0.995 | 1.000 |
| AgentDojo | family=ignore_previous | 40/14 | 0.856±0.061 | 0.943±0.040 | 0.925±0.043 | 0.214±0.071 | 0.804 | 0.966 | 0.825 |
| AgentDojo | family=important_instructions | 40/14 | 0.896±0.096 | 0.985±0.012 | 0.967±0.029 | 0.190±0.218 | 0.949 | 1.000 | 1.000 |
| AgentDojo | family=injecagent | 40/14 | 0.907±0.060 | 0.976±0.010 | 0.958±0.014 | 0.143±0.189 | 0.899 | 0.986 | 0.975 |
| AgentDojo | family=tool_knowledge | 40/14 | 0.705±0.245 | 0.854±0.204 | 0.967±0.038 | 0.524±0.436 | 0.975 | 1.000 | 1.000 |
| **AgentDojo** | **mean over folds** | | 0.798±0.115 | 0.915±0.080 | 0.882±0.153 | 0.258±0.138 | 0.908±0.064 | 0.981±0.025 | 0.942±0.082 |
| InjecAgent | attacker_cases=0 | 27/3 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.000±0.000 | 1.000 | 1.000 | 1.000 |
| InjecAgent | attacker_cases=1 | 57/3 | 0.965±0.060 | 1.000±0.000 | 1.000±0.000 | 0.111±0.192 | 1.000 | 1.000 | 1.000 |
| InjecAgent | attacker_cases=2 | 51/3 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.000±0.000 | 1.000 | 1.000 | 1.000 |
| InjecAgent | attacker_cases=3 | 51/3 | 0.802±0.027 | 0.996±0.008 | 0.948±0.011 | 0.111±0.192 | 0.818 | 1.000 | 0.941 |
| **InjecAgent** | **mean over folds** | | 0.942±0.094 | 0.999±0.002 | 0.987±0.026 | 0.056±0.064 | 0.955±0.091 | 1.000±0.000 | 0.985±0.029 |

## Certified accuracy (seed-0 main model)

**A2A sandbox** (100 windows, n=500)

| sigma | abstain | r=0.0 | r=0.02 | r=0.05 | r=0.1 | r=0.25 |
|---|---|---|---|---|---|---|
| 0.05 | 0.01 | 0.950 | 0.940 | 0.900 | 0.490 | 0.000 |
| 0.1 | 0.08 | 0.870 | 0.840 | 0.670 | 0.460 | 0.000 |
| 0.25 | 0.34 | 0.550 | 0.520 | 0.390 | 0.230 | 0.030 |

**MCPTox** (100 windows, n=500)

| sigma | abstain | r=0.0 | r=0.02 | r=0.05 | r=0.1 | r=0.25 |
|---|---|---|---|---|---|---|
| 0.05 | 0.02 | 0.680 | 0.650 | 0.600 | 0.420 | 0.000 |
| 0.1 | 0.05 | 0.630 | 0.600 | 0.550 | 0.400 | 0.000 |
| 0.25 | 0.19 | 0.530 | 0.480 | 0.390 | 0.240 | 0.000 |

**MCP-SafetyBench** (59 windows, n=500)

| sigma | abstain | r=0.0 | r=0.02 | r=0.05 | r=0.1 | r=0.25 |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 0.627 | 0.492 | 0.356 | 0.102 | 0.000 |
| 0.1 | 0.34 | 0.508 | 0.390 | 0.288 | 0.102 | 0.000 |
| 0.25 | 0.54 | 0.373 | 0.288 | 0.169 | 0.068 | 0.017 |

**AgentDojo** (100 windows, n=500)

| sigma | abstain | r=0.0 | r=0.02 | r=0.05 | r=0.1 | r=0.25 |
|---|---|---|---|---|---|---|
| 0.05 | 0.03 | 0.930 | 0.930 | 0.930 | 0.760 | 0.000 |
| 0.1 | 0.02 | 0.930 | 0.930 | 0.920 | 0.710 | 0.000 |
| 0.25 | 0.09 | 0.880 | 0.870 | 0.870 | 0.680 | 0.240 |

**InjecAgent** (100 windows, n=500)

| sigma | abstain | r=0.0 | r=0.02 | r=0.05 | r=0.1 | r=0.25 |
|---|---|---|---|---|---|---|
| 0.05 | 0.02 | 0.980 | 0.980 | 0.980 | 0.960 | 0.000 |
| 0.1 | 0.02 | 0.980 | 0.980 | 0.980 | 0.960 | 0.000 |
| 0.25 | 0.03 | 0.970 | 0.960 | 0.960 | 0.960 | 0.690 |

## Latency (forward pass on the GPU; payload encoding listed separately)

| Benchmark | batch-1 ms/window | best msg/s | encoder ms/msg (batch 1) | encoder ms/msg (batched) |
|---|---|---|---|---|
| A2A sandbox | 6.14 | 23312 | 8.15 | 1.92 |
| MCPTox | 5.41 | 33371 | 18.61 | 2.01 |
| MCP-SafetyBench | 5.95 | 25458 | 8.68 | 3.50 |
| AgentDojo | 5.45 | 21486 | 8.72 | 2.20 |
| InjecAgent | 5.56 | 36558 | 8.11 | 1.31 |

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
