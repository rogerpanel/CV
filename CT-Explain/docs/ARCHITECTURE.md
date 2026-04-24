# Architecture

This document walks through each module in `ct_explain/` and links it to
the exact equation, algorithm, and table in the manuscript
*CT-Explain: Explainable Continuous-Time Dynamic Graph Neural Networks
with Conformal Safety Certification for Human-AI Collaborative Intrusion
Detection in Security Operations Centers*.

## 1. Graph construction — §3 · Fig. 2

`ct_explain.data.graph_builder` maintains the continuously-evolving
heterogeneous graph G(t) = (V(t), E(t), X(t)). Nodes are network entities
(IP addresses, services, users), edges are timestamped communications with
per-edge feature vectors e_{vu} ∈ ℝ^{d_e}.

## 2. CT-TGNN backbone — Eq. 1 · §3.1

`ct_explain.models.ct_tgnn.CTTemporalGNN` implements

    dh_v/dt = f_θ(h_v(t), ⊕_{u ∈ N(v,t)} α_{vu}(t) · g_φ(h_u(t), e_{vu}), t)

with

* `TemporalMultiHeadAttention` (§3.1, Eq. 2) for α_{vu}(t),
* `EdgeConditionedMessage` (g_φ), and
* `NeuralODEFunc` (f_θ).

Integration uses Dormand–Prince (dopri5) via `torchdiffeq` with adjoint
sensitivity. On environments without torchdiffeq, a fixed-step Euler
fallback preserves functionality at reduced accuracy.

## 3. SDE extension — Eq. 3 · §3.1

`ct_explain.models.sde_tgnn.SDETemporalGNN` adds σ_ψ(h) dW_t so that
the Fokker–Planck moment closure in §3B can propagate variance to the
terminal time T.

## 4. Temporal attention — Eq. 2 · §3.1

See `ct_explain.models.temporal_attention`; the multi-scale time
encoding φ(Δt) uses learnable time constants {τ_k}.

## 5. Explainability engine — §3.2

| Sub-component                      | File                                     | Eq.     |
| ---------------------------------- | ---------------------------------------- | ------- |
| 3A. Attention flow + kill-chain    | `explainers/attention_flow.py`           | §3A     |
| 3B. Fokker–Planck uncertainty      | `explainers/uncertainty_attribution.py`  | Eq. 4–5 |
| 3C. Counterfactual trajectories    | `explainers/counterfactual.py`           | Eq. 6   |
| 3D. Game-theoretic strategy        | `explainers/game_theory.py`              | §3D     |
| Calibrated explanations            | `explainers/calibrated.py`               | Eq. 9   |
| MITRE ATT&CK mapper                | `explainers/mitre_attack.py`             | §3A     |

## 6. ConformalGuard — §4

| Component                           | File                                 | Eq.     |
| ----------------------------------- | ------------------------------------ | ------- |
| Dynamic execution graph             | `conformal/execution_graph.py`       | §4.1    |
| Graph conformal score / Theorem 1   | `conformal/graph_conformal.py`       | Eq. 7   |
| E-value martingale (Ville)          | `conformal/e_value.py`               | Eq. 8   |
| Orchestrator (`.calibrate/.certify/.monitor`) | `conformal/conformal_guard.py` | §4.3 |
| Split-conformal helper              | `conformal/calibration.py`           | §4      |

## 7. Training — §5

`ct_explain.training.trainer.CTExplainTrainer` runs a standard cross-
entropy/focal-loss loop with AdamW, cosine LR schedule, and gradient
clipping. The adjoint helper in `training/adjoint.py` provides Eq. 6's
backward ODE when torchdiffeq is unavailable.

## 8. Evaluation — §6

* `evaluation/detection.py` — F1/precision/recall/MCC/κ/Brier/ECE
* `evaluation/explanation.py` — F-Fidelity, Max-Sensitivity, Effective
  Complexity (with optional Quantus verification).
* `evaluation/conformal.py` — Empirical coverage, Worst-Slab Coverage,
  SSCV, bootstrap 95% CI.
* `evaluation/statistical.py` — Friedman + Nemenyi CD, Wilcoxon,
  Bonferroni.

## 9. SOC layer — §5.4, §7

`ct_explain.soc` hosts the analyst-facing objects — alert triage queue,
investigation panel (four explanations + calibrated CIs + conformal
verdict), active-learning buffer + Bayesian linear-head updater, and a
vendor-neutral SIEM plugin with a default NetFlow-v2 translator.

## 10. REST API — §7

`ct_explain.api.server` wires the SOC layer onto a Flask blueprint.
Endpoints follow the RobustIDPS.ai naming convention (`/api/v1/ct-explain/
<action>`).

## 11. RobustIDPS.ai registration — §7

`ct_explain.integration.robustidps.register(app, collab, registry)` adds
the plugin metadata to the platform's model registry and mounts all
CT-Explain routes on the host Flask app.

## 12. Hyperparameters

All defaults live in `configs/default.yaml`; sweeps for conformal study
in `configs/conformal.yaml`; dataset manifest in `configs/datasets.yaml`.
