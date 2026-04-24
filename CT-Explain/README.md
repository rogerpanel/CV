# CT-Explain

**Explainable Continuous-Time Dynamic Graph Neural Networks with Conformal
Safety Certification for Human-AI Collaborative Intrusion Detection in
Security Operations Centers.**

Reproducibility artifact for the manuscript of the same name (companion repo
[`rogerpanel/CT-Explain`](https://github.com/rogerpanel/CT-Explain)). The
module here is the end-to-end implementation that the paper's empirical
section relies on: the CT-TGNN / SDE-TGNN detection backbone, the four
explanation techniques, the ConformalGuard safety-certification layer, the
Calibrated-Explanations bridge, and the SOC-analyst dashboard / active
learning loop. It is designed to be imported as a plugin into the larger
[RobustIDPS.ai](../robustidps_web_app) platform that hosts all models the
author has developed.

This folder is the *reproducibility link* that can be cited verbatim in the
IEEE submission — every equation, algorithm, and evaluation protocol in the
manuscript maps to a module documented below.

---

## 1. Framework overview

The paper's Problem 1 / Problem 2 decomposition is preserved one-to-one:

```
raw network traffic ──► Graph Construction ──► CT-TGNN / SDE-TGNN ──► verdict + uncertainty
                                                   │
                                                   ├──► Explainability Engine (4 techniques)
                                                   │       ├─ Temporal attention flow
                                                   │       ├─ Fokker–Planck uncertainty attribution
                                                   │       ├─ Counterfactual trajectory (adjoint)
                                                   │       └─ Game-theoretic adversary strategy
                                                   │
                                                   └──► ConformalGuard
                                                           ├─ Graph conformal prediction (Thm. 1)
                                                           ├─ E-value martingale (Ville's inequality)
                                                           └─ Calibrated Explanations (attribution CIs)
                                                   │
                                                   ▼
                                           SOC analyst dashboard
                                           (alert triage, investigation,
                                            active-learning feedback)
```

Module map:

| Paper component                       | Package path                                          |
| ------------------------------------- | ----------------------------------------------------- |
| Graph construction G(t)=(V,E,X)       | `ct_explain.data.graph_builder`                       |
| CT-TGNN Neural ODE backbone (Eq. 1)   | `ct_explain.models.ct_tgnn`                           |
| Temporal attention (Eq. 2)            | `ct_explain.models.temporal_attention`                |
| SDE-TGNN diffusion extension (Eq. 3)  | `ct_explain.models.sde_tgnn`                          |
| Dormand–Prince adjoint integration    | `ct_explain.training.adjoint`                         |
| Temporal-attention flow explainer (3A)| `ct_explain.explainers.attention_flow`                |
| Fokker–Planck uncertainty (3B, Eq. 4–5)| `ct_explain.explainers.uncertainty_attribution`      |
| Counterfactual trajectories (3C, Eq. 6)| `ct_explain.explainers.counterfactual`               |
| Game-theoretic strategy (3D)          | `ct_explain.explainers.game_theory`                   |
| MITRE ATT&CK kill-chain alignment     | `ct_explain.explainers.mitre_attack`                  |
| Calibrated Explanations (Eq. 9)       | `ct_explain.explainers.calibrated`                    |
| Graph conformal scoring (Eq. 7, Thm. 1)| `ct_explain.conformal.graph_conformal`               |
| E-value martingale (Eq. 8)            | `ct_explain.conformal.e_value`                        |
| Dynamic agent execution graph         | `ct_explain.conformal.execution_graph`                |
| ConformalGuard orchestrator           | `ct_explain.conformal.conformal_guard`                |
| Split conformal calibration           | `ct_explain.conformal.calibration`                    |
| Training loop / loss                  | `ct_explain.training.trainer`, `.losses`              |
| Detection / explanation / conformal metrics | `ct_explain.evaluation.*`                        |
| SOC dashboard & active learning       | `ct_explain.soc.*`                                    |
| REST API + SIEM plugin                | `ct_explain.api.server`, `ct_explain.soc.siem_plugin` |
| RobustIDPS.ai model-registry hook     | `ct_explain.integration.robustidps`                   |

---

## 2. Installation

```bash
cd CT-Explain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Prefer a single-file download? Grab `CT-Explain.zip` (repo root or inside
the `CT-Explain/` folder) — it contains the whole `CT-Explain/` directory
so reviewers can reproduce the manuscript without cloning the full CV
monorepo.

Minimum Python 3.10. GPU is optional; the CT-TGNN backbone works on CPU at
reduced throughput. Heavy dependencies (`torchdiffeq`, `torch_geometric`,
`torchsde`) are listed as *optional-but-recommended* in `requirements.txt` —
the package degrades gracefully when they are absent (see
`ct_explain.utils.compat`).

---

## 3. Datasets

Six benchmarks × 84.2 M records × 83 attack classes are used. The loaders in
`ct_explain.data.datasets` accept either (a) the raw CSV/PCAP files placed
under `data/` or (b) the Kaggle-hosted bundles referenced in the manuscript.

| Bundle                            | DOI                                                | Loader                         |
| --------------------------------- | -------------------------------------------------- | ------------------------------ |
| General network traffic (3 sets)  | `10.34740/kaggle/dsv/12483891`                     | `GeneralNetworkTrafficDataset` |
| Cloud / microservice / edge (3 sets) | `10.34740/KAGGLE/DSV/12479689`                  | `CloudEdgeDataset`             |

Individual constituents (CIC-IoT-2023, CSE-CICIDS2018, UNSW-NB15, MS GUIDE,
Container-NID, Edge-IIoT) are addressable through the same loader by name.
See `docs/DATASETS.md` for the feature dictionary and splits.

```bash
python scripts/download_data.py --bundle general --dest data/
python scripts/download_data.py --bundle cloud-edge --dest data/
```

---

## 4. Training, explanation, certification

```bash
# Train CT-TGNN on CIC-IoT-2023
python scripts/train.py --config configs/default.yaml --dataset cic-iot-2023

# Calibrate ConformalGuard on 30% held-out split
python scripts/calibrate.py --checkpoint runs/latest/ckpt.pt --alpha 0.05

# Explain a single alert with all four techniques + calibrated intervals
python scripts/explain.py --checkpoint runs/latest/ckpt.pt --alert examples/alert_42.json

# Reproduce main table (all 6 datasets, 20 cal/test splits, bootstrap CI)
python scripts/evaluate.py --config configs/default.yaml --all-datasets
```

All CLI scripts write deterministic output under `runs/<timestamp>/`;
seeds are set via `ct_explain.utils.seed.set_global_seed`.

---

## 5. Integration with RobustIDPS.ai

`ct_explain.integration.robustidps.register()` exposes CT-Explain as a
first-class model in the RobustIDPS.ai registry alongside SurrogateIDS,
MambaShield, CL-RL Unified, FedGTD, CyberSecLLM, Multi-Agent PQC-IDS, and the
rest of the 13-model ensemble. It wires:

* `/api/models/ct_explain/predict` — detection + uncertainty,
* `/api/models/ct_explain/explain/<alert_id>` — four-way explanation,
* `/api/models/ct_explain/certify` — ConformalGuard verdict + e-value,
* `/api/models/ct_explain/feedback` — Bayesian active-learning updates,

and registers a SOC-dashboard widget pack (alert triage, investigation,
counterfactual explorer, game-theoretic narrative).

---

## 6. Reproducing the paper's tables

Every table in the manuscript has an entry-point:

| Table / figure                          | Reproducer                                                       |
| --------------------------------------- | ---------------------------------------------------------------- |
| Main detection table (6 datasets)       | `scripts/evaluate.py --all-datasets`                             |
| Explanation quality on CIC-IoT-2023     | `scripts/evaluate.py --suite explanation --dataset cic-iot-2023` |
| Critical-difference diagram             | `scripts/evaluate.py --suite cd-diagram`                         |
| ConformalGuard coverage (α = 0.05)      | `scripts/evaluate.py --suite conformal`                          |
| Lab study + field deployment summary    | `scripts/evaluate.py --suite user-study`                         |

Artifacts (PDF plots, LaTeX tables) land in `runs/<timestamp>/report/`.

---

## 7. Citation

If you use this codebase, please cite the manuscript and the companion
platform:

```bibtex
@article{anaedevha2026ctexplain,
  title   = {CT-Explain: Explainable Continuous-Time Dynamic Graph Neural
             Networks with Conformal Safety Certification for Human-AI
             Collaborative Intrusion Detection in Security Operations Centers},
  author  = {Anaedevha, Roger Nick},
  journal = {IEEE (under review)},
  year    = {2026}
}
```

Platform: RobustIDPS.ai — DOI `10.5281/zenodo.19129512`.
Datasets: Kaggle DOIs `10.34740/kaggle/dsv/12483891` and `10.34740/KAGGLE/DSV/12479689`.

License: MIT (see `LICENSE`).
