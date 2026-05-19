# Stochastic ODE-Guard (SODE-Guard)

**A Neural Stochastic Differential Equation Framework with Anti-Concentration Bounds for Adversarially Robust Network Intrusion Detection.**

This repository is the official reproducibility package for the manuscript:

> Anaedevha, R. N. *Stochastic ODE-Guard: A Neural Stochastic Differential Equation Framework with Anti-Concentration Bounds for Adversarially Robust Network Intrusion Detection.* Submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2026.

It accompanies the LaTeX manuscript hosted at
[`rogerpanel/SODE-ExtractGuard-Models`](https://github.com/rogerpanel/SODE-ExtractGuard-Models) and extends the
`robustidps.ai` v3 deployment platform documented in the same upstream repository. SODE-Guard is registered as
the temporal-stochastic detector inside the broader RobustIDPS ensemble alongside SurrogateIDS-7B, SDE-TGNN,
SSL-GraphAnomaly, PPFOT-IDS, LipMamba, Mamba–CrossAttn–MoE, and the federated graph-temporal-dynamics
suite (`fedgtd`).

## 1. What is SODE-Guard?

SODE-Guard treats network flows as paths of an Itô stochastic differential equation
```
dX_t = f_θ(X_t, t) dt + g_θ(X_t, t) dW_t,    t ∈ [0, 1]
```
with jointly learned drift `f_θ` and diffusion `g_θ`. Flow records are embedded to ℝ¹²⁸ via an
**E-GraphSAGE** edge encoder, evolved under the SDE for unit horizon, and classified by a
linear head on the expected terminal state. The diffusion satisfies an ellipticity floor
`g g^⊤ ⪰ λ₀ I` (default λ₀=10⁻³), which together with spectral normalisation of the drift
yields a **Bismut–Elworthy–Li** gradient estimator and, via a Wiener-chaos / Carbery–Wright
argument, the anti-concentration certificate

> **Proposition (Anti-Concentration Certificate).** For any classifier margin `g`, smoothed
> through SODE-Guard with chaos degree `d*`, the perturbation `δ` satisfies
> `Pr[ |g(x+δ) − g(x)| ≤ β ] ≤ C · d* · (β / (L_g · ε))^{1/d*}`.

The dimension enters only via the chaos degree `d*` (default 4) rather than linearly,
which is what allows SODE-Guard to keep 93.1% macro-F1 under PGD-40 at `ε=0.03` on
flow benchmarks containing 18.9M records, against 87.2% for the strongest internal
baseline SDE-TGNN.

## 2. Repository layout

```
SODE-Guard/
├── configs/                   # YAML hyperparameter files (one per experiment family)
├── docs/                      # Methodology, dataset cards, theory notes, references
├── experiments/               # Logs, checkpoints, result CSVs (gitignored by default)
├── notebooks/                 # Jupyter walkthroughs (training, certification, attacks)
├── scripts/                   # Data download, run-all, plotting, certificate computation
├── src/
│   ├── attacks/               # FGSM, PGD, C&W, DeepFool, Gaussian, label-mask poisoning
│   ├── baselines/             # E-GraphSAGE, RTIDS, CNN-LSTM, IDS-GraphMamba, SDE-TGNN,
│   │                          # SurrogateIDS-7B, Llama-Guard adapter, Snort/Suricata stubs
│   ├── data/                  # Loaders for ICS3D, IIS3D, IDS-PQC, CIC-IDS2017/2018/2023,
│   │                          # UNSW-NB15, NSL-KDD, NF-ToN-IoT-V2, CIC-DDoS-2019
│   ├── evaluation/            # Macro-F1, ECE, certified radius, Friedman + McNemar
│   ├── models/                # SODE-Guard, E-GraphSAGE encoder, classifier heads
│   ├── regularizers/          # Anti-concentration loss, spectral normalisation, ellipticity
│   ├── sde/                   # Euler–Maruyama, virtual Brownian tree, stochastic adjoint
│   ├── training/              # Train loop, AC schedule, AMP, distributed
│   └── utils/                 # Seeds, logging, config IO, metrics helpers
└── tests/                     # Unit tests for SDE solver, AC bound, attack budgets
```

## 3. Datasets

SODE-Guard is evaluated on three Kaggle benchmark suites published alongside the manuscript,
plus the canonical public IDS corpora used by the RobustIDPS platform.

| Dataset | Flows | Classes | DOI / URL |
|---|---|---|---|
| **ICS3D** (Azure cloud + Edge-IIoT + K8s) | 18.9 M | 17 | [10.34740/kaggle/dsv/12483891](https://doi.org/10.34740/kaggle/dsv/12483891) |
| **IIS3D** (UNSW-NB15 + CIC-IDS2018/2023) | 13.4 M | 23 | [10.34740/kaggle/dsv/12479689](https://doi.org/10.34740/kaggle/dsv/12479689) |
| **IDS-PQC** (TLS 1.3 + post-quantum handshakes) | 3.1 M | 9 | [10.34740/kaggle/dsv/15424420](https://doi.org/10.34740/kaggle/dsv/15424420) |
| CICIDS2017 | 2.8 M | 15 | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| CIC-IDS2018 | 16.2 M | 14 | <https://www.unb.ca/cic/datasets/ids-2018.html> |
| CIC-IDS2023 | 13.4 M | 33 | <https://www.unb.ca/cic/datasets/ids-2023.html> |
| CIC-IoT-2023 | 46.7 M | 33 | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> |
| CIC-DDoS-2019 | 50.0 M | 13 | <https://www.unb.ca/cic/datasets/ddos-2019.html> |
| UNSW-NB15 | 2.5 M | 10 | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| NSL-KDD | 148 K | 5 | <https://www.unb.ca/cic/datasets/nsl.html> |
| NF-ToN-IoT-V2 | 16.9 M | 10 | <https://staff.itee.uq.edu.au/marius/NIDS_datasets/> |

Stratified splits are 70/15/15. Standardisation, missing-value imputation, and the 83-dim
feature vector definition follow `src/data/feature_engineering.py`. Run
`bash scripts/download_data.sh --all` to fetch every corpus (requires a Kaggle API token
in `~/.kaggle/kaggle.json`).

## 4. Quickstart

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch a small subset (NSL-KDD ≈ 18 MB) for smoke tests
bash scripts/download_data.sh --datasets nslkdd

# 3. Train SODE-Guard on the smoke set with all five reproducibility seeds
python -m src.training.train --config configs/sode_guard_smoke.yaml

# 4. Evaluate clean + PGD-40 robustness and compute the anti-concentration certificate
python -m src.evaluation.run_eval --config configs/sode_guard_smoke.yaml \
    --attacks pgd40 --epsilons 0.005 0.01 0.02 0.03 0.05 0.10 \
    --certify --chaos-degree 4
```

The full TNNLS reproduction (Tables 2–4, Figure 5) is launched with
`bash scripts/reproduce_paper.sh` which sweeps the five seeds {42, 137, 271, 1729, 2026}
on all three Kaggle benchmarks.

## 5. Citing this work

```bibtex
@article{anaedevha2026sodeguard,
  title  = {Stochastic ODE-Guard: A Neural Stochastic Differential Equation Framework
            with Anti-Concentration Bounds for Adversarially Robust Network
            Intrusion Detection},
  author = {Anaedevha, Roger Nick},
  journal= {IEEE Transactions on Neural Networks and Learning Systems (submitted)},
  year   = {2026}
}
```

## 6. Licence

MIT — see [`LICENSE`](LICENSE). The Kaggle datasets retain their own licences (CC-BY
4.0 for ICS3D / IIS3D / IDS-PQC; see each DOI landing page for the others).

## 7. Reproducibility link

This directory is the **reproducibility artefact** referenced in the manuscript:
<https://github.com/rogerpanel/CV/tree/main/SODE-Guard>
