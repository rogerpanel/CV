# FedLoRAGuard

**Federated Dynamic Graph Neural Networks with Differential-Privacy Certificates
for Supply-Chain Integrity Verification of LoRA Adapter Ecosystems.**

Reference implementation accompanying the manuscript

> R. N. Anaedevha, A. G. Trofimov, Y. V. Borodachev.
> *FedLoRAGuard: Federated Dynamic Graph Neural Networks with Differential-Privacy
> Certificates for Supply-Chain Integrity Verification of LoRA Adapter Ecosystems.*
> Информационные технологии и вычислительные системы (ИТиВС), 2026.

This codebase is the reproducibility artifact cited from the paper. It is hosted in
the author's CV repository at:

`https://github.com/rogerpanel/CV/tree/main/FedLoRAGuard`

The companion manuscript repository (LaTeX source, manuscript PDF, RobustIDPS.ai
documentation) is at `https://github.com/rogerpanel/FedLoRAGuard-Models`.

---

## What FedLoRAGuard does

FedLoRAGuard verifies the integrity of community-shared LoRA adapters across
multiple LoRA-hosting marketplaces (Hugging Face Hub, Civitai, ModelScope, …)
**without requiring any marketplace to share its raw adapter weights**. It models
the LoRA ecosystem as a heterogeneous continuous-time dynamic graph, trains a
federated DyGFormer + HGT verifier under client-level Rényi-Differentially-Private
SGD with secure aggregation and FLTrust trust bootstrapping, and emits both a
calibrated maliciousness probability and a *certified poisoning radius* `k*` —
the maximum number of colluding malicious clients under which the global verdict
is provably invariant.

## Repository layout

```
FedLoRAGuard/
├── fedloraguard/                  # core library
│   ├── graph/                     # heterogeneous CT-DG schema, builder, neighbor sampler
│   ├── encoders/                  # multimodal weight / text / behavioral feature encoders
│   ├── models/                    # DyGFormer, HGT relation-aware attention, DyG-Mamba, verifier
│   ├── federated/                 # client / server / SecAgg / FLTrust / Flower runtime
│   ├── privacy/                   # DP-SGD, RDP & PRV accountants, Theorem 1, Theorem 2
│   ├── calibration/               # temperature / Platt scaling, ECE / Brier
│   ├── investigation/             # DQN active-investigation hook (RobustIDPS.ai bridge)
│   └── utils/                     # seeds, logging, metrics
├── benchmarks/
│   ├── lorachain_2026/            # synthetic 13,500-adapter LoRAchain-2026 generator
│   └── ids/                       # CIC-IDS2017, Edge-IIoTset, UNSW-NB15, TON_IoT loaders
├── baselines/                     # PEFTGuard, ShadowGenes, FedAvg-MLP, DP-FedAvg-MLP,
│                                  # FedGraphNN-HGT, Krum-DyGFormer
├── configs/                       # default / smoke / full / ablation YAMLs
├── scripts/                       # build_benchmark, train_federated, evaluate, ablation, …
├── service/                       # FastAPI /scan_adapter endpoint + Dockerfile
├── tests/                         # unit & smoke tests
├── docs/                          # ARCHITECTURE, DATASETS, HYPERPARAMETERS, ALGORITHMS, INTEGRATION
├── requirements.txt
├── setup.py
├── REPRODUCIBILITY.md
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/rogerpanel/CV
cd CV/FedLoRAGuard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.10/3.11, PyTorch 2.1, CUDA 12.1. CPU-only execution works for
the smoke benchmark.

## Quickstart: 5-minute smoke run

```bash
python scripts/build_benchmark.py --config configs/smoke.yaml --out data/smoke
python scripts/train_federated.py --config configs/smoke.yaml --data data/smoke
python scripts/evaluate.py --config configs/smoke.yaml --data data/smoke \
    --checkpoint runs/smoke/global.pt
```

This produces, in `runs/smoke/`:
- `metrics.json` — ACC / macro-F1 / AUROC / ECE / latency
- `certificate.json` — `(epsilon_T, delta, k*, p_hat_1, p_hat_2)`
- `report.md` — human-readable summary

## Full paper run

```bash
python scripts/build_benchmark.py --config configs/full.yaml --out data/lorachain_2026
python scripts/train_federated.py --config configs/full.yaml --data data/lorachain_2026
python scripts/ablation.py        --config configs/full.yaml --data data/lorachain_2026
python scripts/evaluate.py        --config configs/full.yaml --data data/lorachain_2026 \
    --checkpoint runs/full/global.pt --baselines all
```

The full configuration uses `N=50` clients, `T=100` rounds, `(ε_T, δ) = (5.0, 1e-5)`,
and reproduces the headline result (96.4 % macro-F1, AUROC 0.984, certified
radius `k* = 8`) within ±0.3 pp on three random seeds (42, 137, 2026).

## Production deployment as a RobustIDPS.ai LoRA-integrity module

A FastAPI service that exposes `/scan_adapter` and `/healthz` is provided in
`service/`, and is intended to be plugged into the existing `robustidps_web_app`
docker-compose stack as a sibling container. See `docs/INTEGRATION.md`.

```bash
cd service
docker build -t fedloraguard-svc .
docker run -p 8000:8000 -v $(pwd)/../runs/full:/checkpoints fedloraguard-svc
curl -X POST http://localhost:8000/scan_adapter \
    -H "Content-Type: application/json" \
    -d @examples/adapter.json
```

## Datasets

| Dataset | Role | Where |
| --- | --- | --- |
| **LoRAchain-2026** (this work) | Primary benchmark, 13,500 benign+backdoored LoRA adapters | generated locally by `scripts/build_benchmark.py` |
| BackdoorLLM-LoRA leaderboard | External LoRA-backdoor benchmark | https://github.com/bboylyg/BackdoorLLM |
| PADBench (PEFTGuard) | LoRA-backdoor corpus, FLTrust root set | https://github.com/Z-Sun-RG/PEFTGuard |
| HuggingGraph | Real adapter lineage edges | https://github.com/Mohammadrezaei/HuggingGraph |
| CIC-IDS2017 | Network IDS | https://www.unb.ca/cic/datasets/ids-2017.html |
| Edge-IIoTset | IoT/IIoT IDS | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset |
| UNSW-NB15 | Network IDS | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| TON_IoT | Distributed IoT/IIoT IDS | https://research.unsw.edu.au/projects/toniot-datasets |

See `docs/DATASETS.md` for licensing, citation, schema and download instructions.

## Citing

If FedLoRAGuard is useful in your research, please cite both the manuscript and
this artifact (`CITATION.cff`).

## License

MIT. See `LICENSE`.
