# MambaGuard — Artifact for Anonymous Review

This is the anonymised artifact for the IEEE S&P 2027 submission
*"MambaGuard: Certified Streaming Detection for Multi-Protocol LLM Agent
Security"*. It contains the detector, both certificates, the online
controller, the adaptive attacks, the evaluation harness, and the script
that regenerates the paper's numbers.

The artifact is frozen at the submission deadline and is not updated during
review.

## What is in here

```
mambaguard/
  models/          selective SSM block (S4D-Real), Lipschitz-normalised
                   temporal GATv2, Bochner time encoding, detection head
  data/            protocol-message canonicaliser, heterogeneous protocol
                   graph, temporal splits, dataset loaders
  certification/   smoothing.py           Cohen two-stage certificate
                   sparse_smoothing.py    graph-edit (edge add/delete) certificate
                   certificate.py         composed bound (composed_certificate_v2)
                   hedge.py               Hedge controller, eta = sqrt(8 ln N / T) / B
                   empirical_lipschitz.py gradient-ascent lower bound on L_f
                   lipschitz_bounds.py    analytic per-block Lipschitz factors
                   stackelberg.py         Stackelberg LP (Conitzer-Sandholm)
  attacks/         FGSM, PGD, C&W, DeepFool, MCP attack templates
  attacks/adaptive embed_pgd.py (EoT PGD), token_gcg.py (GCG interface)
  training/        focal loss, Lipschitz regulariser, trainer
  evaluation/      metrics, latency/throughput harness, ablation runner
benchmarks/baselines/  baseline model wrappers
configs/               model / training / data / certification YAML
scripts/               train, evaluate, certify, prepare_data,
                       benchmark_latency, run_sp2027_experiments,
                       reproduce_all.sh
reproducibility/       experiments_sp2027.json (recorded outputs),
                       DATASETS.md, ENVIRONMENT.md
tests/                 35 unit and end-to-end tests
```

## Install

Python 3.10 or newer.

```bash
pip install -r requirements.txt
pip install -e .
```

`mamba-ssm`, `torch-geometric` and `sentence-transformers` are optional. Without
them the code falls back to a pure-PyTorch selective scan, a pure-PyTorch
GATv2, and pre-computed embeddings. The fallbacks are numerically equivalent
but slower; latency numbers in the paper's GPU rows use the fused kernels.

## Quick check (CPU, about 10 minutes)

```bash
bash scripts/reproduce_all.sh
```

This runs the test suite and then `scripts/run_sp2027_experiments.py`, which
exercises every certificate, the Hedge controller, the latency harness and the
adaptive attacks on synthetic inputs, and writes
`reproducibility/experiments_sp2027.json`. Compare your output with the
committed copy of that file; values agree up to seed-independent rounding and
machine-dependent timing.

## Full reproduction

```bash
bash scripts/reproduce_all.sh --full --data-root data/raw
```

`--full` additionally trains, evaluates and certifies MambaGuard on each
benchmark found under `data/raw/` (see `reproducibility/DATASETS.md`) with
seeds 0–4. It needs one GPU with at least 24 GB of memory; the paper used a
single A100 80 GB.

## Where each paper result comes from

| Paper element | Command / function | Output key |
|---|---|---|
| Hedge regret bound (Theorem 1) | `run_sp2027_experiments.py`, `HedgeDefender` | `hedge_corrected` |
| Composed certificate (Theorem 2) | `composed_certificate_v2` | `composed_certificate_v2` |
| Cohen certified-accuracy curve | `smoothing.cohen_certify`, `certified_accuracy_curve` | `cohen_smoothing` |
| Lipschitz table, row c = 0.5 (block product, residual bound, probe) | `empirical_lipschitz`, `compute_lipschitz_bound` | `empirical_lipschitz` |
| Latency table, CPU row | `evaluation.benchmarks`, `run_sp2027_experiments.py` | `latency_harmonised` |
| Graph-edit certificate check | `sparse_smoothing.sparse_certify` | `sparse_smoothing` |
| E1 embedding-PGD check | `attacks.adaptive.EmbedPGD` | `embed_pgd_asr` |
| Per-benchmark detection table, ablation, adaptive-attack tables, latency GPU row | `reproduce_all.sh --full` | `outputs/<benchmark>/seed<k>/` |

The first seven rows run on synthetic inputs and need no data. The last row
needs the benchmark data and a GPU.

## Datasets

All datasets are public and are listed with their sources in
`reproducibility/DATASETS.md`. None is redistributed here.

## Notes for reviewers

- The smoothing certificate holds in the sentence-transformer feature space,
  not for arbitrary text edits; see the paper's "Operational Meaning and
  Non-Guarantees" subsection.
- `attacks/adaptive/token_gcg.py` defines the GCG attack interface. Running it
  requires a token-level embedder for the payload encoder; the hook is
  documented in the file.
- The adaptive-attack code is released as research scaffolding and is not
  configured to target any live third-party service.

## License

MIT. See `LICENSE`.
