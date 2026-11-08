# MambaGuard — Artifact for Anonymous Review

This is the anonymised artifact for the IEEE S&P 2027 submission
*"Templates, Not Attacks: A Measurement Study of Learned Detectors for
LLM-Agent Protocol Traffic"*. It contains the benchmark loaders, the
measurement harness (grouped splits, held-out-template folds, lexical,
length and novelty controls, conformal and adaptive conformal alarms,
pre-registered statistics), the neural detector under test (MambaGuard) and
its ablations, the certificates, the adaptive attacks, and the scripts that
regenerate the paper's numbers.

The artifact is frozen at the submission deadline and is not updated during
review.

## What is in here

```
mambaguard/
  models/          selective SSM block (S4D-Real), Lipschitz-normalised
                   temporal GATv2, Bochner time encoding, detection head
  data/            protocol-message canonicaliser, heterogeneous protocol
                   graph, splits, dataset loaders
  data/protocol_traces/  loaders for MCPTox, MCP-SafetyBench, MCPSecBench,
                   AgentDojo, InjecAgent, and the A2A-sandbox generator
                   (11 A2ABreak findings)
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
                       lexical_baseline, summarise_kaggle_results,
                       analyse_v3, calibration_v2, reproduce_all.sh
kaggle/                run_kaggle_study.py (the GPU benchmark study),
                       setup_benchmarks.sh, requirements-kaggle.txt
reproducibility/       kaggle_gpu_v3/ (five-seed study, transcribed tables),
                       kaggle_gpu_v2/ (three-seed study: every run, prediction,
                       certificate and latency file), calibration_v2.json,
                       kaggle_results.json,
                       lexical_baseline.json, experiments_sp2027.json,
                       PROTOCOL_BENCHMARKS.md, DATASETS.md, ENVIRONMENT.md
tests/                 51 unit and end-to-end tests
```

## Install

Python 3.10 or newer.

```bash
pip install -r requirements.txt
pip install -e .
```

`mamba-ssm`, `torch-geometric` and `sentence-transformers` are optional. Without
them the code falls back to a pure-PyTorch selective scan, a pure-PyTorch
GATv2, and pre-computed embeddings. All paper numbers, including the GPU
latency table, were measured with these pure-PyTorch fallbacks.

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

## Benchmarks

Clone the benchmark repositories at the pinned commits listed in
`reproducibility/PROTOCOL_BENCHMARKS.md`, then check each one:

```bash
python -m mambaguard.data.protocol_traces mcptox data/raw/MCPTox-Benchmark --show 1
```

The A2A sandbox needs no download (`root = builtin`).

## Full reproduction (one GPU, under three hours)

```bash
bash kaggle/setup_benchmarks.sh data/raw          # clones at pinned commits
python kaggle/run_kaggle_study.py --study all --data-root data/raw --out results_v3_all
python scripts/summarise_kaggle_results.py results_v3_all.zip --latex
python scripts/analyse_v3.py results_v3_all.zip --json analysis_v3.json
```

This is the paper's five-seed study. It trains the detector (flat read-out,
chunk-scoring payload encoder `mambaguard/data/chunked_encoder.py`) with seeds
0-4 on MCPTox, MCP-SafetyBench, AgentDojo, InjecAgent and the A2A sandbox;
splits validation by group into a selection half and a calibration half;
scores the lexical (TF-IDF) and novelty (Mahalanobis) controls on the same
episodes; runs the held-out-template folds (seeds 0-2, A2A seed 0); certifies
the seed-0 models; and times the detector and the encoder. `analyse_v3.py`
then computes, from the saved scores only, the calibrated hybrid, split
conformal and adaptive conformal (ACI) thresholds, paired bootstrap with Holm
correction and the held-out sign test. The paper's run used one NVIDIA T4 and
took 2.8 GPU-hours; results are saved as they finish and can be resumed
(`--resume-from`). `--smoke` is a 5-minute end-to-end check.

The earlier three-seed study (graph read-out), which supplies the encoder and
architecture ablations and the calibration measurement, is reproduced with

```bash
python kaggle/run_kaggle_study.py --study all --main-variant full --seeds 0 1 2 \
    --heldout-seeds 0 1 --skip --no-calibration-split --data-root data/raw --out results_all
python scripts/calibration_v2.py reproducibility/kaggle_gpu_v2 --data-root data/raw
```

Recorded outputs: `reproducibility/kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` holds
the five-seed run's printed tables (its per-episode score files could not be
retrieved from the compute platform, so they are transcribed from the run's
output); `reproducibility/kaggle_gpu_v2/` holds every run, per-episode
prediction, certificate and latency file of the three-seed study;
`reproducibility/kaggle_gpu/` the first run (one MiniLM vector per message).

## Where each paper result comes from

| Paper element | Command / function | Output |
|---|---|---|
| Table 2 (RQ1), Fig. 3a | `run_kaggle_study.py --study all` + `analyse_v3.py` | `kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` (v3 analysis, in-distribution) |
| Table 4, Fig. 3b-c (RQ3), Appendix D | same | `kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` (held-out tables) |
| Fig. 4a, conformal and ACI (RQ4) | `analyse_v3.py` | `kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` (conformal table) |
| Fig. 4b, ECE (RQ4) | `scripts/calibration_v2.py` | `calibration_v2.json` |
| Fig. 2, encoder ablation (RQ2) | three-seed study | `kaggle_gpu_v2/runs/enc__*.json` |
| Table 3, architecture ablation (RQ2) | three-seed study | `kaggle_gpu_v2/runs/arch__*.json` |
| Fig. 5, certified accuracy; Table 5, latency (RQ5) | `run_kaggle_study.py` (`certify`, `latency`) | `kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` |
| Lipschitz table (block product, residual bound, probe) | `empirical_lipschitz`, `compute_lipschitz_bound` | `experiments_sp2027.json: empirical_lipschitz` |
| Hedge regret, composed bound (synthetic check) | `run_sp2027_experiments.py` | `experiments_sp2027.json` |
| Graph-edit certificate check | `sparse_smoothing.sparse_certify` | `experiments_sp2027.json: sparse_smoothing` |
| E1 embedding-PGD check | `attacks.adaptive.EmbedPGD` | `experiments_sp2027.json: embed_pgd_asr` |

The last four rows run on synthetic inputs on a CPU and need no data.

## Datasets

All datasets are public and are listed with their sources in
`reproducibility/DATASETS.md`. None is redistributed here.

## Notes for reviewers

- The smoothing certificate holds in the MiniLM payload-embedding space,
  not for arbitrary text edits; see the paper's "Certificates" subsection
  and Section 4.5.
- `attacks/adaptive/token_gcg.py` defines the GCG attack interface. Running it
  requires a token-level embedder for the payload encoder; the hook is
  documented in the file.
- The adaptive-attack code is released as research scaffolding and is not
  configured to target any live third-party service.

## License

MIT. See `LICENSE`.
