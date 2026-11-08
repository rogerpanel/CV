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
                       reproduce_all.sh
kaggle/                run_kaggle_study.py (the GPU benchmark study),
                       setup_benchmarks.sh, requirements-kaggle.txt
reproducibility/       kaggle_gpu_v2/ (the reported GPU run: every run, prediction,
                       certificate and latency file), kaggle_results.json,
                       lexical_baseline.json, experiments_sp2027.json,
                       PROTOCOL_BENCHMARKS.md, DATASETS.md, ENVIRONMENT.md
tests/                 43 unit and end-to-end tests
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

## Full reproduction (one GPU, about four hours)

```bash
bash kaggle/setup_benchmarks.sh data/raw          # clones at pinned commits
python kaggle/run_kaggle_study.py --study main    --data-root data/raw --out results_main
python kaggle/run_kaggle_study.py --study heldout --data-root data/raw --out results_heldout
python scripts/summarise_kaggle_results.py results_main.zip results_heldout.zip --latex
```

`--study main` uses the chunked payload encoder (`mambaguard/data/chunked_encoder.py`)
and also runs the lexical baseline and the encoder and architecture
ablations; `--study heldout` runs the leave-template-out evaluation for the
detector and the lexical baseline. `reproducibility/kaggle_gpu/` holds the
first GPU run (one MiniLM vector per message).

The study trains the detector with seeds 0, 1, 2 on MCPTox, MCP-SafetyBench,
AgentDojo, InjecAgent and the A2A sandbox, runs the five ablations with seeds
0 and 1 on the A2A sandbox and MCPTox, certifies the seed-0 models and times
the forward pass. It needs `sentence-transformers` (frozen MiniLM payload
encoder) and one CUDA GPU; the paper's runs used one NVIDIA T4 (16 GB) and
took 4.0 GPU-hours for the first run; the reported v2 run (`--study all`) took 7.1 GPU-hours. It saves each result as soon as it finishes and can resume
(`--resume-from`). `python kaggle/run_kaggle_study.py --smoke` is a 5-minute
end-to-end check. The recorded outputs of the paper's run are in
`reproducibility/kaggle_gpu_v2/`; `summarise_kaggle_results.py` also accepts that
folder.

## Where each paper result comes from

| Paper element | Command / function | Output |
|---|---|---|
| Hedge regret bound (Theorem 1) | `run_sp2027_experiments.py`, `HedgeDefender` | `experiments_sp2027.json: hedge_corrected` |
| Composed certificate (Theorem 2), synthetic check | `composed_certificate_v2` | `experiments_sp2027.json: composed_certificate_v2` |
| Lipschitz table (block product, residual bound, probe) | `empirical_lipschitz`, `compute_lipschitz_bound` | `experiments_sp2027.json: empirical_lipschitz` |
| Graph-edit certificate check | `sparse_smoothing.sparse_certify` | `experiments_sp2027.json: sparse_smoothing` |
| E1 embedding-PGD check | `attacks.adaptive.EmbedPGD` | `experiments_sp2027.json: embed_pgd_asr` |
| Detection table (Table 1, MambaGuard columns) | `kaggle/run_kaggle_study.py` | `kaggle_gpu_v2/runs/main__*.json` |
| Detection table (Lexical column) | `scripts/lexical_baseline.py` | `kaggle_gpu_v2/runs/*__lexical__*.json` |
| Certified-accuracy figure | `run_kaggle_study.py` (`certify`) | `kaggle_gpu_v2/certify/*.json` |
| GPU latency table | `run_kaggle_study.py` (`latency`) | `kaggle_gpu_v2/latency/*.json` |
| Encoder ablation table | `run_kaggle_study.py --study main` | `kaggle_gpu_v2/runs/enc__*.json` |
| Architecture ablation table | `run_kaggle_study.py --study main` | `kaggle_gpu_v2/runs/arch__*.json` |
| Held-out-template table | `run_kaggle_study.py --study heldout` | `kaggle_gpu_v2/runs/heldout__*.json` |

The first five rows run on synthetic inputs on a CPU and need no data.

## Datasets

All datasets are public and are listed with their sources in
`reproducibility/DATASETS.md`. None is redistributed here.

## Notes for reviewers

- The smoothing certificate holds in the MiniLM payload-embedding space,
  not for arbitrary text edits; see the paper's "Operational Meaning and
  Non-Guarantees" subsection.
- `attacks/adaptive/token_gcg.py` defines the GCG attack interface. Running it
  requires a token-level embedder for the payload encoder; the hook is
  documented in the file.
- The adaptive-attack code is released as research scaffolding and is not
  configured to target any live third-party service.

## License

MIT. See `LICENSE`.
