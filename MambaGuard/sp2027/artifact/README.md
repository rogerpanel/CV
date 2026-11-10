# MambaGuard: What LLM Agent-Protocol Intrusion Detectors Actually Learn

Anonymised artifact for an IEEE S&P 2027 submission. It reproduces every
table and figure of the paper.

The artifact is frozen at the submission deadline and is not updated during
review.

---

## 1. What the paper measures

LLM agents exchange tool descriptions, tool results and delegated tasks over
the Model Context Protocol (MCP) and the Agent-to-Agent protocol (A2A). The
paper asks what learned intrusion detectors for this traffic actually learn.
It renders five benchmarks as protocol episodes and compares the following
models on identical episodes:

- MambaGuard, a neural detector: a selective state-space model, graph
  attention, and chunk-level payload scoring;
- an episode-length control;
- a TF-IDF lexical classifier;
- a label-free novelty score (Mahalanobis distance);
- a calibrated hybrid of the three scoring models.

The evaluation covers grouped splits, held-out attack-template folds,
cross-benchmark transfer, four independent detectors (two public prompt-injection
classifiers, an LLM judge, a fine-tuned DeBERTa-v3), five seeds, three alarm rules (validation-tuned, split conformal, adaptive
conformal) and randomised-smoothing certificates.

| Research question | Paper | Where in this artifact |
|---|---|---|
| RQ1 Does the detector beat simple controls and independent detectors? | §4.1, Tables 2–3, Fig. 4 | `kaggle/run_kaggle_study.py`, `kaggle/run_detector_baselines.py`, `scripts/analyse_v3.py` |
| RQ2 Which signal carries detection? | §4.2, Figs. 2–3, Tables 4–5 | encoder/architecture ablations; `scripts/template_audit.py` |
| RQ3 Do models recognise unseen templates and other benchmarks? | §4.3, Tables 3, 6, 7, Fig. 4 | held-out folds and transfers in both runners |
| RQ4 Can false alarms be controlled? Are scores calibrated? | §4.4, Fig. 5 | `analyse_v3.py` (conformal, ACI), `scripts/calibration_v4.py` |
| RQ5 What do certificates and latency deliver? | §4.5, Fig. 6, Table 8 | `mambaguard/certification/`, `run_kaggle_study.py` (`certify`, `latency`) |

## 2. Repository layout

```
mambaguard/                     the library
  data/protocol_traces/           benchmark loaders: MCPTox, MCP-SafetyBench, MCPSecBench,
                                  AgentDojo, InjecAgent, and the A2A-sandbox generator
  data/chunked_encoder.py         chunk-level payload encoder (one chunk per tool / sentence run)
  models/                         MambaGuard: selective SSM (S4D-Real), LipschitzNorm GATv2,
                                  chunk scorer, graph and flat read-outs
  certification/                  randomised smoothing, sparse (graph-edit) smoothing,
                                  composed bound, Hedge controller, Lipschitz bounds and probe
  attacks/, attacks/adaptive/     FGSM/PGD/C&W, EoT embedding PGD, GCG interface
  training/, evaluation/          trainer (focal loss, Lipschitz penalty), metrics, latency
kaggle/
  run_kaggle_study.py             the whole GPU study (main runs, held-out folds, controls,
                                  certificates, latency); resumable
  run_detector_baselines.py       independent detectors (zero-shot, LLM judge, fine-tuned) on
                                  the same splits, held-out folds and transfers; resumable
  run_on_kaggle.ipynb             the same on a free Kaggle T4 (Section 6)
  setup_benchmarks.sh             clones the four benchmark repositories at pinned commits
scripts/
  reproduce_all.sh                one entry point for the three stages of Section 4
  analyse_v3.py                   pre-registered analysis: hybrid, split conformal, ACI,
                                  paired bootstrap + Holm, sign test (reads saved scores only)
  summarise_kaggle_results.py     tables from a results folder or zip
  template_audit.py               CPU: top features, feature removal, template similarity
  summarise_detectors.py          one AUROC table for every detector, from saved scores
  calibration_v4.py               ECE of every detector, from saved scores (no data needed)
  expert_mixture.py, lexical_baseline.py, run_sp2027_experiments.py, ...
reproducibility/                recorded outputs of the paper's runs (Section 5)
tests/                          51 unit and end-to-end tests
configs/, benchmarks/           YAML configs, baseline wrappers
```

## 3. Requirements

- Python 3.10 or newer; `pip install -r requirements.txt && pip install -e .`
- CPU stages: any laptop. Stage 2 needs the benchmark clones (about 1 GB).
- GPU stage: one CUDA GPU with 16 GB of memory. The paper used an NVIDIA T4
  in fp32. `sentence-transformers` is required (frozen `all-MiniLM-L6-v2`,
  downloaded automatically, about 90 MB).
- `mamba-ssm` and `torch-geometric` are optional. All reported numbers,
  including latency, were measured with the pure-PyTorch fallbacks.

## 4. Reproducing the results

```bash
bash kaggle/setup_benchmarks.sh data/raw         # benchmark clones at pinned commits (A2A is generated)
bash scripts/reproduce_all.sh                    # stage 1
bash scripts/reproduce_all.sh --audit data/raw   # stages 1-2
bash scripts/reproduce_all.sh --gpu   data/raw   # stages 1 and 3
```

| Stage | What it does | Hardware / time | Output (compare with) |
|---|---|---|---|
| 1 | 51 tests; certificates, Hedge, Lipschitz probe and embedding-PGD on synthetic inputs | CPU, ~10 min | `outputs/experiments_sp2027.json` (`reproducibility/experiments_sp2027.json`) |
| 2 | Template audit (Table 4, Fig. 3, Appendix C); ECE of every detector from the recorded scores (Fig. 5b) | CPU, ~25 min | `outputs/template_audit.json`, `outputs/calibration_v4.json` (same names in `reproducibility/`) |
| 3 | Five-seed GPU study and its analysis (Tables 2, 7, 8; Figs. 4, 5a, 6; Appendices C–D) | 1 GPU, ~3 h on a T4 | `outputs/summary_v3.md`, `outputs/analysis_v3.md` (`reproducibility/kaggle_gpu_v4/summary_v4.md`, `analysis_v4.md`) |
| 3b | Independent detectors (ProtectAI, PIGuard, Prompt-Guard-2 with a token, an LLM judge, a fine-tuned DeBERTa-v3, lexical) on the same splits, held-out folds and four cross-benchmark transfers; merged comparison (Tables 3, 6, 11) | 1 GPU, ~2 h on a T4 | `outputs/results_baselines/summary.md`, `outputs/detectors.md` (`reproducibility/detectors_v4/summary.md`) |

Stage 2 is deterministic. Its lexical AUROCs equal those of the GPU study on
all 36 splits. Stage 3 was run twice, on Kaggle and on Colab (T4 and L4), and
reproduced every table except latency exactly. Other GPUs or library versions
may differ slightly; expect macro-F1 within about ±0.02 of the means for most
benchmarks, and more on MCPTox and MCP-SafetyBench, whose seed standard
deviations are 0.08 and 0.14.

**The earlier three-seed study** (graph read-out) supplies the encoder and
architecture ablations (Fig. 2, Table 5, rows 1–6). To rerun it:

```bash
python kaggle/run_kaggle_study.py --study all --main-variant full --seeds 0 1 2 \
    --heldout-seeds 0 1 --skip --no-calibration-split --data-root data/raw --out outputs/results_all
```

Useful flags of `run_kaggle_study.py`:

| Flag | Effect |
|---|---|
| `--smoke` | 5-minute end-to-end check |
| `--study main\|heldout\|all` | which part of the study to run |
| `--seeds`, `--heldout-seeds` | override the seed lists |
| `--resume-from <folder or zip>` | skip jobs that are already finished |
| `--time-budget-h` | stop launching new jobs after this many hours |

## 5. Recorded outputs

| File | Content |
|---|---|
| `reproducibility/kaggle_gpu_v4.zip` | Five-seed study: every run record, per-episode score (`.npz`, including the lexical and novelty experts' scores on the same episodes), certificate and latency file, plus `summary_v4.md`, `analysis_v4.md`/`.json`. |
| `reproducibility/detectors_v4.zip` | Independent detectors: every run record (AUROC with 95% CI, threshold metrics) and per-episode score, plus `summary.md`. |
| `reproducibility/calibration_v4.json` | ECE of every detector, raw and Platt-recalibrated (stage 2). |
| `reproducibility/kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` | The first five-seed run's printed output, whose score files were lost; the v4 rerun reproduced it exactly. |
| `reproducibility/kaggle_gpu_v2.zip` | Three-seed study: every run record, per-episode prediction (`.npz`), certificate and latency file. |
| `reproducibility/kaggle_gpu.zip` | First GPU run (one MiniLM vector per message). |
| `reproducibility/template_audit.json` | Template audit (stage 2). |
| `reproducibility/calibration_v2.json` | ECE of the three-seed study (superseded by `calibration_v4.json`). |
| `reproducibility/expert_mixture_v2.json`, `lexical_baseline.json`, `kaggle_results_v2.json` | Offline mixture, lexical baseline and summary of the three-seed study. |
| `reproducibility/experiments_sp2027.json` | Synthetic-input checks (stage 1). |
| `reproducibility/cpu_checks_v2.zip` | CPU pipeline checks with a hashing encoder. These are not paper results. |
| `reproducibility/PROTOCOL_BENCHMARKS.md` | Pinned commits, episode construction, expected counts and shortcut checks for every benchmark. |

All scripts accept these zips directly. For example:
`python scripts/summarise_kaggle_results.py reproducibility/kaggle_gpu_v2.zip`.

## 6. Running on Kaggle (free T4)

1. Upload this artifact's zip as a Kaggle Dataset, or use the anonymous
   mirror's download link.
2. Import `kaggle/run_on_kaggle.ipynb` (File → Import Notebook).
3. Set Accelerator = GPU T4 x2 and Internet = On. Attach the dataset, or
   paste the link into `ARTIFACT_ZIP_URL` in cell 2.
4. Run cells 1–7 by hand. Cell 6 is a smoke test. Cell 7 is the CPU template
   audit.
5. Choose **Save Version → Save & Run All (Commit)**.

Only a committed version keeps its files. The editor's `/kaggle/working`
folder is erased whenever the editor session restarts. When the version
finishes, open it from the notebook's *Versions* list and download
`results_v3_all.zip` from its **Output** section. Cell 9 of the committed
version also prints all tables.

## 7. Where each result comes from

| Paper element | Command | Recorded output |
|---|---|---|
| Table 2 (RQ1) | stage 3 | `kaggle_gpu_v4.zip: analysis_v4.md` → "In-distribution" |
| Fig. 4 (in-distribution vs held-out AUROC, six detectors) | stages 3 + 3b | `scripts/summarise_detectors.py` on `kaggle_gpu_v4.zip` and `detectors_v4.zip` |
| Table 3, Table 11 (independent detectors), Table 6 (transfer) | stage 3b | `detectors_v4.zip` + `kaggle_gpu_v4.zip`, via `scripts/summarise_detectors.py` |
| Fig. 2 (encoder ablation) | three-seed study | `kaggle_gpu_v2.zip: runs/enc__*.json` |
| Table 4, Fig. 3 (template audit) | stage 2 | `template_audit.json` |
| Table 5 (architecture ablation) | three-seed study (rows 1–6), stage 3 (last row) | `kaggle_gpu_v2.zip: runs/arch__*.json` |
| Table 7, Table 10 (held-out) | stage 3 | `kaggle_gpu_v4.zip: summary_v4.md`, `analysis_v4.md` → "Held-out templates" |
| Fig. 5a, Tables 12–13 (conformal, ACI, paired bootstrap) | stage 3 | `analysis_v4.md` → "Conformal", "Hybrid minus lexical" |
| Fig. 5b (ECE) | stage 2 | `calibration_v4.json` |
| Fig. 6, Table 14 (certified accuracy) | stage 3 | `summary_v4.md` → "Certified accuracy" |
| Table 8 (latency) | stage 3 | `kaggle_gpu_v3/RESULTS_TRANSCRIBED.md` → "Latency" (the Kaggle T4 run; `summary_v4.md` has the Colab timings) |
| Table 9, Appendix B (Lipschitz bounds and probe) | stage 1 | `experiments_sp2027.json: empirical_lipschitz` |
| Graph-edit certificate, embedding PGD (synthetic checks) | stage 1 | `experiments_sp2027.json` |

## 8. Datasets and licences

| Benchmark | Licence | Notes |
|---|---|---|
| MCPTox | none stated | read from your own clone at the pinned commit; not redistributed |
| MCP-SafetyBench | none stated | read from your own clone at the pinned commit; not redistributed |
| AgentDojo | see source repository | recorded runs only; no agent is re-run |
| InjecAgent | see source repository | |
| MCPSecBench | see source repository | message-level checks only |
| A2A sandbox | this artifact | generated deterministically by `mambaguard/data/protocol_traces/` |

`kaggle/setup_benchmarks.sh` and `reproducibility/DATASETS.md` list the
sources and commits. Server source files are parsed statically and never
executed.

## 9. Scope and caveats

- The smoothing certificates hold in the payload-embedding feature space,
  not for text edits (paper §3.7 and §4.5).
- `scripts/analyse_v3.py` was fixed before the five-seed run. Its sign test
  counts ties as non-wins (paper §6).
- The adaptive-attack code is research scaffolding and targets no live
  service. Only the embedding-space attack was run for the paper.

## 10. Licence

MIT (`LICENSE`).
