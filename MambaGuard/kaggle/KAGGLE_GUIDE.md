# Running the MambaGuard GPU study on Kaggle

This folder runs every experiment that needs a GPU and returns one small zip of
results that can be summarised (and pasted into the paper) with
`scripts/summarise_kaggle_results.py`.

| File | What it is |
|---|---|
| `mambaguard_kaggle.ipynb` | The notebook to import into Kaggle (9 steps, top to bottom) |
| `run_kaggle_study.py` | The study runner: train/eval, ablations, certificates, latency; resumable |
| `setup_benchmarks.sh` | Downloads the four benchmark repos at their pinned commits |
| `requirements-kaggle.txt` | The two packages Kaggle lacks (`einops`, `sentence-transformers`) |
| `build_kaggle_package.sh` | Builds `dist/mambaguard-kaggle.zip` for upload as a Kaggle Dataset (Option B) |

## What the full run does

| Stage | Scope (defaults) |
|---|---|
| Main | 5 benchmarks (A2A sandbox, MCPTox, MCP-SafetyBench, AgentDojo, InjecAgent) × seeds 0, 1, 2, MiniLM encoder, class-balanced sampling, early stopping (patience 5) |
| Ablations | SSM only, GAT only, flat readout, Transformer backbone, no Lipschitz penalty × seeds 0, 1 on A2A sandbox and MCPTox |
| Certificates | Cohen smoothing on payload embeddings of 100 test windows, σ ∈ {0.05, 0.1, 0.25}, n₀ = 32, n = 500, α = 10⁻³ |
| Latency | Batch 1 / 8 / 32 / 128 on the GPU |

Every run reports macro-F1 at 0.5 and at a threshold tuned on the validation
split, AUROC, average precision, attack recall, false-positive rate, and a
1,000-sample bootstrap 95% CI. Each result is saved the moment it finishes.

How long it takes on a T4 has not been measured yet; the log prints the seconds
per run, so the smoke test (step 7) gives a first estimate. On this CPU
container the slowest benchmark (AgentDojo) took ~15 min for 6 epochs with the
hashing encoder. The runner stops starting new runs after 11 h (Kaggle's limit
is 12 h) and zips what it has; you can resume in a new session (step D below).

---

## A. One-time Kaggle setup (5 min)

1. Sign in at <https://www.kaggle.com>.
2. **Verify your phone**: profile picture → *Settings* → *Phone verification*.
   Without it the GPU and Internet switches stay greyed out.
3. GPU quota is about 30 h per week; *Settings* shows what is left.

## B. Create the notebook

1. *Create* (the **+** button, top left) → *New Notebook*.
2. *File* → *Import Notebook* → upload `kaggle/mambaguard_kaggle.ipynb`
   (download it from this repository, or paste its GitHub URL:
   `https://github.com/rogerpanel/CV/blob/claude/implement-mambaguard-detection-zCgu6/MambaGuard/kaggle/mambaguard_kaggle.ipynb`).
3. Right-hand panel → *Session options* (or *Settings*):
   - **Accelerator: GPU T4 x2.** Avoid P100: Kaggle's current PyTorch build
     may no longer support it (step 1 of the notebook warns you).
   - **Internet: On.**
   - Keep the notebook **Private** (the run downloads MCPTox and
     MCP-SafetyBench, which have no licence for redistribution).
4. Rename it (top left), e.g. `mambaguard-gpu-study`.

### Getting the code: two options

- **Option A, GitHub (default, nothing to do).** Step 2 of the notebook clones
  branch `claude/implement-mambaguard-detection-zCgu6` of `rogerpanel/CV`
  (public) and keeps only `MambaGuard/`.
- **Option B, upload a zip.** Use this if you want a frozen copy, or the repo
  becomes private.
  1. Build the zip here: `bash kaggle/build_kaggle_package.sh` → `kaggle/dist/mambaguard-kaggle.zip`
     (it is also committed in the repo).
  2. Kaggle: *Create* → *New Dataset* → drag the zip in → title `mambaguard-code`,
     visibility **Private** → *Create*.
  3. In the notebook: right panel → *+ Add Input* → *Your Datasets* → `mambaguard-code`.
  Step 2 then finds the code under `/kaggle/input/` and uses it instead of GitHub.
  Internet is still needed for the benchmark downloads and MiniLM.

## C. Run it

1. **Interactive check.** Run cells 1 to 7 one by one (Shift+Enter). Expected:
   - Cell 1 prints `Tesla T4` and `bf16 supported: False` (the runner then uses fp32).
   - Cell 5 prints episode counts matching `reproducibility/PROTOCOL_BENCHMARKS.md`
     (e.g. MCPTox attack 1,312 / benign 533; MCP-SafetyBench 238 / 162;
     InjecAgent 1,054 / 17; A2A sandbox 2,200 / 2,200; AgentDojo 550 / 97).
   - Cell 7 (smoke test, ~5 min) ends with `wrote /kaggle/working/results_smoke.zip`
     and the summary table, with **no `ERROR` lines**. Smoke numbers are
     meaningless (24 test windows, 2 epochs); they only show the pipeline works.
2. **Full run as a background commit.** Top right: *Save Version* →
   *Save & Run All (Commit)* → *Save*. Kaggle runs the whole notebook again
   on a fresh machine, up to 12 h, and you can close the browser. Progress:
   the notebook page → *Versions* (or *View Active Events*, bottom left) →
   the running version → *Logs*.
   - If you prefer to run it interactively instead, keep the tab open: an
     interactive session is stopped after a period without activity.
3. **Shorter run** (if quota is tight): before committing, set in cell 8
   `EXTRA = "--seeds 0 1 --skip-ablations"` or `"--epoch-scale 0.5"` or
   `"--benchmarks a2a_sandbox mcptox"`.

## D. If it hits the time limit: resume

1. Open the finished version → *Output* and check what is in `results/runs/`.
2. Edit the notebook → *+ Add Input* → *Notebook Output Files* (or *Your Work*)
   → pick this notebook's own previous version.
3. In cell 8 set `RESUME = "/kaggle/input/<that-notebook>/results"` (the right
   panel shows the exact path; a `results.zip` path also works).
4. *Save & Run All (Commit)* again. Finished runs are copied and skipped.

## E. Download the results

Open the finished version → *Output* tab (or the *Data* / output pane on the
right) → download:

- **`results.zip`** — small (JSON per run, test-set predictions, certificates,
  latency, `environment.json`, `summary.json`). **This is the file to bring back.**
- `results_checkpoints.zip` — the seed-0 full-model weights per benchmark
  (~15 MB each). Only needed to re-certify or re-evaluate later.
- `results_smoke.zip` — the smoke test; can be ignored.

Command-line alternative: `pip install kaggle`, put your API token in
`~/.kaggle/kaggle.json` (*Settings* → *API* → *Create New Token*), then
`kaggle kernels output <your-username>/mambaguard-gpu-study -p ./kaggle_out`.

## F. Bring the results back here for assessment

Put `results.zip` in the repository so this session can read it:

1. GitHub → `rogerpanel/CV` → switch to branch
   `claude/implement-mambaguard-detection-zCgu6` → open `MambaGuard/kaggle/results/`
   → *Add file* → *Upload files* → drop `results.zip` → *Commit directly to the branch*.
   (The web upload limit is 25 MB; `results.zip` is normally far below it.
   Do not upload the checkpoints zip there.)
2. Then ask here, e.g. "the Kaggle results are in kaggle/results/results.zip".
   The analysis is:

   ```bash
   python scripts/summarise_kaggle_results.py kaggle/results/results.zip --latex \
       --json reproducibility/kaggle_results.json
   ```

   It prints the main table (macro-F1, AUROC, recall, FPR, length-shortcut
   baseline, 95% CI), the ablation table, certified-accuracy curves, latency,
   any errors, and LaTeX rows for `sp2027/MambaGuard_sp2027.tex`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GPU: NONE` in cell 1 | Accelerator is off, or the weekly quota is used up |
| `Could not resolve host github.com` / Hugging Face timeouts | Internet is off (needs phone verification) |
| `CUDA error: no kernel image is available` | P100 with a newer PyTorch: switch to GPU T4 x2 |
| `CUDA out of memory` | Add `--batch-size 16` to `EXTRA` (and `--cert-batch 25`) |
| A benchmark shows `skip ...: not found` | Re-run cell 4; check its folder under `/tmp/mambaguard_data` |
| `ERROR` on one run | The message is in `results/runs/<run>.error.txt` and in the zip; other runs continue |
| Session ended before the end | Resume (section D) |
| pip warns about dependency conflicts | Harmless as long as cell 3 prints both versions; never `pip install torch` |

## Notes

- Benchmarks are cloned to `/tmp/mambaguard_data`, outside `/kaggle/working`,
  so they are not saved with the notebook output and not redistributed.
- No benchmark repository code is executed; loaders read JSON/CSV/YAML and
  parse server source files with Python's `ast`.
- The latency stage times an untrained model of the same architecture (speed
  does not depend on the weights).
- Running elsewhere: any CUDA machine works the same way:
  `bash kaggle/setup_benchmarks.sh /data && python kaggle/run_kaggle_study.py --data-root /data --out results`.
