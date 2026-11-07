# Running the MambaGuard GPU study on Kaggle (version 2)

This folder runs every experiment that needs a GPU and returns small zips of
results that `scripts/summarise_kaggle_results.py` turns into the paper's
tables.

| File | What it is |
|---|---|
| `mambaguard_kaggle.ipynb` | The notebook to import into Kaggle (9 steps, top to bottom) |
| `run_kaggle_study.py` | The study runner (two studies, resumable) |
| `setup_benchmarks.sh` | Downloads the four benchmark repos at their pinned commits |
| `requirements-kaggle.txt` | The two packages Kaggle lacks (`einops`, `sentence-transformers`) |
| `build_kaggle_package.sh` | Builds `dist/mambaguard-kaggle.zip` for upload as a Kaggle Dataset (Option B) |

## What changed since the first run

1. **Payload encoder.** Each message is split into chunks (one per tool in a
   tool list, one per short run of sentences in long text, plus a structural
   header chunk). Every chunk gets a MiniLM embedding and a hashed word vector.
   The detector now scores *each chunk* with a small network and pools the
   scores per message (max and mean: multiple-instance pooling), so one
   poisoned tool description or one injected sentence can decide the message
   instead of being averaged away. Code: `mambaguard/data/chunked_encoder.py`
   and `MambaGuardConfig.chunk_dim`. This variant (`chunk_mil`) was chosen
   over plain pooled chunks (`chunk_lex`) on MCPTox *validation* scores; the
   encoder ablation reports all four (`minilm`, `chunk`, `chunk_lex`,
   `chunk_mil`).
2. **Held-out-template evaluation.** For each fold, the attacks written in one
   wording or family are removed from training and validation, and the test
   split contains only those attacks plus benign episodes from unseen groups.
   This tests whether a model recognises *reworded* attacks rather than
   memorised text. The lexical baseline runs on exactly the same folds.
3. The lexical baseline now runs inside the GPU study, so all comparisons come
   from one place.

## The two studies

| Study | What runs | Estimated T4 time* |
|---|---|---|
| `main` | Chunk-scoring detector (`chunk_mil`) × 5 benchmarks × seeds 0, 1, 2; lexical baseline; encoder ablation (`minilm`, `chunk`, `chunk_lex`) on MCPTox, MCP-SafetyBench, AgentDojo × seeds 0, 1; architecture ablation (5 variants) on A2A and MCPTox × seeds 0, 1; certificates; latency (detector *and* encoder) | ~5 h |
| `heldout` | Held-out folds, detector (seeds 0, 1; A2A seed 0) and lexical baseline: MCPTox by wording style (`important_tag`, `plain`, `ignore_previous`) and by paradigm (Template-1/2/3); AgentDojo by attack family (6 families, GPT-4o recorded runs, which are the only runs with many attack templates); InjecAgent by attacker case (4 groups); MCP-SafetyBench by attack family (4); A2A by vulnerability class (11, leave-one-finding-out) | ~2.5 h |

\*Estimated from the first run's timings; the new encoder adds a few minutes
of embedding per benchmark. The log prints the seconds per job.

Every detector run reports macro-F1 at 0.5 and at a validation-tuned
threshold, AUROC, average precision, attack recall, false-positive rate and a
bootstrap 95% CI. Each result is saved the moment it finishes.

---

## A. One-time Kaggle setup (already done if you ran version 1)

1. Sign in at <https://www.kaggle.com>.
2. **Verify your phone**: profile picture → *Settings* → *Phone verification*.
3. GPU quota is about 30 h per week; *Settings* shows what is left. This
   study needs about 8 h of it.

## B. Get the new notebook

You can reuse your existing Kaggle notebook, but its cells must be replaced
with the new version:

1. Open your notebook on Kaggle → *File* → *Import Notebook* → upload the new
   `kaggle/mambaguard_kaggle.ipynb` (or paste its GitHub URL:
   `https://github.com/rogerpanel/CV/blob/claude/implement-mambaguard-detection-zCgu6/MambaGuard/kaggle/mambaguard_kaggle.ipynb`).
   Importing into a fresh notebook works the same way.
2. Right-hand panel → *Session options*: **Accelerator: GPU T4 x2**,
   **Internet: On**, notebook **Private**.
3. Code: step 2 of the notebook clones the branch from GitHub, so it always
   picks up the latest code. If you use the uploaded-zip route (Option B in
   version 1), **upload the new `kaggle/dist/mambaguard-kaggle.zip` as a new
   version of your Kaggle Dataset** first, or the notebook will run the old code.

## C. Run it

1. **Interactive check (~20 min).** Run cells 1–7 one by one (Shift+Enter):
   - Cell 1: `Tesla T4`. Cell 5: episode counts as in `PROTOCOL_BENCHMARKS.md`.
   - Cell 7 is the smoke test of **both** studies (tiny data, 2 epochs, 2 folds
     per benchmark). It must end with `wrote /kaggle/working/results_all_smoke.zip`
     and **no `ERROR` lines**. Its numbers are meaningless.
2. **Version 1: main study.** In cell 8 leave `STUDY = "main"`. Top right:
   *Save Version* → *Save & Run All (Commit)* → *Save*. You can close the browser.
3. **Version 2: held-out study.** When version 1 has finished (or in parallel,
   if your quota allows two sessions), edit cell 8 to `STUDY = "heldout"` and
   commit again (*Save & Run All*).
   - Alternatively set `STUDY = "all"` and commit once (~7.5 h).
4. Progress: notebook page → *Versions* → the running version → *Logs*.
   Each finished job prints a line like
   `macro-F1 0.912  AUROC 0.960  recall 0.900  FPR 0.050  (95s)`.

## D. If a session hits the time limit

1. Edit the notebook → *+ Add Input* → *Notebook Output Files* → pick the
   version that timed out.
2. In cell 8 keep the same `STUDY` and set
   `RESUME = "/kaggle/input/<that-notebook>/results_<study>"` (the right panel
   shows the exact path; the `.zip` path also works).
3. *Save & Run All (Commit)*. Finished jobs are copied and skipped.
4. To shorten a study: `EXTRA = "--seeds 0 1"` or `EXTRA = "--skip arch"`.

## E. Download

From each finished version's *Output* tab download:

- **`results_main.zip`** (from version 1) and **`results_heldout.zip`** (from
  version 2). These are the files to bring back; each is well under 5 MB.
- `results_main_checkpoints.zip`: seed-0 model weights, only for later
  re-certification. Do not upload it to GitHub.
- `results_all_smoke.zip`: the smoke test; ignore it.

## F. Bring the results back here

Upload both zips to the repository, as last time:

GitHub → `rogerpanel/CV` → branch `claude/implement-mambaguard-detection-zCgu6`
→ `MambaGuard/kaggle/results/` → *Add file* → *Upload files* → drop
`results_main.zip` and `results_heldout.zip` → *Commit directly to the branch*.
You can also attach them in the chat. Then the analysis is:

```bash
python scripts/summarise_kaggle_results.py kaggle/results/results_main.zip \
    kaggle/results/results_heldout.zip --latex --json reproducibility/kaggle_results_v2.json
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GPU: NONE` in cell 1 | Accelerator is off, or the weekly quota is used up |
| `Could not resolve host` / Hugging Face timeouts | Internet is off |
| `CUDA error: no kernel image is available` | P100 with a newer PyTorch: switch to GPU T4 x2 |
| `CUDA out of memory` | `EXTRA = "--batch-size 16 --cert-batch 25"` |
| A benchmark shows `skip ...: not found` | Re-run cell 4 |
| `ERROR` on one job | The traceback is in `results_<study>/runs/<job>.error.txt` (also in the zip); other jobs continue |
| Smoke test prints `ERROR` | Stop and send me the error file before committing the real run |

## Notes

- Benchmarks are cloned to `/tmp/mambaguard_data`, outside `/kaggle/working`,
  so they are not saved with the notebook output.
- On a T4, mixed precision is now off ("auto" picks bfloat16 only on GPUs that
  support it natively), so runs are in fp32.
- The held-out AgentDojo folds use the recorded GPT-4o runs, capped at 100 /
  25 / 40 attack episodes per family for train / val / test, with all 97 benign
  episodes; the main study keeps the Claude 3.5 Sonnet runs as before.
