# ConformalGuard

Reference implementation for *ConformalGuard: Distribution-Free False-Block
Guarantees for Multi-Agent LLM Systems via Conformal Prediction on Dynamic
Execution Graphs*.

ConformalGuard is a runtime guard for multi-agent LLM systems. It represents an
execution as a heterogeneous continuous-time graph (four node types, six edge
types), scores each proposed action against that graph, and sets its blocking
threshold by conformal prediction, so that the rate at which **safe** actions
are blocked is at most a level `alpha` chosen by the operator. The guarantee
bounds false blocks only. How many violations are blocked is an empirical
property of the learned scorer.

## What is in this repository, and how far it has been verified

| Component | Location | Status |
|---|---|---|
| Split-conformal threshold, one-step-per-trace and trace-level calibration, two-sided guard | `conformalguard/conformal/core.py` | NumPy only. Unit-tested, including Monte-Carlo checks of the bounds (`tests/test_core.py`). |
| Adaptive conformal inference exactly as in Algorithm 1 (no clipping), with delayed labels | `conformalguard/adaptive/gibbs_candes.py` | Unit-tested: the Proposition 2 inequality is asserted at every horizon on shifted, adversarial and delayed-feedback sequences (`tests/test_adaptive.py`). |
| Controlled simulations behind Figure 2 and the calibration-unit comparison | `scripts/sim_checks.py` | Synthetic scores only; fixed seeds; prints the numbers reported in the paper and draws the figure. |
| Figure 2 generator | `scripts/make_fig2.py` | Runs on logged scores; refuses to plot a curve that violates Proposition 2. `--demo` uses synthetic scores and watermarks the output. |
| **R-Judge experiment** (real data, third-party labels): converter, four causal scorers, split protocols, conformal evaluation | `conformalguard/rjudge/`, `scripts/rjudge_run.py`, `docs/RJUDGE_EXPERIMENT.md` | Unit-tested, including that no scorer sees anything after the action it judges. Runs on CPU with TF-IDF features or on GPU with a sentence encoder (`notebooks/rjudge_colab.ipynb`). |
| LLM-judge baseline on R-Judge | `scripts/rjudge_llm_judge.py` | Prompt construction and parsing unit-tested; model loading needs a GPU and has not been executed here. |
| Execution graph, builder, framework hooks | `conformalguard/graph/` | Graph and builder unit-tested. The AutoGen / MetaGPT / LangGraph / CrewAI hooks are **untested** against the real frameworks. |
| DyGFormer-HGT encoder | `conformalguard/encoder/` | Forward pass smoke-tested on CPU. |
| Training / calibration / evaluation scripts | `scripts/` | Run end to end on the synthetic generator. See the limitations below. |
| Runtime, analyst routing, FastAPI server | `conformalguard/deployment/` | Routing unit-tested; server not load-tested. |

Run the tests with `pip install -e ".[dev]" && pytest`.

## Measured results

- `results/rjudge_kaggle_t4/{minilm,bge}/`: R-Judge with sentence-encoder features, 100 random and
  100 prompt-grouped splits, two encoders (GPU run): `results.json` with every per-split number and
  `summary.md` with the tables. These are the numbers in the paper.
- `results/rjudge_lsa_cpu/RESULTS.md`: the same experiment with TF-IDF features on CPU, 40 splits,
  with per-split JSON.
- In short: the false-block guarantee holds on exchangeable splits for every scorer and fails under
  prompt reuse and domain shift; the guard blocks 63-69% of unsafe records at alpha = 0.05; the
  graph scorer is not better than a sequence model on this benchmark.

## What the paper uses, and what it does not

The revised paper rests on three things, all in this repository and all reproducible: the conformal
layer and its tests, the R-Judge experiment, and the simulations (`SUPPLEMENTARY_README.md` maps each
table and figure to a command).

The following are **not** used by the paper and carry no results. They are scaffolding written for
the AgentChain-26 experiments of the submitted version, which have been withdrawn because their data
and runs could not be recovered:

- `conformalguard/data/` (a loader and a procedural generator that assigns labels at random),
  `conformalguard/training/`, `conformalguard/evaluation/` (baseline stubs that return random
  decisions), and the scripts `train.py`, `calibrate.py`, `evaluate.py`, `reproduce_results.py`,
  `download_datasets.py`;
- `conformalguard/deployment/` (runtime, routing, FastAPI server) and the framework hooks in
  `conformalguard/graph/instrumentation.py`, which have not been tested against the real frameworks;
- `docs/METHODOLOGY.md`, `ALGORITHMS.md`, `HYPERPARAMETERS.md`, `DATASETS.md`, which describe that
  earlier design.

`scripts/make_anonymous_release.py` leaves all of this out of the package prepared for reviewers.

## Layout

```
conformalguard/
  conformal/    core.py (guarantees), calibration.py, nonconformity.py, prediction_set.py
  adaptive/     gibbs_candes.py (ACI), drift_monitor.py
  graph/        execution graph, builder, framework hooks
  encoder/      DyGFormer-HGT
  data/         AgentChain-26 loader, synthetic generator, external dataset registry
  training/     trainer, losses
  evaluation/   metrics, baseline adapters
  deployment/   runtime guard, analyst routing, FastAPI server
configs/  scripts/  tests/  examples/  docs/  notebooks/
```

## Quick start

```bash
pip install -e ".[dev]"
pytest                                    # guarantees and plumbing
python scripts/sim_checks.py --figure fig2_aci_simulation.pdf   # Figure 2 and its table (synthetic scores)
python scripts/make_fig2.py --cal-scores cal.npy --stream stream.csv   # on logged scores
```

## External datasets

URLs and licences for R-Judge, AgentDojo, ToolEmu, InjecAgent and the other
corpora referenced by the paper are listed in `conformalguard/data/external.py`
and `docs/DATASETS.md`. None is redistributed here.

## License

MIT, see [LICENSE](LICENSE).
