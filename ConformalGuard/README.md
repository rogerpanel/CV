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
| Figure 2 generator | `scripts/make_fig2.py` | Runs on logged scores; refuses to plot a curve that violates Proposition 2. `--demo` uses synthetic scores and watermarks the output. |
| Execution graph, builder, framework hooks | `conformalguard/graph/` | Graph and builder unit-tested. The AutoGen / MetaGPT / LangGraph / CrewAI hooks are **untested** against the real frameworks. |
| DyGFormer-HGT encoder | `conformalguard/encoder/` | Forward pass smoke-tested on CPU. |
| Training / calibration / evaluation scripts | `scripts/` | Run end to end on the synthetic generator. See the limitations below. |
| Runtime, analyst routing, FastAPI server | `conformalguard/deployment/` | Routing unit-tested; server not load-tested. |

Run the tests with `pip install -e ".[dev]" && pytest` (34 tests).

## Limitations of this release: read before citing it as the source of the paper's numbers

1. **No benchmark data is included.** `conformalguard/data/synthetic.py` is a
   procedural generator that matches only the marginal statistics of
   AgentChain-26 (platform mix, trace length) and assigns labels **at random**.
   It exists to exercise the pipeline. Nothing trained or evaluated on it says
   anything about the results in the paper.
2. **The scorer here is not yet the one the paper describes.** The paper scores
   candidate *actions* at each step and trains with a next-action likelihood
   plus an unlikelihood term. `conformalguard/training/` currently trains a
   12-way trace-level classifier over violation classes. The conformal layer is
   independent of this and is correct as it stands; the scorer, the step-level
   dataset and the training loss still have to be brought in line.
3. **Baselines are stubs.** `conformalguard/evaluation/baselines.py` returns
   random decisions unless the real model is wired in through the documented
   environment variables. The learned agent guardrails (GuardAgent, AGrail,
   ShieldAgent, Progent), the rule-based provenance monitor, the sequence
   Transformer control, the predicate matcher, the R-Judge converter and the
   closed-loop harness are **not implemented here**.
4. Consequently **no table or figure of the paper can be regenerated from this
   repository alone.**

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
python scripts/make_fig2.py --demo        # exact ACI on a synthetic stream
python scripts/make_fig2.py --cal-scores cal.npy --stream stream.csv   # on logged scores
```

## External datasets

URLs and licences for R-Judge, AgentDojo, ToolEmu, InjecAgent and the other
corpora referenced by the paper are listed in `conformalguard/data/external.py`
and `docs/DATASETS.md`. None is redistributed here.

## License

MIT, see [LICENSE](LICENSE).
