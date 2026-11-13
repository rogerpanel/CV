# What can and cannot be reproduced from this code

## Reproducible here

| What | Command | Notes |
|---|---|---|
| Unit tests of the guarantees (Propositions 1 and 2, the trace-level corollary, the two-sided guard) | `pip install -e ".[dev]" && pytest` | NumPy for the conformal layer; PyTorch for the encoder smoke tests. |
| Figure 2 and its table (online adaptation, controlled simulation) and the calibration-unit simulation | `python scripts/sim_checks.py --figure fig2_aci_simulation.pdf` | **Synthetic scores**, not agent traces. Tests the update rule and its cost, not the detector. About 3 minutes on a laptop CPU. |
| The same figure from your own logged scores | `python scripts/make_fig2.py --cal-scores cal.npy --stream stream.csv` | Asserts Proposition 2 at every horizon before plotting. |

## Not reproducible here

The benchmark tables of the paper (main results and ablations) **cannot** be
regenerated from this repository. It does not contain the AgentChain-26 traces,
their labels, the predicate matcher, the R-Judge conversion, trained weights,
or working baselines.

`scripts/download_datasets.py`, `train.py`, `calibrate.py`, `evaluate.py` and
`reproduce_results.py` run end to end, but on a **procedural generator that
assigns labels at random** and against **stub baselines**. They exercise the
plumbing. Their output is not evidence about the method and must not be
reported as a result.

## Determinism

`scripts/sim_checks.py` uses fixed seeds (0 to 19 for the online simulation, 0
for the calibration-unit simulation) and prints the same numbers on every run.
