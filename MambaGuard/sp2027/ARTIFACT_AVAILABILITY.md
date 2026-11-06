# Artifact Availability

An anonymised reviewer mirror of the MambaGuard codebase is available at
`https://anonymous.4open.science/r/mambaguard-sp2027` for the duration
of the IEEE S&P 2027 Cycle 2 review process. The mirror is frozen at the
full-paper submission deadline (2026-11-17 23:59 AoE); no post-deadline
changes will be pushed to it, and its tree hash is recorded in the
submission's supplementary material. The mirror contains: (i) the full
training, evaluation, and certification code for the MambaGuard detector
and projector, under `mambaguard/`; (ii) the adaptive-attack scaffolds
described in Section 5 and in `ETHICS.md`, under
`mambaguard/attacks/adaptive/`; (iii) the configuration files for all
experiments reported in the paper, under `configs/`; (iv) scripts to
reproduce every table and figure, under `scripts/`; (v) the frozen
prompt templates and cached LLM responses used to generate the
synthetic traces, under `mambaguard/data/`; (vi) the subset of each
public benchmark we use, pinned by commit hash in
`reproducibility/benchmark_pins.txt`; and (vii) a top-level
`reproducibility/` directory with seeds, environment pins, and
container recipes. Authors, institutional affiliations, deployment
endpoints, and DOIs that resolve to known authors have been removed by
the `anonymize.sh` harness shipped alongside the paper's supplementary
material; the mirror is intended only to allow reviewers to inspect and
re-run our experiments. A full-attribution, archivally stable release
will be deposited after notification and before the camera-ready
deadline.
