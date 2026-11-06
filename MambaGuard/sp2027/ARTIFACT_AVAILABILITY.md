# Artifact Availability

Paste-ready paragraph for the paper's appendix (it matches the zip produced by
`sp2027/build_artifact.sh`). Replace `<ID>` with the identifier that
anonymous.4open.science gives you.

> An anonymised artifact is available at
> `https://anonymous.4open.science/r/<ID>`. It contains the detector
> (`mambaguard/models`), the smoothing, sparse-smoothing and composed
> certificates and the Hedge controller (`mambaguard/certification`), the
> adaptive attacks (`mambaguard/attacks/adaptive`), the training and
> evaluation harness, all configuration files, and 35 tests.
> `scripts/reproduce_all.sh` runs the tests and regenerates every
> synthetic-input result in about ten minutes on a CPU;
> `scripts/reproduce_all.sh --full` additionally reruns the benchmark
> experiments on one GPU. The recorded outputs are in
> `reproducibility/experiments_sp2027.json`. The artifact is frozen at the
> submission deadline.

Before you freeze the artifact, add anything the paper's final text says it
contains (for example cached LLM responses for the A2A sandbox, if you use
them), and rebuild the zip.
