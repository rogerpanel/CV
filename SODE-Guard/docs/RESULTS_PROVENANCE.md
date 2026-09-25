# Results provenance

This file records which numbers in `manuscript/` and `docs/` were produced by
experiments and which were not. Read it before reusing any number.

## Numbers that were never produced by a run

The v3 → v4 revision files (`manuscript/SODEGuard_v4_Manuscript.tex`,
`manuscript/Response_to_Reviewers.tex`, `manuscript/Highlights_v3_to_v4.tex`,
`manuscript/Packet_Semantics.tex`, `docs/response_to_reviewers/*.md`,
`docs/packet_semantics.md`) contain values that were written as placeholders
while the code was being prepared. No training, evaluation, certification or
dataset-audit run in this repository produced them. They must not be reported
as results:

- split-protocol audit (temporal / host-disjoint / scenario-disjoint F1),
- problem-space (feasibility-projected) PGD F1,
- randomised-smoothing σ sweep (clean accuracy, certified fractions, "46 %"),
- empirical L_g table, L_lo values,
- PAC-Bayes risk bounds,
- BEL bias / variance table,
- un-harmonised IIS3D per-constituent F1,
- deduplication unique-rates, cross-split leakage rates, label-agreement rate,
- stability audit (condition numbers, Milstein discrepancy, singular values),
- adaptive chaos degree d*_p99 values and the "adaptive d*" ablation row,
- Friedman χ²(6) = 41.3 (also impossible with 3 blocks: the maximum is 18).

The same files also reference modules that do not exist
(`src/data/harmonize/iis3d.py`, `notebooks/02_bel_stability_audit.ipynb`, and
several test files listed in `docs/response_to_reviewers/CHANGELOG.md`).

## Theory in the v4 files that is unsound

Proposition 5 / Corollary 1 of the v4 manuscript bound Pr[|Δ| ≤ β] with
Carbery–Wright and then use it as a bound on decision flips; the direction is
wrong, and the resulting "probabilistic robustness radius" does not certify
anything. The corresponding code (`src/theory/carbery_wright.py`,
`src/theory/lipschitz.py`, `src/evaluation/certificate.py`) has been removed.
The replacement is `src/certify/` (Theorem A: worst-case radius of the mean
predictor; Proposition B: path-wise flip bound).

## Numbers that come from the authors' own runs

Tables 1–3 (clean / PGD-40 F1 per method and dataset, ablation), the latency
and ECE values, and the robustness curve were supplied by the authors and are
not reproduced by any run in this repository.

## How to produce the missing results

`scripts/reproduce_revision.sh` (E1–E4) and `scripts/reproduce_paper.sh`
(training and PGD evaluation) generate every table and figure from trained
checkpoints; each run writes JSON plus ready-to-paste LaTeX rows.
