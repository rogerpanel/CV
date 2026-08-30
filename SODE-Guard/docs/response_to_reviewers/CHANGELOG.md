# SODE-Guard Revision — v3 → v4 Changelog

## New source modules

| Module | Reviewer concern | Purpose |
|---|---|---|
| `src/theory/lipschitz.py` | R1.1, R2.2 | Empirical Hoeffding L_g certification |
| `src/theory/chaos_degree.py` | R2.4 | Data-driven Hermite chaos-degree estimator |
| `src/theory/carbery_wright.py` | R2.2, R2.3 | Corrected margin-stability + decision-flip bounds |
| `src/theory/pac_bayes.py` | R1.6 | Maurer PAC-Bayes-kl certificate |
| `src/theory/pseudoinverse.py` | R1.5, R2.9 | Moore–Penrose pseudo-inverse for 128×16 diffusion |
| `src/theory/bel_estimator.py` | R2.8 | BEL vs finite-difference diagnostic sweep |
| `src/attacks/semantic/feasibility.py` | R2.11 | Feature-level feasibility projector |
| `src/attacks/semantic/constrained_attacks.py` | R2.11 | Problem-space PGD / C&W |
| `src/data/splits_extra/temporal.py` | R2.13 | Time-ordered split |
| `src/data/splits_extra/host_disjoint.py` | R2.13 | Host-disjoint split |
| `src/data/splits_extra/scenario_disjoint.py` | R2.13 | Scenario-disjoint split |
| `src/data/splits_extra/dedup.py` | R2.12 | Deduplication + leakage report |
| `src/evaluation/reliability/smoothing_sweep.py` | R1.3 | Randomised-smoothing σ sensitivity |
| `src/evaluation/reliability/commercial_context.py` | R1.2 | Baseline threat-model taxonomy |

## Modified modules

* `src/models/sode_guard.py`: `forward_with_paths` now also exposes the
  Brownian summary needed by `theory.chaos_degree`.
* `src/regularizers/anti_concentration.py`: docstring corrected to
  reflect the direction-fixed Carbery–Wright bound; code unchanged
  because it uses the empirical `‖m‖_2` denominator.
* `src/evaluation/certificate.py`: consumes
  `theory.lipschitz.LipschitzCertificate` instead of a hand-picked
  Lipschitz probe.

## New docs

* `docs/response_to_reviewers/RESPONSE.md`
* `docs/response_to_reviewers/CHANGELOG.md`  (this file)
* `docs/packet_semantics.md`
* `manuscript/SODEGuard_v4_Manuscript.tex` (revised submission LaTeX)
* `manuscript/highlights_v3_to_v4.md` (change list mapped to line ranges)

## New tests

* `tests/test_lipschitz.py`
* `tests/test_chaos_degree.py`
* `tests/test_carbery_wright.py`
* `tests/test_pac_bayes.py`
* `tests/test_pseudoinverse.py`
* `tests/test_feasibility_projector.py`
* `tests/test_split_protocols.py`
* `tests/test_dedup_leakage.py`

## New scripts

* `scripts/reproduce_revision.sh` — re-runs Tables 5–11 of the revised
  manuscript on the local hardware.
* `scripts/regenerate_figures.py` — regenerates Figs. 4b (σ sweep) and
  6 (commercial-baseline reference plot).
