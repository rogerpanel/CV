# Model Card — LipMamba

## Overview

Lipschitz-constrained selective state-space language model trained with a
PAC-Bayes adversarial objective, with a two-sided clamp on the discretisation
step that removes the HiSPA collapse mechanism Ā_t → 0.

* **Variants**: 130M (24 layers), 370M (48 layers); 1.3B configuration provided, not trained.
* **Base checkpoints**: `state-spaces/mamba-130m` / `-370m` (The Pile, GPT-NeoX tokenizer), converted by `scripts/todo3_init_from_hf_mamba.py`.
* **Fine-tuning / evaluation**: RoBench-25, HarmBench, JailbreakBench, WildJailbreak; IDS transfer in Appendix F.
* **License**: MIT.

## What is guaranteed, and what is not (Remarks 4 and 5)

* Theorem 1 gives an explicit but **non-tight global** Lipschitz constant on
  the bounded-input domain; with the stated constants it is ≈ 10⁸ per block,
  so any radius under it is vacuous.  Reported radii are **empirical
  local-Lipschitz radii (LL-Acc)**, not certificates.
* Theorem 2 bounds state **norm**, not content.  With the stated constants
  the certified trigger length is ℓ\* ≈ 1 token; an adversary can overwrite
  content at unchanged norm (measured by `attacks/adaptive_clamp.py`, objective `overwrite`).
* Theorem 3 is instantiated with the local estimate; it is a bound
  conditional on L_loc upper-bounding the true local constant on the sampled balls.

## Results as reported in the manuscript (Table 1; three seeds; **regenerate before quoting**)

| Method | ACC | PACC (HiSPA ℓ=24) | ASR ↓ | LL-Acc@0.18 | ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| Mamba (unconstrained) | 89.7 | 23.4 | 92.1 | 19.2 | 0.058 |
| GloRo-Mamba (head only) | placeholder — produce with `scripts/todo4_gloro_mamba_baseline.py` or delete | | | | |
| CLASP | 89.7 | 67.1 | 38.5 | — | — |
| SpectralGuard (Bonetto 2026) | 89.4 | 71.4 | 35.6 | — | — |
| LipMamba-130M | 90.8 | 82.7 | 32.4 | 77.1 | 0.038 |
| LipMamba-370M | 91.9 | 85.3 | 30.9 | 80.4 | 0.034 |

These numbers come from the manuscript, not from a run of this repository;
the scripts under `scripts/todo*.py` regenerate them from the seeds
{42, 137, 2026}.  PACC must additionally be evaluated under the adaptive
attack (`todo2_adaptive_attack.py`) before it can be described as worst-case.

## Intended use

Research on robustness of selective SSMs; a defensive primitive, not a
stand-alone safeguard.  The attack code is for evaluating one's own systems.

## Limitations

* Global certificate vacuous at depth; local radii are empirical.
* Retention bound is per-input and, for the stated constants, ≈ 1 token.
* HiSPA, CLASP, SpectralGuard, RoBench-25 are unrefereed 2026 preprints.
* The HarmBench-CLS classifier is not bundled (licence); the rule-based
  stand-in in `attacks/jailbreak.py` is for unit tests only.
