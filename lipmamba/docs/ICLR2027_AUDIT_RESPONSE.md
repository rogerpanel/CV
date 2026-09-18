# Response to the ICLR 2027 Editorial–Technical Audit — what changed in the code and what the numbers say

This file maps every audit finding (A–N, Tasks 1–5) and every red `\todo`
marker of `lipmamba_iclr2027.tex` to (i) the code change in this repository
and (ii) the script that produces the number or artefact the manuscript
needs.  It is deliberately *not* part of the anonymised bundle
(`scripts/todo6_make_anonymous_bundle.py` omits it).

## 0. Two numerical facts the authors must act on before anything else

Both are produced by `python scripts/report_constants.py` with the
manuscript's stated 130M constants and are covered by unit tests
(`tests/test_constants.py`).

| Fact | Value | Consequence |
| --- | --- | --- |
| Theorem 2, Eq. (lstar) with (s_B, Δ_max, λ_max, X_max, α_min, ‖h_{t0}‖) = (1, 0.5, 1, 1, 0.5, 4) | **ℓ\* = 0.95** (integer 0) | Confirms the TODO in Remark 5. The Figure-4 shaded region at ℓ ≈ 24 is not reproducible. Certifying ℓ\* ≥ 24 needs Δ_max λ_max ≤ 0.029 (now 0.5). |
| Theorem 1, L_block with Δ_min = 10⁻³, λ_min = 0.05 | **1.1 × 10⁸ per block**, 10¹⁹³ at 24 blocks | Figure 2's "∼10¹⁰ at 24 layers" is *not* the worst-case Theorem-1 product; 1/(1−ρ_max) = 2 × 10⁴ dominates. Only the data-dependent Algorithm-1 quantity can be ∼10¹⁰. The caption and Remark 4 must name the curve. |

The second fact is *worse* than the audit's item H: the text says "the
analytical product reaches ∼10¹⁰"; with the stated constants it reaches
10¹⁹³.  Either the plotted curve is the data-dependent one (then say so, and
the worst-case must be added or its magnitude stated), or the constants in
the Implementation paragraph are not the ones used (then correct them).

## 1. Model changes (Sections 3–4 of the manuscript)

| Manuscript | Code | Test |
| --- | --- | --- |
| Eq. (2) two-sided clamp Δ_t ∈ [Δ_min, Δ_max) | `models/clipped_delta.py` (was one-sided) | `test_clipped_delta.py` (bounds, 1-Lipschitz) |
| Assumption 1 ‖x_t‖ ≤ X_max | `models/input_clip.py` (projection, 1-Lipschitz), applied inside `SelectiveSSM` | `test_scan_trace_records_bounds_within_theory` |
| Lemma 1 H = c/(1−ρ_max) | `certificates/constants.py::ConstraintSet.H` | same test checks ‖h_t‖ ≤ H at run time |
| Theorem 1 (γ, H, no ‖h‖_∞) | `ConstraintSet.l_block`, `l_network` | `test_theorem1_constant_with_paper_constants_is_astronomical` |
| Algorithm 1 online D_t tracking | `SelectiveSSM.forward` (`ScanTrace.d_t`), `data_dependent_block_bound` | `dd ≤ worst case` asserted |
| Theorem 2 with (1+κ), ρ_min = e^{−Δ_max λ_max} | `ConstraintSet.ell_star`; `poisoning_immunity.py` (old ρ_min = e^{−Δ_min λ_min} was **wrong** — fixed) | `test_retention_bound_matches_eq_and_is_consistent_with_ell_star` |
| Remark 5 per-input ℓ\* distribution | `ell_star_distribution`, `ell_star_from_trace` | `test_ell_star_from_trace` |
| Theorem 3 with L_ℓ; Eq. (5) with ½ L_ℓ L_loc ε | `certificates/pac_bayes.py` (`l_ell`, `objective_half`) | `test_pac_bayes.py` |
| Appendix E local estimator (r = 0.3, 20 steps, 8 starts) | `certificates/local_lipschitz.py` | `test_local_lipschitz_and_ll_radius` |
| GloRo "⊥" logit | `models/glorot_head.py` | smoke |
| Diagonal-A: σ_min vs spectral radius (item G) | `ScanTrace.a_bar_min` / `a_bar_max` recorded separately | trace test |
| Trainer uses L_loc, not the global constant (Remark 4) | `training/trainer.py::lipschitz_mode = local\|global\|fixed` | — |

## 2. Audit items A–N → code / script

| Item | Status | Where |
| --- | --- | --- |
| A ‖h‖_∞ in Thm 1 | fixed (H from Lemma 1) | `constants.py` |
| B Δ_min > 0 by construction | fixed (two-sided clamp) | `clipped_delta.py` |
| C "tight" | dropped; docstrings say "upper bound, no tightness claim" | `THEORY.md` |
| D proofs / Gama | manuscript side; code exposes each proof step's constant | — |
| E norm ≠ content | `AdaptiveClampAttack(objective="overwrite")` is that adversary; report its `overwrite_distance` next to retention | `attacks/adaptive_clamp.py` |
| F ℓ\* algebra | (1+κ) denominator; sign handled; test | `constants.py` |
| G σ_min vs ρ | both recorded; docs state diagonal-A caveat | `selective_ssm.py` |
| **H** L_SSM = 6.5 vs 10¹⁰ | all radii use L_loc; `ll_accuracy` reports LL-Acc **and** the accuracy under the global constant side by side (the latter ≈ 0) | `evaluation/ll_acc.py` |
| **I** empirical LB within 2 % of UB | `todo5_fig2_lipschitz_depth.py` computes worst-case, data-dependent, op-norm and a true attack-based **lower** bound and refuses to call the plot consistent unless LB ≪ bound | `scripts/todo5_*` |
| **J** ℓ\* ≈ 24 circular | `todo1_ell_star.py`: per-input distribution + sweep; worst-case region drawn from constants only | `scripts/todo1_*` |
| **K** no adaptive attack | `AdaptiveClampAttack` (saturate / overwrite / margin; continuous + GCG-discrete) + Z-HiSPA + M-HiSPA (GA) | `scripts/todo2_*` |
| L LLM + IDS in one table | IDS loaders kept but scripts/docs treat IDS as Appendix F; anonymised bundle omits the platform doc | `todo6_*` |
| **M** baselines | `baselines.preset("gloro_mamba" \| "unconstrained_mamba" \| "naive_sn_mamba")` share the code path; randomized smoothing (Cohen et al.) added | `baselines/`, `scripts/todo4_*` |
| N production / perplexity provenance | `PerplexityProvenance` makes base id, revision, corpus, tokenizer mandatory; `todo3_*` builds LipMamba from `state-spaces/mamba-130m/370m` | `scripts/todo3_*`, `scripts/perplexity_overhead.py` |
| Task 1 attributions | RoBench-25 loader/registry now: 120 abstracts + 240 T/F questions, no separate trigger release; SpectralGuard → Bonetto (2026) in docs | `data/robench.py`, `data/registry.py`, `DATASETS.md` |
| Task 3 dead `rogerpanel/LipMamba` URL; de-anonymisation | anonymised bundle builder with leak check | `scripts/todo6_*` |

## 3. The seven red `\todo` markers → script → LaTeX snippet

`python scripts/regenerate_all.py --config … --checkpoint … --tokens …`
runs 1, 2, 4, 5 with seeds {42, 137, 2026} and then
`scripts/fill_paper_numbers.py` renders `paper/todo_snippets.tex` — the
replacement paragraphs with the *measured* numbers.  Nothing is invented: a
snippet appears only if its JSON exists.

| # | Marker (line in .tex) | Script | Output consumed by the snippet |
| --- | --- | --- | --- |
| 1 | Remark 5: recompute ℓ\* | `todo1_ell_star.py` | worst-case & data-dependent ℓ\* percentiles, required Δ_max λ_max |
| 2 | Baselines: adaptive white-box attack | `todo2_adaptive_attack.py` | Δ-saturation, retention, overwrite distance, label flips at each ℓ |
| 3 | Implementation: corpus + base checkpoints | `todo3_init_from_hf_mamba.py` + `perplexity_overhead.py` | base id, HF revision, clipped-eigenvalue fraction, PPL overhead |
| 4 | Table 1: GloRo-Mamba row | `todo4_gloro_mamba_baseline.py` | ACC/PACC/LL-Acc@0.18/ECE/latency, Friedman + Holm |
| 5 | Fig. 2: verify lower-bound curve | `todo5_fig2_lipschitz_depth.py` | four curves + pgfplots coordinates |
| 6 | Anonymous repo link | `todo6_make_anonymous_bundle.py` | zip for anonymous.4open.science |
| 7 | Fig. 4 shaded region | (same as 1) | pgfplots `\addplot` for the region |

**What the demo run (random 4-layer model, CPU) already shows** — pipeline
checks only, not results:

* TODO 5: log₁₀ K at depth 24 — worst case 193.0, data-dependent 5.8,
  op-norm −1.2, empirical LB −0.03.  The lower bound is ≈ 6 orders below the
  data-dependent bound, as it must be.
* TODO 2: the adaptive `saturate` adversary pins Δ_t at 88 % of Δ_max and
  drives retention to 0.14 at ℓ = 24 on an untrained model; the `overwrite`
  adversary moves the state by 1.2 relative units at unchanged norm — the
  behaviour Remark 5 warns about, now measurable.
* TODO 1: ℓ\* is < 1 for the stated constants at every observed ‖h_{t0}‖.

## 4. Things only the authors can decide (flagged, not decided here)

1. **Which Figure-2 curve to keep.** If the data-dependent curve is kept,
   state its magnitude and that the worst case is 10¹⁹³; if the worst case
   is kept, the constants must change (Δ_min λ_min ≈ 0.1–0.7).
2. **Whether to re-train with constants that certify a useful ℓ\*.**
   Δ_max λ_max ≤ 0.03 makes ρ_min ≥ 0.97; the Δ_max = 0.25 ablation row
   suggests clean accuracy tolerates smaller steps, but 0.03 is untested.
   Alternative: keep the constants and report the honest per-input
   distribution (median ℓ\* ≈ 1) and drop the "until the trigger is long
   enough to dominate the retention bound" sentence.
3. **Base checkpoints.** `todo3_*` proposes `state-spaces/mamba-130m` and
   `-370m` (The Pile, GPT-NeoX tokenizer) with WikiText-103 validation for
   the overhead; if the models were pre-trained from scratch instead, the
   corpus and tokenizer must be stated and the script's provenance block
   edited accordingly.
4. **Dual submission.** Withdraw from INJOIT / SiSI before uploading, and
   add one sentence delimiting the delta from the ESWA 2026 MambaShield paper.
5. **The HarmBench ASR column** is produced by the jailbreak harness with the
   real HarmBench-CLS classifier, which is not bundled (licence); the
   rule-based stand-in in `attacks/jailbreak.py` is for unit tests only.

## 5. Suggested honest framing for the abstract (one sentence each)

* Theorem 1: "an explicit, non-tight Lipschitz constant on the bounded-input
  domain; it is structurally informative and numerically vacuous at depth."
* Theorem 2: "a state-*norm* retention length ℓ\* that is per-input and, for
  the trained constants, of order one token; the clamp removes the Ā_t → 0
  mechanism, not content overwrite."
* Radii: "empirical local-Lipschitz radii (LL-Acc), not certificates."
* Adaptive evaluation: "a white-box adversary optimising against the clamp
  reduces retention to X and flips Y % of labels at ℓ = 24" — filled by
  `todo2_*`.
