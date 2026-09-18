# Demo outputs — pipeline checks only

Produced by `python scripts/regenerate_all.py --demo` on a **random,
untrained 2–4-layer model with random tokens on CPU**.  The numbers are
meaningless as results; they exist to show that every script runs end to end
and that the rendered LaTeX (`todo_snippets_DEMO.tex`) has the right shape.

What they do show correctly, because it is structural:

* `todo1_ell_star.json` — Theorem-2 ℓ\* is < 1 for the manuscript's constants;
  the (Δ_max, λ_max) sweep and the required product Δ_max λ_max ≤ 0.029 for ℓ\* = 24.
* `todo5_fig2.json/.png` — log₁₀ K at depth 24: worst case 193.0, data-dependent
  34.5, op-norm −7.2, attack-based lower estimate ≈ 0.  The lower estimate is
  tens of orders below either bound, as a correct Figure 2 must show.
* `todo2_adaptive.json` — the adaptive `saturate` adversary pins Δ_t at ~88 % of
  Δ_max; the `overwrite` adversary moves the state by > 1 relative unit at
  unchanged norm (Remark 5's caveat is measurable).
* `todo4_baselines.json` — all four presets train, evaluate (ACC, PACC, LL-Acc,
  global-constant accuracy = 0, ECE, latency) and pass through Friedman/Holm.

Regenerate with real checkpoints as described in `docs/REPRODUCIBILITY.md`.
