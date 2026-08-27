# Resolution plan (camera-ready, EMNLP-accepted)

| # | Reviewer issue | Resolution | Camera-ready action |
|---|---|---|---|
| 1 | `XX%` DP+SecAgg overhead placeholder | **MEASURE** with codebase; use actual number | Replace with measured value + method sentence |
| 2 | ε_T=5.0 at (q=0.2, σ=1.1, T=100, δ=1e-5) is inconsistent (Opacus gives ~13.6) | **VERIFIED**: reviewer correct. Report actual PRV/Opacus ε_T; keep σ=1.1 as experimental value | Update ε_T value throughout: abstract, results table, ethics, limitations, discussion; add reproduced-run note in appendix |
| 3 | DP-SGD terminology mixed (per-example vs client-level) | Paper actually applies **client-update-level** clipping (Algorithm 1 line 6 clips whole `g_i`). Theorem 1 as-stated says "per-example clipping … averaged over the local minibatch" which conflates. **Rewrite** Theorem 1 statement and proof to explicitly use client-update-level clipping. Drop the `1/|D_i|` factor. | Rewrite Theorem 1, Corollary, sensitivity bound, proof sketch, and the Reproducibility "training-time overhead" paragraph; make Algorithm 1 clip line consistent |
| 4 | k*=25 derivation too compressed | Strengthen certificate proof: state exactly which random variable is smoothed, what perturbation corresponds to one malicious client, why k clients induce assumed perturbation, why FLTrust doesn't invalidate smoothing, where 2 e^{ε_r} comes from. | Rewrite Section 5 certificate proof + Appendix A.2 with explicit five-item derivation |
| 5 | Theorem stronger than "arbitrary 33% Byzantine" abstract claim | Paper defines (k, ε_wt)-bounded-update attacks; certificate is against these, not arbitrary Byzantine. | Update abstract + contributions to say "certifies robustness against coalitions of up to k* clients under the bounded-update threat model" |
| 6 | 33% vs 50% inconsistency | Certified k*=25/50 = 50%; empirical FLTrust stress test tolerates ρ ≤ 0.33. Distinct results. | Add explicit distinction in Results and abstract |
| 7 | "97.9% of PEFTGuard upper bound" | Change to "within 1.9pp of PEFTGuard's reported centralised macro-F1" | Rewrite one sentence in Results + Discussion |
| 8 | Benchmark provenance (real vs synthetic components) | Add explicit provenance table | Add Table in Appendix |
| 9 | Data leakage / lineage-disjoint splits | Add note that we also report lineage-disjoint, base-model-disjoint, attack-family-held-out splits in Appendix; report main-paper numbers under the strictest split; acknowledge if not run, note as limitation | Add split-strategy description + note in Limitations; add table in Appendix |
| 10 | 50/50 balanced macro-F1 too easy | Add FPR, precision, recall, PR-AUC at operating points | Add operating-point table in Appendix |
| 11 | IDS experiment for NLP venue | Keep as short cross-domain sanity check in appendix (already done: `app:ids`) | No changes needed — already appendix-only |
| 12 | Too many claims for evidence | Hierarchize claim structure | Add explicit primary/secondary/theoretical claim structure in intro |
| 13 | Statistical reporting weak | Add unit-of-analysis clarification; add mean±95%CI to primary metrics | Add note in Experiments about statistical unit; add CI to Table 1 |
| 14 | Ablation should distinguish detection vs certification | Emphasize DP removal destroys certificate (k*=0) while improving F1 (0.5pp) — beautiful result | Add explicit sentence in ablation subsection |
| 15 | "no inference-time cost" is too absolute | Rewrite to "the privacy mechanism introduces no additional model-training step at verification time; verification itself requires ~0.7s per adapter" | Rewrite one sentence in Reproducibility |
| 16 | A100/A10G/T4 latency should be in main paper | Already reported in Appendix; also add short mention in Discussion | Add one sentence to Discussion |
| 17-19 | Anonymity issues | **Paper is CAMERA-READY, ACCEPTED at EMNLP** — anonymity issues no longer apply. Remove "Anonymous version for review" comment. Keep authors, funding, GitHub URL. | Update file header |
| 20 | "the first formal model" — too strong | Change to "to our knowledge, the first formal model" | Rewrite one sentence in Contributions |
| 21 | LoRAchain-2026 provenance detail | Add explicit provenance table, checksums note, license note | Add Appendix subsection |

## Overhead measured value
See running measurement; will insert once complete.

## Camera-ready specific choices

* **Anonymous version for review** comment: **REMOVE** — paper accepted at EMNLP.
* Authors, affiliations, acknowledgements, funding, GitHub URL: **KEEP**.
* Use `[final]` acl option (already there).
