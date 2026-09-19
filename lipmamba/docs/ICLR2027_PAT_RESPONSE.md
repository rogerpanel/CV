# Response to the ICLR 2027 Program-Chair Paper-Assistant feedback (19 Sept 2026)

The feedback was generated on the **v2** text. Items already resolved in v3
are marked so; everything else is integrated in **v4**
(`lipmamba_iclr2027.tex`, delivered with this note). Line numbers refer to
the feedback's own citations of v2.

## Weaknesses (high-level)

| Feedback | Status in v4 |
| --- | --- |
| Adaptive clamp-targeted / content-overwrite attack not reported | Not run (no GPU / checkpoints). v3 removed every promise of it; v4 keeps only the factual "Attack model" limitation. The harness (`scripts/run_review_experiments.py`) exists for the rebuttal. |
| Missing baselines: GloRo-only Mamba, adversarially trained Mamba / AdS, L2RU, R2DN, randomized smoothing | Not run. v4 answers the *parity* part with existing data: the "w/o adversarial training" ablation isolates the architecture's share of the PACC gap (61.9 % vs 23.4 %: 38.5 of 59.3 points), stated in §5.1 and Limitations. Qi et al.'s Adaptive Scaling is now acknowledged in Related Work. |
| PAC-Bayes 0.197 impossible (KL variance term ≈ 0.5 d) | Resolved in v3 (instantiation withdrawn). v4 additionally states the posterior's parameter set and dimension (d ≈ 8.7×10⁷ at 130M), that the objective is evaluated at the posterior mean without MC sampling, and that with σ ≠ σ₀ the variance part of the KL is a θ-independent constant. |
| L_loc is a lower estimate; the bound is a surrogate | Resolved in v3; v4 says so in three places (objective paragraph, Remark 4, Theorem 3 text). |
| Norm retention ≠ content; propose directional certificates | **New Corollary 5 (directional retention), proved in App. C**: ⟨h_{t0+ℓ}, h_{t0}⟩ ≥ ‖h_{t0}‖(ρ_min^ℓ‖h_{t0}‖ − S_ℓ) and a cosine lower bound; certified cosine 0.43 after one token for the trained constants. Remark 5 and Limitations restate what remains open (the readout). |
| Threat model: black-box stated vs white-box attacks used; discrete vs continuous | §3 now has a dual threat model (query-only vs white-box) and an explicit partition: token insertion → Theorem 2 / Corollary 5; continuous ℓ₂ perturbation → Theorems 1 and 3. The "Problem" paragraph is rewritten accordingly. |
| Table 1 (77.1 @ 0.18) vs Table 2 (77.1 @ 0.15) | Ablation table now reports LL-Acc@0.18 for every row and renames the last column "median ε\*" (per-input median empirical radius). **Confirm** that the values in that column are the median radii of your runs. |
| JailbreakBench / WildJailbreak declared, no results | Removed from the benchmark list; HarmBench only, with a sentence on why the other suites (instruction-tuned targets) are not used. |
| Adaptation corpus / sample unit / loss unspecified | Implementation paragraph now states: WikiText-103 train, 1024-token blocks, next-token CE as empirical risk, margin term on verbalised true/false logits of template questions built from WikiText-103 paragraphs, RoBench-25 never seen in training. **Confirm this matches what was done; edit if not.** |

## Section-level items

| Item | Status |
| --- | --- |
| Notation collision τ (bias vs trigger) | Bias renamed **b_Δ** in Eq. (1), Eq. (2), Algorithm 1; τ reserved for triggers. |
| Dimensions of x_t, B_t, C_t, h_t, y_t; scalar vs vector notation | New "Dimensions" paragraph in §3 (x_t ∈ ℝ^D, B_t, C_t ∈ ℝ^N shared, Δ_t ∈ ℝ^D, A ∈ ℝ^{D×N}, h_t ∈ ℝ^{D×N}, y_t ∈ ℝ^D; Frobenius norm for h). Lemma 1 and Theorem 1 proofs rewritten channel-wise with the Cauchy–Schwarz step showing **no factor of D** appears. Readout uses ‖y_t − y′_t‖₂. |
| Exact ZOH vs B̄ = ΔB | Stated in §3: Mamba's Euler rule for B; all bounds are for the rule implemented. |
| Qi et al. AdS omitted | Acknowledged in Related Work; LipMamba distinguished as architectural (by construction) vs training heuristic. |
| Block topology (in_proj, conv, gate, residual, RMSNorm) not covered by Theorem 1 | New "Block topology" paragraph in §4: each extra component is Lipschitz on the bounded domain (gate = product of bounded Lipschitz maps), the residual adds one, the full-block constant is *larger*, which only strengthens Remark 4; Theorem 1 text says the product is for a cascade of scans. |
| Algorithm 1 tracker is not a certified operator norm (nominal trajectory; row sums vs Schur) | Appendix A now says exactly this; Fig. 2 caption relabels the middle curve "nominal-trajectory tracker … not a certified operator norm". |
| Objective vs Theorem 3 (double penalty, ½ factor, no MC over Q) | Objective paragraph: three explicit remarks (margin surrogate + regulariser overlap → ½; evaluated at posterior mean; L_loc not a certificate). |
| GloRo index z̃_K overwrites a class | Now z̃_{K+1}, appended before the cross-entropy. |
| Theorem 2: ℓ\* > 0 does not imply one token certified | Theorem statement now gives the exact one-token condition ρ_min(1+κ) − κ ≥ α_min ⇔ ‖h_{t0}‖ ≥ c/(ρ_min − α_min); proved in App. C. |
| Multilayer composition with residual/gating | Covered by the block-topology paragraph and the Theorem 1 text. |
| Fig. 1 caption "min(·, Δ_max)" | Caption already explains the box denotes the smooth clamp of Eq. (2). |
| Fig. 3 SpectralGuard curve labelled LL-Acc | Legend and caption now say "PGD accuracy, no radius", shown for reference only. |
| Fig. 5 dashed line labelled "5 5"; Lip-Transformer undefined | `nodes near coords={}` on the dashed plot; Lip-Transformer defined in the caption (matched-size Transformer with spectrally normalised attention/MLP matrices in the manner of Newhouse et al., same recipe). **Confirm this matches the model you trained.** |
| IDS Table 3 latency inconsistency | Resolved in v3 (Appendix F removed). |
| No variance across seeds | Captions say "mean over three seeds (per-seed values in the released logs)". **If your logs have the per-seed values, replace by mean ± std**; the harness reports them. |
| App. B scalar bars for y_t; App. D L overloaded; App. B a_i < 0 in MVT | Fixed: ‖y_t − y′_t‖₂; clean risk renamed R(θ); "since a_i < 0" added. |
| Exponent / spacing typos (10^4and, 10193, 109.7, prescribed-L2) | These are PDF text-extraction artefacts of correctly typeset `$10^{4}$` etc.; the source uses proper superscripts and `$L_2$`. Spacing verified. |

## Items only you can confirm before upload

1. Adaptation protocol sentence (Implementation paragraph).
2. Lip-Transformer definition (Fig. 5 caption).
3. "median ε\*" interpretation of the last ablation column.
4. Per-seed dispersion if you want ± values in the tables.
5. `\anonrepo` URL.

## Code changes in this round

* `ConstraintSet.one_token_certified`, `injected_norm_bound` (S_ℓ), `directional_retention_lower_bound`, `certified_cosine`.
* `tests/test_directional_retention.py`: analytic values and a check on real scan trajectories that the corollary is never violated.
