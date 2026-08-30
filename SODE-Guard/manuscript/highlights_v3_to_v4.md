# Highlighted-PDF Change Map (v3 → v4)

This file catalogues every yellow-highlighted change the IEEE Access
resubmission checklist expects to see in the Highlighted PDF. Line
numbers refer to `SODEGuard_v4_Manuscript.tex`.

## Global terminology sweeps (highlight every occurrence)

| From                                | To                                             | Where |
|---                                  |---                                             |---    |
| certified radius                    | probabilistic robustness radius                | Abstract, §6.5, §7, caption of Fig. 4 |
| chaos degree $d^\star=4$            | 99-th percentile empirical degree $d^\star_{p99}$ (typically 5) | §4.1, §5.3, §6.4 |
| $g^{-1}$                            | $g^{+}$ (Moore–Penrose pseudo-inverse)          | §3.2 (Thm 2), §4.3 (Alg 1), §5.3 |
| small random walk in feature space  | SDE on the embedded state (input is not perturbed) | §1 (last paragraph), §4.1 |
| preserve packet semantics           | feature-space perturbation bounded by the extractor noise floor | Abstract, §1 |

## Block-level additions (highlight the entire block)

* **Abstract** — added the sentence starting "The chaos truncation degree is now selected adaptively per example …" and the last sentence describing problem-space projection.
* **§1 (Introduction)** — added the "What is genuinely new in the composition" paragraph after C0.
* **§3.2 (BEL identity)** — added Equation (2) defining $g^{+}$ and re-stated Theorem 2 with the pseudo-inverse.
* **§3.3 (Carbery–Wright)** — added the sentence "The denominator is the exact $L^{2}$ norm of $p$. The revised Proposition 5 uses a Hoeffding lower confidence bound …".
* **§3.4 (PAC-Bayes)** — added the reference to Reeb–Seldin kl-inversion.
* **§4 (Framework)** — added Assumption 1 (analytic ↔ implementation).
* **§4.1 (Flow representation)** — added the italicised note "The SDE evolves the embedded state $X_t$ …".
* **§4.3 (Training procedure)** — added Equation (7) with the explicit $\mathcal L_{\mathrm{AC}}$ formula.
* **§4** — added Algorithm 2 (adaptive chaos-degree estimator).
* **§5 (Theory)** — added Definition 4 (three robustness quantities), Lemma 3 (closed-form $L_g$), Corollary 1 (decision-flip bound), Remark 8 (why differs from randomised smoothing), Remark 9 (why $L_{\mathrm{lo}}$ not $L_g$).
* **§6.1 (Datasets)** — added the "Provenance, deduplication, leakage" paragraph.
* **§6.2 (Baselines)** — split into "threat-model-matched neural" and "reference" groups.
* **§6.4 (Ablation)** — added the row "$-$ Adaptive $d^\star_{p99}$ (fix $d^\star=4$)".
* **§6.5.1 (σ sweep)** — new Table 6.
* **§6.6 (Split-Protocol Audit)** — new §6.6, new Tables 7 and 8, new Table 9.
* **§6.7 (Reference baselines)** — moved the commercial + LLM-policy comparison into its own subsection with Table 10.
* **§6.8 (Diagnostics)** — new subsection with Tables 11 (certified $L_g$), 12 (PAC-Bayes + ECE), 13 (BEL bias / variance).
* **Appendices A–F** — all new.

## Row-level table edits

* **Table 1 (formerly Table I: aggregate F1)** — commercial rows moved to Table 10.
* **Table 3 (formerly Table III: ablation)** — added the `-` Adaptive $d^\star_{p99}$ row.
* **Table 4 (formerly efficiency)** — unchanged.
* **New Table 5 (σ sweep), Table 6 (Lipschitz certificate), Table 7 (PAC-Bayes)**, etc.

## Figure updates

* **Fig. 2 (radar)** — recomputed using only threat-model-matched baselines.
* **Fig. 4 (CDF)** — added the "randomised smoothing at best σ per benchmark" curve.

## Bibliography

* Added: `reeb2014klinv`.
* No entries removed; no entries reordered.

## Version banner

The pre-title `%% Version` header block was updated to `v3 → v4 — IEEE Access resubmission` with a note pointing to `docs/response_to_reviewers/RESPONSE.md`.
