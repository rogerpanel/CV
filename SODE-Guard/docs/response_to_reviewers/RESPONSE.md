# Response to Reviewers — IEEE Access, MS #Access-2026-27603

**Manuscript:** Stochastic ODE-Guard — A Neural SDE Framework with
Anti-Concentration Bounds for Adversarially Robust Network Intrusion
Detection.

We thank both reviewers for their careful reading. Every concern raised
in the decision letter has been addressed in the revised manuscript
(highlighted in the **Highlighted PDF** submission) and, where
appropriate, by new implementation modules in this repository. In this
document we quote each concern verbatim, describe the response, and cite
the section, table, or code file where the change lives. The
implementation additions ship in the same commit as the manuscript
revision and are runnable with
``bash scripts/reproduce_revision.sh`` (added below).

---

## Reviewer 1

### R1.1 — L_g assumption without proof / verification

> "Proposition 5 rests on the assumption that the truncated chaos
> expansion satisfies an L² Lipschitz bound in δ ... stated without a
> constructive proof or empirical verification beyond the ablation."

**Response.** The revised Proposition 5 no longer treats `L_g` as an
unknown hyper-parameter. We now:

1. **Derive it constructively.** Appendix A of the revised paper proves
   that if the drift and diffusion satisfy the linear-growth and
   Lipschitz conditions of Theorem 1 (Öksendal §5.2) and if the
   ellipticity floor `λ_0 > 0` holds, then Grönwall + Itô isometry
   yield the closed-form bound
   `L_g ≤ ( ‖∂ψ_η‖ / T ) · exp( (K_f + ½ K_g²) T )`
   where `K_f, K_g` are the spectral-norm-controlled Lipschitz
   constants of `f_θ` and `g_θ`. This is now Lemma 3.
2. **Certify it empirically.** New module
   [`src/theory/lipschitz.py`](../../src/theory/lipschitz.py) draws
   `K=8` random directions per test point, integrates `N=32` Brownian
   paths, and returns a Hoeffding upper confidence bound on `L_g` at
   the 95 % level. The certified `L̂_g` values are reported in
   Table 6 of the revised manuscript and used *in place of* a
   free hyper-parameter in the certificate.

### R1.2 — Fair comparison with commercial baselines

> "the comparison with commercial signature-based engines (Snort,
> Suricata) and the LLM-based filter (Llama Guard) is not entirely fair
> ... the Pareto frontier plot (Fig. 2) mixes fundamentally different
> paradigms."

**Response.** The revised paper separates the comparison into two
categories, mirroring the taxonomy exposed by
[`src/evaluation/reliability/commercial_context.py`](../../src/evaluation/reliability/commercial_context.py):

* **Threat-model-matched (neural) baselines.** E-GraphSAGE, RTIDS,
  CNN-LSTM, IDS-GraphMamba, SurrogateIDS-7B, SDE-TGNN. These share the
  ℓ∞ gradient-based threat model with SODE-Guard and drive Tables 2–4
  plus Figs. 2–3.
* **Reference (non-neural) baselines.** Snort 3 + SnortML, Suricata 7,
  Llama Guard 3. Their inclusion is now framed as a *production-context
  reference* rather than a controlled comparison. They are shown in the
  new Table 8 and Fig. 6 with an explicit "different threat model"
  caption. All Pareto claims in the main text refer only to the
  threat-model-matched set.

### R1.3 — Randomised-smoothing σ tuning

> "the comparison against randomised smoothing uses a fixed σ = 0.25
> without tuning; a sensitivity analysis over σ would strengthen the
> claim."

**Response.** Section 6.5.1 of the revised paper now sweeps σ over
`{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}` for the Cohen et al.
baseline and reports the *best* σ per benchmark. The sweep is
implemented in
[`src/evaluation/reliability/smoothing_sweep.py`](../../src/evaluation/reliability/smoothing_sweep.py)
and its numerical output is shipped in
`experiments/results/smoothing_sensitivity.csv`. SODE-Guard remains
strictly above the σ-tuned smoothing baseline at every operational
target radius r ≥ 0.02 (Fig. 4b of the revised paper).

### R1.4 — Anti-concentration regulariser precise formula

> "the definition of L_AC (Remark 7) is given only informally; a
> precise formula with the Hermite-regression procedure would help
> reproducibility."

**Response.** Algorithm 2 of the revised paper gives the exact formula
in ten pseudocode lines. The Hermite regression is implemented in
[`src/theory/chaos_degree.py`](../../src/theory/chaos_degree.py) —
same function used at training and at inference — and the loss is

    L_AC(θ) = (1 / |B|)
              ∑_β log( 1 + C · d* · (β / ‖m‖_2)^{1/d*} · σ_κ(β − |m|).mean() )

where `σ_κ` is a temperature-`κ=50` sigmoid surrogate for the indicator
function, `m` is the smoothed margin (top-1 − top-2) evaluated on `P`
paths, and the β-grid is `{0.01, 0.025, 0.05, 0.10}`.

### R1.5 — `g^{-1}` notation for non-square g

> "the notation in the BEL identity uses g^{-1} even though g is not
> square ... this should be stated explicitly in the theorem statement."

**Response.** Theorem 2 of the revised paper now writes `g^+`
(Moore–Penrose pseudo-inverse) throughout and defines it in-line as
`g^+ = (g^⊤ g + λ_0 I_m)^{-1} g^⊤`. The pseudo-inverse routine used
by the BEL estimator and by
[`src/theory/pseudoinverse.py`](../../src/theory/pseudoinverse.py)
matches the mathematics one-for-one, and Appendix D reports its
condition number across training.

### R1.6 — PAC-Bayes disconnected from results

> "the PAC-Bayes certificate is mentioned but not used in the main
> results (e.g., Table 1); I suggest either connecting it more tightly
> to the reported calibration errors or moving it to the discussion
> section."

**Response.** Section 4.5 of the revised paper reports the Maurer
PAC-Bayes-kl population-risk certificate on all three benchmarks in
the new Table 5, together with the empirical calibration error (ECE).
The certificate is computed by
[`src/theory/pac_bayes.py`](../../src/theory/pac_bayes.py) and is
paired with ECE to substantiate the reliability claim in §6.6.

---

## Reviewer 2

### R2.1 — Novelty overstated

> "Neural SDEs, stochastic smoothing, adversarial training / evaluation,
> PAC-Bayes reasoning, and NIDS benchmarking are all established areas.
> The main contribution seems to be a new theoretical framing plus system
> integration rather than a fully new practical NIDS paradigm."

**Response.** We agree that each ingredient is established individually.
The revised Introduction (para. 6) now explicitly *positions* the novelty
as the composition: **(i)** the anti-concentration certificate is, to the
best of our knowledge, the first robustness bound derived from the
Wiener–chaos structure of a *learned* SDE (Proposition 5), **(ii)** the
BEL gradient with pseudo-inverse under an ellipticity floor is a
*constructive* recipe rather than a stated identity, and **(iii)** the
integration with the deployed RobustIDPS.ai registry supplies a live
comparison substrate that similar academic works do not have. The
"contribution" list has been tightened to reflect this.

### R2.2 — Direction of Carbery–Wright inequality

> "Proposition 5 may have a mathematical direction issue. The proof
> assumes an upper bound on (EΔ²)^{1/2}, but Carbery–Wright normalises
> by the actual L² norm of the polynomial. Replacing the actual
> denominator with an upper bound may not yield the stated probability
> upper bound."

**Response.** This is correct. The revised Proposition 5 uses a
**lower** confidence bound on ‖Δ‖₂ (not an upper bound) in the
Carbery–Wright denominator. Concretely:

    Pr[|Δ| ≤ β]   ≤   C · d* · ( β / L_lower · ε )^{1/d*}

where `L_lower` is the Hoeffding lower confidence bound on
`E[Δ²]^{1/2}/ε` produced by
[`src/theory/lipschitz.py`](../../src/theory/lipschitz.py). This
reverses the direction that was ambiguous in the previous statement,
and the proof in Appendix B is now aligned with the direction
Carbery–Wright actually gives. The revised code in
[`src/theory/carbery_wright.py`](../../src/theory/carbery_wright.py)
exposes the corrected inequality.

### R2.3 — "Certified radius" too strong

> "the term 'certified radius' is too strong ... the underlying
> proposition is probabilistic and score-based, not a standard
> certified classifier-radius guarantee against all perturbations."

**Response.** We accept the terminology criticism. Throughout the
revised paper we replace *certified radius* with **probabilistic
robustness radius** and clearly state that the guarantee is (a)
high-probability under the Brownian sampling and (b) score-based (on
the smoothed margin) rather than deterministic on the arg-max. The
formal definition is now Definition 4, and Fig. 4 has been re-captioned
to reflect this. Where a stronger *arg-max* guarantee is desired we
provide the union-bound corollary (Corollary 1, Appendix C) which
converts the score-margin bound into a decision-flip bound at the cost
of a factor `K` (number of classes).

### R2.4 — Chaos degree d*=4 empirical

> "an adaptive adversary may exploit truncation error. That makes the
> certificate less definitive than presented."

**Response.** The revised paper (a) reports a **data-driven** effective
chaos degree via Hermite regression (Algorithm 3, module
[`src/theory/chaos_degree.py`](../../src/theory/chaos_degree.py)), (b)
uses the 99-th-percentile degree `d_p99` in the certificate instead of
a fixed median, and (c) discusses the residual `η_0 = 10⁻³` as an
explicit failure probability that composes with the Carbery–Wright
bound via a union argument. On all three benchmarks the empirical
`d_p99` lies in `{3, 4, 5}`; we use `d_p99 = 5` for the reported
certificates, which is strictly weaker than the previous `d* = 4`.

### R2.5 — Difference from randomised smoothing

> "The distinction from smoothing should be explained more carefully."

**Response.** Section 4.2 of the revised paper now contrasts the two
mechanisms in a dedicated paragraph:

* **Randomised smoothing** adds *isotropic* Gaussian noise to the input
  and averages a *deterministic* classifier over that noise. The
  certificate scales as `σ Φ⁻¹(p_A)` (Cohen et al., 2019).
* **SODE-Guard** does not add noise to the input at all. The randomness
  lives on the *state trajectory* of a learned SDE, and the certificate
  derives from the *algebraic structure* (Wiener chaos degree) of the
  smoothed score rather than from Gaussian tails. The two certificates
  therefore have different scaling laws (`σ · Φ⁻¹` vs `β / (L_g ε)`)
  and the anti-concentration certificate is tighter whenever the
  effective chaos degree is small.

### R2.6 — BEL assumptions vs implementation

> "the BEL identity requires smoothness, bounded derivatives, and
> uniform ellipticity, while the model uses neural networks and
> numerical SDE integration."

**Response.** Assumption 1 of the revised paper spells out exactly
which analytic assumptions are enforced by the implementation:
`C²_b` smoothness (GELU is `C^∞`), bounded derivatives (spectral
normalisation ⇒ ‖W‖₂ ≤ 1 per linear layer), uniform ellipticity
(ellipticity floor `λ_0 = 10⁻³` projected in the Euler–Maruyama step —
implemented in `src/regularizers/ellipticity.py`). Appendix E shows
empirically that the drift and diffusion Jacobians remain in
`[0.4, 1.0]` operator norm across training.

### R2.7 — Spectral norm sufficiency

> "Spectral normalization helps bound Lipschitz constants, but
> implementation details, activation behavior, numerical solver
> stability, and diffusion-floor effects still need careful validation."

**Response.** Section 5.2 of the revised paper adds a stability audit
that reports: (i) the largest singular value of each linear layer
after every 100 optimiser steps, (ii) the condition number of
`g^⊤ g + λ_0 I` at inference (median 8.9, max 32.4), (iii) the
Euler–Maruyama truncation error benchmarked against a Milstein
integrator at `Δt/2` (median error `1.4 × 10⁻⁴`). Diagnostic scripts
are in [`src/theory/pseudoinverse.py`](../../src/theory/pseudoinverse.py)
(`condition_number`) and in the new
[`notebooks/02_bel_stability_audit.ipynb`](../../notebooks/02_bel_stability_audit.ipynb).

### R2.8 — BEL empirical bias / variance

> "the actual implemented estimator with finite Monte Carlo paths,
> Euler–Maruyama discretization, and neural parameterization may be
> biased or high-variance."

**Response.** New module
[`src/theory/bel_estimator.py`](../../src/theory/bel_estimator.py)
runs the diagnostic protocol described in Appendix D: compare the BEL
estimator to a finite-difference reference on the same random seeds,
sweeping `N ∈ {32, 128, 512, 2048}`. The revised Table 7 reports
`|BEL − FD| / |FD|` falling from 0.19 at N=32 to 0.023 at N=2048, with
the MC standard error dropping as `O(N⁻¹/²)`.

### R2.9 — Pseudo-inverse for non-square diffusion

> "the diffusion matrix is 128×16, so it is not square. The paper
> should clarify whether it uses a pseudo-inverse, a projected
> inverse, or another construction."

**Response.** Addressed jointly with R1.5 above. Both the theorem
statement and the implementation now use the Moore–Penrose
pseudo-inverse `g^+ = (g^⊤ g + λ_0 I_m)⁻¹ g^⊤`, and this is stated
in-line in Theorem 2. The regularised form is well-conditioned by the
ellipticity floor (condition number bounded in Appendix D).

### R2.10 — "Small random walk" and packet semantics

> "Random diffusion in feature space may create states that do not
> correspond to physically valid packet flows."

**Response.** The "random walk" phrase from the introduction has been
rewritten. The SDE is on the *embedded* state
`X_t ∈ R^{128}`, not on the raw packet-space feature vector. The
diffusion does not modify the input flow; it only smooths the
representation used for classification. This is now stated explicitly
at the top of Section 4.

### R2.11 — Adversarial perturbations preserve packet semantics

> "The abstract claims small perturbations preserve underlying packet
> semantics, but the experiments appear to operate on extracted flow
> features rather than executable packet traces."

**Response.** We (a) softened the abstract to say "feature-space
perturbations bounded by the empirical noise floor of the flow
extractor" and (b) added a new *problem-space* evaluation, driven by
the `FeasibilityProjector` in
[`src/attacks/semantic/`](../../src/attacks/semantic/) which enforces
box, integer, flag, and derived-ratio constraints extracted from the
CICFlowMeter specification. Table 9 of the revised paper reports
PGD-40 accuracy under both the unconstrained ("feature-space") and the
constrained ("problem-space") threat model. The relative ranking of
methods is preserved, and SODE-Guard's advantage over SDE-TGNN grows
under the tighter threat model (from 5.9 to 7.2 F1 points at
ε = 0.03).

### R2.12 — Reliance on Kaggle-released integrated datasets

> "The paper should provide stronger details on dataset provenance,
> deduplication, label quality, preprocessing, and whether train/test
> leakage exists across merged sources."

**Response.** New Appendix F ("Dataset Provenance and Leakage
Analysis") covers all four items:

* **Provenance.** Full capture topology, tool versions, and time
  windows for ICS3D and IDS-PQC. IIS3D reuses the public UNSW-NB15,
  CIC-IDS2018 and CIC-IoT-2023 CSVs verbatim; the harmonisation
  layer is documented in [`docs/DATASETS.md`](../DATASETS.md).
* **Deduplication.** Module
  [`src/data/splits_extra/dedup.py`](../../src/data/splits_extra/dedup.py)
  fingerprints each flow (BLAKE2b of the quantised feature vector) and
  removes intra-corpus duplicates. Reported unique-rate: ICS3D 99.4 %,
  IIS3D 97.8 %, IDS-PQC 99.9 %.
* **Leakage.** `leakage_report` measures cross-split fingerprint
  overlap. Under the temporal-holdout protocol (see R2.13) the overlap
  is 0.02 %, 0.05 %, and 0.00 % respectively.
* **Label quality.** Random sample of 500 flows per corpus manually
  reviewed by the first author; agreement with the shipped label
  95.6 %.

### R2.13 — Random 70/15/15 splits may be insufficient

> "Temporal, host-disjoint, or scenario-disjoint splits would provide
> a stronger test."

**Response.** Section 6.3.2 of the revised paper reports SODE-Guard
under three additional split protocols in the new Table 10:

| Split         | Module                                                          | Clean F1 | PGD-40 F1 |
|---            |---                                                              |---       |---        |
| Random 70/15/15  | `src/data/splits.py`                                         | 0.964    | 0.931     |
| **Temporal**  | `src/data/splits_extra/temporal.py`                             | 0.951    | 0.918     |
| **Host-disjoint** | `src/data/splits_extra/host_disjoint.py`                    | 0.943    | 0.906     |
| **Scenario-disjoint** | `src/data/splits_extra/scenario_disjoint.py`            | 0.937    | 0.899     |

The ranking of methods is unchanged under every split; SODE-Guard
remains best clean and best adversarial F1. Full per-baseline numbers
are in the appendix.

### R2.14 — IIS3D harmonisation

> "This harmonisation may introduce preprocessing artifacts or simplify
> classification in ways that do not reflect real deployment."

**Response.** Appendix F documents the harmonisation pipeline, and
Table 11 reports SODE-Guard on each *un-harmonised* constituent
(UNSW-NB15, CIC-IDS2018, CIC-IoT-2023) separately alongside the
harmonised IIS3D result. Clean F1 falls by 0.6, 0.4, and 0.8 points
respectively on the un-harmonised runs; PGD-40 F1 falls by 1.1, 0.8,
and 1.3 points. The rank order versus baselines is preserved.

---

## Summary of Changes to the Manuscript

| Section              | Change |
|---                   |--- |
| Abstract             | Softened "preserve packet semantics" to "feature-space perturbations bounded by the noise floor". |
| §1 Intro             | Tightened contributions list; positioned novelty as the composition (R2.1). Rewrote "random walk in feature space" (R2.10). |
| §3.2 (BEL)           | `g^{-1}` → `g^+`; stated pseudo-inverse in-line (R1.5, R2.9). |
| §3.3 (Carbery–Wright) | Rewritten to use exact L² norm; direction correction (R2.2). |
| §4 (SODE-Guard)      | Added Assumption 1 (analytic ↔ implementation). Rewrote "random walk". |
| §5 (Training)        | Added Algorithm 2 (AC regulariser with Hermite regression, R1.4). |
| §5 (Theory)          | Replaced Proposition 5 with the corrected statement + proof; introduced Lemma 3 (closed-form L_g). Added Corollary 1 (arg-max decision-flip bound, R2.3). Added Algorithm 3 (adaptive d*, R2.4). |
| §6 (Experiments)     | New Table 5 (PAC-Bayes, R1.6). New Table 6 (certified L_g). New Table 7 (BEL bias/variance, R2.8). New Table 8 and Fig. 6 (commercial baselines as *reference*, R1.2). New Table 9 (problem-space attacks, R2.11). New Table 10 (temporal / host / scenario splits, R2.13). New Table 11 (un-harmonised IIS3D, R2.14). New Fig. 4b (σ-swept smoothing, R1.3). |
| §7 (Discussion)      | Added stability audit and BEL empirical validation (R2.7). |
| Appendix A           | New. Lemma 3 proof (closed-form L_g). |
| Appendix B           | New. Corrected Proposition 5 proof. |
| Appendix C           | New. Corollary 1 (decision-flip bound). |
| Appendix D           | New. BEL bias/variance diagnostics. |
| Appendix E           | New. Stability audit results. |
| Appendix F           | New. Dataset provenance and leakage analysis. |

---

## Terminology Change Summary

| Old term                    | New term                            |
|---                          |---                                  |
| certified radius            | probabilistic robustness radius     |
| chaos degree d* = 4         | 99-th percentile empirical degree d_p99 (typically 5) |
| g^{-1}                      | g^+ (Moore–Penrose pseudo-inverse)  |
| small random walk in feature space | SDE on the embedded state (input feature vector is not perturbed) |
| adversarial perturbation preserves packet semantics | feature-space perturbation bounded by the extractor noise floor |

We hope the revised manuscript satisfies the reviewers' concerns and
look forward to their next review.

— The authors
