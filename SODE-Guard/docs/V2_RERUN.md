# SODE-Guard v2 re-run

The v2 pipeline implements modules 1–3 of the research report
(`reports/SODE Guard improvement modules.md` at the repository root) and re-runs
SODE-Guard on ICS3D over five seeds. The ICLR-rebuttal pipeline
(`scripts/rebuttal_run.py`) is unchanged, so the rebuttal numbers stay reproducible.

* Script: `scripts/v2_run.py` (stages `prepare`, `train`, `attack`, `certify`, `aggregate`, `report`)
* Colab notebook: `notebooks/v2_rerun_ics3d_colab.ipynb`
* Tests: `tests/test_v2_modules.py`

## Modules

| # | Module | Code | Used by |
|---|---|---|---|
| 1 | Logit-adjusted CE (Menon et al., 2021); focal and class-balanced CE and square-root sampling as ablations | `src/training/imbalance.py` | every neural model |
| 1 | AdamW, cosine LR, 60 epochs, early stopping on validation macro-F1 (patience 12) | `scripts/v2_run.py::train_neural` | every neural model |
| 1 | Post-hoc logit adjustment `-τ log π`, τ tuned on the selection half; per-class biases as an ablation | `src/training/imbalance.py::tune_tau` | every model, trees included |
| 1 | Differentiable quantile input encoding (ablation) | `src/models/pipeline.py::QuantileEncoder` | `sode_quant` |
| 2 | EOT-matched adversarial training: N-FGSM through the 8-path mean logits with fresh paths; PGD-5 and TRADES as ablations; ε ramp over the first 15% of epochs; early stopping on PGD-10 validation macro-F1 | `src/training/robust.py` | `sode_at`, `ode_at`, `mlp_at` |
| 3 | Gaussian input noise (σ = 0.25, a pure-diffusion prefix of the SDE) with consistency regularisation (Jeong & Shin, 2020) | `src/training/robust.py::gaussian_consistency_loss` | `sode_rs`, `mlp_rs`, `ode_rs` |
| 3 | Randomised-smoothing certificate over input noise and Brownian paths jointly | `src/certify/smoothing_sde.py` | the `*_rs` models |

The post-hoc bias and the optional encoder sit inside `Pipeline`, which is the
deployed predictor, so every attack differentiates through them.

**Why the certificate is sound for a stochastic base classifier.** For SODE-Guard,
f(z, W) is the class of the 8-path mean logits at input z with Brownian paths W,
drawn independently of z. With h_c(z) = P_W[f(z, W) = c] ∈ [0, 1], the smoothed
classifier is G(x) = argmax_c E_ε[h_c(x + σε)]. Cohen et al.'s bound holds for
[0, 1]-valued base classifiers (Salman et al., 2019, Lemma 2). Sampling (ε, W)
jointly therefore gives a valid Clopper–Pearson bound and the radius
σ Φ⁻¹(p_A). The certified object is the smoothed classifier G, not the 8-path
mean-logit predictor. Theorem A is not used in v2.

## Protocol

* **Data.** `Containers_Dataset.csv`, capped at 20,000 flows per class. Identifier columns and `Src Port` are dropped. Exact duplicates and feature vectors with conflicting labels are removed *before* splitting (counts in `meta.json`).
* **Splits.** Every seed draws a new stratified 60/20/20 split. Validation is halved into a selection half (early stopping, τ) and a calibration half (held back for calibration and conformal work). An optional per-class chronological split (`--split class_time`) replaces the degenerate global temporal split.
* **Models (core).** `sode_base` (the rebuttal recipe on the new split), `sode`, `ode`, `mlp`, `lgbm`, `xgb`, `rf`, `sode_at`, `ode_at`, `mlp_at`, `sode_rs`, `mlp_rs`. Ablations are listed in `ABLATIONS`.
* **Attacks.** ℓ∞ ε = 0.03 in z-units, on a class-stratified test subset; macro-F1 is also reported reweighted to the test class mix. The `lite` profile (all seeds) runs PGD-40, PGD-100 ×2, APGD-CE/DLR (EOT 4), EOT-PGD-50 (EOT 8), a 64-path expected-margin attack, Square (1,000 queries) and transfer from `mlp` and `mlp_at`. The `full` profile matches the rebuttal (PGD-200 ×5, EOT 32, 256 paths, Square 5,000) and is run on seed 0 as an attack-strength check. Trees get Square and transfer only. On seed 0, ε and step sweeps run as gradient-masking checks.
* **Certification.** σ = 0.25, n₀ = 100, n = 2,000 (lite) or 10,000 (full), α = 0.001, on one class-stratified subset of 500 flows per seed shared by all certified models. Abstentions count as errors. The script reports certified accuracy, ACR and per-class certified recall, plus a version reweighted to the test mix.
* **Statistics.** Mean ± 95% t-interval over seeds. Each contrast declared in `CONTRASTS` gets a paired bootstrap (resampling flows within each seed, the same indices for both models), and the family is Holm-adjusted.

## Reading the results

`summary.md` holds four tables: clean and worst-case macro-F1, certified accuracy,
per-class clean F1, and the declared contrasts. The contrasts answer three questions:

1. Does module 1 lift clean macro-F1 over the rebuttal recipe (`sode` vs `sode_base`), and how far is it from the trees (`sode` vs `lgbm`)?
2. Does module 2 close the worst-case gap (`sode_at` vs `sode`), and does SODE-Guard differ from an MLP or a neural ODE trained the same way (`sode_at` vs `mlp_at`, `ode_at`)?
3. Does module 3 bring SODE-Guard's certified accuracy to the smoothed MLP's (`sode_rs` vs `mlp_rs` at r = 0.273)?

The research report expects parity, not superiority, on questions 2 and 3, so
null contrasts there are an expected and reportable result.

## Running on Kaggle (unattended)

`notebooks/v2_rerun_ics3d_kaggle.ipynb` runs the same plan as a background
Kaggle job (**Save Version → Save & Run All**), so it does not depend on a
browser tab staying open.

* **Settings:** Accelerator *GPU T4 x2*, Internet *On*, and the ICS3D dataset added as an input (read from `/kaggle/input`, no API token).
* `scripts/v2_driver.py` spreads the seeds over the two GPUs. For each seed it runs train → attack → certify, then aggregates.
* The driver stops every process before a time budget (default 11 h) runs out, so the job finishes inside Kaggle's 12-hour limit and its output is saved.
* All writes are atomic (temporary file, then rename), so a stopped process never leaves a truncated checkpoint or `results.json`.
* **To continue in a further run:** attach the notebook's own previous output (*Add Input → Notebook Output Files*) and commit again. Finished models, attacks and certificates are skipped.
* **Why not a TPU or the P100:** the code is CUDA PyTorch, and the SDE solver and the tree models do not run on a TPU without a rewrite. Recent PyTorch builds no longer support the P100's architecture.
