# EMNLP 2026 camera-ready manuscript

This directory holds the final camera-ready sources for

> Anaedevha, Trofimov, Borodachev (2026).
> *FedLoRAGuard: Federated Dynamic Graph Neural Networks with
> Differential-Privacy Certificates for LoRA Adapter Integrity
> Verification.*  EMNLP 2026 (Budapest, Hungary).

## Files

| File | Purpose |
| --- | --- |
| `FedLoRAGuard_camera_final.tex` | Ready-to-compile Overleaf source with all reviewer fixes applied |
| `REVIEWER_RESOLUTIONS.md` | Resolution table: reviewer point → action taken |
| `CAMERA_READY_CHECKLIST.md` | **Pre-submission checklist mapped to the EMNLP 2026 Program-Chair notification** — reference audit, page-limit compliance on Overleaf, Responsible NLP Checklist section-number updates, presenter/visa/registration data. Open this before uploading to OpenReview. |
| `measurements/overhead_v{2,3,4}.json` | Raw wall-clock measurements from `scripts/measure_dp_overhead.py` used to derive the ~4% overhead figure in Appendix E |

## Overleaf compile checklist

1. Upload `FedLoRAGuard_camera_final.tex` to the EMNLP camera-ready template
   project on Overleaf; the template ships `acl.sty`, so keep the existing
   `\usepackage[final]{acl}` line unchanged.
2. Copy `figures/fig1_FedLoraGuard_arch.pdf` and
   `figures/fig2_mean.pdf` into the Overleaf `figures/` folder
   (same relative paths the .tex expects).
3. Compile with pdfLaTeX; two passes are enough
   (`bibitem`s are inline, no BibTeX needed).

## Key camera-ready changes vs the review draft

- **DP accounting** ε_T=5.0 → **ε_T ≈ 13.6** (Opacus PRV at
  σ=1.1, q=0.2, T=100, δ=10⁻⁵).  The manuscript now reports the
  honest PRV output and explains what the earlier draft got wrong.
- **DP terminology** unified as **client-update-level** DP-SGD
  (Algorithm 1 line 5 clips the whole client update, Theorem 1
  sensitivity is 2S by two-sided clipping and the graph-structural
  refinement is stated as a supplementary tighter bound).
- **Certified radius k\*** derivation strengthened to a
  four-step composition (PixelDP → coalition-bounded perturbation
  → CRFL → RDP composition) with an explicit statement of what is
  smoothed and where each factor comes from.
- **Certified 50% vs empirical 33%** distinction made explicit in
  §5, §6.1 and the Limitations.
- **"97.9% of PEFTGuard upper bound"** replaced by
  *within 1.9 pp of the centralised PEFTGuard reference number*.
- **Benchmark provenance table** added (Appendix D).
- **Lineage-controlled splits** (main-paper number now under the
  lineage-component-disjoint split; four splits reported in
  Appendix C).
- **Security-operating-point metrics** (FPR, precision, PR-AUC,
  detection@FPR=1%) added in Appendix C.
- **"No inference-time cost"** softened to the honest
  ~0.7s/A100/adapter statement.
- **XX% placeholder** replaced by the measured pooled median of
  ~4% across three runs, with the per-run breakdown archived under
  `measurements/`.
- **"First formal model"** softened to "to our knowledge, the
  first formal model".
- **Anonymity artifacts** removed (`Anonymous version for review`
  comment stripped; camera-ready authors, affiliations, funding,
  and GitHub URL kept).

## Reproducing the overhead measurement

```bash
python scripts/measure_dp_overhead.py \
    --config configs/overhead.yaml \
    --data /path/to/overhead_bench \
    --warmup 5 --rounds 60 \
    --out runs/overhead_new.json
```

Reproducing the ε_T verification:

```bash
python -c "from opacus.accountants import PRVAccountant
acc = PRVAccountant()
for _ in range(100):
    acc.step(noise_multiplier=1.1, sample_rate=0.2)
print(acc.get_epsilon(delta=1e-5))"    # -> ~12.35
```
