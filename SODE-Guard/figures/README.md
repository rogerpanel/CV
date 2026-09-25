# Figure sources

Each file is a standalone LaTeX document (`pdflatex <name>.tex`).

| File | Content |
|---|---|
| `fig1_architecture.tex` | Model architecture; wraps `fig1_architecture_body.tex`, which the paper can also `\input` directly |
| `fig2_robustness.tex` | Macro-F1 vs. l∞ budget under PGD-40 (values recovered from the original vector figure; they match Table 1) |
| `fig3_guarantees.tex` | Schematic of the mean-predictor certificate and the path-wise flip bound (no data) |
| `fig4_certified_accuracy.tex` | Certified-accuracy template; reads the CSV written by `scripts/run_revision_experiments.py e4` |
