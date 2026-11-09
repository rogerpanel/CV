# Overleaf project: MambaGuard: What LLM Agent-Protocol Intrusion Detectors Actually Learn

## Upload

Overleaf → **New Project → Upload Project** → choose `paper-overleaf.zip`.

Settings (Menu, top left):
- **Compiler:** pdfLaTeX
- **TeX Live version:** 2023 or newer
- **Main document:** `MambaGuard_sp2027.tex`

The first compile takes about a minute, because all six figures are drawn
with TikZ/pgfplots. Later compiles are faster.

## Files

| File | What it is |
|---|---|
| `MambaGuard_sp2027.tex` | The paper (IEEEtran compsoc conference, anonymised). The bibliography is inline (`thebibliography`), so no `.bib` file is needed. |
| `figures/fig-preamble.tex` | TikZ/pgfplots set-up and the chart colours, shared by the paper and the standalone figures. |
| `figures/fig1-framework.tex` … `fig6-certified-accuracy.tex` | Drawing code of each figure. The paper `\input`s them, so edit a figure here. |
| `figures/standalone.tex` | Compiles one figure on its own (instructions at the top of the file). |
| `figures/pdf/*.pdf`, `*.png` | Ready-made figures (vector PDF, and 300-dpi PNG) for slides, posters or a rebuttal. They are not used by the paper. |

| Figure | File | Width |
|---|---|---|
| 1 Measurement framework | `fig1-framework` | two columns |
| 2 Payload-encoder ablation | `fig2-encoder-ablation` | one column |
| 3 Template audit | `fig3-template-audit` | two columns |
| 4 AUROC radar and generalisation gap | `fig4-radar-generalisation` | two columns |
| 5 False alarms and calibration error | `fig5-alarms-calibration` | two columns |
| 6 Certified accuracy | `fig6-certified-accuracy` | one column |

## Faster drafts

To compile quickly while editing text, swap a figure's `\input{figures/...}`
line for `\includegraphics[width=\textwidth]{figures/pdf/<name>.pdf}`, using
`\columnwidth` for figures 2 and 6. Switch back before submitting, so the
figure and the text share fonts exactly.

## Page budget (IEEE S&P 2027)

The paper is 14 pages: the main text ends on page 11, followed by
references and appendices. The limit is 13 pages of main text and 18 pages
in total.
