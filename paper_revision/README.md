# Paper Revision — IEEE Access Access-2026-27615

**Manuscript:** *Self-Supervised Graph Neural Networks for Network Intrusion
Detection with Conformal Safety Certification*

**Authors:** R. N. Anaedevha, A. G. Trofimov, Y. V. Borodachev
(National Research Nuclear University MEPhI, Moscow, Russia)

**Status:** Revised after first-round peer review (Rejected with encouragement
to resubmit, 11 Aug 2026).

## Files in this folder

| File | Purpose |
|---|---|
| `SSLGraphAnomaly_v3_original.tex` | Verbatim copy of the originally submitted manuscript. Kept for diffing only — do not resubmit this. |
| `SSLGraphAnomaly_v4_revised.tex` | **Main Manuscript** ready for resubmission. All 16 Reviewer 2 concerns and Reviewer 1's reference-sufficiency concern addressed. Every change is called out in `CHANGES.md`. |
| `response_to_reviewers.tex` | The "Author's Response Files" document required by IEEE Access at resubmission. One entry per reviewer concern with (a) concern verbatim, (b) response, (c) action taken. |
| `CHANGES.md` | Change log mapping every original-vs-revised delta to the reviewer concern that motivated it. Read this first when cross-checking on Overleaf. |
| `README.md` | This file. |

## Overleaf workflow

1. Create a new Overleaf project from the IEEE Access template
   ([direct link](https://www.overleaf.com/gallery/tagged/ieee-official)).
2. Delete the template's `main.tex` and upload `SSLGraphAnomaly_v4_revised.tex`
   in its place (rename to whatever the template's main-doc name is).
3. Upload `figures/fig1_sslGraphAnomaly_arch.png`,
   `figures/p2fig2_radar.pdf`, `figures/p2fig3_advrob.pdf`,
   `figures/p2fig4_drift.pdf`, and `figures/p2fig5_coverage.pdf` from
   `../ssl_graph_anomaly/figures/` (or your existing figures folder).
4. Upload `author1.jpeg`, `author2.jpeg`, `author3.png` if biographies
   should render photographs.
5. Compile with `pdflatex` and inspect: (a) that Figures 1–5 and
   Tables I–III do not float past the Conclusion, (b) that every
   citation resolves to a blue clickable link, and (c) that Fig. 1's
   caption ends with the "not AI-generated" disclaimer.

## Response-to-reviewers PDF

`response_to_reviewers.tex` is a standalone LaTeX document and can be
compiled locally with `pdflatex response_to_reviewers.tex` (three passes
to resolve refs). The resulting PDF is what IEEE Access expects to be
uploaded under *Author's Response Files*.

## Highlighted PDF

IEEE Access asks for a highlighted PDF where every change is highlighted
yellow. The recommended procedure:

1. Compile `SSLGraphAnomaly_v4_revised.tex` to `SSLGraphAnomaly_v4_revised.pdf`.
2. Open the PDF in Adobe Acrobat Pro.
3. Use `CHANGES.md` as your checklist: for each row, locate the sentence
   in the compiled PDF and apply the yellow highlight tool.
4. Save as `SSLGraphAnomaly_v4_highlighted.pdf` and upload under
   *Highlighted PDF*.

Alternatively, the `latexdiff` tool can auto-produce a colour-coded diff:
```bash
latexdiff SSLGraphAnomaly_v3_original.tex \
          SSLGraphAnomaly_v4_revised.tex > SSLGraphAnomaly_v4_diff.tex
pdflatex SSLGraphAnomaly_v4_diff.tex
```

## Reproducibility artefact

Every result quoted in the manuscript is reproducible end-to-end from the
codebase at
[`github.com/rogerpanel/CV/tree/main/ssl_graph_anomaly`](https://github.com/rogerpanel/CV/tree/main/ssl_graph_anomaly).
Section V-D of the revised manuscript ("Reproducibility") itemises the
exact command that regenerates each table and figure.

## Contact

Roger Nick Anaedevha, `ar006@campus.mephi.ru`
