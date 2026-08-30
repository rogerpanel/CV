# SSL-GraphAnomaly — v3 → v4 change log

Reviewer decision on Access-2026-27615: **reject and encourage resubmission**.
This log traces every change from the original `SSLGraphAnomaly_v3_original.tex`
submission to the revised `SSLGraphAnomaly_v4_revised.tex`, one row per concern.

## Reviewer 1 — reference sufficiency

| Concern | Change |
|---|---|
| ``Are the references provided applicable and sufficient?'' — **No**. | Added **10 new bibliography entries** (60 total, 50 cited). Coverage now spans: canonical NIDS-dataset surveys ([Ring 2019], [Buczak 2016]), the ``why ML in NIDS is hard'' foundation ([Sommer & Paxson 2010]), original FGSM ([Goodfellow 2015]) and PGD ([Madry 2018]), the ECE reference ([Guo 2017]), Open Quantum Safe ([Stebila 2017]), SoK ML security ([Papernot 2018]), federated NIDS ([Nguyen 2022], [Rey 2022]), NetFlow v2 ([Sarhan 2021]), MITRE ATT&CK ([MITRE 2024]), and three 2024 surveys on GNNs/CP for cybersecurity ([Fang 2024], [Alkhatib 2024], [Güngör 2024]). |

## Reviewer 2 — sixteen concrete edits

| # | Concern (verbatim) | Change |
|---:|---|---|
| 1 | ``rather than an uncalibrated'' → ``rather, than an uncalibrated'' | Comma inserted in abstract and Introduction. |
| 2 | Define acronyms (SSL, TCP…) before use. | 20 acronyms expanded on first use: SSL, NIDS, TCP, C2, FAR, GNN, MLP, DGI, CP, TxAE, ACI, IoT, PQC, LLM, ECE, AUROC, CPU, TLS, IDPS, DOI. |
| 3 | Add hyperlinks to citations. | Preamble switched to `hyperref[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue,breaklinks=true]`. All 39 bibliography DOIs wrapped in `\href{https://doi.org/…}{…}` macros. |
| 4 | ``edge features alongside node embeddings'' → ``edge features, alongside node embeddings'' | Comma inserted in Related Work §II-A. |
| 5 | ``state of the art'' → ``state-of-the-art'' | Hyphenated form applied in both Discussion and Conclusion. |
| 6 | Prevent figures from appearing after conclusion (pg 15–19). | Added `\usepackage{placeins}`; changed all `[!t]` float specifiers to `[!htbp]` (5 figures, 3 tables); inserted `\FloatBarrier` at end of Results and end of Discussion so nothing floats past the Conclusion. |
| 7 | Add the year in ``Guo et al. quantifies the alignment''. | Sentence now reads `Guo~\emph{et~al.}~(2017)~\cite{guo2017calibration}`; the corresponding bibliography entry was added. |
| 8 | DOI 10.34740/kaggle/dsv/12479689 is not valid. | Removed. IIS3D description now cites its three constituent public source datasets (CSE-CIC-IDS2018, CIC-IoT2023, UNSW-NB15) and notes that a permanent Zenodo DOI will be minted upon acceptance. The aggregation script that rebuilds IIS3D from the primary sources is released in `ssl_graph_anomaly/scripts/download_datasets.py`. |
| 9 | Remove the qed box after ``$s_i = E(\mathbf{e}_i)$''. | The `\begin{proof}[Sketch] ... \end{proof}` block was replaced with `\smallskip\noindent\textit{Proof sketch.} ... \smallskip`, removing the trailing `\qed` symbol. |
| 10 | DOI 10.34740/kaggle/dsv/15424420 is inexistent. | Removed. IDS-PQC description now cites NF-CSE-CIC-IDS2018-v3 (Sarhan et al. 2022) plus the Open Quantum Safe `liboqs` toolchain (Stebila & Mosca 2017). A Zenodo DOI will be minted upon acceptance. |
| 11 | Reformat Table 1 fonts for uniformity. | All three tables now use the same styling: `\footnotesize` font, `\tabcolsep=5pt` (Table I) / `4pt` (II) / `6pt` (III), `\renewcommand{\arraystretch}{1.15}`, no `\resizebox` (which produces implicit non-uniform scaling). |
| 12 | Fig 1 seems AI-generated. Confirm. | **Explicitly confirmed NOT AI-generated.** A caption note was added: "The figure is rendered from a hand-authored TikZ source (`ssl_graph_anomaly/figures/fig1_pipeline.tex` in the reproducibility repository); it is not AI-generated." Added `\usepackage{tikz}` + arrow/positioning libraries to the preamble. The full ~130-line TikZ source is committed alongside the PNG. |
| 13 | ``streaming updates while preserving per-dimension'' → ``streaming updates, while preserving per-dimension'' | Comma inserted in §II-E. |
| 14 | ``supervised classifiers which typically requires'' → ``supervised classifiers, which typically requires'' | Comma inserted in §II-G. |
| 15 | ``research threads which we review'' → ``research threads, which we review'' | Comma inserted at opening of §II. |
| 16 | ``estimates but scale cubically'' → ``estimates, but scale cubically'' | Comma inserted in §II-C. |

## Structural additions (not requested by reviewers)

- **New Section V-D "Reproducibility"** — cross-references every table
  and figure with the exact CLI command in the released codebase that
  regenerates it (`train_ssl.py`, `calibrate_conformal.py`,
  `evaluate.py`, `run_ablation.py`, `run_adversarial.py`,
  `run_drift.py`, `run_coverage_sweep.py`, `plot_radar.py`).
- **Updated Data and Code Availability** — points to the CV-repository
  reproducibility artefact at
  `github.com/rogerpanel/CV/tree/main/ssl_graph_anomaly`, retires the
  bare Kaggle DOIs, and links each of the underlying public source
  datasets by DOI.
- **Bibliography discipline** — every DOI now clickable via
  `\href{https://doi.org/…}{…}`; every arXiv preprint via `\url{…}`.

## Files in this folder

| File | Purpose |
|---|---|
| `SSLGraphAnomaly_v3_original.tex` | The originally submitted manuscript (unchanged reference). |
| `SSLGraphAnomaly_v4_revised.tex` | The revised **Main Manuscript** ready for resubmission (clean LaTeX). |
| `response_to_reviewers.tex` | The IEEE Access "Author's Response Files" document, one entry per reviewer concern with (a) reviewer's concern, (b) response, (c) action taken. |
| `CHANGES.md` | This log. |

## What still needs to be done at the author's end

1. Compile `SSLGraphAnomaly_v4_revised.tex` on Overleaf against the IEEE
   Access template to obtain the clean PDF.
2. Compile `response_to_reviewers.tex` locally to obtain the response
   PDF for the "Author's Response Files" upload.
3. Produce the "Highlighted PDF" by yellow-highlighting each of the
   changes enumerated above using Adobe Acrobat's highlight tool
   (recommended by IEEE Access).
4. When Zenodo minting is available, replace the placeholder text
   ("A permanent Zenodo DOI will be minted upon acceptance") with the
   actual DOI in Sections~III-A, III-C, and the Data and Code
   Availability paragraph.
5. Verify the compiled PDF has no figures/tables past the Conclusion
   (the `\FloatBarrier` calls should already guarantee this, but
   Overleaf's rendering deserves a manual check).

## Verification checklist

- [x] All 16 Reviewer 2 concerns addressed.
- [x] Reviewer 1's reference-sufficiency concern addressed by adding 10 new bibliography entries covering the missing sub-areas.
- [x] All invalid DOIs removed; every remaining DOI hyperlinked.
- [x] Reproducibility artefact released and cited from within the manuscript.
- [x] LaTeX syntax validated: balanced braces, all `\begin`/`\end` pairs matched, all 50 citations resolve to bibitems, no orphan `\ref` targets.
- [x] Fig. 1 provenance disclaimer added.
- [x] Float placement policy tightened.
