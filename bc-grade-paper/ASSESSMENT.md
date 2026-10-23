# Assessment of the v0.1 runbook, research and manuscript, and what changed in v1.0

Date: 18 October 2026. Scope: `runbook_claude_code_v0.1`, the `bc-grade-opendata` v0.1 package, the
v0.1 manuscript (PDF + Overleaf) and the merged P2+P3 note.

## 1. Runbook

The runbook is well built: each step has an action, the files it touches and an acceptance check, and
CLAUDE.md's rules (no hand-typed numbers, in-pipeline preprocessing, leakage columns quarantined,
one row per patient) are the right ones for this paper. Problems found:

| Item | Issue | Resolution in v1.0 |
|---|---|---|
| R2 | — | Passed: clean re-download (checksums unchanged) + re-run reproduces `numbers.tex` byte-for-byte. |
| R5 TabPFN | TabPFN weights are now gated (Hugging Face login + licence acceptance); the build sandbox cannot reach huggingface.co. | Code path added (`make_model("tabpfn")`, optional rung); status logged to `results/optional_models_status.json`. Needs a run on a machine with an HF token. |
| R6 Saudi anchors | Requires reading full-text tables; journal hosts were unreachable from the sandbox (search engine only). | 3 more Saudi series added with `verified: snippet` flags, random-effects pooled anchor, age-share sensitivity, bootstrap CIs. **A human must re-read all anchor tables (V2).** |
| R7 sample size | Acceptance "required n ≤ 2386, or the paper says it is not met". | Computed (Riley 2020 criteria per pairwise sub-model, Pate 2023): **not met**, driven by the grade 1 vs 2 sub-model. v0.1 Methods claimed the opposite; corrected and discussed. |
| R8–R9 imaging | Kaggle and TCIA image hosts unreachable from the sandbox. | Not run; stated as a limitation. |
| R10 Duke | Runbook cites a "published radiomics benchmark AUC 0.722" for Duke. The paper it points to (Wang 2022, *Breast Cancer: Targets and Therapy*) reports 0.722, but no record shows it used the Duke data. | Duke clinical + 529 radiomics obtained from a checksummed GitHub mirror of the TCIA XLSX files; Duke added as second full-feature external cohort, to LODO, to the leakage audit and as a radiomics-fusion experiment (E9). Wang 2022 cited only as single-centre radiomics work. |
| V1 | DOI resolvers blocked. | 79 entries checked against search-indexed PubMed/publisher records; 4 VERIFY items fixed (MBCproject has no peer-reviewed paper → medRxiv preprint cited; kim2025grade title/authors corrected; Kussaibi DOI found). Provenance kept in `paper/refs_annotated.bib`. |
| Missing from runbook | Paired model comparisons; multiclass calibration; CIs for the transport estimate; STROBE checklist; native-LaTeX figures. | First four added (STROBE checklist still to do); all charts now pgfplots from CSV. |

## 2. Research and manuscript (v0.1): substantive issues

1. **"Logistic regression = XGBoost" was asserted, not tested.** v1.0 adds paired-bootstrap Δ macro-AUC vs LR internally and in every external setting.
2. **The sample-size sentence was wrong** (see R7).
3. **The Saudi "over-prediction" conclusion rested on one development cohort.** With Duke added, the frozen METABRIC model *under*-predicts grade 3 in MSK-2018 and *over*-predicts it in Duke; the Saudi-weighted grade-3 share depends strongly on which Western cohort is re-weighted. What is robust: (a) Saudi case-mix shifts every cohort towards more grade 3; (b) a METABRIC-developed model would predict far more grade 3 than published Saudi series; (c) local intercept recalibration is mandatory. The Discussion now says exactly this.
4. **Undisclosed assumption**: the share of Saudi patients aged ≤50 (50%) was not reported by the source; it is now labelled an assumption with a 40–60% sensitivity analysis.
5. **Anchor provenance**: the Kussaibi grade split (8.7/54.5/36.7%) could not be re-confirmed; the medRxiv preprint (a different data cut, n = 291) gives 8.2/54.6/37.1%. Must be checked against Table 1 of the published paper.
6. **Methods/Results gaps**: no figure of the study design (now Figure 1); single imputation only (MI added, m = 20); one full-feature external cohort (now two); radiomics claim untested (now tested: no gain).

## 3. Inputs needed from the authors

1. Author list, affiliations, corresponding author, funding, competing interests, CRediT, PPI statement.
2. Target journal (recommendation in the cover note) — drives word limits and template.
3. PDFs (or a double-read) of the Saudi anchor papers: Kussaibi 2021 (Tables 1–2), Durrani 2025, Alamri 2024, Albasri 2014, and ideally Elkum 2025 / Almohanna 2025 / Aamir 2026 which may report grade with larger n.
4. For R5/R8–R9: a run on a machine with internet access to Hugging Face (TabPFN token), Kaggle and TCIA; or a network policy for this environment that allows those hosts.
5. GitHub repository name + Zenodo DOI for the Code availability statement.

## 4. Update v1.0.1 (23 October 2026)

- **Anchors verified** against the three PDFs supplied (see `anchors_verification.md`). Kussaibi 2021 is
  confirmed in full. Durrani 2025 counts are confirmed, but its printed ER/HER2 percentages do not match
  its counts, so the counts are used. Albasri 2014 now enters the pooled estimate with full grade counts
  (IDC only). Alamri 2024 could not be located and was removed.
- **Author** set to Dr Aminat Abiola Ajibola (University of Hafr Al-Batin), corresponding author.
- **Target journal:** *Computers in Biology and Medicine*. It is Elsevier, Q1 (JIF ≈ 6.3, 5-year 6.5, per
  third-party metrics; check JCR) and hybrid, so the subscription route has no APC. The manuscript now
  uses the `elsarticle` template, with:
  - an abstract ≤ 250 words;
  - 5 highlights;
  - a graphical abstract;
  - a separate title page;
  - CRediT, competing-interest, funding and generative-AI declarations.
  IEEE JBHI was rejected because of mandatory overlength page charges beyond 8 pages.
- **Double-anonymised build:** `manuscript_anonymised.tex` produces a line-numbered PDF with no
  identifying information. The repository export is identity-free; the author files are kept out of it.
- **Internal history removed:** text about the earlier MOH drafts was replaced by neutral "design
  principles", which is appropriate for a journal and preserves blinding.
- **TabPFN:** `make tabpfn` now adds it in one command once a Hugging Face token is available.
