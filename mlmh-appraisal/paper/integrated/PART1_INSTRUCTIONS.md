# Part I: what you must do yourself, step by step

Everything below is human work (searching licensed databases, dual screening, full-text
extraction). The plumbing is done: the study list is a CSV, and the table and figure in the
manuscript regenerate from it with one command, so you never edit LaTeX or plot by hand.

## 0. Before you start (one hour)
1. Register the closed-scope review on PROSPERO (https://www.crd.york.ac.uk/prospero/) or, if
   PROSPERO declines it as non-health-outcome, on OSF Registries. Paste the text from
   `paper/review/prospero_registration.md`, replacing the field-wide scope with: "all studies
   developing or validating a supervised model on the DEPRESJON, PSYKOSE, HYPERAKTIV or
   OBF-Psychiatric datasets". Put the ID into `manuscript_AB.tex` where it says `[PROSPERO/OSF ID]`.
2. Open `paper/review/P1_extraction_appraisal.xlsx` and re-key the TRIPOD_AI sheet to the official
   27 items (download the checklist from https://www.tripod-statement.org/). Keep one row per
   study ID from `part1_studies.csv`.

## 1. Run the searches (half a day)
Run each of these, record the date and count in `paper/review/search_log.csv`, and export RIS:
- Scopus: `TITLE-ABS-KEY("Depresjon" OR "Psykose" OR "Hyperaktiv" OR "OBF-Psychiatric")` plus
  "Cited by" on the four dataset papers (DOIs 10.1145/3204949.3208125, 10.1109/CBMS49503.2020.00064,
  10.1145/3458305.3478454, 10.1038/s41597-025-04384-3).
- Web of Science: `TS=("Depresjon" OR "Psykose" OR "Hyperaktiv" OR "OBF-Psychiatric")` plus
  "Citing articles" on the same four DOIs.
- PubMed: `Depresjon[tiab] OR Psykose[tiab] OR Hyperaktiv[tiab] OR "OBF-Psychiatric"[tiab]`.
- IEEE Xplore and ACM Digital Library: full-text search for the four names.
- Google Scholar: "Cited by" on the four dataset papers (export with Publish or Perish).
Expect 150 to 400 records before de-duplication.

## 2. Screen in Rayyan (two reviewers, one to two days)
Import all RIS files, de-duplicate, blind on. Include if: supervised model trained or evaluated on
one of the four datasets' activity data, reports a performance metric, 2018 or later, English.
Record kappa from Rayyan's "Agreement" panel.

## 3. Extract into the CSV (the only file that feeds the paper)
For every included study add one row to `paper/integrated/part1_studies.csv` with these columns:
`id, first_author, year, venue, cohorts, task, best_model, split_unit, best_metric_name, best_metric,
calibration, external_validation, shared_controls_flag, code, confidence, source`.
Rules: `split_unit` must start with `participant`, `day` or `unstated`; `best_metric_name` is
`accuracy`, `AUROC`, `F1` or `MCC` (accuracy rows are plotted); `calibration` and
`external_validation` are `yes`/`no`; `shared_controls_flag` is `n/a`, `recognised` or
`not recognised` (only for studies using both DEPRESJON and PSYKOSE); set `confidence` to `high`
once verified against the full text and remove the word `verify` from every cell (that is what
turns a red cell black).
The 28 seed rows are already there; verify each against its full text and correct it.

## 4. Appraise (PROBAST+AI and TRIPOD+AI, two reviewers)
Fill the PROBAST and TRIPOD_AI sheets of the workbook for every study ID. In the Reporting_Outcomes
sheet, add the EPV column: minority-class participants divided by the number of candidate
predictors stated in the paper.

## 5. Regenerate the table and figure (one command)
```bash
cd mlmh-appraisal
python scripts/part1_figure.py        # rewrites paper/integrated/part1_studies.tex and figures/part1_claims_vs_bands.pdf
python scripts/part2_extra_figures.py # rewrites tables/sota_comparison.tex from the same CSV
```
The script prints the counts (split units, calibration, external validation, code). Paste them
into the three red sentences in Section 2.2/2.3 of `manuscript_AB.tex`, add the PROBAST+AI domain
proportions and TRIPOD+AI per-item adherence from the workbook (Wilson 95% CIs: `statsmodels`
`proportion_confint(k, n, method="wilson")`), and delete the red brackets.

## 6. PRISMA flow
Put the counts into `paper/review/prisma_counts.tex` (the macros do the arithmetic) and include
the flow diagram as Supplementary Figure S1.

That is all of Part I. Nothing else in the manuscript depends on it; Part II, the SOTA table and
all figures are already final and regenerate from the results directory.
