# Venue targeting and alignment with the review's categorisation

*Prepared 2026-09-05. Journal metrics quoted here were taken from public indexing pages found by
web search on that date and must be re-verified in JCR / Scopus before submission; they change every
June.*

## 1. What each paper is, in the eyes of an editor

| | Paper A | Paper B |
|---|---|---|
| Article type | Systematic review with critical appraisal (PRISMA 2020; PROBAST+AI; TRIPOD+AI) | Prediction-model methods study / empirical demonstration (TRIPOD+AI as author) |
| Primary claim | The ML-for-mental-health literature is methodologically unsound and incompletely reported in specific, quantifiable ways | Those specific failures cost a measurable amount of discrimination and calibration on real, public data |
| Reader | Methodologists, editors, reviewers, clinical-AI developers | ML-for-health practitioners, digital-phenotyping researchers |
| Reporting checklist the venue will demand | PRISMA 2020 (+ PROSPERO ID) | TRIPOD+AI checklist as a supplement |

## 2. Candidate venues

Fit is judged on (a) whether the journal has published PROBAST/TRIPOD-style appraisals or
validation-methodology studies before, (b) whether the mental-health readership is reached,
(c) selectivity relative to the likely strength of the result, and (d) open-access cost.

### Paper A (review + appraisal)

| Rank | Venue | Why it fits | Risk / cost | Evidence of precedent |
|---|---|---|---|---|
| 1 | **npj Digital Medicine** (Nature Portfolio; IF ~18 per 2026 listing) | Publishes systematic reviews with PROBAST critical appraisal of ML models; digital-health methods critique is core scope; high visibility for a "state of the field" paper | Very selective; needs the full 80-200 study yield and PROBAST+AI (2025) applied, not legacy PROBAST; APC | "Machine learning in vascular surgery: a systematic review and critical appraisal" (npj Digit Med 2021) |
| 2 | **JMIR Mental Health** (IF ~7.4, Q1 Psychiatry) | Exact readership; publishes methodological reviews of digital mental-health AI; fast; accepts long supplements (extraction tables) | APC; less methodological prestige than 1 | Frequent ML-for-mental-health reviews; JMIR family publishes TRIPOD-adherence audits |
| 3 | **Journal of Clinical Epidemiology** (Elsevier; Q1) | Home journal of the PROBAST/TRIPOD community; the natural venue if the paper's centre of gravity is the appraisal instrument (PROBAST+AI first use at breadth, EPV >= 200 finding) | Readers are methodologists, not psychiatrists; mental-health framing becomes secondary | Andaur Navarro et al. (BMJ 2021) and Dhiman et al. (J Clin Epidemiol 2022) ML-appraisal reviews |
| 4 | **Molecular Psychiatry** / **Translational Psychiatry** | Meehan et al. 2022 (308 psychiatric prediction models; 22.1 % reported calibration) set the precedent there | Meehan is also the closest prior work; the paper must show what it adds (ML/digital modalities, PROBAST+AI, leakage outcome, EPV >= 200) | Meehan 2022; "Generalizability of clinical prediction models in mental health" (Mol Psychiatry 2025) |
| 5 | **Journal of Biomedical Informatics** (brief's primary; 2026 listing shows IF 5.9, **Q2**) | Methodological informatics; would accept; good for Paper B pairing | Lower quartile than the brief assumed; weaker mental-health readership | Regularly publishes ML validation methodology |
| 6 | International Journal of Medical Informatics (IF ~5.0, Q1) | Safe Q1 fallback | Less visible | — |

**Recommendation for A.** Target npj Digital Medicine if the search yields >= 100 included studies
and PROBAST+AI is used; otherwise JMIR Mental Health. Keep J Clin Epidemiol as the methodological
fallback. Do not send to JBI first: for a review, a Q2 informatics venue under-sells the work.

### Paper B (empirical companion)

| Rank | Venue | Why it fits | Risk / cost | Evidence of precedent |
|---|---|---|---|---|
| 1 | **JMIR AI** | Published "Participant-Aware Model Validation for Repeated-Measures Data: Comparative Cross-Validation Study" (2026), i.e. exactly the E1 question in another domain; welcomes calibration and validation-methodology papers | Young journal (indexing still maturing) | JMIR AI 2026;e87728 |
| 2 | **IEEE Journal of Biomedical and Health Informatics** (Q1) | Wearable/actigraphy ML readership; the Simula cohorts are well known there; methodological rigour valued | Long review times; needs strong engineering presentation | HYPERAKTIV / PSYKOSE follow-ups appear in IEEE venues |
| 3 | **Journal of Biomedical Informatics** | Methods-oriented; comfortable with "negative"/cautionary empirical results; pairs with A if A goes elsewhere | Q2 per 2026 listing | — |
| 4 | **npj Digital Medicine** (brief research / short communication) | If E1-E3 effect sizes are large and clean, a short paper here alongside A in the same journal is the highest-impact pairing | Selectivity; APC x2 | — |
| 5 | Computers in Biology and Medicine (IF ~8.4) / Artificial Intelligence in Medicine | Accept broad ML-in-medicine work; CBM's IF is high but its scope is engineering-general | Reviewer pool less attuned to PROBAST/TRIPOD framing | — |

**Recommendation for B.** JMIR AI first (fastest route to a venue that has already validated the
question), IEEE JBHI second. Submit B only after A is at least under review with a preprint DOI
(medRxiv for A, arXiv cs.LG or medRxiv for B) so the cross-citation is verifiable.

## 3. Alignment matrix: review outcome -> appraisal item -> experiment -> code -> test

This is the "categorisation and directions" the user asked to be honoured: every primary outcome
of Paper A maps onto one PROBAST(+AI) signalling question, one TRIPOD+AI item, one Paper B
experiment, one code module and one test that enforces it.

| Paper A primary outcome | PROBAST(+AI) | TRIPOD+AI item (workbook numbering) | Paper B experiment | Code | Enforced by |
|---|---|---|---|---|---|
| Any external validation performed | 4.8 (optimism), applicability | 10e, 24 | **E2** train-on-A / test-on-B, no refitting | `evaluation/external.py` | `assert_cohorts_disjoint` + `test_external_runner_refuses_overlapping_subjects` |
| Calibration reported | 4.7 (both calibration AND discrimination) | 12, 24 | **E3** slope, intercept, Brier, ECE, reliability curves for every model | `evaluation/metrics.py` | `test_metrics.py` (slope/intercept recover known miscalibration) |
| AUROC rather than accuracy alone | 4.7 | 12 | all: AUROC, AUPRC, macro-F1, MCC alongside accuracy | `evaluation/metrics.py` | `test_binary_metrics_keys_and_ranges` |
| Leakage-prone evaluation design (record-wise splitting of repeated measures) | 4.8; PROBAST+AI data-leakage question | 10d, 11 | **E1** record-wise vs subject-wise, paired difference | `evaluation/splitters.py` | `test_no_subject_leakage.py` (fails the build for any non-E1 config using `record_wise`) |
| EPV >= 200 expectation for ML | 4.1 | 8 | reported per cohort in the E2 table (`epv` column) | `experiments._epv` | — (reported, not enforced) |
| Class prevalence reported | 1.2 / 3.4 | 22 | prevalence in every metrics row (`prevalence`) | `metrics.binary_metrics` | — |
| Resampling inside CV folds | 4.8 | 11 | SMOTE optional, imblearn Pipeline only | `models/pipelines.py` | `test_pipeline_fit_order.py` |
| Code / data availability | — | 19, 20 | this repository; `data/README.md`; manifests with git SHA and checksums | `reporting/manifest.py` | — |
| Near-perfect performance as leakage case studies | 4.8 | 24 | E1 inflation size is the empirical counterpart | `experiments.run_e1` | paired subject bootstrap |
| Uncertainty intervals | 4.8 | 24 | subject-level BCa bootstrap | `evaluation/bootstrap.py` | `test_subject_bootstrap_resamples_subjects_not_windows` |

Two directions in the workbook are **not** covered by the code and must be handled in the manuscript
text: TRIPOD+AI item 14/25 (fairness by subgroup; sex is available for DEPRESJON, PSYKOSE, HYPERAKTIV
and can be added as a stratified analysis) and item 18 (pre-registration; an OSF registration of the
E1-E3 analysis plan before the real data is run is strongly recommended and cheap).

## 4. Consequences for the manuscripts' structure

* Paper A follows PRISMA 2020 section order; the appraisal results are organised **by outcome**
  (the six primary outcomes above), not by study, and each outcome section ends with one sentence
  pointing to the Paper B experiment that quantifies its cost.
* Paper B follows TRIPOD+AI section order; Methods are organised **by experiment** (E1-E3) and each
  experiment opens with the Paper A proportion it responds to.
* Both papers share one figure concept: "prevalence of the practice in the literature (A)" beside
  "cost of the practice on data (B)".
