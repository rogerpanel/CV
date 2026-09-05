# Gap review: what is already known, what the benchmarks actually show, and what is genuinely open

*Compiled 2026-09-05 from web-search evidence (the sandbox cannot open full texts; every quantitative
claim below carries its source and should be checked against the paper before it is cited in a
manuscript). Items marked **[verify]** could not be confirmed beyond an abstract or a secondary
citation.*

## 1. Existing reviews the proposal must position against

| Review | What it did | What it did *not* do (the gap) |
|---|---|---|
| Hasan MJ et al. *Front Digit Health* 2026 (doi 10.3389/fdgth.2025.1724348) | Descriptive review of ML methods for mental-health state detection; Springer, ScienceDirect, IEEE, PubMed; Jan 2015-Dec 2024; 3,320 records screened; taxonomy of algorithms and datasets | No risk-of-bias or reporting appraisal; no PROBAST/TRIPOD; no outcome on validation design, calibration or leakage |
| Meehan AJ et al. *Mol Psychiatry* 2022 (doi 10.1038/s41380-022-01528-4) | 308 clinical prediction models in psychiatry, two decades; only 22.1 % (68/308) tested calibration, of which 52.9 % showed a plot and 22.1 % used only Hosmer-Lemeshow; 75.6 % used some internal validation but only 11.4 % of all models did so to an adequate standard | Focus on *clinical prediction models* (largely regression on cohort/EHR data); pre-dates PROBAST+AI and TRIPOD+AI; digital-phenotyping, sensor, speech and social-media ML is largely outside its net; no leakage-design outcome; no EPV >= 200 criterion. **This is the closest prior work and must be cited in the first paragraph of Paper A.** |
| "Generalizability of clinical prediction models in mental health", *Mol Psychiatry* 2025 (doi 10.1038/s41380-025-02950-0) | Review of external validation / transportability of mental-health prediction models | Not an ML-specific appraisal; no TRIPOD+AI adherence; confirms external validation is rare, which is Paper A outcome 1 [verify exact proportions] |
| Madububambachu et al. *Clin Pract Epidemiol Ment Health* 2024 | 30 studies of ML to predict mental-health diagnoses | Small; descriptive |
| ML for mental-illness detection on social media: SR of biases (arXiv 2410.16204) | Applied PROBAST to social-media ML studies | Single modality |
| ML models predicting adolescent mental-health crises from EHR (2025; PMC12426581) | PROBAST on 5 studies | Tiny, single setting |
| Prediction models in first-episode psychosis / at-risk mental state (BJPsych; PMC11480010) | PROBAST; only 2/48 psychosis models at low risk of bias | Single disorder |
| Individualized prediction models in ADHD, *Mol Psychiatry* 2024 (doi 10.1038/s41380-024-02606-5) | SR + meta-regression with PROBAST | Single disorder |
| Responsible Evaluation of AI for Mental Health, ACL 2026 (arXiv 2602.00065) | 135 NLP papers; over-reliance on generic metrics; little clinician involvement | NLP only; not a PROBAST/TRIPOD appraisal |

**Net.** No review applies PROBAST+AI (BMJ 2025) and TRIPOD+AI (BMJ 2024) across all modalities and
disorders, none quantifies leakage-prone evaluation design as an outcome, and none pairs the
appraisal with an empirical measurement of the cost. That combination is the novelty; the
descriptive question ("which algorithms, how accurate") is fully occupied.

## 2. Instrument-level facts to get right (they are currently wrong or ambiguous in the package)

* **PROBAST+AI** (Moons KGM, Damen JAA, Collins GS, et al. BMJ 2025; PMID 40127903) supersedes
  PROBAST 2019 for AI/ML models. The workbook's PROBAST sheet reproduces the 2019 signalling
  questions. Either appraise with PROBAST+AI (recommended, and a novelty claim) or state explicitly
  why 2019 PROBAST is used. PROBAST+AI adds explicit questions on data leakage, model
  transportability and, in the quality domain, on sample size for ML **[verify the exact wording and
  the EPV/sample-size guidance in the BMJ paper before quoting the >= 200 figure; the ">= 200 for ML"
  rule of thumb is cited in the brief without a page reference]**.
* **TRIPOD+AI** (Collins GS et al. BMJ 2024;385:e078378) has 27 items with 52 sub-items. The
  workbook's TRIPOD_AI sheet uses a **custom 27-line numbering that does not match the official
  checklist** (e.g. official item 4 is "Source of data", 6 is "Outcome", 8 is "Sample size",
  10 is "Data preparation", 12 is "Analytical methods", 16 is "Class imbalance", 17 is "Fairness",
  18 is "Model output", 19 is "Training vs evaluation", 20 is "Ethical approval", 21 is "Open
  science", 22 is "Patient & public involvement", 23 is "Participants", 24 is "Model development",
  25 is "Model specification", 26 is "Model performance", 27 is "Model updating") **[verify against
  the published checklist]**. Reviewers check adherence percentages item by item; the sheet must be
  re-keyed to the official numbering before extraction starts, or every adherence figure will be
  challenged.
* The PRISMA figure in the earlier draft does not reconcile (14 vs 15). Regenerate from the search
  log once counts exist (`paper/review/search_log.csv`).

## 3. Dataset-level benchmarks: what "state of the art" on these cohorts actually is

### DEPRESJON (23 depressed, 32 controls; 693 days)

| Study | Unit / split | Best reported | Calibration | External validation | Note |
|---|---|---|---|---|---|
| Garcia-Ceja et al. 2018 (dataset paper) | per-day features; leave-one-patient-out; RF vs DNN; SMOTE/oversampling compared | F1 0.73, MCC 0.44 | none | none | The honest baseline; everything higher needs scrutiny of the split unit |
| Frogner et al. 2019, 1D-CNN (ACM MMHealth) | per-day segments; leave-one-participant-out reported | [verify numbers] | none | none | — |
| 2D-CNN on time series, 2022 (PMC9495338) | per-window images | accuracy 76.7 % | none | none | — |
| Transfer learning screening tool, arXiv 2303.07847 | "modified leave-one-out" | accuracy 0.96 | none | claims an independent set | Split modification is exactly the kind of design the review flags |
| Explainable XGBoost, *JMIR Ment Health* 2025;e72038 | statistical + demographic features | accuracy 84.9 % (binary), 85.9 % (multiclass severity) | none | none | Split unit not stated in abstract **[verify: day-wise k-fold would be record-wise leakage]** |
| Vision-transformer / CoAtNet on actigraphy images, arXiv 2512.00103 | three-fold subject-wise | CoAtNet-Tiny best | none | none | One of few that states subject-wise |

### PSYKOSE (22 schizophrenia, 32 controls **shared with DEPRESJON**)

| Study | Unit / split | Best reported | Note |
|---|---|---|---|
| Jakobsen et al. 2020 (dataset paper) | leave-one-patient-out, classical ML | [verify F1/MCC] | baseline |
| Multi-branch DL, *Soft Computing* 2024; DL + XAI, *Biomed Signal Process Control* 2024 | day-level | accuracy ~0.94 (depression/schizophrenia vs control) | Day-level splits imply record-wise leakage unless stated otherwise **[verify]** |
| Night-time signal classification, 2022 (PMC9318635) | night windows | — | Uses both DEPRESJON and PSYKOSE: check whether the shared controls were double-counted |

### HYPERAKTIV (51 ADHD + 52 clinical controls; activity for 85)

| Study | Split | Best reported | Note |
|---|---|---|---|
| Hicks et al. 2021 (dataset paper) | 10-fold on features | modest (table in paper) | Baseline code on GitHub simula/hyperaktiv |
| OBF-Psychiatric ADHD case study, medRxiv 2025.08.26.25332257 | — | — | Addresses label/data uncertainty |

### OBF-Psychiatric (162 participants, 1,565 days; Sci Data 2025)

Ships a feature file and multi-class baselines; the paper's technical validation uses standard
classifiers. It is the natural canonical source for E1-E2 and it resolves the duplicated-control
problem by listing each participant once.

### DAIC-WOZ / E-DAIC (secondary, EULA)

The 2025-2026 audit literature is now strong enough that DAIC should be a *secondary* cohort:
"Most DAIC-WOZ depression classifiers are invalid, they don't learn task-specific features"
(ICMI 2025 companion, doi 10.1145/3747327.3763034) and "A multi-probe audit of clinical-interview
depression detection benchmarks" (arXiv 2605.23977: across 96 configurations the best
cross-validation model ranks 20th on the official test split, the official-test winner ranks 41st
by CV, top-3 overlap is zero; text models rise on symptom-dense slices while audio stays flat). Add
"DAIC-WOZ: on the validity of using the therapist's prompts" (arXiv 2404.14463). These are ready-made
"leakage case studies" for Paper A and a reason not to build Paper B's main claim on DAIC.

## 4. The gaps, stated as testable claims

1. **Unit-of-analysis leakage is common and its size on these cohorts is unmeasured.** Every study
   above that reports > 0.9 accuracy on DEPRESJON/PSYKOSE does so with day- or window-level splits
   or without stating the unit; the dataset authors' own subject-wise baseline is F1 0.73. No paper
   reports the paired record-wise vs subject-wise difference on the same data. (E1.)
2. **Cross-cohort external validation of actigraphy models has not been reported, and the obvious
   pair is contaminated.** DEPRESJON -> PSYKOSE shares 32 controls; a naive transfer leaks them.
   The codebase detects this from the series themselves and splits the controls. (E2; the
   shared-control assignment table is itself a reportable finding.)
3. **Calibration is essentially never reported for sensor-based mental-health models** (Meehan:
   22 % across psychiatric prediction models generally; the actigraphy studies above: none). No
   reliability curve for any DEPRESJON/PSYKOSE model exists in the literature. (E3.)
4. **Sample size is far below any EPV guidance, and window-level counts hide it.** With 23 cases
   and ~30 engineered features the subject-level EPV on DEPRESJON is < 1; window-level n
   (hundreds of days) makes it look 10-20x larger. Reporting both, with subject-level bootstrap
   intervals, is itself a corrective. (E2 table `epv` column.)
5. **Reproducibility.** The dataset papers release code; most follow-ups do not; none release run
   manifests. Every run here writes git SHA, config hash, seeds, versions and data checksums.
6. **Multiclass and transdiagnostic framing is under-examined.** OBF-Psychiatric enables
   "any psychiatric vs control" and 5-class tasks with a single, de-duplicated control group;
   the E2 pair DEPRESJON -> HYPERAKTIV is a deliberate label-shift test (ADHD vs *clinical* controls)
   that should fail, and saying so is informative.

## 5. Corrections to the brief's own dataset table

* HYPERAKTIV is "ADHD patients **and clinical controls** (51 + 52), activity available for 85,
  HRV for 80", not "ADHD patients" only; its controls are psychiatric patients without ADHD, so it
  is not a healthy-control cohort.
* PSYKOSE's control group is DEPRESJON's control group (32 people, identical files).
* OBF-Psychiatric is on Zenodo (record 13754984), not only at the Nature URL.
* DAIC-WOZ: 189 sessions in the full DAIC; the WOZ subset used in AVEC has 142 participants with
  PHQ-8 labels in the official train/dev/test split **[verify the exact counts from the EULA
  documentation]**.

## 6. Search-string notes for Paper A (from the protocol)

The four strings in `paper/review/protocol.md` are usable as written. Two additions are needed for
the reframed question: (i) terms that catch validation-methodology papers that are not themselves
prediction studies should be *excluded* at screening, not by the string; (ii) add
`"actigraph*" OR "wearable*" OR "smartphone*" OR "speech" OR "social media"` to the ML block in
Scopus/WoS only if the yield without them is implausibly low for sensor studies; pre-register any
change.
