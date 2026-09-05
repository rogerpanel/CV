# PROSPERO registration text (Paper A)

Register at https://www.crd.york.ac.uk/prospero/ **before** running the searches.

**Title.** Methodological quality and reporting completeness of machine learning models for mental
health prediction: a systematic review and PROBAST+AI / TRIPOD+AI appraisal.

**Review question.** How methodologically sound and completely reported are supervised machine
learning models developed or validated to predict, detect, classify or diagnose mental health
disorders, across populations, settings and data modalities?

**Searches.** PubMed, Scopus, Web of Science Core Collection, IEEE Xplore (strings in
`protocol.md`); PsycINFO if institutional access is obtained (declare otherwise); 1 Jan 2018 to
31 Dec 2026; English; backward and forward citation chasing of included studies and of the
overlapping reviews (Hasan 2026; Meehan 2022; Mol Psychiatry 2025 generalizability review).

**Condition or domain.** Depressive, anxiety, stress-related, psychotic, bipolar, eating and
substance use disorders, ADHD, general psychological distress.

**Participants.** Human participants of any age, any setting, any country.

**Index model.** Supervised ML/DL prediction models with at least one quantitative performance
metric. Any data source (EHR, survey, wearable/actigraphy, imaging, speech, social media text).

**Primary outcomes.** Risk of bias by PROBAST+AI domain; TRIPOD+AI adherence per item.

**Secondary outcomes.** Proportion with any external validation (temporal, geographical,
independent cohort); proportion reporting calibration (plot, slope/intercept, Brier, ECE);
proportion reporting AUROC vs accuracy only; proportion reporting class prevalence; proportion with
a leakage-prone evaluation design (record-wise splitting of repeated measures, resampling or
feature selection before splitting, target-derived predictors); events-per-variable relative to
current guidance; code and data availability.

**Data extraction.** Two reviewers independently using `P1_extraction_appraisal.xlsx` after a
five-study pilot; disagreements by discussion or a third reviewer; Cohen's kappa reported.

**Risk of bias.** PROBAST+AI, two reviewers independently.

**Synthesis.** Descriptive-quantitative synthesis of appraisal outcomes with 95 % CIs for
proportions; stratified by modality, disorder and year. Meta-analysis of AUROC only for clinically
homogeneous subsets reporting variance; otherwise reasons stated. Studies reporting >= 99 %
accuracy without external validation examined as leakage case studies.

**Companion study.** The empirical companion (Paper B; repository `mlmh-appraisal`) quantifies
the cost of the three most prevalent deficiencies on public actigraphy cohorts and is cross-cited.
