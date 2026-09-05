# TRIPOD+AI self-audit (auto-generated)

Runs found: 4
- `E1` git=9bb891b9 dirty=True config=58955ed9b797a0a1 seeds=[0, 1, 2, 3, 4] synthetic=True
- `E1_cnn` git=ef78278a dirty=True config=727797a16647ccf8 seeds=[0, 1] synthetic=True
- `E2` git=23f3001a dirty=True config=3d86339d63d7390d seeds=[0, 1, 2, 3, 4] synthetic=True
- `E3` git=ef78278a dirty=True config=4593148d5bedc11c seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] synthetic=True

**WARNING: at least one run used the synthetic fixture. Nothing below is a reportable result.**

| Item | Requirement | Status | Where satisfied |
|---|---|---|---|
| 1 | Title identifies development/validation of a prediction model, target population, outcome | MANUAL: write in manuscript | Title |
| 2 | Abstract reports objective, data, population, outcome, predictors, sample size, model, performance | MANUAL: write in manuscript | Abstract |
| 3a | Background and rationale | MANUAL: write in manuscript | Introduction |
| 3b | Objectives incl. development/validation | MANUAL: write in manuscript | Introduction |
| 4a | Data source described | satisfied by code/manifest | Methods: data (cohort loaders, data/README.md) |
| 4b | Study dates | MANUAL: write in manuscript | Methods: data (dataset publication dates) |
| 5a | Setting, eligibility, recruitment | MANUAL: write in manuscript | Methods: data (source publications) |
| 5b | Treatments received | MANUAL: write in manuscript | Methods: data |
| 6 | Outcome definition and timing | satisfied by code/manifest | Methods: label = diagnostic group (subjects table) |
| 7 | Predictors defined incl. measurement | satisfied by code/manifest | Methods: features (src/mlmh/features/actigraphy.py) |
| 8 | Sample size justified, EPV stated | satisfied by code/manifest | Results: n_subjects / n_windows per cohort, EPV in tables |
| 9 | Missing data handling | satisfied by code/manifest | Median imputation inside pipeline; window validity threshold |
| 10a | Pre-processing | satisfied by code/manifest | Pipeline steps in manifest config |
| 10b | Model types and rationale | satisfied by code/manifest | Model registry |
| 10c | Hyperparameter tuning | satisfied by code/manifest | Fixed a priori; no tuning (registry docstring) |
| 10d | Internal validation (CV scheme, repeats, seeds) | satisfied by code/manifest | Splitter + seeds in manifest |
| 10e | External validation method | satisfied by code/manifest | E2 runner |
| 11 | Class imbalance handling; resampling inside folds | satisfied by code/manifest | resample setting in manifest; imblearn Pipeline |
| 12 | Performance measures incl. discrimination AND calibration | satisfied by code/manifest | metrics.py |
| 13 | Model updating / recalibration | MANUAL: write in manuscript | Discussion (none performed unless E3 recalibration arm run) |
| 14 | Fairness / subgroup performance | MANUAL: write in manuscript | Sex-stratified analysis where sex is available |
| 15 | Interpretability methods | MANUAL: write in manuscript | Not a study aim; state so |
| 16 | Funding | MANUAL: write in manuscript | Declarations |
| 17 | Conflicts of interest | MANUAL: write in manuscript | Declarations |
| 18 | Protocol / registration | MANUAL: write in manuscript | OSF pre-registration of E1-E3 recommended |
| 19 | Data availability | satisfied by code/manifest | data/README.md: sources, terms, checksums |
| 20 | Code availability | satisfied by code/manifest | This repository; git SHA in manifest |
| 21 | Participant flow | satisfied by code/manifest | Windows retained per subject (prepare step log) |
| 22 | Participant characteristics incl. prevalence | satisfied by code/manifest | Cohort table |
| 23 | Full model specification | satisfied by code/manifest | Pipelines + registry, seeds recorded |
| 24 | Performance with uncertainty intervals | satisfied by code/manifest | Subject-level BCa bootstrap CIs |
| 25 | Subgroup / fairness results | MANUAL: write in manuscript | Results |
| 26 | Limitations | MANUAL: write in manuscript | Discussion |
| 27 | Interpretation and implications | MANUAL: write in manuscript | Discussion |
