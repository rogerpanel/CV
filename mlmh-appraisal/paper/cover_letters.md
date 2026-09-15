# Cover letter drafts

## Integrated paper — Artificial Intelligence in Medicine (Elsevier; subscription route, no APC)

Dear Editor,

We submit "Machine Learning for Mental Disorder Detection from Actigraphy: Published Claims versus Subject-wise, Cross-cohort and Calibrated Evaluation" for consideration as a Research Paper.

The three Simula actigraphy cohorts are the most used open benchmarks for machine-learning detection of mental disorders from wearables, and reported accuracies on them have climbed from the dataset authors' leave-one-patient-out 0.7 to above 0.9. Our paper asks whether that is progress. Part I is a closed-scope systematic review of every published model on the cohorts, appraised with PROBAST+AI and TRIPOD+AI, with outcomes that can be checked against the data: unit of splitting, external validation, calibration, sample size, and double counting of the control group that DEPRESJON and PSYKOSE share. Part II re-evaluates the cohorts with fixed models, ten seeds and subject-level bootstrap intervals under the designs the review scores as high and low risk. Record-wise splitting inflates AUROC by up to 0.17 and turns a chance-level ADHD classifier into an apparently useful one; frozen cross-cohort transfer loses up to 0.21 AUROC in one direction with calibration collapse; calibration is poor even under honest validation and is never reported in the literature on these cohorts.

All analysis code, run manifests and generated tables are public. The TRIPOD+AI checklist is attached; the Part I appraisal is reported following PRISMA 2020 principles and its identification procedure and limitations are stated in the manuscript. All figures and tables were regenerated independently on Google Colab from the public repository with identical results. The manuscript is not under consideration elsewhere. We choose the subscription publication route.

Corresponding author: Dr Aminat Abiola Ajibola, College of Computer Science and Engineering, University of Hafr Al-Batin, Kingdom of Saudi Arabia (aajibola@uhb.edu.sa). Co-author: Roger Nick Anaedevha, Institute of Cyber Intelligence Systems, National Research Nuclear University MEPhI, Moscow (roger@robustidps.ai).

## Paper A — npj Digital Medicine (alternative: JMIR Mental Health)

Dear Editor,

We submit "Methodological quality and reporting completeness of machine learning models for
mental health prediction: a systematic review and PROBAST+AI / TRIPOD+AI appraisal" for
consideration as an Article.

Descriptive reviews of machine learning for mental health already exist; the most recent screened
more than 3,000 records and catalogued algorithms and datasets. None has asked whether the models
are trustworthy. We applied PROBAST+AI and TRIPOD+AI, in duplicate, across [N] studies spanning all
disorders and data modalities, and pre-specified six outcomes: external validation, calibration,
AUROC versus accuracy-only reporting, leakage-prone evaluation design, events per candidate
predictor, and code/data availability. [One sentence with the headline proportions.]

What distinguishes this review is a paired empirical companion (submitted to [journal]; preprint
DOI [..]) that measures, on three public actigraphy cohorts, what the most prevalent deficiencies
cost: record-wise splitting inflated AUROC by up to 0.17 and manufactured signal where there was
none, external validation exposed losses of up to 0.21 AUROC with calibration collapse, and no
published model on those cohorts reports calibration. The review measures how widespread the
problem is; the companion measures how much it matters.

The protocol was registered on PROSPERO ([CRD]) before searching. All extraction data and code are
public. The manuscript follows PRISMA 2020; the checklist is attached. It is not under
consideration elsewhere. Suggested reviewers: [names from the PROBAST/TRIPOD community and from
digital psychiatry].

## Paper B — JMIR AI (alternative: IEEE JBHI)

Dear Editor,

We submit "How much do leaky splits, missing external validation and missing calibration cost? An
empirical companion to a PROBAST+AI / TRIPOD+AI appraisal of machine learning for mental-health
prediction, on three public actigraphy cohorts" as an Original Paper.

JMIR AI recently published a comparative cross-validation study showing that stratified k-fold
overestimates performance on repeated-measures data through participant-level leakage. Our study
asks the same question in mental-health prediction, where day-level splitting of actigraphy is
routine, and extends it to the two other practices a companion systematic review found to be
widespread: absence of external validation and absence of calibration assessment.

On DEPRESJON, PSYKOSE and HYPERAKTIV (162 participants, same recording device) with fixed models
and ten seeds, record-wise splitting inflated AUROC by 0.02 to 0.17 with subject-level bootstrap
intervals excluding zero, and turned a chance-level ADHD classifier into an apparently useful one.
Freezing a model and applying it to another cohort lost up to 0.21 AUROC in one direction while
the other direction held, and calibration slopes fell to 0.3. We also report that the PSYKOSE and
DEPRESJON control groups are the same 32 people, a fact that contaminates any naive cross-cohort
validation and that we detect from the data rather than the documentation.

The study follows TRIPOD+AI as authors (checklist attached); every table is generated by public
code from manifests that record commit, seeds, package versions and data checksums. The companion
review is [under review at / preprinted as] [..]. The manuscript is not under consideration
elsewhere.
