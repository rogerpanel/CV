# Paper 1 — Review protocol package

**Contents:** overlap check and its consequence · recommended research question · inclusion/exclusion criteria (PICOS) · four ready-to-paste search strings · PROSPERO registration content · expected workload

---

## 1. Overlap check — read this before anything else

I searched for existing reviews at the scope you selected (all populations, all mental health disorders). It is heavily occupied. Four directly competing reviews, one of them published two months ago:

| Review | Scope | Year |
|---|---|---|
| Hasan et al., *Frontiers in Digital Health* | ML methods for mental health state detection; Springer, ScienceDirect, IEEE, PubMed; 3,320 records screened | **Jan 2026** |
| Systematic review of AI in mental health care | Diagnosis, monitoring, intervention; 5 databases; 85 studies included | 2024 |
| Madububambachu et al., *Clin Pract Epidemiol Ment Health* | ML to predict mental health diagnoses; 30 studies | 2024 |
| Comprehensive review of predictive analytics for mental illness | PubMed + Scopus, 2020–2025 | 2025 |

The Hasan review in particular is close to identical to the scope you chose: same breadth of population, same disorder coverage, three of your four databases, published January 2026.

**What this means.** A descriptive review asking "which ML algorithms predict mental health conditions, and how accurately?" at this scope will very likely be rejected on novelty grounds at any Q1 venue, regardless of how well it is executed. An editor's first action is to check whether the question has already been answered. Here it has been, four times.

**This is fixable, and your scope choice is still the right one** — but the *question* has to change.

---

## 2. Recommended research question

Keep the broad population and disorder scope you selected. Change what the review asks.

> **How methodologically sound and completely reported are machine learning models for mental health prediction?** A systematic review and critical appraisal using PROBAST and TRIPOD+AI.

Three reasons this works where the descriptive version does not:

**It is genuinely unoccupied at this breadth.** The PROBAST-based reviews I found are narrow — one covered adolescent EHR crisis prediction and included five studies; another covered depression/anxiety trajectories. Nobody has appraised the field as a whole.

**Breadth becomes an asset instead of a liability.** For a performance comparison, pooling schizophrenia MRI studies with student stress surveys is a weakness — the numbers are not comparable. For a methodological appraisal, that heterogeneity *is* the sample. You are characterising a field, and the field is broad.

**It converts your existing findings into contributions.** The problems already visible in your draft — 100% accuracy claims, accuracy reported on imbalanced outcomes without AUC, no external validation, no calibration — stop being awkward observations and become the results.

### What you would actually report

- Proportion of studies reporting **any external validation**
- Proportion reporting **calibration** (a near-universal omission)
- Proportion reporting **AUC** rather than bare accuracy on imbalanced outcomes
- Proportion reporting **class prevalence** at all
- **PROBAST** risk-of-bias distribution across the four domains
- **TRIPOD+AI** adherence, item by item
- Proportion making **code or data available**
- Studies reporting near-perfect performance, examined as **data-leakage case studies**
- Events-per-variable against the ≥20 (statistical) / ≥200 (ML) rule of thumb now recommended by Moons et al.

That last point matters: recent PROBAST guidance sets a much higher EPV expectation for ML models than most published mental-health prediction studies meet. Quantifying that gap across a large sample would be a real finding.

---

## 3. Inclusion and exclusion criteria (PICOS)

This replaces Table 2 in the current draft, which was imported from an unrelated review on online extremism.

| Element | Include | Exclude |
|---|---|---|
| **Population** | Human participants of any age, any setting (community, clinical, educational, occupational), any country | Animal studies; simulation-only work with no human data |
| **Condition** | Any mental health disorder or state: depression, anxiety, stress, PTSD, bipolar disorder, schizophrenia and other psychoses, eating disorders, substance use disorders, general psychological distress | Neurological conditions without a psychiatric outcome (dementia, epilepsy); physical health outcomes |
| **Index model** | A supervised machine learning or deep learning model developed and/or validated to predict, detect, classify or diagnose a mental health outcome | Purely descriptive statistics; unsupervised clustering with no predictive target; conventional regression with no ML component; AI chatbots or interventions with no prediction model |
| **Comparator** | Not required. Where present: other ML models, conventional regression, clinical judgement, or screening instruments | — |
| **Outcomes** | At least one quantitative performance metric (accuracy, AUC, sensitivity, specificity, F1, precision, recall, calibration) | Studies reporting no performance metric |
| **Study design** | Prediction model development, validation, or development-with-validation studies. Prospective or retrospective. Any data source: EHR, survey, wearable, imaging, speech, social media text | Reviews, editorials, commentaries, protocols, conference abstracts without full text, preprints |
| **Timeframe** | Published 1 January 2018 – 31 December 2026 | Before 2018 |
| **Language** | English | Non-English |
| **Availability** | Full text obtainable | Full text unobtainable after two contact attempts |

### Justifying the boundaries

The 2018 start is defensible: it captures the deep-learning era in mental health prediction and aligns with the period covered by the reviews above. State it as a deliberate choice, not a convenience.

The English-only restriction is a limitation you must name explicitly in the Limitations section. It is standard but not costless.

Excluding preprints is a judgement call. Given that reporting quality is your outcome, including them would arguably strengthen the paper — non-peer-reviewed work would be expected to report worse, which is itself informative. If you include them, mark them clearly and analyse them as a separate stratum.

### LaTeX version for the manuscript

```latex
\begin{table}[!htbp]
\centering
\caption{Inclusion and exclusion criteria.}
\label{tab:criteria}
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabularx}{\linewidth}{@{}
  >{\raggedright\arraybackslash\hspace{0pt}}p{62pt}
  >{\raggedright\arraybackslash\hspace{0pt}}X
  >{\raggedright\arraybackslash\hspace{0pt}}X @{}}
\toprule
\textbf{Element} & \textbf{Inclusion} & \textbf{Exclusion}\\
\midrule
Population & Human participants, any age or setting & Animal studies; simulation-only work\\
Condition & Any mental health disorder or state & Neurological conditions without psychiatric outcome\\
Index model & Supervised ML or DL model predicting a mental health outcome & Descriptive statistics; unsupervised clustering; conventional regression alone\\
Comparator & Not required & ---\\
Outcomes & At least one quantitative performance metric & No performance metric reported\\
Design & Model development and/or validation studies & Reviews, editorials, protocols, abstracts\\
Timeframe & 2018--2026 & Before 2018\\
Language & English & Non-English\\
\bottomrule
\end{tabularx}
\end{table}
```

Requires `\usepackage{tabularx}` and `\usepackage{array}`.

---

## 4. Search strings

Three concept blocks combined with AND: **(ML) AND (mental health) AND (prediction)**. Run each verbatim, record the date, and save the result count — PRISMA 2020 requires all three per database.

### 4.1 PubMed

```
("Machine Learning"[Mesh] OR "Artificial Intelligence"[Mesh] OR "Deep Learning"[Mesh]
OR "Neural Networks, Computer"[Mesh] OR "Support Vector Machine"[Mesh]
OR "machine learning"[tiab] OR "deep learning"[tiab] OR "artificial intelligence"[tiab]
OR "neural network*"[tiab] OR "random forest"[tiab] OR "support vector"[tiab]
OR "gradient boosting"[tiab] OR XGBoost[tiab] OR "naive Bayes"[tiab]
OR "k-nearest"[tiab] OR "decision tree*"[tiab] OR "ensemble learning"[tiab]
OR "predictive model*"[tiab] OR "prediction model*"[tiab])
AND
("Mental Disorders"[Mesh] OR "Mental Health"[Mesh] OR "Depression"[Mesh]
OR "Depressive Disorder"[Mesh] OR "Anxiety Disorders"[Mesh]
OR "Stress, Psychological"[Mesh] OR "Stress Disorders, Post-Traumatic"[Mesh]
OR "Schizophrenia"[Mesh] OR "Bipolar Disorder"[Mesh]
OR "Feeding and Eating Disorders"[Mesh] OR "Substance-Related Disorders"[Mesh]
OR "mental health"[tiab] OR "mental illness"[tiab] OR "mental disorder*"[tiab]
OR depress*[tiab] OR anxiety[tiab] OR "psychological distress"[tiab]
OR PTSD[tiab] OR "post-traumatic stress"[tiab] OR schizophreni*[tiab]
OR "bipolar disorder"[tiab] OR psychosis[tiab] OR "eating disorder*"[tiab]
OR "substance use disorder*"[tiab])
AND
(predict*[tiab] OR detect*[tiab] OR classif*[tiab] OR screen*[tiab]
OR diagnos*[tiab] OR prognos*[tiab] OR identif*[tiab])
AND
("2018/01/01"[Date - Publication] : "2026/12/31"[Date - Publication])
AND english[Language]
```

### 4.2 Scopus

```
TITLE-ABS-KEY(("machine learning" OR "deep learning" OR "artificial intelligence"
OR "neural network*" OR "random forest" OR "support vector" OR "gradient boosting"
OR "XGBoost" OR "naive Bayes" OR "decision tree*" OR "ensemble learning"
OR "predictive model*" OR "prediction model*")
AND
("mental health" OR "mental illness" OR "mental disorder*" OR depress*
OR anxiety OR "psychological distress" OR "PTSD" OR "post-traumatic stress"
OR schizophreni* OR "bipolar disorder" OR psychosis OR "eating disorder*"
OR "substance use disorder*")
AND
(predict* OR detect* OR classif* OR screen* OR diagnos* OR prognos*))
AND PUBYEAR > 2017 AND PUBYEAR < 2027
AND (LIMIT-TO(LANGUAGE,"English"))
AND (LIMIT-TO(DOCTYPE,"ar") OR LIMIT-TO(DOCTYPE,"cp"))
```

### 4.3 Web of Science (Core Collection)

```
TS=(("machine learning" OR "deep learning" OR "artificial intelligence"
OR "neural network*" OR "random forest" OR "support vector" OR "gradient boosting"
OR "XGBoost" OR "naive Bayes" OR "decision tree*" OR "ensemble learning"
OR "predictive model*" OR "prediction model*")
AND
("mental health" OR "mental illness" OR "mental disorder*" OR depress*
OR anxiety OR "psychological distress" OR "PTSD" OR "post-traumatic stress"
OR schizophreni* OR "bipolar disorder" OR psychosis OR "eating disorder*"
OR "substance use disorder*")
AND
(predict* OR detect* OR classif* OR screen* OR diagnos* OR prognos*))
```

Then refine: Publication Years `2018–2026`; Document Types `Article` or `Proceedings Paper`; Language `English`.

### 4.4 IEEE Xplore (Command Search)

IEEE Xplore caps query complexity and rejects deeply nested strings, so this block is deliberately shorter. If it still errors, split the ML block in two and combine the result sets manually.

```
("All Metadata":"machine learning" OR "All Metadata":"deep learning"
OR "All Metadata":"artificial intelligence" OR "All Metadata":"neural network"
OR "All Metadata":"random forest" OR "All Metadata":"support vector")
AND
("All Metadata":"mental health" OR "All Metadata":"depression"
OR "All Metadata":"anxiety" OR "All Metadata":"PTSD"
OR "All Metadata":"schizophrenia" OR "All Metadata":"bipolar")
AND
("All Metadata":"prediction" OR "All Metadata":"detection"
OR "All Metadata":"classification")
```

Then filter: year range `2018–2026`; content type `Journals` and `Conferences`.

### 4.5 Also required

**Backward and forward citation chasing** of every included study, plus of the four overlapping reviews in Section 1. PRISMA 2020 asks you to report records found by methods other than database searching, and reviewers check for it.

**A note on PsycINFO.** It is the single most important database for this topic and it is not in your list. If your institution has any route to it — interlibrary access, a colleague's login, a trial — use it. If not, say so explicitly in the Limitations; omitting the field's primary psychology database is the first thing a methods reviewer will notice.

---

## 5. PROSPERO registration

Register at `https://www.crd.york.ac.uk/prospero/` **before running the searches**. Retrospective registration is visible to editors and reads as a weaker signal. Approval typically takes a few weeks.

Content for the main fields:

**Review title.** Methodological quality and reporting completeness of machine learning models for mental health prediction: a systematic review and PROBAST appraisal

**Review question.** How methodologically sound and completely reported are machine learning models developed to predict mental health disorders across populations?

**Searches.** PubMed, Scopus, Web of Science, IEEE Xplore, from 1 January 2018 to 31 December 2026, English language, supplemented by backward and forward citation chasing. Full strategies in the registration attachment.

**Condition or domain.** Mental health disorders, including depressive, anxiety, stress-related, psychotic, bipolar, eating and substance use disorders.

**Participants.** Human participants of any age in any setting.

**Index test / model.** Supervised machine learning or deep learning models predicting, detecting, classifying or diagnosing a mental health outcome.

**Primary outcomes.** Risk of bias by PROBAST domain (participants, predictors, outcome, analysis); adherence to TRIPOD+AI reporting items.

**Secondary outcomes.** Proportion of studies performing external validation; proportion reporting calibration; proportion reporting AUC versus accuracy alone; proportion reporting class prevalence; code and data availability; events-per-variable relative to current guidance.

**Data extraction.** Two reviewers independently, disagreements resolved by discussion or a third reviewer. Inter-rater agreement reported as Cohen's κ.

**Risk of bias.** PROBAST, applied independently by two reviewers.

**Synthesis.** Narrative and descriptive-quantitative synthesis of appraisal results. Meta-analysis of performance will be undertaken only if a sufficient subset reports AUC with variance estimates and is clinically homogeneous; otherwise the reason for not pooling will be stated.

---

## 6. Expected workload — plan for this honestly

At this breadth, expect roughly **8,000–15,000 records** before deduplication and **5,000–9,000** after. The Hasan review screened 3,320 with a narrower ML block; yours is wider.

That is a substantial screening burden, and it has to be dual-screened. Three ways to make it tractable:

1. **Use Rayyan** (`rayyan.ai`, free) for dual screening. It handles blinding, conflict resolution and κ calculation automatically.
2. **Recruit a second screener.** Non-negotiable for a Q1 methods review. If nobody is available, the defensible fallback is a second screener on a random 20% sample with κ reported on that subset — declared as a limitation.
3. **Consider narrowing the ML block** if the count is unmanageable — for example requiring the model terms in title/abstract only rather than all metadata. Any narrowing must be pre-registered, not applied after seeing the results.

**A realistic target is 80–200 included studies.** The current draft's 15 will not support a review at this scope, and a reviewer comparing your yield against Hasan's 3,320 screened records would immediately question the search.

---

## 7. What happens next

1. Confirm the reframed research question (Section 2), or tell me to keep the descriptive version and I will note the novelty risk in the manuscript.
2. Register on PROSPERO using Section 5.
3. Run the four searches, record counts and dates.
4. Screen in Rayyan.
5. Extract and appraise with PROBAST and TRIPOD+AI.

When you have the search counts, send them and I will build the PRISMA 2020 flow diagram with reconciling arithmetic — the current one does not add up, giving 14 where the text says 15.

I can draft the PROBAST and TRIPOD+AI extraction sheet next, so it is ready before screening finishes.
