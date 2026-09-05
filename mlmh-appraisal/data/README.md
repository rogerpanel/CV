# Data: how to obtain each dataset, where to put it, and how to verify it

Nothing under `data/raw/`, `data/interim/`, `data/processed/` or `data/synthetic/` is ever committed.
Redistribution terms differ per dataset and the DAIC-WOZ licence forbids it outright. This file is
the only thing that goes into git: sources, exact download steps, the expected on-disk layout, and
the SHA-256 checksums that `python -m mlmh prepare` records into `data/processed/real/checksums.json`
(copy that file's contents into the table at the bottom once you have run it, so a reader can
confirm they hold byte-identical inputs).

The Claude Code sandbox that built this repository **cannot reach any of the dataset hosts**
(`datasets.simula.no`, `zenodo.org`, `dcapswoz.ict.usc.edu`, `wwwn.cdc.gov`, `uni-siegen.de`,
`dartmouth.edu` are all blocked by the egress policy). Download on your own machine with the links
below, then hand the archives back to the session as chat uploads (see "Bringing the data back into
the session" at the end).

---

## Tier 1 (primary cohorts, actigraphy, immediately available, no agreement to sign)

All three come from the same group (Simula / Haukeland University Hospital / University of Bergen)
and were recorded with the same Actiwatch device, which is what makes cross-cohort validation a test
of *population* shift rather than *measurement* shift. They are "fully open for research and
educational purposes"; commercial use and competitions need written permission; the source
publication must be cited.

| Cohort | Landing page (authoritative) | Archive | Source publication to cite |
|---|---|---|---|
| DEPRESJON | https://datasets.simula.no/depresjon/ | link labelled "Download" on the page (currently `https://datasets.simula.no/downloads/depresjon.zip`) | Garcia-Ceja E, Riegler M, Jakobsen P, et al. *Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients.* ACM MMSys 2018. https://doi.org/10.1145/3204949.3208125 |
| PSYKOSE | https://datasets.simula.no/psykose/ | "Download" on the page (`https://datasets.simula.no/downloads/psykose.zip`) | Jakobsen P, Garcia-Ceja E, Stabell LA, et al. *PSYKOSE: a motor activity database of patients with schizophrenia.* IEEE CBMS 2020. https://doi.org/10.1109/CBMS49503.2020.00064 |
| HYPERAKTIV | https://datasets.simula.no/hyperaktiv/ (code: https://github.com/simula/hyperaktiv) | "Download" on the page (`https://datasets.simula.no/downloads/hyperaktiv.zip`) | Hicks SA, Stautland A, Fasmer OB, et al. *HYPERAKTIV: an activity dataset from patients with attention-deficit/hyperactivity disorder (ADHD).* ACM MMSys 2021. https://doi.org/10.1145/3458305.3478454 |
| OBF-Psychiatric (all three, harmonised) | https://zenodo.org/records/13754984 (DOI 10.5281/zenodo.13754984) | Zenodo file list on the record page | Garcia-Ceja E, et al. *OBF-Psychiatric, a motor activity dataset of patients diagnosed with major depression, schizophrenia, and ADHD.* Sci Data 2025;12. https://doi.org/10.1038/s41597-025-04384-3 |

**Recommendation.** Download all four. Use OBF-Psychiatric as the canonical, harmonised copy
(same devices, standardised CSVs, five group folders, 162 participants / 1,565 days) and the three
originals for the clinical score files (MADRS, BPRS, neuropsychological test output) and to
cross-check that the harmonisation did not change any series. `python -m mlmh prepare` hashes every
participant's activity series and reports any participant present in more than one cohort.

**Known hazard: the PSYKOSE control group is the DEPRESJON control group.** The 32 healthy controls
are the same people in both archives (this is why OBF-Psychiatric has 162 participants:
23 depression + 22 schizophrenia + 85 HYPERAKTIV + 32 controls). A model trained on DEPRESJON and
"externally validated" on PSYKOSE without removing them has seen half of the test set. The E2
runner refuses to run if it detects the overlap and, with the default policy, assigns each shared
control to exactly one cohort.

### Expected layout after unzipping

Place each archive's *contents* in the folder named below. The loaders sniff delimiters and
normalise column names, so minor differences (`;` vs `,`, `TIME` vs `timestamp`) are fine; the
folder names are what matter.

```
data/raw/
├── depresjon/
│   ├── condition/condition_1.csv … condition_23.csv     # timestamp, date, activity
│   ├── control/control_1.csv … control_32.csv
│   └── scores.csv          # number, days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2
├── psykose/
│   ├── patient/patient_1.csv … patient_22.csv           # (older archives name this folder "condition/")
│   ├── control/control_1.csv … control_32.csv
│   ├── patients_info.csv   # number, days, gender, age, schtype, migraine, ... bprs ...
│   └── schizophrenia-features.csv (optional, not used)
├── hyperaktiv/
│   ├── activity_data/patient_activity_01.csv … (semicolon separated: TIME;ACTIVITY)
│   ├── hrv_data/ (not used)
│   ├── patient_info.csv    # ID;SEX;AGE;ACC;ACC_TIME;ACC_DAYS;HRV;...;ADHD;ADD;BIPOLAR;UNIPOLAR;ANXIETY;...
│   └── features/ (not used)
└── obf_psychiatric/
    ├── adhd/  clinical/  control/  depression/  schizophrenia/   # per-participant csv in each
    └── (any patients_info / features csv shipped with the record)
```

If your archive differs, run `python -m mlmh verify-data` and paste the output into the chat; the
loader will be adjusted rather than the data.

---

## Tier 2 (secondary cohorts, other modalities; confirm access terms before committing effort)

| Dataset | Modality / label | Access | Where |
|---|---|---|---|
| DAIC-WOZ / E-DAIC | 189 clinical interviews (audio, transcripts, facial features), PHQ-8 | End-user licence agreement; academics only; academic e-mail required; **start now, takes weeks** | https://dcapswoz.ict.usc.edu/ (form on the page; E-DAIC / AVEC 2019 on request via the same route) |
| NHANES DPQ (PHQ-9) | Population survey, PHQ-9 total >= 10 | Fully open, no registration | Cycle pages under https://wwwn.cdc.gov/nchs/nhanes/ ; questionnaire file `DPQ_J.XPT` (2017-18), `P_DPQ.XPT` (2017-Mar 2020), `DPQ_L.XPT` (2021-23) plus `DEMO_*.XPT`; read with `pandas.read_sas` |
| WESAD | Wearable physiology, stress vs baseline vs amusement, 15 subjects | Open, cite required, 2.5 GB | https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html (also on UCI ML repository) |
| StudentLife | Smartphone sensing + PHQ-9/PSS in 48 students | Open, ~5 GB | https://studentlife.cs.dartmouth.edu/dataset.html (`dataset.tar.bz2`); R package `studentlife` on CRAN |
| MODMA | EEG + speech, MDD vs controls | Application form | https://modma.lzu.edu.cn/data/index/ (GitHub: UAIS-LANZHOU/MODMA-Dataset) |
| CLPsych / eRisk | Social-media text | Signed agreements; ethical review needed | https://www.clpsych.org/ ; https://erisk.irlab.org/ |

Only Tier 1 is wired into the loaders today. Tier 2 loaders are added when a cohort is actually in
hand, one loader per dataset into the same schema (`src/mlmh/data/loaders.py`).

---

## Step-by-step: bringing the data back into the session

1. On your machine, download the three Simula archives and the OBF-Psychiatric Zenodo files.
2. Do **not** unzip them into git. Keep the `.zip` files.
3. In the Claude Code chat, attach the archives as uploads (they appear under
   `/root/.claude/uploads/...`). Simula archives are tens of megabytes each; if an upload limit
   bites, upload DEPRESJON and PSYKOSE first (smallest, and enough to run E1-E3), then HYPERAKTIV,
   then OBF.
4. Ask the session to run:
   ```bash
   cd mlmh-appraisal
   unzip -q <upload>.zip -d data/raw/depresjon   # one folder per cohort, as in the layout above
   python -m mlmh verify-data                    # prints what was recognised, exits non-zero on problems
   python -m mlmh prepare                        # windows, features, checksums, shared-participant report
   python -m mlmh run configs/e1_split_leakage.yaml
   python -m mlmh run configs/e2_external_val.yaml
   python -m mlmh run configs/e3_calibration.yaml
   python -m mlmh tripod
   python scripts/update_readme.py               # refreshes the results section of README.md
   ```
5. Commit `results/real/**` (tables, figures, manifests, CSV summaries) and the regenerated
   `paper/empirical/tables/*.tex`. Never commit anything under `data/`.

---

## Checksums (fill in after the first real `prepare`)

`data/processed/real/checksums.json` holds one SHA-256 per raw CSV. Paste a summary here:

| Cohort | Files | Combined SHA-256 of the sorted per-file hashes | Date obtained |
|---|---|---|---|
| depresjon | | | |
| psykose | | | |
| hyperaktiv | | | |
| obf_psychiatric | | | |
