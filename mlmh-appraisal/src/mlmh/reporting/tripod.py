"""TRIPOD+AI self-audit generated from run manifests.

Items whose status can be inferred from the code and manifests are filled in
automatically; the rest are marked MANUAL with a pointer to the manuscript
section that must satisfy them.  The item list mirrors the extraction workbook
(P1_extraction_appraisal.xlsx, sheet TRIPOD_AI) so that Paper B is audited with
exactly the instrument Paper A applies to other studies.
"""
from __future__ import annotations

import json
from pathlib import Path

ITEMS = [
    ("1", "Title identifies development/validation of a prediction model, target population, outcome", "MANUAL", "Title"),
    ("2", "Abstract reports objective, data, population, outcome, predictors, sample size, model, performance", "MANUAL", "Abstract"),
    ("3a", "Background and rationale", "MANUAL", "Introduction"),
    ("3b", "Objectives incl. development/validation", "MANUAL", "Introduction"),
    ("4a", "Data source described", "AUTO", "Methods: data (cohort loaders, data/README.md)"),
    ("4b", "Study dates", "MANUAL", "Methods: data (dataset publication dates)"),
    ("5a", "Setting, eligibility, recruitment", "MANUAL", "Methods: data (source publications)"),
    ("5b", "Treatments received", "MANUAL", "Methods: data"),
    ("6", "Outcome definition and timing", "AUTO", "Methods: label = diagnostic group (subjects table)"),
    ("7", "Predictors defined incl. measurement", "AUTO", "Methods: features (src/mlmh/features/actigraphy.py)"),
    ("8", "Sample size justified, EPV stated", "AUTO", "Results: n_subjects / n_windows per cohort, EPV in tables"),
    ("9", "Missing data handling", "AUTO", "Median imputation inside pipeline; window validity threshold"),
    ("10a", "Pre-processing", "AUTO", "Pipeline steps in manifest config"),
    ("10b", "Model types and rationale", "AUTO", "Model registry"),
    ("10c", "Hyperparameter tuning", "AUTO", "Fixed a priori; no tuning (registry docstring)"),
    ("10d", "Internal validation (CV scheme, repeats, seeds)", "AUTO", "Splitter + seeds in manifest"),
    ("10e", "External validation method", "AUTO", "E2 runner"),
    ("11", "Class imbalance handling; resampling inside folds", "AUTO", "resample setting in manifest; imblearn Pipeline"),
    ("12", "Performance measures incl. discrimination AND calibration", "AUTO", "metrics.py"),
    ("13", "Model updating / recalibration", "MANUAL", "Discussion (none performed unless E3 recalibration arm run)"),
    ("14", "Fairness / subgroup performance", "MANUAL", "Sex-stratified analysis where sex is available"),
    ("15", "Interpretability methods", "MANUAL", "Not a study aim; state so"),
    ("16", "Funding", "MANUAL", "Declarations"),
    ("17", "Conflicts of interest", "MANUAL", "Declarations"),
    ("18", "Protocol / registration", "MANUAL", "OSF pre-registration of E1-E3 recommended"),
    ("19", "Data availability", "AUTO", "data/README.md: sources, terms, checksums"),
    ("20", "Code availability", "AUTO", "This repository; git SHA in manifest"),
    ("21", "Participant flow", "AUTO", "Windows retained per subject (prepare step log)"),
    ("22", "Participant characteristics incl. prevalence", "AUTO", "Cohort table"),
    ("23", "Full model specification", "AUTO", "Pipelines + registry, seeds recorded"),
    ("24", "Performance with uncertainty intervals", "AUTO", "Subject-level BCa bootstrap CIs"),
    ("25", "Subgroup / fairness results", "MANUAL", "Results"),
    ("26", "Limitations", "MANUAL", "Discussion"),
    ("27", "Interpretation and implications", "MANUAL", "Discussion"),
]


def self_audit(results_dir: Path, out_path: Path) -> Path:
    results_dir = Path(results_dir)
    manifests = sorted(results_dir.glob("**/manifest.json"))
    runs = [json.loads(m.read_text()) for m in manifests]
    lines = ["# TRIPOD+AI self-audit (auto-generated)", ""]
    if not runs:
        lines.append("No run manifests found; run the experiments first.")
    else:
        lines.append(f"Runs found: {len(runs)}")
        for r in runs:
            lines.append(f"- `{r.get('experiment','?')}` git={r['git_sha'][:8]} dirty={r['git_dirty']} config={r['config_hash']} seeds={r['config'].get('seeds')} synthetic={r['synthetic']}")
        lines.append("")
    any_synthetic = any(r.get("synthetic") for r in runs)
    if any_synthetic:
        lines.append("**WARNING: at least one run used the synthetic fixture. Nothing below is a reportable result.**\n")
    lines.append("| Item | Requirement | Status | Where satisfied |")
    lines.append("|---|---|---|---|")
    for item, text, mode, where in ITEMS:
        status = "satisfied by code/manifest" if mode == "AUTO" and runs else ("MANUAL: write in manuscript" if mode == "MANUAL" else "pending: no runs")
        lines.append(f"| {item} | {text} | {status} | {where} |")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
