"""Refresh the auto-generated results section of README.md from results/<real|synthetic>/.

Usage: python scripts/update_readme.py [--synthetic]
The section between <!-- RESULTS:START --> and <!-- RESULTS:END --> is replaced; everything
else in README.md is left untouched, so the narrative can be edited by hand as results change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"majority": "Majority class", "logreg": "Logistic regression", "rf": "Random forest", "xgboost": "XGBoost", "mlp": "MLP", "cnn1d": "1D-CNN"}


def _f(v, nd=3):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def _ci(row, key, level="window"):
    est = row.get(f"{level}_{key}_est")
    if est is None or pd.isna(est):
        est = row.get(f"{level}_{key}_mean")
    lo, hi = row.get(f"{level}_{key}_ci_lo"), row.get(f"{level}_{key}_ci_hi")
    if pd.isna(est):
        return "--"
    if pd.isna(lo) or pd.isna(hi):
        return _f(est)
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def e1_section(rdir: Path) -> str:
    p = rdir / "E1" / "e1_results.csv"
    if not p.exists():
        return "_E1 not run yet._\n"
    t = pd.read_csv(p)
    main = t[t["splitter"].isin(["subject_wise", "record_wise"])]
    lines = ["| Cohort | Model | AUROC subject-wise (window) | AUROC record-wise (window) | Inflation (paired, window) | AUROC subject-wise (subject-level) | AUROC record-wise (subject-level) |", "|---|---|---|---|---|---|---|"]
    for (c, m), g in main.groupby(["cohort", "model"], sort=False):
        sw = g[g["splitter"] == "subject_wise"]
        rw = g[g["splitter"] == "record_wise"]
        inf = t[(t["cohort"] == c) & (t["model"] == m) & (t["splitter"] == "inflation") & t["window_auroc_mean"].notna()]
        lines.append(
            f"| {c} | {LABELS.get(m, m)} | {_ci(sw.iloc[0], 'auroc') if len(sw) else '--'} | {_ci(rw.iloc[0], 'auroc') if len(rw) else '--'} | "
            f"{_ci(inf.iloc[0], 'auroc') if len(inf) else '--'} | {_ci(sw.iloc[0], 'auroc', 'subject') if len(sw) else '--'} | {_ci(rw.iloc[0], 'auroc', 'subject') if len(rw) else '--'} |"
        )
    return "\n".join(lines) + "\n"


def e2_section(rdir: Path) -> str:
    p = rdir / "E2" / "e2_results.csv"
    if not p.exists():
        return "_E2 not run yet._\n"
    t = pd.read_csv(p)
    lines = ["| Train -> Test | Model | EPV (train) | Internal AUROC | External AUROC | Delta | Ext. cal. slope | Ext. cal. intercept | Ext. Brier |", "|---|---|---|---|---|---|---|---|---|"]
    for (a, b, m), g in t.groupby(["train", "test", "model"], sort=False):
        i, e = g[g["arm"] == "internal"].iloc[0], g[g["arm"] == "external"].iloc[0]
        lines.append(
            f"| {a} -> {b} | {LABELS.get(m, m)} | {_f(i['epv'], 1)} | {_ci(i, 'auroc')} | {_ci(e, 'auroc')} | {(e.get('window_auroc_est') if pd.notna(e.get('window_auroc_est', float('nan'))) else e['window_auroc_mean']) - (i.get('window_auroc_est') if pd.notna(i.get('window_auroc_est', float('nan'))) else i['window_auroc_mean']):+.3f} | "
            f"{_f(e['window_calibration_slope_mean'], 2)} | {_f(e['window_calibration_intercept_mean'], 2)} | {_f(e['window_brier_mean'])} |"
        )
    return "\n".join(lines) + "\n"


def e3_section(rdir: Path) -> str:
    p = rdir / "E3" / "e3_calibration.csv"
    if not p.exists():
        return "_E3 not run yet._\n"
    t = pd.read_csv(p)
    t = t[t["level"] == "window"]
    lines = ["| Exp. | Run | AUROC | Brier | Cal. slope | Cal. intercept | ECE |", "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['source']} | `{r['run']}` | {_f(r['auroc'])} | {_f(r['brier'])} | {_f(r['calibration_slope'], 2)} | {_f(r['calibration_intercept'], 2)} | {_f(r['ece'])} |")
    return "\n".join(lines) + "\n"


def e4_section(rdir: Path) -> str:
    p = rdir / "E4" / "e4_fairness.csv"
    if not p.exists():
        return "_E4 not run yet._\n"
    t = pd.read_csv(p)
    lines = ["| Cohort | Model | Sex | Participants (cases) | AUROC [95% CI] | Cal. slope | ECE |", "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['cohort']} | {LABELS.get(r['model'], r['model'])} | {r['sex']} | {r['n_subjects']} ({r['n_cases']}) | {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] | {_f(r['calibration_slope'], 2)} | {_f(r['ece'])} |")
    return "\n".join(lines) + "\n"


def e5_section(rdir: Path) -> str:
    p = rdir / "E5" / "e5_obf.csv"
    if not p.exists():
        return "_E5 not run yet._\n"
    t = pd.read_csv(p)
    lines = ["| Arm | Model | Split | AUROC / macro-AUROC [95% CI] | Accuracy | Macro-F1 |", "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        if r["arm"] == "any_psychiatric_vs_control":
            lines.append(f"| any psychiatric vs control | {LABELS.get(r['model'], r['model'])} | {r['splitter']} | {_ci(r, 'auroc')} | {_f(r['window_accuracy_mean'])} | {_f(r['window_macro_f1_mean'])} |")
        else:
            lines.append(f"| five classes | {LABELS.get(r['model'], r['model'])} | {r['splitter']} | {r['macro_auroc_ovr_mean']:.3f} [{r['macro_auroc_ovr_ci_lo']:.3f}, {r['macro_auroc_ovr_ci_hi']:.3f}] | {_f(r['accuracy_mean'])} | {_f(r['macro_f1_mean'])} |")
    return "\n".join(lines) + "\n"


def cnn_section(rdir: Path) -> str:
    p = rdir / "E1_cnn" / "e1_results.csv"
    if not p.exists():
        return "_1D-CNN arm not run yet._\n"
    t = pd.read_csv(p)
    main = t[t["splitter"].isin(["subject_wise", "record_wise"]) & (t["model"] == "cnn1d")]
    lines = ["| Cohort | AUROC subject-wise (window) | AUROC record-wise (window) | Inflation (paired) |", "|---|---|---|---|"]
    for c, g in main.groupby("cohort", sort=False):
        sw, rw = g[g["splitter"] == "subject_wise"], g[g["splitter"] == "record_wise"]
        inf = t[(t["cohort"] == c) & (t["model"] == "cnn1d") & (t["splitter"] == "inflation") & t["window_auroc_mean"].notna()]
        lines.append(f"| {c} | {_ci(sw.iloc[0], 'auroc') if len(sw) else '--'} | {_ci(rw.iloc[0], 'auroc') if len(rw) else '--'} | {_ci(inf.iloc[0], 'auroc') if len(inf) else '--'} |")
    return "\n".join(lines) + "\n"


def e6_section(rdir: Path) -> str:
    p = rdir / "E6" / "e6_ablation.csv"
    if not p.exists():
        return "_E6 not run yet._\n"
    t = pd.read_csv(p)
    lines = ["| Cohort | Model | Variant | k | AUROC [95% CI] | Subject AUROC | Cal. slope | ECE |", "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['cohort']} | {LABELS.get(r['model'], r['model'])} | {r['variant']} | {r['n_features']} | {_ci(r, 'auroc')} | {_f(r['subject_auroc_est'])} | {_f(r['window_calibration_slope_est'], 2)} | {_f(r['window_ece_est'])} |")
    return "\n".join(lines) + "\n"


def manifests(rdir: Path) -> str:
    rows = []
    for m in sorted(rdir.glob("*/manifest.json")):
        j = json.loads(m.read_text())
        rows.append(f"| {j.get('experiment')} | `{j['git_sha'][:10]}`{' (dirty)' if j['git_dirty'] else ''} | `{j['config_hash']}` | {j['config'].get('seeds')} | {j['config'].get('n_boot')} | {j['created_utc'][:19]} | {'yes' if j['synthetic'] else 'no'} |")
    if not rows:
        return "_No manifests yet._\n"
    return "| Exp. | git | config hash | seeds | bootstrap | created (UTC) | synthetic |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rdir = ROOT / "results" / ("synthetic" if a.synthetic else "real")
    banner = (
        "> **These numbers come from the SYNTHETIC fixture** (`python -m mlmh synth`). They demonstrate that the pipeline runs end to end and show the *mechanism* of leakage inflation on fingerprinted data. They are not results and must not be quoted.\n\n"
        if a.synthetic
        else "> Results below were produced from the real cohorts; every table is traceable to a manifest in `results/real/`.\n\n"
    )
    body = (
        "<!-- RESULTS:START -->\n"
        f"_Last refreshed by `scripts/update_readme.py` from `{rdir.relative_to(ROOT)}`._\n\n"
        + banner
        + "#### E1: leakage inflation (record-wise minus subject-wise)\n\n" + e1_section(rdir) + "\n"
        + "#### E2: internal versus external validation\n\n" + e2_section(rdir) + "\n"
        + "#### E3: calibration alongside discrimination\n\n" + e3_section(rdir) + "\n"
        + "#### E4: sex-stratified performance (subject-wise E1 models)\n\n" + e4_section(rdir) + "\n"
        + "#### E5: OBF-Psychiatric transdiagnostic and five-class arms\n\n" + e5_section(rdir) + "\n"
        + "#### E6: ablation (feature groups, SMOTE inside folds, LOSO)\n\n" + e6_section(rdir) + "\n"
        + "#### Supplementary arm: 1D-CNN on the raw 1440-minute series (5 seeds)\n\n" + cnn_section(rdir) + "\n"
        + "#### Run manifests\n\n" + manifests(rdir)
        + "<!-- RESULTS:END -->"
    )
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = text.index("<!-- RESULTS:START -->"), text.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
    readme.write_text(text[:start] + body + text[end:])
    print(f"[update_readme] refreshed results section from {rdir}")


if __name__ == "__main__":
    main()
