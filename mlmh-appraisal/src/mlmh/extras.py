"""Additional analyses on top of E1-E3.

E4  sex-stratified (fairness) performance from the stored E1 subject-wise predictions
E5  OBF-Psychiatric transdiagnostic (any psychiatric vs healthy control) and 5-class arms,
    again under subject-wise vs record-wise CV
cohort_table  participant characteristics for the manuscript (TRIPOD+AI item 22)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .data.schema import WindowedDataset
from .evaluation.bootstrap import subject_bootstrap_ci
from .evaluation.cv import seed_averaged
from .evaluation.metrics import binary_metrics, subject_level
from .evaluation.splitters import make_splitter
from .experiments import ROOT, load_processed, processed_dir, results_dir
from .models.pipelines import make_pipeline
from .models.registry import MODEL_LABELS
from .reporting.manifest import write_manifest
from .reporting.tables import fmt_ci, write_latex_table

SEX_CODE = {1: "female", 2: "male", "1": "female", "2": "male", "female": "female", "male": "male", 0: "female", "0": "female"}


# ------------------------------------------------------------- demographics
def demographics(cfg: dict) -> pd.DataFrame:
    """One row per participant across all cohorts, keyed by series_hash, with sex, age, days, scales.

    OBF-Psychiatric carries harmonised *-info.csv files for every participant, so it is used as the
    single source of demographics and matched to the other cohorts by activity-series hash.
    """
    pdir = processed_dir(cfg)
    obf = pd.read_csv(pdir / "obf_psychiatric.subjects.csv")
    cols = [c for c in ("series_hash", "subject_id", "group", "gender", "age", "days", "madrs1", "madrs2", "bprs", "afftype", "adhd", "asrs", "wurs", "hads_a", "hads_d") if c in obf.columns]
    d = obf[cols].rename(columns={"subject_id": "obf_id", "group": "obf_group"})
    if "gender" in d:
        d["sex"] = d["gender"].map(SEX_CODE)
    return d


def cohort_table(cfg: dict) -> pd.DataFrame:
    pdir = processed_dir(cfg)
    demo = demographics(cfg)
    summary = json.loads((pdir / "prepare_summary.json").read_text())
    rows = []
    for cohort in cfg["cohorts"]:
        p = pdir / f"{cohort}.subjects.csv"
        if not p.exists():
            continue
        subj = pd.read_csv(p).merge(demo, on="series_hash", how="left", suffixes=("", "_demo"))
        for c in ("sex", "age", "days", "madrs1", "bprs", "asrs", "gender"):
            if f"{c}_demo" in subj.columns:
                subj[c] = subj[c].where(subj[c].notna(), subj[f"{c}_demo"])
        n_windows = summary.get(cohort, {}).get("n_windows", np.nan)
        for label, name in ((1, "cases"), (0, "controls")):
            g = subj[subj["label"] == label]
            sex = g["sex"].value_counts(dropna=True) if "sex" in g else pd.Series(dtype=int)
            days = pd.to_numeric(g["days"], errors="coerce") if "days" in g else pd.Series(dtype=float)
            scale = ""
            if "madrs1" in g and g["madrs1"].notna().any() and label == 1 and cohort == "depresjon":
                scale = f"MADRS {pd.to_numeric(g['madrs1'], errors='coerce').median():.0f} (IQR {pd.to_numeric(g['madrs1'], errors='coerce').quantile(.25):.0f}-{pd.to_numeric(g['madrs1'], errors='coerce').quantile(.75):.0f})"
            if "bprs" in g and g["bprs"].notna().any() and label == 1 and cohort == "psykose":
                scale = f"BPRS {pd.to_numeric(g['bprs'], errors='coerce').median():.0f} (IQR {pd.to_numeric(g['bprs'], errors='coerce').quantile(.25):.0f}-{pd.to_numeric(g['bprs'], errors='coerce').quantile(.75):.0f})"
            if "asrs" in g and g["asrs"].notna().any() and cohort == "hyperaktiv":
                scale = f"ASRS {pd.to_numeric(g['asrs'], errors='coerce').median():.0f} (IQR {pd.to_numeric(g['asrs'], errors='coerce').quantile(.25):.0f}-{pd.to_numeric(g['asrs'], errors='coerce').quantile(.75):.0f})"
            rows.append(
                {
                    "Cohort": cohort,
                    "Group": f"{name} ({', '.join(sorted(g['group'].unique()))})",
                    "n": int(len(g)),
                    "Female, n (%)": f"{int(sex.get('female', 0))} ({100 * sex.get('female', 0) / max(len(g), 1):.0f})" if len(sex) else "--",
                    "Age (modal band)": str(g["age"].mode().iloc[0]) if "age" in g and g["age"].notna().any() else "--",
                    "Recording days, median (range)": f"{days.median():.0f} ({days.min():.0f}-{days.max():.0f})" if days.notna().any() else "--",
                    "Valid day-windows": int(subj[subj["label"] == label]["n_days"].sum()) if "n_days" in subj else "--",
                    "Clinical scale, median (IQR)": scale or "--",
                }
            )
    df = pd.DataFrame(rows)
    tdir = ROOT / cfg.get("tables_dir", "paper/empirical/tables")
    write_latex_table(df, tdir / "cohort_characteristics.tex", "Participant characteristics per cohort. HYPERAKTIV controls are psychiatric patients without ADHD (clinical controls); DEPRESJON and PSYKOSE share the same 32 healthy controls. Sex, age band and recording days from the harmonised OBF-Psychiatric metadata; day-windows are the analysis units after the 80\\% validity threshold.", "tab:cohorts", synthetic=bool(cfg.get("synthetic")), column_format="llrlllrl")
    df.to_csv(results_dir(cfg, "tables") / "cohort_characteristics.csv", index=False)
    return df


# ------------------------------------------------------------------- E4
def run_e4(cfg: dict) -> pd.DataFrame:
    """Sex-stratified discrimination and calibration from E1 subject-wise predictions (TRIPOD+AI 14/25)."""
    out = results_dir(cfg, "E4")
    demo = demographics(cfg)[["series_hash", "sex"]]
    pdir = processed_dir(cfg)
    rows = []
    for path in sorted((results_dir(cfg, "E1") / "predictions").glob("*.subject_wise.csv")):
        cohort, model, _ = path.stem.split(".")
        if model == "majority":
            continue
        subj = pd.read_csv(pdir / f"{cohort}.subjects.csv")[["subject_id", "series_hash"]].merge(demo, on="series_hash", how="left")
        pred = seed_averaged(pd.read_csv(path)).merge(subj[["subject_id", "sex"]], on="subject_id", how="left")
        for sex, g in pred.groupby("sex"):
            if g["y"].nunique() < 2 or g["subject_id"].nunique() < 6:
                continue
            m = binary_metrics(g["y"], g["p"])
            est, lo, hi = subject_bootstrap_ci(g, lambda f: roc_auc_score(f["y"], f["p"]) if f["y"].nunique() == 2 else np.nan, n_boot=int(cfg.get("n_boot", 1000)))
            s = subject_level(g)
            rows.append({"cohort": cohort, "model": model, "sex": sex, "n_subjects": int(g["subject_id"].nunique()), "n_cases": int(s["y"].sum()), "auroc": est, "auroc_lo": lo, "auroc_hi": hi, "brier": m["brier"], "calibration_slope": m["calibration_slope"], "calibration_intercept": m["calibration_intercept"], "ece": m["ece"]})
            print(f"[E4] {cohort:>10} {model:>8} {sex:>6}  n={rows[-1]['n_subjects']:3d} cases={rows[-1]['n_cases']:2d}  AUROC={est:.3f} [{lo:.3f},{hi:.3f}]  slope={m['calibration_slope']:.2f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "e4_fairness.csv", index=False)
    show = pd.DataFrame(
        {
            "Cohort": table["cohort"],
            "Model": table["model"].map(lambda m: MODEL_LABELS.get(m, m)),
            "Sex": table["sex"],
            "Participants (cases)": [f"{a} ({b})" for a, b in zip(table["n_subjects"], table["n_cases"])],
            "AUROC [95\\% CI]": [fmt_ci(a, b, c) for a, b, c in zip(table["auroc"], table["auroc_lo"], table["auroc_hi"])],
            "Brier": table["brier"].map(lambda v: f"{v:.3f}"),
            "Cal. slope": table["calibration_slope"].map(lambda v: f"{v:.2f}"),
            "ECE": table["ece"].map(lambda v: f"{v:.3f}"),
        }
    )
    write_latex_table(show, ROOT / cfg.get("tables_dir", "paper/empirical/tables") / "e4_fairness_sex.tex", "E4: sex-stratified performance of the subject-wise E1 models (window level, subject-level bootstrap intervals). Subgroups with fewer than six participants or a single class are omitted.", "tab:e4", synthetic=bool(cfg.get("synthetic")), column_format="lllllrrr")
    write_manifest(out, cfg, extra={"experiment": "E4", "n_rows": len(table)}, checksums_path=pdir / "checksums.json")
    return table


# ------------------------------------------------------------------- E5
def _multiclass_oof(ds: WindowedDataset, model: str, splitter_name: str, seeds: list[int], n_splits: int) -> pd.DataFrame:
    classes = np.unique(ds.y)
    rows = []
    for seed in seeds:
        splitter = make_splitter(splitter_name, n_splits=n_splits, random_state=seed)
        pipe = make_pipeline(model, seed=seed, n_classes=len(classes))
        for fold, (tr, te) in enumerate(splitter.split(ds.X, ds.y, ds.subject_id)):
            if not splitter.leaks_subjects:
                assert not set(ds.subject_id[tr]) & set(ds.subject_id[te])
            est = clone(pipe).fit(ds.X[tr], ds.y[tr])
            P = est.predict_proba(ds.X[te])
            df = pd.DataFrame(P, columns=[f"p{c}" for c in est.classes_])
            df.insert(0, "y", ds.y[te])
            df.insert(0, "subject_id", ds.subject_id[te])
            df.insert(0, "window_id", ds.window_id[te])
            df.insert(0, "fold", fold)
            df.insert(0, "seed", seed)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _mc_metrics(pred: pd.DataFrame) -> dict[str, float]:
    pcols = [c for c in pred.columns if c.startswith("p")]
    P = pred[pcols].to_numpy()
    y = pred["y"].to_numpy()
    yhat = np.array([int(c[1:]) for c in pcols])[P.argmax(1)]
    try:
        auc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[int(c[1:]) for c in pcols])
    except ValueError:
        auc = np.nan
    return {"macro_auroc_ovr": auc, "accuracy": accuracy_score(y, yhat), "macro_f1": f1_score(y, yhat, average="macro")}


def run_e5(cfg: dict) -> pd.DataFrame:
    """OBF-Psychiatric: transdiagnostic binary arm and 5-class arm, subject-wise vs record-wise."""
    out = results_dir(cfg, "E5")
    (out / "predictions").mkdir(exist_ok=True)
    seeds = list(cfg["seeds"])
    pdir = processed_dir(cfg)
    rows = []
    # --- binary transdiagnostic arm reuses the standard machinery
    from .evaluation.cv import oof_predictions
    from .experiments import _summarise

    ds = load_processed(cfg, "obf_psychiatric")
    for model in cfg["models"]:
        for splitter in cfg["splitters"]:
            pred = oof_predictions(ds, model, splitter, seeds, n_splits=cfg.get("n_splits", 5))
            pred.to_csv(out / "predictions" / f"obf_any_vs_control.{model}.{splitter}.csv", index=False)
            row = {"arm": "any_psychiatric_vs_control", "model": model, "splitter": splitter, "n_subjects": ds.n_subjects, "n_windows": ds.n_windows}
            row.update(_summarise(pred, cfg))
            rows.append(row)
            print(f"[E5] any-vs-control {model:>8} {splitter:>12}  AUROC(window)={row['window_auroc_mean']:.3f} AUROC(subject)={row['subject_auroc_mean']:.3f}")
    # --- 5-class arm: relabel by group
    subj = pd.read_csv(pdir / "obf_psychiatric.subjects.csv")
    groups = ["control", "depression", "schizophrenia", "adhd", "clinical"]
    gmap = dict(zip(subj["subject_id"], subj["group"].map(groups.index)))
    y5 = np.array([gmap[s] for s in ds.subject_id])
    ds5 = WindowedDataset(X=ds.X, y=y5, subject_id=ds.subject_id, cohort=ds.cohort, window_id=ds.window_id, feature_names=ds.feature_names, subjects=ds.subjects)
    for model in [m for m in cfg["models"] if m != "majority"]:
        for splitter in cfg["splitters"]:
            pred = _multiclass_oof(ds5, model, splitter, seeds, cfg.get("n_splits", 5))
            pred.to_csv(out / "predictions" / f"obf_5class.{model}.{splitter}.csv", index=False)
            per_seed = pd.DataFrame([_mc_metrics(g) for _, g in pred.groupby("seed")])
            row = {"arm": "five_class", "model": model, "splitter": splitter, "n_subjects": ds5.n_subjects, "n_windows": ds5.n_windows}
            for c in per_seed.columns:
                row[f"{c}_mean"], row[f"{c}_sd"] = float(per_seed[c].mean()), float(per_seed[c].std(ddof=1))
            avg = pred.groupby(["window_id", "subject_id"]).agg({**{c: "mean" for c in pred.columns if c.startswith("p")}, "y": "first"}).reset_index()
            est, lo, hi = subject_bootstrap_ci(avg, lambda f: _mc_metrics(f)["macro_auroc_ovr"], n_boot=int(cfg.get("n_boot", 1000)))
            row.update({"macro_auroc_ovr_ci_lo": lo, "macro_auroc_ovr_ci_hi": hi})
            rows.append(row)
            print(f"[E5] 5-class        {model:>8} {splitter:>12}  macroAUROC={row['macro_auroc_ovr_mean']:.3f} [{lo:.3f},{hi:.3f}]  acc={row['accuracy_mean']:.3f}  macroF1={row['macro_f1_mean']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "e5_obf.csv", index=False)
    # LaTeX
    show = []
    for _, r in table.iterrows():
        if r["arm"] == "any_psychiatric_vs_control":
            show.append({"Arm": "Any psychiatric vs control", "Model": MODEL_LABELS.get(r["model"], r["model"]), "Split": r["splitter"].replace("_", "-"), "AUROC / macro-AUROC": fmt_ci(r["window_auroc_mean"], r.get("window_auroc_ci_lo"), r.get("window_auroc_ci_hi")), "Accuracy": f"{r['window_accuracy_mean']:.3f}", "Macro-F1": f"{r['window_macro_f1_mean']:.3f}", "Cal. slope": f"{r['window_calibration_slope_mean']:.2f}"})
        else:
            show.append({"Arm": "Five diagnostic groups", "Model": MODEL_LABELS.get(r["model"], r["model"]), "Split": r["splitter"].replace("_", "-"), "AUROC / macro-AUROC": fmt_ci(r["macro_auroc_ovr_mean"], r["macro_auroc_ovr_ci_lo"], r["macro_auroc_ovr_ci_hi"]), "Accuracy": f"{r['accuracy_mean']:.3f}", "Macro-F1": f"{r['macro_f1_mean']:.3f}", "Cal. slope": "--"})
    write_latex_table(pd.DataFrame(show), ROOT / cfg.get("tables_dir", "paper/empirical/tables") / "e5_obf_transdiagnostic.tex", "E5: OBF-Psychiatric (162 participants, single control group). Transdiagnostic binary arm (any psychiatric group vs healthy control; window-level AUROC with subject-level BCa intervals) and five-class arm (macro one-vs-rest AUROC), each under subject-wise and record-wise CV.", "tab:e5", synthetic=bool(cfg.get("synthetic")), column_format="lllrrrr")
    write_manifest(out, cfg, extra={"experiment": "E5", "n_rows": len(table)}, checksums_path=pdir / "checksums.json")
    return table


# ------------------------------------------------------------------- E6
FEATURE_GROUPS = {
    "distributional": ["mean", "sd", "cv", "median", "p10", "p90", "max", "skew", "kurtosis", "prop_zero", "mean_log1p", "sd_log1p"],
    "day_night": ["day_mean", "night_mean", "night_day_ratio", "night_prop_zero"],
    "circadian": ["M10", "L5", "relative_amplitude", "IV", "m10_onset_sin", "m10_onset_cos"],
    "temporal_spectral": ["acf_5", "acf_60", "psd_mean", "dominant_period_min", "n_transitions", "missing_frac"],
}


def run_e6(cfg: dict) -> pd.DataFrame:
    """Ablation under subject-wise CV: feature groups (drop-one and only-one), SMOTE inside folds,
    leave-one-subject-out instead of 5-fold, and window-validity threshold sensitivity is reported
    from the prepare summary. Fixed models, seeds as configured."""
    from .evaluation.cv import oof_predictions
    from .experiments import _summarise

    out = results_dir(cfg, "E6")
    seeds = list(cfg["seeds"])
    rows = []
    for cohort in cfg["cohorts"]:
        ds = load_processed(cfg, cohort)
        names = ds.feature_names
        variants = {"all_features": names}
        for g, cols in FEATURE_GROUPS.items():
            variants[f"drop_{g}"] = [n for n in names if n not in cols]
            variants[f"only_{g}"] = [n for n in names if n in cols]
        for model in cfg["models"]:
            for vname, cols in variants.items():
                idx = [names.index(c) for c in cols]
                sub = WindowedDataset(X=ds.X[:, idx], y=ds.y, subject_id=ds.subject_id, cohort=ds.cohort, window_id=ds.window_id, feature_names=cols, subjects=ds.subjects)
                pred = oof_predictions(sub, model, "subject_wise", seeds, n_splits=cfg.get("n_splits", 5))
                row = {"cohort": cohort, "model": model, "variant": vname, "n_features": len(cols), "splitter": "subject_wise", "resample": "none"}
                row.update(_summarise(pred, cfg))
                rows.append(row)
                print(f"[E6] {cohort:>10} {model:>8} {vname:>24} k={len(cols):2d}  AUROC={row['window_auroc_est']:.3f} [{row['window_auroc_ci_lo']:.3f},{row['window_auroc_ci_hi']:.3f}]  ECE={row['window_ece_est']:.3f}")
            # SMOTE inside folds
            pred = oof_predictions(ds, model, "subject_wise", seeds, n_splits=cfg.get("n_splits", 5), resample="smote")
            row = {"cohort": cohort, "model": model, "variant": "all_features+smote_in_fold", "n_features": len(names), "splitter": "subject_wise", "resample": "smote"}
            row.update(_summarise(pred, cfg)); rows.append(row)
            print(f"[E6] {cohort:>10} {model:>8} {'smote_in_fold':>24} k={len(names):2d}  AUROC={row['window_auroc_est']:.3f}  ECE={row['window_ece_est']:.3f}")
            # LOSO
            pred = oof_predictions(ds, model, "loso", seeds[:1], n_splits=cfg.get("n_splits", 5))
            row = {"cohort": cohort, "model": model, "variant": "all_features+loso", "n_features": len(names), "splitter": "loso", "resample": "none"}
            row.update(_summarise(pred, cfg)); rows.append(row)
            print(f"[E6] {cohort:>10} {model:>8} {'loso':>24} k={len(names):2d}  AUROC={row['window_auroc_est']:.3f}  ECE={row['window_ece_est']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "e6_ablation.csv", index=False)
    show = pd.DataFrame({
        "Cohort": table["cohort"], "Model": table["model"].map(lambda m: MODEL_LABELS.get(m, m)),
        "Variant": table["variant"].str.replace("_", "\\_"), "k": table["n_features"],
        "AUROC [95\\% CI]": [fmt_ci(a, b, c) for a, b, c in zip(table["window_auroc_est"], table["window_auroc_ci_lo"], table["window_auroc_ci_hi"])],
        "Subject AUROC": table["subject_auroc_est"].map(lambda v: f"{v:.3f}"),
        "Brier": table["window_brier_est"].map(lambda v: f"{v:.3f}"),
        "Cal. slope": table["window_calibration_slope_est"].map(lambda v: f"{v:.2f}"),
        "ECE": table["window_ece_est"].map(lambda v: f"{v:.3f}"),
    })
    write_latex_table(show, ROOT / cfg.get("tables_dir", "paper/empirical/tables") / "e6_ablation.tex", "E6: ablation under subject-wise CV. Feature-group ablations (drop-one and only-one), SMOTE applied inside training folds, and leave-one-subject-out (LOSO) in place of five-fold CV. Window-level estimates with subject-level BCa intervals.", "tab:e6", synthetic=bool(cfg.get("synthetic")), column_format="lllrlrrrr")
    write_manifest(out, cfg, extra={"experiment": "E6", "n_rows": len(table)}, checksums_path=processed_dir(cfg) / "checksums.json")
    return table
