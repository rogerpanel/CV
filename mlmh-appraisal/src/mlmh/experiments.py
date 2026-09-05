"""Experiment runners E1, E2, E3 and the data preparation step."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data.io import sha256_file
from .data.loaders import load_cohort
from .data.schema import WindowedDataset
from .data.windowing import make_day_windows
from .evaluation.bootstrap import paired_subject_bootstrap_diff, subject_bootstrap_ci
from .evaluation.cv import oof_predictions, seed_averaged
from .evaluation.external import external_predictions, split_shared_controls
from .evaluation.metrics import all_metrics, binary_metrics, subject_level
from .evaluation.splitters import shared_subjects_across_cohorts
from .features.actigraphy import build_features
from .models.registry import MODEL_INPUT, MODEL_LABELS
from .reporting.figures import inflation_plot, reliability_plot, roc_plot
from .reporting.manifest import write_manifest
from .reporting.tables import fmt_ci, write_latex_table

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base_path = path.parent / cfg.get("inherit", "base.yaml")
    if base_path.exists() and base_path.resolve() != path.resolve():
        base = yaml.safe_load(base_path.read_text()) or {}
        merged = {**base, **cfg}
    else:
        merged = cfg
    merged["_config_path"] = str(path)
    return merged


def data_root(cfg: dict) -> Path:
    return ROOT / (cfg.get("synthetic_root", "data/synthetic") if cfg.get("synthetic") else cfg.get("data_root", "data/raw"))


def processed_dir(cfg: dict) -> Path:
    d = ROOT / cfg.get("processed_dir", "data/processed") / ("synthetic" if cfg.get("synthetic") else "real")
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(cfg: dict, experiment: str) -> Path:
    d = ROOT / cfg.get("results_dir", "results") / ("synthetic" if cfg.get("synthetic") else "real") / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- prepare
def prepare(cfg: dict, cohorts: list[str] | None = None) -> dict[str, dict]:
    """Load raw cohorts, window, featurise and cache to data/processed/.

    Also records SHA-256 checksums of every raw file and the cross-cohort
    shared-participant table (activity-series hashes).
    """
    root = data_root(cfg)
    out = processed_dir(cfg)
    cohorts = cohorts or cfg["cohorts"]
    summary, checksums, minute_tables = {}, {}, {}
    for name in cohorts:
        croot = root / name
        if not croot.exists():
            print(f"[prepare] {name}: {croot} not found -- skipped (see data/README.md)")
            continue
        loader_kwargs = cfg.get("loader_kwargs", {}).get(name, {})
        minutes, subjects = load_cohort(name, croot, **loader_kwargs)
        for f in sorted(croot.rglob("*.csv")):
            checksums[str(f.relative_to(root))] = sha256_file(f)
        windows = make_day_windows(minutes, min_minutes=cfg.get("min_minutes_per_day", 1152), drop_edge_days=cfg.get("drop_edge_days", True))
        minute_tables[name] = minutes
        for rep in cfg.get("representations", ["engineered"]):
            feats, names = build_features(windows, representation=rep)
            feats = feats.merge(subjects[["subject_id", "label", "group"]], on="subject_id", how="left")
            feats.to_parquet(out / f"{name}.{rep}.parquet") if _has_parquet() else feats.to_csv(out / f"{name}.{rep}.csv", index=False)
            (out / f"{name}.{rep}.features.json").write_text(json.dumps(names))
        subjects.to_csv(out / f"{name}.subjects.csv", index=False)
        kept = windows.drop_duplicates("window_id").groupby("subject_id").size()
        summary[name] = {
            "n_subjects": int(subjects.shape[0]),
            "n_cases": int(subjects["label"].sum()),
            "n_windows": int(windows["window_id"].nunique()),
            "windows_per_subject_median": float(kept.median()),
            "subjects_with_no_valid_window": int((~subjects["subject_id"].isin(kept.index)).sum()),
        }
        print(f"[prepare] {name}: {summary[name]}")
    (out / "checksums.json").write_text(json.dumps(checksums, indent=1))
    if len(minute_tables) > 1:
        dup = shared_subjects_across_cohorts(minute_tables)
        dup.to_csv(out / "shared_subjects.csv", index=False)
        if len(dup):
            print(f"[prepare] WARNING: {dup['series_hash'].nunique()} participant(s) appear in more than one cohort (see shared_subjects.csv). E2 handles this with policy '{cfg.get('shared_control_policy', 'split')}'.")
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def load_processed(cfg: dict, cohort: str, representation: str = "engineered") -> WindowedDataset:
    out = processed_dir(cfg)
    p_parq, p_csv = out / f"{cohort}.{representation}.parquet", out / f"{cohort}.{representation}.csv"
    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"{cohort} ({representation}) not prepared; run `python -m mlmh prepare` first")
    names = json.loads((out / f"{cohort}.{representation}.features.json").read_text())
    subjects = pd.read_csv(out / f"{cohort}.subjects.csv")
    return WindowedDataset.from_frame(df, names, subjects=subjects)


# --------------------------------------------------------------- helpers
def _metric_fn(key: str, level: str):
    """Single-metric evaluator (cheaper than all_metrics inside the bootstrap loop)."""
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

    from .evaluation.metrics import calibration_intercept, calibration_slope, expected_calibration_error

    def fn(pred: pd.DataFrame) -> float:
        frame = subject_level(pred) if level == "subject" else pred
        y, p = frame["y"].to_numpy().astype(int), frame["p"].to_numpy(dtype=float)
        if key == "auroc":
            return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        if key == "accuracy":
            return float(accuracy_score(y, (p >= 0.5).astype(int)))
        if key == "macro_f1":
            return float(f1_score(y, (p >= 0.5).astype(int), average="macro"))
        if key == "brier":
            return float(brier_score_loss(y, p))
        if key == "calibration_slope":
            return calibration_slope(y, p)
        if key == "calibration_intercept":
            return calibration_intercept(y, p)
        if key == "ece":
            return expected_calibration_error(y, p)
        return binary_metrics(y, p)[key]

    return fn


def _summarise(pred: pd.DataFrame, cfg: dict, seed_for_boot: int = 0) -> dict:
    """Per-seed mean/sd for every metric plus subject-bootstrap BCa CIs for the primary ones."""
    per_seed = pd.DataFrame([all_metrics(g) for _, g in pred.groupby("seed")])
    row = {}
    for c in per_seed.columns:
        row[f"{c}_mean"] = float(per_seed[c].mean())
        row[f"{c}_sd"] = float(per_seed[c].std(ddof=1)) if len(per_seed) > 1 else 0.0
    avg = seed_averaged(pred)
    n_boot = int(cfg.get("n_boot", 1000))
    for key in cfg.get("ci_metrics", ["auroc", "accuracy", "brier", "calibration_slope"]):
        for level in ("window", "subject"):
            est, lo, hi = subject_bootstrap_ci(avg, _metric_fn(key, level), n_boot=n_boot, seed=seed_for_boot, method=cfg.get("ci_method", "bca"))
            row[f"{level}_{key}_ci_lo"], row[f"{level}_{key}_ci_hi"] = lo, hi
    return row


def _epv(ds: WindowedDataset) -> tuple[int, int, float]:
    labels = ds.subject_labels()
    events = int(min(labels.sum(), len(labels) - labels.sum()))
    n_pred = int(ds.X.shape[1])
    return events, n_pred, events / n_pred if n_pred else np.nan


# --------------------------------------------------------------------- E1
def run_e1(cfg: dict) -> pd.DataFrame:
    run_name = cfg.get("run_name", "E1")
    out = results_dir(cfg, run_name)
    (out / "predictions").mkdir(exist_ok=True)
    seeds = list(cfg["seeds"])
    rows, preds_store = [], {}
    for cohort in cfg["cohorts"]:
        for model in cfg["models"]:
            rep = MODEL_INPUT[model]
            try:
                ds = load_processed(cfg, cohort, rep)
            except FileNotFoundError as e:
                print(f"[E1] skip {cohort}/{model}: {e}")
                continue
            arm = {}
            for splitter in cfg["splitters"]:
                pred = oof_predictions(ds, model, splitter, seeds, n_splits=cfg.get("n_splits", 5), resample=cfg.get("resample"))
                pred.to_csv(out / "predictions" / f"{cohort}.{model}.{splitter}.csv", index=False)
                arm[splitter] = pred
                preds_store[(cohort, model, splitter)] = seed_averaged(pred)
                row = {"cohort": cohort, "model": model, "splitter": splitter, "n_subjects": ds.n_subjects, "n_windows": ds.n_windows}
                row.update(_summarise(pred, cfg))
                rows.append(row)
                print(f"[E1] {cohort:>15} {model:>9} {splitter:>12}  AUROC(window)={row['window_auroc_mean']:.3f}  AUROC(subject)={row['subject_auroc_mean']:.3f}  acc={row['window_accuracy_mean']:.3f}")
            # paired inflation CI
            if "record_wise" in arm and "subject_wise" in arm:
                for key in ("auroc", "accuracy", "macro_f1"):
                    for level in ("window", "subject"):
                        est, lo, hi = paired_subject_bootstrap_diff(seed_averaged(arm["record_wise"]), seed_averaged(arm["subject_wise"]), _metric_fn(key, level), n_boot=int(cfg.get("n_boot", 1000)))
                        rows.append({"cohort": cohort, "model": model, "splitter": "inflation", f"{level}_{key}_mean": est, f"{level}_{key}_ci_lo": lo, f"{level}_{key}_ci_hi": hi})
    table = pd.DataFrame(rows)
    table.to_csv(out / "e1_results.csv", index=False)
    suffix = "" if run_name == "E1" else "_" + run_name.lower()
    _e1_tables(table, cfg, out, suffix)
    _e1_figures(preds_store, table, cfg, out, suffix)
    write_manifest(out, cfg, extra={"experiment": run_name, "n_rows": len(table)}, checksums_path=processed_dir(cfg) / "checksums.json")
    return table


def _e1_tables(table: pd.DataFrame, cfg: dict, out: Path, suffix: str = "") -> None:
    syn = bool(cfg.get("synthetic"))
    tdir = ROOT / cfg.get("tables_dir", "paper/empirical/tables")
    main = table[table["splitter"].isin(["subject_wise", "record_wise"])]
    rows = []
    for (cohort, model), g in main.groupby(["cohort", "model"], sort=False):
        sw = g[g["splitter"] == "subject_wise"].iloc[0] if (g["splitter"] == "subject_wise").any() else None
        rw = g[g["splitter"] == "record_wise"].iloc[0] if (g["splitter"] == "record_wise").any() else None
        inf = table[(table["cohort"] == cohort) & (table["model"] == model) & (table["splitter"] == "inflation")]
        r = {"Cohort": cohort, "Model": MODEL_LABELS.get(model, model)}
        for level, lab in (("window", "Window"), ("subject", "Subject")):
            if sw is not None:
                r[f"{lab} AUROC subject-wise"] = fmt_ci(sw[f"{level}_auroc_mean"], sw.get(f"{level}_auroc_ci_lo"), sw.get(f"{level}_auroc_ci_hi"))
            if rw is not None:
                r[f"{lab} AUROC record-wise"] = fmt_ci(rw[f"{level}_auroc_mean"], rw.get(f"{level}_auroc_ci_lo"), rw.get(f"{level}_auroc_ci_hi"))
            if len(inf):
                col = f"{level}_auroc_mean"
                ii = inf[inf[col].notna()]
                if len(ii):
                    r[f"{lab} inflation"] = fmt_ci(ii.iloc[0][col], ii.iloc[0][f"{level}_auroc_ci_lo"], ii.iloc[0][f"{level}_auroc_ci_hi"])
        rows.append(r)
    df = pd.DataFrame(rows)
    write_latex_table(df, tdir / f"e1_auroc_inflation{suffix}.tex", "E1: discrimination under subject-wise versus record-wise cross-validation. Cells give the mean over seeds with subject-level BCa 95\\% bootstrap intervals; inflation is the paired record-wise minus subject-wise difference.", f"tab:e1{suffix}", synthetic=syn)
    # accuracy/F1/calibration companion
    rows = []
    for _, r in main.iterrows():
        rows.append(
            {
                "Cohort": r["cohort"],
                "Model": MODEL_LABELS.get(r["model"], r["model"]),
                "Split": r["splitter"].replace("_", "-"),
                "Accuracy": f"{r['window_accuracy_mean']:.3f}",
                "Macro-F1": f"{r['window_macro_f1_mean']:.3f}",
                "AUROC": f"{r['window_auroc_mean']:.3f}",
                "Brier": f"{r['window_brier_mean']:.3f}",
                "Cal. slope": f"{r['window_calibration_slope_mean']:.2f}",
                "Cal. intercept": f"{r['window_calibration_intercept_mean']:.2f}",
                "ECE": f"{r['window_ece_mean']:.3f}",
            }
        )
    write_latex_table(pd.DataFrame(rows), tdir / f"e1_full_metrics{suffix}.tex", "E1: window-level discrimination and calibration for every model, cohort and splitting design (mean over seeds).", f"tab:e1full{suffix}", synthetic=syn)


def _e1_figures(preds_store: dict, table: pd.DataFrame, cfg: dict, out: Path, suffix: str = "") -> None:
    syn = bool(cfg.get("synthetic"))
    fdir = ROOT / cfg.get("figures_dir", "paper/empirical/figures")
    main = table[table["splitter"].isin(["subject_wise", "record_wise"])]
    if main.empty:
        return
    piv = main.pivot_table(index=["cohort", "model"], columns="splitter", values="window_auroc_mean").reset_index()
    piv.columns = [c if isinstance(c, str) else c for c in piv.columns]
    piv = piv.rename(columns={"subject_wise": "auroc_subject_wise", "record_wise": "auroc_record_wise"})
    if {"auroc_subject_wise", "auroc_record_wise"} <= set(piv.columns):
        inflation_plot(piv, fdir / f"e1_inflation_auroc{suffix}.pdf", synthetic=syn)
    for cohort in cfg["cohorts"]:
        sub = {f"{MODEL_LABELS.get(m, m)} / {s.replace('_', '-')}": p for (c, m, s), p in preds_store.items() if c == cohort and m != "majority"}
        if sub:
            reliability_plot(sub, fdir / f"e1_reliability_{cohort}{suffix}.pdf", title=cohort, synthetic=syn)
            roc_plot(sub, fdir / f"e1_roc_{cohort}{suffix}.pdf", title=cohort, synthetic=syn)


# --------------------------------------------------------------------- E2
def run_e2(cfg: dict) -> pd.DataFrame:
    out = results_dir(cfg, "E2")
    (out / "predictions").mkdir(exist_ok=True)
    seeds = list(cfg["seeds"])
    rows, preds_store = [], {}
    for pair in cfg["pairs"]:
        a_name, b_name = pair["train"], pair["test"]
        for model in cfg["models"]:
            rep = MODEL_INPUT[model]
            try:
                a, b = load_processed(cfg, a_name, rep), load_processed(cfg, b_name, rep)
            except FileNotFoundError as e:
                print(f"[E2] skip {a_name}->{b_name}/{model}: {e}")
                continue
            policy = cfg.get("shared_control_policy", "split")
            assign = pd.DataFrame()
            if policy == "split":
                a, b, assign = split_shared_controls(a, b, seed=int(cfg.get("shared_control_seed", 0)))
            elif policy == "drop_from_test":
                shared = set(a.subjects["series_hash"]) & set(b.subjects["series_hash"])
                b = b.drop_subjects(b.subjects[b.subjects["series_hash"].isin(shared)]["subject_id"])
            if len(assign):
                assign.to_csv(out / f"shared_control_assignment.{a_name}-{b_name}.csv", index=False)
            # internal estimate on A (subject-wise CV)
            internal = oof_predictions(a, model, "subject_wise", seeds, n_splits=cfg.get("n_splits", 5), resample=cfg.get("resample"))
            external = external_predictions(a, b, model, seeds, resample=cfg.get("resample"))
            internal.to_csv(out / "predictions" / f"{a_name}.{model}.internal.csv", index=False)
            external.to_csv(out / "predictions" / f"{a_name}-to-{b_name}.{model}.external.csv", index=False)
            preds_store[(a_name, b_name, model, "internal")] = seed_averaged(internal)
            preds_store[(a_name, b_name, model, "external")] = seed_averaged(external)
            for arm, pred, ds in (("internal", internal, a), ("external", external, b)):
                ev, npred, epv = _epv(a)
                row = {"train": a_name, "test": b_name, "model": model, "arm": arm, "n_subjects_eval": ds.n_subjects, "n_windows_eval": ds.n_windows, "train_events": ev, "n_predictors": npred, "epv": epv}
                row.update(_summarise(pred, cfg))
                rows.append(row)
                print(f"[E2] {a_name:>10}->{b_name:<10} {model:>9} {arm:>8}  AUROC(window)={row['window_auroc_mean']:.3f} AUROC(subject)={row['subject_auroc_mean']:.3f} slope={row['window_calibration_slope_mean']:.2f} int={row['window_calibration_intercept_mean']:.2f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "e2_results.csv", index=False)
    _e2_tables(table, cfg)
    _e2_figures(preds_store, cfg)
    write_manifest(out, cfg, extra={"experiment": "E2", "n_rows": len(table)}, checksums_path=processed_dir(cfg) / "checksums.json")
    return table


def _e2_tables(table: pd.DataFrame, cfg: dict) -> None:
    syn = bool(cfg.get("synthetic"))
    tdir = ROOT / cfg.get("tables_dir", "paper/empirical/tables")
    rows = []
    for (a, b, m), g in table.groupby(["train", "test", "model"], sort=False):
        i = g[g["arm"] == "internal"].iloc[0]
        e = g[g["arm"] == "external"].iloc[0]
        rows.append(
            {
                "Train $\\rightarrow$ Test": f"{a} $\\rightarrow$ {b}",
                "Model": MODEL_LABELS.get(m, m),
                "EPV": f"{i['epv']:.1f}",
                "Internal AUROC": fmt_ci(i["window_auroc_mean"], i.get("window_auroc_ci_lo"), i.get("window_auroc_ci_hi")),
                "External AUROC": fmt_ci(e["window_auroc_mean"], e.get("window_auroc_ci_lo"), e.get("window_auroc_ci_hi")),
                "$\\Delta$": f"{e['window_auroc_mean'] - i['window_auroc_mean']:+.3f}",
                "Int. slope": f"{i['window_calibration_slope_mean']:.2f}",
                "Ext. slope": f"{e['window_calibration_slope_mean']:.2f}",
                "Ext. intercept": f"{e['window_calibration_intercept_mean']:.2f}",
                "Ext. Brier": f"{e['window_brier_mean']:.3f}",
            }
        )
    df = pd.DataFrame(rows)
    path = tdir / "e2_external_validation.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=False, na_rep="--")
    cap = ("E2: internal (subject-wise CV on the training cohort) versus external (unchanged model applied to the second cohort) performance. "
           "Window-level estimates, mean over seeds, subject-level BCa 95\\% intervals. EPV = minority-class subjects per candidate predictor in the training cohort.")
    if syn:
        cap = "[SYNTHETIC FIXTURE -- NOT A RESULT] " + cap
    path.write_text("% Auto-generated by mlmh -- do not edit by hand.\n\\begin{table}[!htbp]\n\\centering\n" f"\\caption{{{cap}}}\n\\label{{tab:e2}}\n\\footnotesize\n{body.strip()}\n\\end{{table}}\n")


def _e2_figures(preds_store: dict, cfg: dict) -> None:
    syn = bool(cfg.get("synthetic"))
    fdir = ROOT / cfg.get("figures_dir", "paper/empirical/figures")
    pairs = {(a, b) for (a, b, _, _) in preds_store}
    for a, b in pairs:
        sub = {f"{MODEL_LABELS.get(m, m)} / {arm}": p for (aa, bb, m, arm), p in preds_store.items() if (aa, bb) == (a, b) and m != "majority"}
        if sub:
            reliability_plot(sub, fdir / f"e2_reliability_{a}_to_{b}.pdf", title=f"{a} $\\rightarrow$ {b}", synthetic=syn)
            roc_plot(sub, fdir / f"e2_roc_{a}_to_{b}.pdf", title=f"{a} $\\rightarrow$ {b}", synthetic=syn)


# --------------------------------------------------------------------- E3
def run_e3(cfg: dict) -> pd.DataFrame:
    """Calibration report over every stored E1 (subject-wise arm) and E2 prediction set.

    Optionally adds a recalibration arm: logistic (Platt) recalibration fitted
    on the training cohort's out-of-fold predictions and applied to the external
    cohort, to show how much of the external miscalibration is fixable without
    refitting the model.
    """
    from sklearn.linear_model import LogisticRegression

    from .evaluation.metrics import _logit

    out = results_dir(cfg, "E3")
    syn = bool(cfg.get("synthetic"))
    base = out.parent
    rows, curves = [], {}
    for path in sorted(base.glob("E1*/predictions/*.subject_wise.csv")) + sorted((base / "E2" / "predictions").glob("*.csv")):
        pred = seed_averaged(pd.read_csv(path))
        name = path.stem
        m = binary_metrics(pred["y"], pred["p"])
        rows.append({"source": path.parent.parent.name, "run": name, "level": "window", **m})
        s = subject_level(pred)
        rows.append({"source": path.parent.parent.name, "run": name, "level": "subject", **binary_metrics(s["y"], s["p"])})
        curves[name] = pred
    # recalibration arm
    if cfg.get("recalibrate", True):
        for ext in sorted((base / "E2" / "predictions").glob("*.external.csv")):
            stem = ext.stem  # a-to-b.model.external
            a_to_b, model, _ = stem.split(".")
            a = a_to_b.split("-to-")[0]
            internal = base / "E2" / "predictions" / f"{a}.{model}.internal.csv"
            if not internal.exists():
                continue
            pi, pe = seed_averaged(pd.read_csv(internal)), seed_averaged(pd.read_csv(ext))
            if pi["y"].nunique() < 2:
                continue
            lr = LogisticRegression(C=1e6, max_iter=5000).fit(_logit(pi["p"]).reshape(-1, 1), pi["y"])
            pe2 = pe.copy()
            pe2["p"] = lr.predict_proba(_logit(pe["p"]).reshape(-1, 1))[:, 1]
            rows.append({"source": "E2", "run": f"{stem}+recal", "level": "window", **binary_metrics(pe2["y"], pe2["p"])})
            curves[f"{stem}+recal"] = pe2
    table = pd.DataFrame(rows)
    table.to_csv(out / "e3_calibration.csv", index=False)
    tdir = ROOT / cfg.get("tables_dir", "paper/empirical/tables")
    show = table[table["level"] == "window"][["source", "run", "auroc", "brier", "calibration_slope", "calibration_intercept", "ece"]].copy()
    show.columns = ["Exp.", "Run", "AUROC", "Brier", "Cal. slope", "Cal. intercept", "ECE"]
    for c in ("AUROC", "Brier", "ECE"):
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    for c in ("Cal. slope", "Cal. intercept"):
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    write_latex_table(show, tdir / "e3_calibration.tex", "E3: calibration alongside discrimination for every model in E1 (subject-wise arm) and E2, plus logistic recalibration of external predictions.", "tab:e3", synthetic=syn, column_format="llrrrrr")
    fdir = ROOT / cfg.get("figures_dir", "paper/empirical/figures")
    pairs = sorted({k.split(".")[0] for k in curves if "external" in k})
    for pair in pairs:
        ext = {k.split(".", 1)[1].replace(".external", ""): v for k, v in curves.items() if k.startswith(pair + ".") and "external" in k and "majority" not in k}
        if ext:
            reliability_plot(ext, fdir / f"e3_reliability_{pair.replace('-to-', '_to_')}.pdf", title=pair.replace("-to-", " $\\rightarrow$ "), synthetic=syn)
    write_manifest(out, cfg, extra={"experiment": "E3", "n_rows": len(table)}, checksums_path=processed_dir(cfg) / "checksums.json")
    return table
