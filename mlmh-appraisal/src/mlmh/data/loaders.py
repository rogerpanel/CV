"""One loader per dataset, each returning the same two tables.

``load_cohort(name, root)`` -> (minutes, subjects)

* ``minutes``  : long table with columns subject_id, cohort, timestamp, activity
* ``subjects`` : one row per subject with subject_id, cohort, label, group,
                 series_hash, n_minutes, n_files and any demographic columns found

Labels are binary at this layer: 1 = the cohort's patient group, 0 = its control
group.  Multiclass labelling for OBF-Psychiatric is available through
``label_scheme="group"``.

Layouts recognised (see data/README.md for the download instructions):

  depresjon/   condition/condition_*.csv  control/control_*.csv  scores.csv
  psykose/     patient/patient_*.csv (or condition/)  control/control_*.csv  schizophrenia-features.csv (optional)
  hyperaktiv/  activity_data/patient_activity_*.csv  patient_info.csv (ADHD column)
  obf_psychiatric/  adhd/  clinical/  control/  depression/  schizophrenia/  (per-subject csv in each)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_activity_csv, read_table, series_hash

COHORT_GROUPS = {
    "depresjon": {"case": "depression", "control": "healthy_control"},
    "psykose": {"case": "schizophrenia", "control": "healthy_control"},
    "hyperaktiv": {"case": "adhd", "control": "clinical_control"},
}


def _collect(files: list[Path], cohort: str, label: int, group: str, id_prefix: str = "") -> tuple[list[pd.DataFrame], list[dict]]:
    minutes, subjects = [], []
    for f in sorted(files, key=_natural_key):
        act = read_activity_csv(f)
        sid = f"{id_prefix}{f.stem}"
        act.insert(0, "cohort", cohort)
        act.insert(0, "subject_id", sid)
        minutes.append(act)
        subjects.append(
            {
                "subject_id": sid,
                "cohort": cohort,
                "label": int(label),
                "group": group,
                "series_hash": series_hash(act["activity"].to_numpy()),
                "n_minutes": int(len(act)),
                "n_days": int(act["timestamp"].dt.normalize().nunique()),
                "source_file": str(f),
            }
        )
    return minutes, subjects


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)]


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------- Simula
def load_depresjon(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    cond_dir = _first_existing(root, ["condition"])
    ctrl_dir = _first_existing(root, ["control"])
    if cond_dir is None or ctrl_dir is None:
        raise FileNotFoundError(f"{root}: expected condition/ and control/ folders")
    m1, s1 = _collect(list(cond_dir.glob("*.csv")), "depresjon", 1, "depression")
    m2, s2 = _collect(list(ctrl_dir.glob("*.csv")), "depresjon", 0, "healthy_control")
    subjects = pd.DataFrame(s1 + s2)
    scores = _first_existing(root, ["scores.csv"])
    if scores is not None:
        sc = read_table(scores)
        if "number" in sc.columns:
            sc = sc.rename(columns={"number": "subject_id"})
            sc["subject_id"] = sc["subject_id"].astype(str)
            subjects = subjects.merge(sc, on="subject_id", how="left")
    return pd.concat(m1 + m2, ignore_index=True), subjects


def load_psykose(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    pat_dir = _first_existing(root, ["patient", "condition", "patients"])
    ctrl_dir = _first_existing(root, ["control", "controls"])
    if pat_dir is None or ctrl_dir is None:
        raise FileNotFoundError(f"{root}: expected patient/ (or condition/) and control/ folders")
    m1, s1 = _collect(list(pat_dir.glob("*.csv")), "psykose", 1, "schizophrenia")
    m2, s2 = _collect(list(ctrl_dir.glob("*.csv")), "psykose", 0, "healthy_control")
    subjects = pd.DataFrame(s1 + s2)
    info = _first_existing(root, ["patients_info.csv", "scores.csv"])
    if info is not None:
        sc = read_table(info)
        if "number" in sc.columns:
            sc = sc.rename(columns={"number": "subject_id"})
            sc["subject_id"] = sc["subject_id"].astype(str)
            subjects = subjects.merge(sc, on="subject_id", how="left")
    return pd.concat(m1 + m2, ignore_index=True), subjects


def load_hyperaktiv(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    act_dir = _first_existing(root, ["activity_data", "activity"])
    info_path = _first_existing(root, ["patient_info.csv", "patients_info.csv"])
    if act_dir is None or info_path is None:
        raise FileNotFoundError(f"{root}: expected activity_data/ and patient_info.csv")
    info = read_table(info_path)
    if "id" not in info.columns or "adhd" not in info.columns:
        raise ValueError(f"{info_path}: expected ID and ADHD columns, found {list(info.columns)}")
    info["id"] = pd.to_numeric(info["id"], errors="coerce").astype("Int64")
    label_by_id = dict(zip(info["id"].astype(int), pd.to_numeric(info["adhd"], errors="coerce").fillna(0).astype(int)))
    minutes, subjects = [], []
    for f in sorted(act_dir.glob("*.csv"), key=_natural_key):
        m = re.search(r"(\d+)", f.stem)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in label_by_id:
            continue
        label = label_by_id[pid]
        mm, ss = _collect([f], "hyperaktiv", label, "adhd" if label == 1 else "clinical_control")
        minutes += mm
        subjects += ss
    subjects = pd.DataFrame(subjects)
    subjects["patient_no"] = subjects["subject_id"].str.extract(r"(\d+)").astype(int)
    extra = info.rename(columns={"id": "patient_no"})
    subjects = subjects.merge(extra, on="patient_no", how="left", suffixes=("", "_info"))
    return pd.concat(minutes, ignore_index=True), subjects


OBF_GROUPS = ["adhd", "clinical", "control", "depression", "schizophrenia"]


def load_obf_psychiatric(root: Path, label_scheme: str = "any_psychiatric_vs_control") -> tuple[pd.DataFrame, pd.DataFrame]:
    """OBF-Psychiatric (Sci Data 2025): five group folders of per-subject CSVs.

    label_scheme:
      any_psychiatric_vs_control : 1 for adhd/clinical/depression/schizophrenia, 0 for control
      group                      : multiclass 0..4 in OBF_GROUPS order
    """
    root = Path(root)
    minutes, subjects = [], []
    found = [g for g in OBF_GROUPS if (root / g).is_dir()]
    if not found:
        raise FileNotFoundError(f"{root}: none of the OBF group folders {OBF_GROUPS} found")
    for g in found:
        if label_scheme == "group":
            label = OBF_GROUPS.index(g)
        else:
            label = 0 if g == "control" else 1
        mm, ss = _collect(list((root / g).rglob("*.csv")), "obf_psychiatric", label, g, id_prefix=f"{g}_")
        minutes += mm
        subjects += ss
    return pd.concat(minutes, ignore_index=True), pd.DataFrame(subjects)


LOADERS = {
    "depresjon": load_depresjon,
    "psykose": load_psykose,
    "hyperaktiv": load_hyperaktiv,
    "obf_psychiatric": load_obf_psychiatric,
}


def load_cohort(name: str, root: Path, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name not in LOADERS:
        raise KeyError(f"unknown cohort {name!r}; known: {sorted(LOADERS)}")
    minutes, subjects = LOADERS[name](Path(root), **kwargs)
    _validate(minutes, subjects, name)
    return minutes, subjects


def _validate(minutes: pd.DataFrame, subjects: pd.DataFrame, name: str) -> None:
    for c in ("subject_id", "cohort", "timestamp", "activity"):
        if c not in minutes.columns:
            raise ValueError(f"{name}: minutes table missing {c}")
    for c in ("subject_id", "cohort", "label", "group", "series_hash"):
        if c not in subjects.columns:
            raise ValueError(f"{name}: subjects table missing {c}")
    if subjects["subject_id"].duplicated().any():
        raise ValueError(f"{name}: duplicated subject_id")
    if subjects["label"].nunique() < 2:
        raise ValueError(f"{name}: only one class present")
