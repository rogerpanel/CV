"""Common schema every dataset loader must map into.

The unit of analysis is a *window* (by default one calendar day of minute-level
activity counts).  Every window carries the identifier of the participant it was
cut from, and that identifier is what all splitting, bootstrapping and
aggregation is keyed on.  Losing ``subject_id`` anywhere in the pipeline is the
single most consequential bug this codebase can have, so the schema is explicit
and validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# Column names of the long-format per-minute table produced by every loader.
MINUTE_COLUMNS = ["subject_id", "cohort", "timestamp", "activity"]

# Column names of the per-subject metadata table.
SUBJECT_COLUMNS = ["subject_id", "cohort", "label", "group"]


@dataclass
class WindowedDataset:
    """A feature matrix with the bookkeeping needed for honest evaluation.

    Attributes
    ----------
    X : (n_windows, n_features) float array of window-level features.
    y : (n_windows,) int array of labels (0 = control, 1 = case, or multiclass).
    subject_id : (n_windows,) array of participant identifiers (str).
    cohort : (n_windows,) array of cohort names (str), e.g. "depresjon".
    window_id : (n_windows,) array of unique window identifiers (str).
    feature_names : list of feature column names, length n_features.
    subjects : per-subject metadata table (one row per subject).
    """

    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    cohort: np.ndarray
    window_id: np.ndarray
    feature_names: list[str]
    subjects: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.validate()

    # ----------------------------------------------------------------- checks
    def validate(self) -> None:
        n = len(self.y)
        if self.X.ndim != 2 or self.X.shape[0] != n:
            raise ValueError(f"X has shape {self.X.shape} but y has length {n}")
        for name in ("subject_id", "cohort", "window_id"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)} but y has length {n}")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names does not match X columns")
        if len(np.unique(self.window_id)) != n:
            raise ValueError("window_id values must be unique")
        # A subject must carry exactly one label: labels are subject-level.
        df = pd.DataFrame({"s": self.subject_id, "y": self.y})
        per_subject = df.groupby("s")["y"].nunique()
        if (per_subject > 1).any():
            bad = per_subject[per_subject > 1].index.tolist()[:5]
            raise ValueError(f"Subjects with more than one label: {bad}")

    # ---------------------------------------------------------------- helpers
    @property
    def n_subjects(self) -> int:
        return int(len(np.unique(self.subject_id)))

    @property
    def n_windows(self) -> int:
        return int(len(self.y))

    def subject_labels(self) -> pd.Series:
        """Label per subject (index = subject_id)."""
        return pd.Series(self.y, index=self.subject_id).groupby(level=0).first()

    def subset(self, mask: np.ndarray) -> "WindowedDataset":
        mask = np.asarray(mask, dtype=bool)
        return WindowedDataset(
            X=self.X[mask],
            y=self.y[mask],
            subject_id=self.subject_id[mask],
            cohort=self.cohort[mask],
            window_id=self.window_id[mask],
            feature_names=list(self.feature_names),
            subjects=self.subjects[self.subjects["subject_id"].isin(np.unique(self.subject_id[mask]))].reset_index(drop=True)
            if len(self.subjects)
            else self.subjects,
        )

    def drop_subjects(self, subject_ids: Iterable[str]) -> "WindowedDataset":
        drop = set(map(str, subject_ids))
        keep = np.array([s not in drop for s in self.subject_id])
        return self.subset(keep)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df.insert(0, "window_id", self.window_id)
        df.insert(0, "cohort", self.cohort)
        df.insert(0, "subject_id", self.subject_id)
        df["label"] = self.y
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame, feature_names: list[str], subjects: pd.DataFrame | None = None) -> "WindowedDataset":
        return cls(
            X=df[feature_names].to_numpy(dtype=float),
            y=df["label"].to_numpy(dtype=int),
            subject_id=df["subject_id"].astype(str).to_numpy(),
            cohort=df["cohort"].astype(str).to_numpy(),
            window_id=df["window_id"].astype(str).to_numpy(),
            feature_names=list(feature_names),
            subjects=subjects if subjects is not None else pd.DataFrame(),
        )


def concat(datasets: list[WindowedDataset]) -> WindowedDataset:
    """Concatenate datasets (e.g. two cohorts) keeping every identifier."""
    if not datasets:
        raise ValueError("nothing to concatenate")
    names = datasets[0].feature_names
    for d in datasets[1:]:
        if d.feature_names != names:
            raise ValueError("feature_names differ between datasets")
    return WindowedDataset(
        X=np.vstack([d.X for d in datasets]),
        y=np.concatenate([d.y for d in datasets]),
        subject_id=np.concatenate([d.subject_id for d in datasets]),
        cohort=np.concatenate([d.cohort for d in datasets]),
        window_id=np.concatenate([d.window_id for d in datasets]),
        feature_names=list(names),
        subjects=pd.concat([d.subjects for d in datasets], ignore_index=True) if all(len(d.subjects) for d in datasets) else pd.DataFrame(),
    )
