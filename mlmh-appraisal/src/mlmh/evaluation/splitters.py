"""Cross-validation splitters.

``SubjectWiseSplit`` is the default everywhere.  ``RecordWiseSplit`` exists only
so that experiment E1 can measure the optimism it produces; it must never be
used outside E1 (``tests/test_no_subject_leakage.py`` enforces this for every
non-E1 config).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold


class SubjectWiseSplit:
    """Stratified k-fold over *subjects*: no subject appears in both train and test.

    Stratification is by the subject-level label, so each fold has cases and
    controls in roughly the population proportion.
    """

    name = "subject_wise"
    leaks_subjects = False

    def __init__(self, n_splits: int = 5, random_state: int | None = 0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        y = np.asarray(y)
        groups = np.asarray(groups)
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in sgkf.split(X, y, groups):
            _assert_disjoint(groups[tr], groups[te])
            yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class LeaveOneSubjectOut:
    """One subject per test fold.  The classical LOSO / leave-one-patient-out design."""

    name = "loso"
    leaks_subjects = False

    def split(self, X, y, groups) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = np.asarray(groups)
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            _assert_disjoint(groups[tr], groups[te])
            yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(len(np.unique(groups)))


class RecordWiseSplit:
    """Stratified k-fold over *windows*, ignoring subject identity.

    Windows from the same participant land in both train and test.  This is the
    leaky design E1 measures.  It is intentionally loud about what it is.
    """

    name = "record_wise"
    leaks_subjects = True

    def __init__(self, n_splits: int = 5, random_state: int | None = 0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in skf.split(X, y):
            yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


SPLITTERS = {
    "subject_wise": SubjectWiseSplit,
    "loso": LeaveOneSubjectOut,
    "record_wise": RecordWiseSplit,
}


def make_splitter(name: str, n_splits: int = 5, random_state: int | None = 0):
    if name not in SPLITTERS:
        raise KeyError(f"unknown splitter {name!r}; choose from {sorted(SPLITTERS)}")
    cls = SPLITTERS[name]
    if cls is LeaveOneSubjectOut:
        return cls()
    return cls(n_splits=n_splits, random_state=random_state)


def _assert_disjoint(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    overlap = np.intersect1d(np.unique(train_groups), np.unique(test_groups))
    if len(overlap):
        raise RuntimeError(f"subject leakage: {len(overlap)} subject(s) in both train and test, e.g. {overlap[:3]}")


def subject_overlap(split_iter, groups) -> int:
    """Return the number of (fold, subject) pairs where a subject crosses the boundary."""
    groups = np.asarray(groups)
    n = 0
    for tr, te in split_iter:
        n += len(np.intersect1d(np.unique(groups[tr]), np.unique(groups[te])))
    return n


def shared_subjects_across_cohorts(minute_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Detect participants whose activity series appear in more than one cohort.

    The Simula PSYKOSE control group is the same 32 people as the DEPRESJON
    control group.  Training on one cohort and 'externally' validating on the
    other therefore leaks those participants unless they are removed from the
    test side.  We detect this from the data itself by hashing each subject's
    activity series, so the check does not rely on documentation being right.
    """
    import hashlib

    rows = []
    for cohort, table in minute_tables.items():
        for sid, g in table.groupby("subject_id"):
            act = g.sort_values("timestamp")["activity"].to_numpy()
            h = hashlib.sha256(np.ascontiguousarray(act, dtype=np.float64).tobytes()).hexdigest()
            rows.append({"cohort": cohort, "subject_id": sid, "series_hash": h, "n_minutes": len(act)})
    df = pd.DataFrame(rows)
    dup = df[df.duplicated("series_hash", keep=False)].sort_values(["series_hash", "cohort"])
    return dup.reset_index(drop=True)
