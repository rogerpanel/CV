"""THE critical test.

1. SubjectWiseSplit and LeaveOneSubjectOut never put a subject on both sides.
2. RecordWiseSplit *does* (so that E1 is actually measuring something).
3. Every experiment config on disk, except those explicitly flagged as E1,
   uses a non-leaking splitter.  A config that requests ``record_wise`` outside
   E1 fails the build.
4. Cross-cohort external validation refuses to run when a subject in the
   training cohort also appears in the test cohort.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlmh.data.schema import WindowedDataset
from mlmh.evaluation.splitters import (
    LeaveOneSubjectOut,
    RecordWiseSplit,
    SubjectWiseSplit,
    make_splitter,
    shared_subjects_across_cohorts,
    subject_overlap,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixture(n_subjects=30, windows_per_subject=8, seed=0):
    rng = np.random.default_rng(seed)
    subj = np.repeat([f"s{i:03d}" for i in range(n_subjects)], windows_per_subject)
    y_subj = rng.integers(0, 2, size=n_subjects)
    y = np.repeat(y_subj, windows_per_subject)
    X = rng.normal(size=(len(y), 6))
    return X, y, subj


@pytest.mark.parametrize("splitter", [SubjectWiseSplit(n_splits=5, random_state=1), LeaveOneSubjectOut()])
def test_subject_wise_never_leaks(splitter):
    X, y, groups = _fixture()
    n_leaks = subject_overlap(splitter.split(X, y, groups), groups)
    assert n_leaks == 0


def test_subject_wise_uses_every_window_once():
    X, y, groups = _fixture()
    seen = np.zeros(len(y), dtype=int)
    for _, te in SubjectWiseSplit(n_splits=5, random_state=3).split(X, y, groups):
        seen[te] += 1
    assert (seen == 1).all()


def test_record_wise_does_leak():
    """If this fails, E1 is not measuring leakage and its result is meaningless."""
    X, y, groups = _fixture()
    n_leaks = subject_overlap(RecordWiseSplit(n_splits=5, random_state=1).split(X, y, groups), groups)
    assert n_leaks > 0
    assert RecordWiseSplit.leaks_subjects is True
    assert SubjectWiseSplit.leaks_subjects is False


def test_every_non_e1_config_uses_non_leaking_splitter():
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict) or "experiment" not in cfg:
            continue  # base.yaml carries defaults only
        is_e1 = cfg.get("experiment") == "E1"
        splitters = cfg.get("splitters", [cfg.get("splitter", "subject_wise")])
        for name in splitters:
            leaks = make_splitter(name).leaks_subjects
            if leaks and not is_e1:
                pytest.fail(f"{path.name} uses leaking splitter {name!r} outside E1")


def test_schema_rejects_subject_with_two_labels():
    X, y, groups = _fixture(n_subjects=4, windows_per_subject=2)
    y = y.copy()
    y[0], y[1] = 0, 1  # same subject, two labels
    with pytest.raises(ValueError):
        WindowedDataset(X=X, y=y, subject_id=groups, cohort=np.array(["c"] * len(y)), window_id=np.arange(len(y)).astype(str), feature_names=[f"f{i}" for i in range(6)])


def test_shared_subject_detection_across_cohorts():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2020-01-01", periods=100, freq="min")
    a = pd.DataFrame({"subject_id": "control_1", "cohort": "A", "timestamp": ts, "activity": rng.integers(0, 500, 100)})
    b = a.assign(cohort="B", subject_id="ctrl_01")  # same person, different naming convention
    c = pd.DataFrame({"subject_id": "control_2", "cohort": "B", "timestamp": ts, "activity": rng.integers(0, 500, 100)})
    dup = shared_subjects_across_cohorts({"A": a, "B": pd.concat([b, c])})
    assert set(dup["subject_id"]) == {"control_1", "ctrl_01"}


def test_external_runner_refuses_overlapping_subjects():
    from mlmh.evaluation.external import assert_cohorts_disjoint

    X, y, groups = _fixture(n_subjects=10, windows_per_subject=2)
    feats = [f"f{i}" for i in range(6)]
    train = WindowedDataset(X=X, y=y, subject_id=groups, cohort=np.array(["A"] * len(y)), window_id=np.array([f"a{i}" for i in range(len(y))]), feature_names=feats)
    test = WindowedDataset(X=X, y=y, subject_id=groups, cohort=np.array(["B"] * len(y)), window_id=np.array([f"b{i}" for i in range(len(y))]), feature_names=feats)
    with pytest.raises(RuntimeError):
        assert_cohorts_disjoint(train, test)
