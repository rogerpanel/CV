"""Loaders -> windows -> features -> CV on the synthetic fixture, small and fast."""
from pathlib import Path

import numpy as np
import pytest

from mlmh.data.loaders import load_cohort
from mlmh.data.schema import WindowedDataset
from mlmh.data.synthetic import generate
from mlmh.data.windowing import make_day_windows
from mlmh.evaluation.cv import oof_predictions
from mlmh.evaluation.external import assert_cohorts_disjoint, split_shared_controls
from mlmh.evaluation.splitters import shared_subjects_across_cohorts
from mlmh.features.actigraphy import build_features


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    generate(root, seed=1, n_days=4, scale=0.3)
    return root


def _dataset(root: Path, name: str) -> tuple[WindowedDataset, object]:
    minutes, subjects = load_cohort(name, root / name)
    windows = make_day_windows(minutes, min_minutes=1152)
    feats, names = build_features(windows)
    feats = feats.merge(subjects[["subject_id", "label"]], on="subject_id")
    return WindowedDataset.from_frame(feats, names, subjects=subjects), minutes


@pytest.mark.parametrize("name", ["depresjon", "psykose", "hyperaktiv", "obf_psychiatric"])
def test_each_loader_round_trips(fixture_root, name):
    ds, minutes = _dataset(fixture_root, name)
    assert ds.n_subjects >= 4 and ds.n_windows >= ds.n_subjects
    assert set(np.unique(ds.y)) == {0, 1}
    assert not np.isnan(ds.X).all(axis=0).any(), "a feature is all-NaN"


def test_shared_controls_detected_and_resolved(fixture_root):
    a, ma = _dataset(fixture_root, "depresjon")
    b, mb = _dataset(fixture_root, "psykose")
    dup = shared_subjects_across_cohorts({"depresjon": ma, "psykose": mb})
    assert dup["series_hash"].nunique() > 0, "fixture should share controls like the real data"
    with pytest.raises(RuntimeError):
        assert_cohorts_disjoint(a, b)
    a2, b2, table = split_shared_controls(a, b, seed=0)
    assert_cohorts_disjoint(a2, b2)
    assert (a2.subject_labels() == 0).sum() > 0 and (b2.subject_labels() == 0).sum() > 0


def test_cv_runs_and_record_wise_beats_subject_wise_on_fingerprinted_data(fixture_root):
    from mlmh.evaluation.metrics import binary_metrics

    ds, _ = _dataset(fixture_root, "depresjon")
    sw = oof_predictions(ds, "rf", "subject_wise", seeds=[0], n_splits=4)
    rw = oof_predictions(ds, "rf", "record_wise", seeds=[0], n_splits=4)
    auc_sw = binary_metrics(sw["y"], sw["p"])["auroc"]
    auc_rw = binary_metrics(rw["y"], rw["p"])["auroc"]
    assert auc_rw >= auc_sw - 0.02  # leakage should not make things worse; usually markedly better
