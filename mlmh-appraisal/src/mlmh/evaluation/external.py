"""Train on cohort A, evaluate unchanged on cohort B."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import WindowedDataset
from ..models.pipelines import make_pipeline


def assert_cohorts_disjoint(train: WindowedDataset, test: WindowedDataset) -> None:
    """Fail if any participant is in both cohorts, by identifier or by activity-series hash."""
    ids = set(train.subject_id) & set(test.subject_id)
    if ids:
        raise RuntimeError(f"{len(ids)} subject_id(s) appear in both cohorts, e.g. {sorted(ids)[:3]}")
    if len(train.subjects) and len(test.subjects) and "series_hash" in train.subjects and "series_hash" in test.subjects:
        h = set(train.subjects["series_hash"]) & set(test.subjects["series_hash"])
        if h:
            a = train.subjects[train.subjects["series_hash"].isin(h)]["subject_id"].tolist()[:3]
            raise RuntimeError(f"{len(h)} participant(s) share an identical activity series across cohorts (e.g. {a}); the PSYKOSE and DEPRESJON control groups are the same people")


def external_predictions(train: WindowedDataset, test: WindowedDataset, model_name: str, seeds: list[int], resample: str | None = None) -> pd.DataFrame:
    assert_cohorts_disjoint(train, test)
    rows = []
    n_classes = int(len(np.unique(train.y)))
    for seed in seeds:
        pipe = make_pipeline(model_name, seed=seed, resample=resample, n_classes=n_classes)
        pipe.fit(train.X, train.y)
        p = pipe.predict_proba(test.X)[:, 1]
        rows.append(pd.DataFrame({"seed": seed, "fold": -1, "window_id": test.window_id, "subject_id": test.subject_id, "cohort": test.cohort, "y": test.y, "p": p}))
    return pd.concat(rows, ignore_index=True)


def split_shared_controls(a: WindowedDataset, b: WindowedDataset, seed: int = 0) -> tuple[WindowedDataset, WindowedDataset, pd.DataFrame]:
    """Assign participants that appear in both cohorts to exactly one of them.

    Returns the two disjoint datasets and a table of the assignments.  Used when
    the two cohorts share a control group (DEPRESJON / PSYKOSE): half the
    shared controls stay with A and half go to B, chosen at random with the
    given seed, so both cohorts keep both classes and no participant crosses.
    """
    if not (len(a.subjects) and len(b.subjects)):
        return a, b, pd.DataFrame()
    ha = a.subjects.set_index("series_hash")["subject_id"]
    hb = b.subjects.set_index("series_hash")["subject_id"]
    shared = sorted(set(ha.index) & set(hb.index))
    if not shared:
        return a, b, pd.DataFrame()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(shared))
    to_a = {shared[i] for i in perm[: len(shared) // 2]}
    drop_from_a = [ha[h] for h in shared if h not in to_a]
    drop_from_b = [hb[h] for h in shared if h in to_a]
    table = pd.DataFrame({"series_hash": shared, "kept_in": [a.cohort[0] if h in to_a else b.cohort[0] for h in shared]})
    return a.drop_subjects(drop_from_a), b.drop_subjects(drop_from_b), table
