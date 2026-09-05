"""Cross-validated out-of-fold predictions, repeated over seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..data.schema import WindowedDataset
from ..models.pipelines import make_pipeline
from .splitters import make_splitter


def oof_predictions(ds: WindowedDataset, model_name: str, splitter_name: str, seeds: list[int], n_splits: int = 5, resample: str | None = None) -> pd.DataFrame:
    """Return a long frame: seed, fold, window_id, subject_id, cohort, y, p."""
    rows = []
    n_classes = int(len(np.unique(ds.y)))
    for seed in seeds:
        splitter = make_splitter(splitter_name, n_splits=n_splits, random_state=seed)
        pipe = make_pipeline(model_name, seed=seed, resample=resample, n_classes=n_classes)
        for fold, (tr, te) in enumerate(splitter.split(ds.X, ds.y, ds.subject_id)):
            if not splitter.leaks_subjects:
                assert not set(ds.subject_id[tr]) & set(ds.subject_id[te]), "subject leakage detected at runtime"
            est = clone(pipe)
            est.fit(ds.X[tr], ds.y[tr])
            p = est.predict_proba(ds.X[te])[:, 1]
            rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "fold": fold,
                        "window_id": ds.window_id[te],
                        "subject_id": ds.subject_id[te],
                        "cohort": ds.cohort[te],
                        "y": ds.y[te],
                        "p": p,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def seed_averaged(pred: pd.DataFrame) -> pd.DataFrame:
    """Average the out-of-fold probability of each window across seeds."""
    g = pred.groupby(["window_id", "subject_id", "cohort"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
    return g
