"""Bootstrap-replicate SDM ensemble.

Implements the manuscript's "10 replicate models with bootstrap resampling;
ensemble average" protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from .maxent_wrapper import MaxEntModel


@dataclass
class EnsembleResult:
    mean_suitability: np.ndarray
    std_suitability: np.ndarray
    mean_auc: float
    mean_tss: float


class EnsembleSDM:
    """Bootstrap-replicate ensemble of MaxEnt models."""

    def __init__(self, replicates: int | None = None, test_fraction: float | None = None):
        cfg = load_config("hyperparameters")["maxent"]
        self.replicates = replicates or cfg["replicates"]
        self.test_fraction = test_fraction or cfg["test_split_fraction"]
        self._models: list[MaxEntModel] = []

    def fit(self, presences: pd.DataFrame, background: pd.DataFrame, seed: int = 42):
        rng = np.random.default_rng(seed)
        self._models.clear()
        self.aucs, self.tsses = [], []
        for r in range(self.replicates):
            n = len(presences)
            idx = rng.choice(n, n, replace=True)
            train = presences.iloc[idx]
            mask = np.ones(n, dtype=bool)
            mask[idx] = False                                  # not strictly OOB but a separate hold-out
            test = presences.iloc[mask] if mask.sum() else presences
            model = MaxEntModel().fit(train, background)
            X_eval = pd.concat([test, background.sample(min(len(background), 1000), random_state=r)])
            y_eval = np.concatenate([np.ones(len(test)),
                                     np.zeros(min(len(background), 1000))])
            ev = model.evaluate(X_eval, y_eval)
            self.aucs.append(ev.auc)
            self.tsses.append(ev.tss)
            self._models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> EnsembleResult:
        preds = np.column_stack([m.predict(X) for m in self._models])
        return EnsembleResult(
            mean_suitability=preds.mean(axis=1),
            std_suitability=preds.std(axis=1, ddof=1),
            mean_auc=float(np.mean(self.aucs)),
            mean_tss=float(np.mean(self.tsses)),
        )
