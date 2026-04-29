"""MaxEnt species-distribution model.

A self-contained logistic-MaxEnt approximation suitable for small occurrence
datasets — used for the manuscript's IUCN species (n = 5).  For full Phillips-
style MaxEnt with regularisation paths, install ``elapid`` and pass
``backend="elapid"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cmhr.utils import load_config


@dataclass
class MaxEntPrediction:
    suitability: np.ndarray
    auc: float
    tss: float
    threshold: float

    def is_good(self, cfg: dict | None = None) -> bool:
        cfg = cfg or load_config("hyperparameters")["maxent"]["evaluation"]
        return self.auc > cfg["auc_min"] and self.tss > cfg["tss_min"]


class MaxEntModel:
    """Lightweight MaxEnt-style classifier with feature engineering."""

    def __init__(self, backend: Literal["sklearn", "elapid"] = "sklearn"):
        self.cfg = load_config("hyperparameters")["maxent"]
        self.backend = backend
        self._scaler = StandardScaler()
        self._model: LogisticRegression | None = None

    # ------------------------------------------------------------ feature eng
    def _feature_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if "quadratic" in self.cfg["feature_classes"]:
            for c in X.columns:
                out[f"{c}_sq"] = X[c] ** 2
        if "hinge" in self.cfg["feature_classes"]:
            for c in X.columns:
                med = X[c].median()
                out[f"{c}_hinge"] = np.maximum(0.0, X[c] - med)
        return out

    def fit(self, presences: pd.DataFrame, background: pd.DataFrame):
        if self.backend == "elapid":
            try:
                import elapid                                  # noqa: F401
                from elapid import MaxentModel
            except ImportError as err:                                # pragma: no cover
                raise ImportError("Install elapid for full MaxEnt: pip install elapid") from err
            X = pd.concat([presences, background], ignore_index=True)
            y = np.concatenate([np.ones(len(presences)), np.zeros(len(background))])
            mx = MaxentModel(beta_multiplier=self.cfg["regularisation_multiplier"])
            mx.fit(X.values, y)
            self._model = mx
            self._presences = presences
            self._background = background
            return self

        X = pd.concat([presences, background], ignore_index=True)
        y = np.concatenate([np.ones(len(presences)), np.zeros(len(background))])
        F = self._feature_matrix(X)
        Xs = self._scaler.fit_transform(F.values)
        self._model = LogisticRegression(
            C=1.0 / self.cfg["regularisation_multiplier"],
            max_iter=self.cfg["max_iterations"],
            tol=self.cfg["convergence_threshold"],
            random_state=42,
        ).fit(Xs, y)
        self._feature_cols = list(F.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit yet")
        if self.backend == "elapid":
            return self._model.predict(X.values)
        F = self._feature_matrix(X[ [c for c in X.columns if c in self._feature_cols or f"{c}_sq" in self._feature_cols ] ])
        Xs = self._scaler.transform(F.reindex(columns=self._feature_cols, fill_value=0).values)
        return self._model.predict_proba(Xs)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test) -> MaxEntPrediction:
        proba = self.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        # Threshold at maximum (sensitivity + specificity − 1) — TSS rule
        thresholds = np.linspace(0.05, 0.95, 19)
        tss_scores = []
        for t in thresholds:
            pred = (proba >= t).astype(int)
            tp = ((pred == 1) & (y_test == 1)).sum()
            tn = ((pred == 0) & (y_test == 0)).sum()
            fp = ((pred == 1) & (y_test == 0)).sum()
            fn = ((pred == 0) & (y_test == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            tss_scores.append(sens + spec - 1)
        best = int(np.argmax(tss_scores))
        return MaxEntPrediction(proba, float(auc), float(tss_scores[best]), float(thresholds[best]))
