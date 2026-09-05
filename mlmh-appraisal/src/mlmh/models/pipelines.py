"""All preprocessing lives inside an imblearn Pipeline.

Imputation and scaling are fitted on the training fold only; SMOTE (when
enabled) runs only on the training fold because imblearn's Pipeline skips
samplers at predict time.  There is no ``fit_transform`` on the full dataset
anywhere in this codebase; ``tests/test_pipeline_fit_order.py`` guards it.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .registry import MODEL_INPUT, make_model


def make_pipeline(model_name: str, seed: int = 0, resample: str | None = None, n_classes: int = 2) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if MODEL_INPUT.get(model_name) == "raw":
        steps = [("impute", SimpleImputer(strategy="constant", fill_value=0.0))]
    if resample == "smote":
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=5)))
    elif resample not in (None, "none", False):
        raise ValueError(f"unknown resample {resample!r}")
    steps.append(("clf", make_model(model_name, seed=seed, n_classes=n_classes)))
    return Pipeline(steps)
