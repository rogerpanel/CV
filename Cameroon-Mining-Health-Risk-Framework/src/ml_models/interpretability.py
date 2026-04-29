"""Model interpretability via SHAP and permutation importance."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def shap_summary(model, X: pd.DataFrame, max_display: int = 20) -> Any:
    """Compute SHAP TreeExplainer / KernelExplainer values.

    Returns the SHAP values; plots can be produced with ``shap.summary_plot``.
    """
    try:
        import shap
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("shap is required: pip install shap") from err

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
    except Exception:
        # Fallback for non-tree models
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X, 50))
        values = explainer.shap_values(X)
    return values


def permutation_importance_df(model, X, y, n_repeats: int = 30, random_state: int = 42,
                              scoring: str = "r2") -> pd.DataFrame:
    """Permutation importance with bootstrapped uncertainty."""
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    feature_names = list(getattr(X, "columns", [f"x{i}" for i in range(X.shape[1])]))
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df
