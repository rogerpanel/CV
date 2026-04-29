"""Adjusted logistic regression for DHS proximity → outcome analysis.

Reproduces the manuscript's covariate set: age, household education,
wealth quintile.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from cmhr.utils import load_config


def adjusted_logit(
    df: pd.DataFrame,
    outcome: str,
    proximity_col: str = "proximity_class",
    covariates: Sequence[str] | None = None,
    reference: str | None = None,
):
    """Fit a logistic regression with ``proximity_class`` as the exposure.

    Returns the fitted ``statsmodels`` results object (call ``.summary()``).
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("statsmodels is required: pip install statsmodels") from err

    cfg = load_config("hyperparameters")["dhs_logistic"]
    covariates = list(covariates or cfg["covariates"])
    reference = reference or cfg["reference_proximity_class"]

    df = df.copy()
    df[proximity_col] = pd.Categorical(
        df[proximity_col],
        categories=[reference] + [c for c in df[proximity_col].unique() if c != reference],
    )
    formula = f"{outcome} ~ C({proximity_col}) + " + " + ".join(covariates)
    model = smf.logit(formula, data=df).fit(disp=False)
    return model


def prevalence_table(df: pd.DataFrame, outcome: str, proximity_col: str = "proximity_class"):
    """Crude prevalence and 95% CI per proximity class."""
    rows = []
    for cls, sub in df.groupby(proximity_col):
        n = len(sub)
        k = int(sub[outcome].sum())
        p = k / n if n else float("nan")
        se = np.sqrt(p * (1 - p) / n) if n else float("nan")
        rows.append({
            "proximity_class": cls,
            "n": n,
            "events": k,
            "prevalence": p,
            "ci_low": max(0.0, p - 1.96 * se),
            "ci_high": min(1.0, p + 1.96 * se),
        })
    return pd.DataFrame(rows)
