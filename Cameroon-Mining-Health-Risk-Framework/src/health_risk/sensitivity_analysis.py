"""Global sensitivity analysis for the health-risk model.

Implements:

* Spearman rank correlation between sampled input parameters and HQ/CR output.
* Sobol' first-order and total-order indices via the Saltelli sampling scheme
  (relies on the optional ``SALib`` package).

Used to reproduce the results in section "Sensitivity Analysis" of the
manuscript.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from cmhr.utils import load_config


def spearman_sensitivity(inputs: pd.DataFrame, output: np.ndarray) -> pd.DataFrame:
    """Spearman rank correlation between every input column and the output."""
    rows = []
    out = np.asarray(output, dtype=float)
    for col in inputs.columns:
        r, p = stats.spearmanr(inputs[col].values, out)
        rows.append({"parameter": col, "spearman_r": float(r), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)


def sobol_sensitivity(
    parameter_specs: Mapping[str, Mapping],
    model: Callable[[Mapping[str, float]], float],
    n_base: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Saltelli-scheme Sobol' indices.

    Parameters
    ----------
    parameter_specs : mapping
        ``{name: {"distribution": ..., **params}}`` — the same schema used in
        ``config/exposure_parameters.yaml``.  Currently supports uniform/normal/
        triangular/lognormal converted to uniform bounds for SALib.
    model : callable
        Takes a mapping of sampled parameters → returns a scalar (HQ, CR, …).
    n_base : int, optional
        Base sample size ``N``; defaults to ``config/exposure_parameters.yaml ::
        sobol.n_base_samples``.
    """
    try:
        from SALib.analyze import sobol as sobol_analyze
        from SALib.sample import saltelli
    except ImportError as err:
        raise ImportError(
            "SALib is required for Sobol' analysis: pip install SALib"
        ) from err

    cfg = load_config("exposure_parameters")["sobol"]
    n_base = n_base or cfg["n_base_samples"]

    bounds = []
    names = []
    for name, spec in parameter_specs.items():
        dist = spec.get("distribution", "uniform")
        if dist == "uniform":
            bounds.append([spec["low"], spec["high"]])
        elif dist == "normal":
            bounds.append([spec["mean"] - 3 * spec["std"], spec["mean"] + 3 * spec["std"]])
        elif dist == "triangular":
            bounds.append([spec["low"], spec["high"]])
        elif dist == "lognormal":
            mu = np.log(spec["mean"])
            sigma = np.log(spec["sigma"]) if spec["sigma"] > 1 else spec["sigma"]
            bounds.append([np.exp(mu - 3 * sigma), np.exp(mu + 3 * sigma)])
        else:
            raise ValueError(f"Unsupported distribution {dist}")
        names.append(name)

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    samples = saltelli.sample(problem, n_base, calc_second_order=True)

    y = np.array([model(dict(zip(names, row))) for row in samples], dtype=float)
    si = sobol_analyze.analyze(
        problem, y, calc_second_order=True,
        num_resamples=cfg["bootstrap_resamples"], conf_level=0.95,
    )
    s1 = pd.DataFrame({"parameter": names, "S1": si["S1"], "S1_conf": si["S1_conf"]})
    st = pd.DataFrame({"parameter": names, "ST": si["ST"], "ST_conf": si["ST_conf"]})
    return {"first_order": s1, "total_order": st}
