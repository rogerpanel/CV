"""Multi-seed statistics for the v2 re-run.

* ``mean_ci``: mean and two-sided 95% Student-t interval over seeds.
* ``paired_bootstrap``: difference of a metric between two models evaluated on
  the same flows in every seed. Each replicate resamples flows within every
  seed (the same indices for both models) and averages the per-seed
  differences; the p-value is two-sided.
* ``holm``: Holm-Bonferroni adjustment over a declared family of contrasts.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import t as t_dist


def mean_ci(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(v.mean())
    if len(v) < 2:
        return {"mean": m, "lo": m, "hi": m, "n": 1}
    h = float(t_dist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": int(len(v))}


def paired_bootstrap(per_seed: list[tuple], metric: Callable, n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """per_seed: list of (y, pred_a, pred_b, weights_or_None) for each seed.
    metric(y, pred, w) -> float. Returns the observed mean difference a - b,
    a 95% percentile interval and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean([metric(y, a, w) - metric(y, b, w) for y, a, b, w in per_seed]))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        diffs = []
        for y, a, b, w in per_seed:
            idx = rng.integers(0, len(y), len(y))
            ww = None if w is None else w[idx]
            diffs.append(metric(y[idx], a[idx], ww) - metric(y[idx], b[idx], ww))
        boots[i] = np.mean(diffs)
    # add-one estimate, so p is never reported as exactly zero
    p = 2 * (min((boots <= 0).sum(), (boots >= 0).sum()) + 1) / (n_boot + 1)
    return {"diff": obs, "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "p": float(min(1.0, p)), "n_seeds": len(per_seed)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out
