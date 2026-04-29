"""Probability distribution sampling for the Monte Carlo engine.

Translates the entries in ``config/exposure_parameters.yaml`` into NumPy random
arrays with a consistent interface.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def sample(spec: Mapping | float | int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` values from the distribution described in ``spec``."""
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return np.full(n, float(spec))

    dist = spec.get("distribution", "deterministic")

    if dist == "deterministic":
        return np.full(n, float(spec["value"]))

    if dist == "normal":
        x = rng.normal(spec["mean"], spec["std"], size=n)
        bounds = spec.get("bounds")
        if bounds:
            x = np.clip(x, bounds[0], bounds[1])
        return x

    if dist == "lognormal":
        # ``mean`` is the median (50th percentile) and ``sigma`` is shape;
        # implement as exp(N(ln(mean), ln(sigma))) for log-shape interpretation.
        mu = np.log(spec["mean"])
        sigma = np.log(spec["sigma"]) if spec["sigma"] > 1 else spec["sigma"]
        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    if dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size=n)

    if dist == "triangular":
        return rng.triangular(spec["low"], spec["mode"], spec["high"], size=n)

    raise ValueError(f"Unknown distribution: {dist}")
