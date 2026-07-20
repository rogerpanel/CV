"""Batch Monte-Carlo runner.

For every (defence, J/S level, seed) the runner samples ``repeats_per_point``
flights from the fixed corpus, replays each under the adversarial contour via
the chosen backend, and records the per-flight completion outcome.  Results
are returned as a tidy per-flight table plus an aggregated per-point table
with Wilson confidence intervals.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pandas as pd

from .analysis import wilson_interval
from .backends import Backend
from .config import BenchmarkConfig
from .corpus import Flight


def _sub_seed(*parts) -> int:
    """Deterministic 63-bit sub-seed from arbitrary parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def run_benchmark(
    cfg: BenchmarkConfig,
    backend: Backend,
    flights: List[Flight],
    js_grid: List[float] | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full sweep.

    Returns
    -------
    per_flight : DataFrame
        One row per flight evaluation (defense, js_db, seed, flight_id,
        mission, receiver, completed).
    per_point : DataFrame
        One row per (defense, js_db): pooled completion mean over all seeds
        with a Wilson 95% interval, plus per-seed means.
    """
    js_grid = js_grid if js_grid is not None else cfg.js_grid()
    n_rep = cfg.repeats_per_point
    corpus = np.asarray(flights, dtype=object)
    n_corpus = len(corpus)

    rows = []
    total_cells = len(cfg.defenses) * len(js_grid) * len(cfg.seeds)
    cell = 0
    for defense in cfg.defenses:
        for js in js_grid:
            for seed in cfg.seeds:
                rng = np.random.default_rng(_sub_seed(defense.id, js, seed))
                idx = rng.integers(0, n_corpus, size=n_rep)
                for j in idx:
                    f: Flight = corpus[j]
                    completed = backend.fly(f, defense.id, float(js), rng)
                    rows.append(
                        (
                            defense.id,
                            float(js),
                            int(seed),
                            int(f.flight_id),
                            f.mission,
                            f.receiver,
                            int(completed),
                        )
                    )
                cell += 1
                if progress and cell % max(1, total_cells // 20) == 0:
                    pct = 100.0 * cell / total_cells
                    print(f"  [{backend.name}] {pct:5.1f}%  "
                          f"({cell}/{total_cells} cells)", flush=True)

    per_flight = pd.DataFrame(
        rows,
        columns=[
            "defense", "js_db", "seed", "flight_id",
            "mission", "receiver", "completed",
        ],
    )

    per_point = _aggregate(cfg, per_flight)
    return per_flight, per_point


def _aggregate(cfg: BenchmarkConfig, per_flight: pd.DataFrame) -> pd.DataFrame:
    z_conf = cfg.confidence
    out = []
    for (defense, js), grp in per_flight.groupby(["defense", "js_db"], sort=True):
        k = int(grp["completed"].sum())
        n = int(len(grp))
        mean = k / n if n else float("nan")
        lo, hi = wilson_interval(k, n, z_conf)
        seed_means = (
            grp.groupby("seed")["completed"].mean().to_dict()
        )
        out.append(
            {
                "defense": defense,
                "js_db": js,
                "n": n,
                "completed": k,
                "completion_mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "ci_halfwidth": (hi - lo) / 2.0,
                "seed_means": seed_means,
            }
        )
    df = pd.DataFrame(out).sort_values(["defense", "js_db"]).reset_index(drop=True)
    return df
