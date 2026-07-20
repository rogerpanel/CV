"""Physical flight corpus generation.

The corpus is the set of base UAV flights generated once (in AirSim/PX4 for
the full-fidelity backend, or as deterministic descriptors for sim-lite) and
then *replayed* under every (defence x J/S) evaluation condition.  Keeping the
corpus fixed means all conditions are compared on the same flights, which is
what makes the paired mission-completion curves meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from .config import BenchmarkConfig


@dataclass(frozen=True)
class Flight:
    flight_id: int
    mission: str
    receiver: str
    # nominal scenario parameters (used by the airsim backend; recorded for
    # provenance in sim-lite)
    waypoint_km: float
    terrain_roughness: float
    wind_mps: float


def _even_split(total: int, k: int) -> List[int]:
    """Split ``total`` into ``k`` parts as evenly as possible (sums to total)."""
    base = total // k
    rem = total - base * k
    return [base + (1 if i < rem else 0) for i in range(k)]


def build_corpus(cfg: BenchmarkConfig, seed: int = 20260101) -> List[Flight]:
    """Deterministically build the base flight corpus.

    Flights are distributed as evenly as possible across the (mission x
    receiver) grid, summing exactly to ``cfg.total_flights``.
    """
    rng = np.random.default_rng(seed)
    cells = [(m, r) for m in cfg.missions for r in cfg.receivers]
    counts = _even_split(cfg.total_flights, len(cells))

    flights: List[Flight] = []
    fid = 0
    for (mission, receiver), n in zip(cells, counts):
        for _ in range(n):
            flights.append(
                Flight(
                    flight_id=fid,
                    mission=mission,
                    receiver=receiver,
                    waypoint_km=float(round(rng.uniform(2.0, 12.0), 3)),
                    terrain_roughness=float(round(rng.uniform(0.0, 1.0), 3)),
                    wind_mps=float(round(rng.uniform(0.0, 9.0), 3)),
                )
            )
            fid += 1
    assert len(flights) == cfg.total_flights
    return flights


def corpus_manifest(cfg: BenchmarkConfig, flights: List[Flight]) -> dict:
    """Summary of the corpus composition for the benchmark manifest."""
    by_mission = {m: 0 for m in cfg.missions}
    by_receiver = {r: 0 for r in cfg.receivers}
    for f in flights:
        by_mission[f.mission] += 1
        by_receiver[f.receiver] += 1
    return {
        "total_flights": len(flights),
        "by_mission": by_mission,
        "by_receiver": by_receiver,
        "example_flight": asdict(flights[0]) if flights else None,
    }
