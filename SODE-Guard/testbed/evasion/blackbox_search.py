"""Black-box evasion search over the knob vector.

A simple (1+lambda) evolutionary search that minimises the target's margin on the
*realised* flows, subject to the budget and the functionality oracle. It needs
only scores, so the same search drives the neural detectors, the trees, and the
platform / signature detectors identically — the fair black-box comparison the
plan asks for.

~50 queries per scenario and model by default. Each query realises a knob
setting end to end (apply knob -> run scenario -> capture -> extract -> score),
so queries are expensive; the budget keeps the count honest.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from transforms import Knobs


@dataclass
class Trial:
    knobs: Knobs
    margin: float          # target margin on realised flows (lower = more evasive)
    functional: bool       # oracle passed
    detected: bool         # any flow raised an alarm


def _mutate(k: Knobs, scale: float, rng: random.Random) -> Knobs:
    v = k.vector()
    spans = [0.5, 50, 25, 1460, 1.0, 20]
    nv = [max(0.0, x + rng.gauss(0, 1) * s * scale) for x, s in zip(v, spans)]
    return Knobs.from_vector(nv)


def search(realise_and_score: Callable[[Knobs], Trial],
           queries: int = 50, lam: int = 4, seed: int = 0) -> tuple[Trial, list[Trial]]:
    """realise_and_score(knobs) -> Trial (runs the whole pipeline once).
    Returns the best *functional* trial and the full history."""
    rng = random.Random(seed)
    history: list[Trial] = []

    parent = realise_and_score(Knobs())          # query 0: unperturbed attack
    history.append(parent)
    best = parent if parent.functional else None
    used = 1
    scale = 1.0
    while used < queries:
        improved = False
        for _ in range(min(lam, queries - used)):
            child = realise_and_score(_mutate(parent.knobs, scale, rng))
            history.append(child)
            used += 1
            # objective: stay functional, then minimise margin (prefer undetected)
            if child.functional:
                if best is None or child.margin < best.margin:
                    best = child
                if child.margin < parent.margin:
                    parent = child
                    improved = True
        scale = min(scale * 1.3, 2.0) if not improved else max(scale * 0.7, 0.2)
    return (best or parent), history


def evasion_success(best: Trial) -> bool:
    """A scenario is evaded iff the best functional trial is undetected."""
    return best.functional and not best.detected
