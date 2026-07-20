"""Calibrated completion-probability model and adversarial contour.

The mission-completion probability of a defence configuration at a given
jamming-to-signal ratio (J/S, dB) is modelled as a monotone-decreasing
function that interpolates the calibrated anchor points from
``config/defenses.yaml`` (PCHIP — shape-preserving piecewise cubic).

This is the *ground truth* of the benchmark.  The Monte-Carlo runner draws
per-flight Bernoulli completion outcomes from this probability; the airsim
backend is calibrated to reproduce it from real 3-D flights.

Physical interpretation
-----------------------
J/S is the standard EW severity axis (jammer power / signal power, dB).
As J/S rises the GNSS position solution degrades and the combined
adversarial contour (spoofing + visual PGD + DRL BIM) increasingly drives
the vehicle outside the DO-326A safe-completion envelope.  Each defence
shifts the whole curve to the right by suppressing one or more attack
channels; the M1+M4+M6+M7 stack shifts it farthest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import BenchmarkConfig, Defense


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CompletionModel:
    """Per-defence J/S -> mission-completion probability curve."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self._anchors_x = np.asarray(cfg.js_anchor_points_db, dtype=float)
        self._curves: Dict[str, PchipInterpolator] = {}
        for d in cfg.defenses:
            y = np.asarray(d.completion_anchors, dtype=float)
            # PchipInterpolator preserves monotonicity of monotone data and
            # passes exactly through the anchor points.
            self._curves[d.id] = PchipInterpolator(self._anchors_x, y, extrapolate=True)

    # -- base probability -------------------------------------------------
    def base_probability(self, defense_id: str, js_db: float) -> float:
        curve = self._curves[defense_id]
        p = float(curve(js_db))
        return min(1.0, max(0.0, p))

    # -- probability with optional documented stratified effects ----------
    def probability(
        self,
        defense_id: str,
        js_db: float,
        mission: str | None = None,
        receiver: str | None = None,
    ) -> float:
        p = self.base_probability(defense_id, js_db)
        if not self.cfg.stratified_enabled:
            return p
        z = _logit(p)
        if receiver is not None:
            z += float(self.cfg.receiver_logit_delta.get(receiver, 0.0))
        if mission is not None:
            z += float(self.cfg.mission_logit_delta.get(mission, 0.0))
        return _sigmoid(z)

    # -- J/S at which the curve crosses a completion threshold ------------
    def crossing_js(self, defense_id: str, threshold: float) -> float:
        """Largest J/S (dB) at which completion >= threshold.

        Found by dense evaluation of the monotone curve over the sweep
        range; returns the sweep max if the curve never drops below the
        threshold, and the sweep min if it starts below it.
        """
        grid = np.linspace(self.cfg.js_min_db, self.cfg.js_max_db, 4001)
        vals = np.clip(self._curves[defense_id](grid), 0.0, 1.0)
        above = np.where(vals >= threshold)[0]
        if above.size == 0:
            return float(self.cfg.js_min_db)
        return float(grid[above[-1]])


@dataclass
class AdversaryContour:
    """Descriptor of the combined adversarial loop applied on replay.

    In sim-lite the contour's effect is already folded into the calibrated
    completion anchors (which were measured under the full contour).  This
    object records the attack parameters for the manifest and is the hook
    the airsim backend uses to actually perturb sensor streams.
    """

    gnss_spoofing: bool = True
    visual_pgd: bool = True
    drl_bim: bool = True
    pgd_epsilon: float = 8 / 255
    pgd_steps: int = 20
    bim_epsilon: float = 8 / 255
    bim_steps: int = 10

    @classmethod
    def from_config(cls, raw: dict) -> "AdversaryContour":
        adv = raw.get("adversary", {}) if raw else {}
        g = adv.get("gnss_spoofing", {}) or {}
        v = adv.get("visual_pgd", {}) or {}
        b = adv.get("drl_bim", {}) or {}
        return cls(
            gnss_spoofing=bool(g.get("enabled", True)),
            visual_pgd=bool(v.get("enabled", True)),
            drl_bim=bool(b.get("enabled", True)),
            pgd_epsilon=float(v.get("epsilon", 8 / 255)),
            pgd_steps=int(v.get("steps", 20)),
            bim_epsilon=float(b.get("epsilon", 8 / 255)),
            bim_steps=int(b.get("steps", 10)),
        )
