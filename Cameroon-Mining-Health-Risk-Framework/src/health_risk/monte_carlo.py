"""10,000-iteration Monte Carlo health-risk simulator.

Mirrors the Anaedevha et al. (2026) implementation: deterministic seed, 5/50/95th
percentile reporting, simultaneous HQ + ILCR computation across the three
USEPA pathways.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from . import add_calculation as _add
from ._distributions import sample
from .cancer_risk import _csf_lookup
from .hazard_quotient import _rfd_lookup


@dataclass
class RiskResult:
    """Container for Monte Carlo output."""
    receptor: str
    element: str
    pathway: str
    samples: np.ndarray = field(repr=False)
    rfd: float | None = None
    csf: float | None = None
    is_carcinogenic: bool = False

    def percentiles(self, q=(5, 50, 95)) -> Dict[float, float]:
        return {p: float(np.percentile(self.samples, p)) for p in q}

    def summary(self) -> Dict[str, float]:
        p = self.percentiles()
        return {
            "receptor": self.receptor,
            "element": self.element,
            "pathway": self.pathway,
            "mean": float(self.samples.mean()),
            "std": float(self.samples.std(ddof=1)),
            "p05": p[5],
            "p50": p[50],
            "p95": p[95],
            "carcinogenic": self.is_carcinogenic,
        }


class MonteCarloRiskAssessment:
    """Probabilistic risk simulator.

    Parameters
    ----------
    receptor : {"adults", "children"}
    n_iterations : int, default = 10_000
    seed : int, optional
        Overrides the seed in ``exposure_parameters.yaml``.
    """

    def __init__(
        self,
        receptor: str = "adults",
        n_iterations: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = load_config("exposure_parameters")
        if receptor not in self.cfg["receptors"]:
            raise ValueError(f"Unknown receptor {receptor!r}")
        self.receptor = receptor
        self.n_iterations = n_iterations or self.cfg["n_iterations"]
        self.seed = seed if seed is not None else self.cfg["random_seed"]
        self._rng = np.random.default_rng(self.seed)
        self._abs = load_config("exposure_parameters")["dermal_absorption_fraction"]

    # ------------------------------------------------------------------ utils
    def _draw(self, key: str) -> np.ndarray:
        spec = self.cfg["receptors"][self.receptor][key]
        return sample(spec, self.n_iterations, self._rng)

    def _averaging_time(self, exposure_duration: np.ndarray, carcinogenic: bool) -> np.ndarray:
        if carcinogenic:
            return np.full_like(exposure_duration, 70 * 365.0)
        return exposure_duration * 365.0

    # -------------------------------------------------------------- pathways
    def run_ingestion(
        self,
        concentration: float | np.ndarray,
        element: str,
        matrix: str = "water",
        carcinogenic: bool = False,
    ) -> RiskResult:
        bw = self._draw("body_weight_kg")
        ef = self._draw("exposure_frequency_days_yr")
        ed = self._draw("exposure_duration_yr")
        if matrix == "water":
            ir = self._draw("ingestion_rate_water_l_day")
        else:
            # convert mg/day → kg/day for soil ingestion
            ir = self._draw("ingestion_rate_soil_mg_day") * 1e-6
        at = self._averaging_time(ed, carcinogenic)
        add = _add.add_ingestion(concentration, ir, ef, ed, bw, at)
        return self._wrap(add, element, "ingestion", carcinogenic)

    def run_dermal(
        self,
        concentration: float | np.ndarray,
        element: str,
        carcinogenic: bool = False,
    ) -> RiskResult:
        sa = self._draw("skin_surface_area_cm2")
        af = self._draw("skin_adherence_factor_mg_cm2")
        ef = self._draw("exposure_frequency_days_yr")
        ed = self._draw("exposure_duration_yr")
        bw = self._draw("body_weight_kg")
        abs_frac = float(self._abs.get(element, 0.001))
        at = self._averaging_time(ed, carcinogenic)
        add = _add.add_dermal(concentration, sa, af, abs_frac, ef, ed, bw, at)
        return self._wrap(add, element, "dermal", carcinogenic)

    def run_inhalation(
        self,
        concentration: float | np.ndarray,
        element: str,
        carcinogenic: bool = False,
    ) -> RiskResult:
        ir = self._draw("inhalation_rate_m3_day")
        ef = self._draw("exposure_frequency_days_yr")
        ed = self._draw("exposure_duration_yr")
        bw = self._draw("body_weight_kg")
        at = self._averaging_time(ed, carcinogenic)
        pef = self.cfg.get("particulate_emission_factor_m3_kg", 1.36e9)
        add = _add.add_inhalation(concentration, ir, ef, ed, bw, at, pef)
        return self._wrap(add, element, "inhalation", carcinogenic)

    # -------------------------------------------------------- HQ / CR helper
    def _wrap(self, add, element, pathway, carcinogenic):
        if carcinogenic:
            csf = _csf_lookup(element, pathway)
            samples = add * csf
            return RiskResult(
                receptor=self.receptor, element=element, pathway=pathway,
                samples=np.asarray(samples), csf=csf, is_carcinogenic=True,
            )
        rfd = _rfd_lookup(element, pathway)
        samples = add / rfd
        return RiskResult(
            receptor=self.receptor, element=element, pathway=pathway,
            samples=np.asarray(samples), rfd=rfd, is_carcinogenic=False,
        )

    # ----------------------------------------------------------------- batch
    def run(
        self,
        concentration_mg_per_l: float | None = None,
        concentration_mg_per_kg: float | None = None,
        element: str = "As",
        pathways: tuple = ("ingestion", "dermal", "inhalation"),
        carcinogenic: bool = False,
    ) -> RiskResult:
        """Convenience: run the requested pathways and return the aggregated HI/CR.

        ``concentration_mg_per_l`` is used for ingestion; ``mg_per_kg`` for
        dermal/inhalation pathways.  If only one is passed it is used for all
        pathways.
        """
        cw = concentration_mg_per_l if concentration_mg_per_l is not None else concentration_mg_per_kg
        cs = concentration_mg_per_kg if concentration_mg_per_kg is not None else cw

        results = []
        if "ingestion" in pathways:
            results.append(self.run_ingestion(cw, element, "water", carcinogenic))
        if "dermal" in pathways:
            results.append(self.run_dermal(cs, element, carcinogenic))
        if "inhalation" in pathways:
            try:
                results.append(self.run_inhalation(cs, element, carcinogenic))
            except KeyError:
                # No inhalation RfD/CSF → skip pathway silently
                pass

        combined = np.sum(np.column_stack([r.samples for r in results]), axis=1)
        return RiskResult(
            receptor=self.receptor,
            element=element,
            pathway="+".join(r.pathway for r in results),
            samples=combined,
            is_carcinogenic=carcinogenic,
        )

    # --------------------------------------------------------------- helper
    def batch_summary(self, runs: Mapping[str, RiskResult]) -> pd.DataFrame:
        return pd.DataFrame([r.summary() for r in runs.values()], index=list(runs))
