"""Average Daily Dose (ADD) calculators for the USEPA exposure framework.

Three pathways:

Ingestion (water and soil)
    .. math::  ADD_{ing} = \\frac{C \\cdot IR \\cdot EF \\cdot ED}{BW \\cdot AT}

Dermal
    .. math::  ADD_{derm} = \\frac{C \\cdot SA \\cdot AF \\cdot ABS \\cdot EF \\cdot ED \\cdot 10^{-6}}{BW \\cdot AT}

Inhalation
    .. math::  ADD_{inh} = \\frac{C \\cdot ET \\cdot EF \\cdot ED}{PEF \\cdot BW \\cdot AT}
"""
from __future__ import annotations

import numpy as np

ArrayLike = float | np.ndarray


def add_ingestion(
    concentration: ArrayLike,
    ingestion_rate: ArrayLike,
    exposure_frequency: ArrayLike,
    exposure_duration: ArrayLike,
    body_weight: ArrayLike,
    averaging_time: ArrayLike,
) -> ArrayLike:
    """Ingestion ADD (mg/kg/day).

    Parameters
    ----------
    concentration : mg/L (water) or mg/kg (soil).
    ingestion_rate : L/day (water) or kg/day (soil).
        For soil, values from the YAML config are in mg/day; convert externally
        with ``ingestion_rate * 1e-6`` to get kg/day before calling.
    exposure_frequency : days/year.
    exposure_duration : years.
    body_weight : kg.
    averaging_time : days.
    """
    return (concentration * ingestion_rate * exposure_frequency * exposure_duration) / (
        body_weight * averaging_time
    )


def add_dermal(
    concentration: ArrayLike,
    skin_surface_area: ArrayLike,
    adherence_factor: ArrayLike,
    absorption_fraction: ArrayLike,
    exposure_frequency: ArrayLike,
    exposure_duration: ArrayLike,
    body_weight: ArrayLike,
    averaging_time: ArrayLike,
) -> ArrayLike:
    """Dermal ADD (mg/kg/day).

    The :math:`10^{-6}` factor converts ``mg/cm² × cm² × mg/kg → mg`` when
    ``adherence_factor`` is in mg/cm² and ``concentration`` in mg/kg.
    """
    return (
        concentration * skin_surface_area * adherence_factor * absorption_fraction
        * exposure_frequency * exposure_duration * 1e-6
    ) / (body_weight * averaging_time)


def add_inhalation(
    concentration: ArrayLike,
    inhalation_rate: ArrayLike,
    exposure_frequency: ArrayLike,
    exposure_duration: ArrayLike,
    body_weight: ArrayLike,
    averaging_time: ArrayLike,
    particulate_emission_factor: float = 1.36e9,
) -> ArrayLike:
    """Inhalation ADD via resuspended particulates (mg/kg/day).

    PEF is the USEPA particulate emission factor (default 1.36 × 10⁹ m³/kg).
    """
    return (
        concentration * inhalation_rate * exposure_frequency * exposure_duration
    ) / (particulate_emission_factor * body_weight * averaging_time)
