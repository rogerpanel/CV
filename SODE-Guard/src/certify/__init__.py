"""Certificates for the SODE-Guard mean predictor and single-path decisions.

    * ``lipschitz_constants``: the Lipschitz constant L of Theorem A, computed
      from the exact spectral norms of the trained weights.
    * ``mean_certificate``: Theorem A with a Monte-Carlo confidence correction
      (worst case over all ||δ||_2 ≤ r for the mean predictor).
    * ``path_flip``: Proposition B, a bound on the single-path decision-flip
      probability for a fixed perturbation δ.
"""
from .lipschitz_constants import model_lipschitz, LipschitzReport
from .mean_certificate import certify_mean, MeanCertificate
from .path_flip import path_flip_bound, clopper_pearson_upper

__all__ = ["model_lipschitz", "LipschitzReport", "certify_mean", "MeanCertificate",
           "path_flip_bound", "clopper_pearson_upper"]
