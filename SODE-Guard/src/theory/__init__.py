"""Theoretical-verification utilities added in the IEEE-Access revision.

These modules give the *empirical* handles the reviewers asked for:

    * ``lipschitz.certify_L2_lipschitz`` — verifies the L^2 Lipschitz
      assumption underpinning Proposition 5 with a two-sample estimator and
      a Hoeffding upper confidence bound, so ``L_g`` is not a free
      hyper-parameter.
    * ``chaos_degree.estimate_effective_degree`` — data-driven selection of
      d* via Hermite-polynomial residual variance, replacing the previously
      fixed d*=4.
    * ``carbery_wright.margin_stability_bound`` — the corrected two-sided
      bound of the revised Proposition 5. Uses the actual L^2 norm of the
      chaos-truncated margin polynomial rather than an upper bound in the
      denominator (fix for Reviewer 2's direction-of-inequality concern).
    * ``pac_bayes.pac_bayes_bound`` — closed-form Maurer PAC-Bayes-kl
      certificate wired into the training / evaluation pipeline.
    * ``pseudoinverse.moore_penrose_diffusion`` — the (128 × 16) diffusion
      pseudo-inverse used in the BEL identity, with numerical checks
      documenting the effect of the ellipticity floor.
    * ``bel_estimator.verify_bel`` — Monte-Carlo variance study of the
      BEL gradient estimator against a finite-difference reference.
"""
from .lipschitz import certify_L2_lipschitz, LipschitzCertificate
from .chaos_degree import estimate_effective_degree, HermiteChaosFit
from .carbery_wright import (
    margin_stability_bound,
    decision_flip_bound,
    two_sided_stability_bound,
    invert_for_radius,
)
from .pac_bayes import pac_bayes_bound
from .pseudoinverse import moore_penrose_diffusion
from .bel_estimator import verify_bel

__all__ = [
    "certify_L2_lipschitz", "LipschitzCertificate",
    "estimate_effective_degree", "HermiteChaosFit",
    "margin_stability_bound", "decision_flip_bound",
    "two_sided_stability_bound", "invert_for_radius",
    "pac_bayes_bound", "moore_penrose_diffusion", "verify_bel",
]
