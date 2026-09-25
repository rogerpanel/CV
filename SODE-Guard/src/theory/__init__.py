"""Stochastic-analysis utilities: the regularised BEL input-gradient estimator
and the Moore–Penrose pseudo-inverse of the (d × m) diffusion."""
from .bel import bel_gradient, bel_input_gradient
from .pseudoinverse import moore_penrose_diffusion, condition_number

__all__ = ["bel_gradient", "bel_input_gradient", "moore_penrose_diffusion", "condition_number"]
