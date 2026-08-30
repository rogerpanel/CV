"""Packet-semantic-preserving adversarial perturbations.

Reviewer 2 raised the concern that feature-space perturbations may not
correspond to realisable network flows. This subpackage adds a
constraint-projection layer for PGD-40 and Carlini–Wagner so the
adversary is restricted to the *feasible* region of the 83-dim feature
vector implied by the packet structure.

    * ``FeasibilityProjector``  — enforces the box, monotonicity, integrality
      and derived-ratio constraints of ``docs/packet_semantics.md``.
    * ``ConstrainedPGD``         — PGD with per-step projection.
    * ``ConstrainedCW``          — Carlini–Wagner with the same projection.
"""
from .feasibility import FeasibilityProjector, FEATURE_CONSTRAINTS
from .constrained_attacks import ConstrainedPGD, ConstrainedCW

__all__ = ["FeasibilityProjector", "FEATURE_CONSTRAINTS",
           "ConstrainedPGD", "ConstrainedCW"]
