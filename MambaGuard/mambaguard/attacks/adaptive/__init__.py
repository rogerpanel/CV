"""Adaptive adversaries for MambaGuard.

These attackers are intended to defeat the certified defence, not just
report an arbitrary ASR number. Two modules are provided:

* :class:`embed_pgd.EmbedPGD` -- PGD in payload-embedding space with
  Expectation-Over-Transformation (EoT) to reduce variance against
  Gaussian smoothing.
* :class:`token_gcg.TokenGCG` -- scaffold for GCG-style discrete token
  search (requires a tokeniser / embedder to be plugged in).
"""
from __future__ import annotations

from .embed_pgd import EmbedPGD
from .token_gcg import TokenGCG

__all__ = ["EmbedPGD", "TokenGCG"]
