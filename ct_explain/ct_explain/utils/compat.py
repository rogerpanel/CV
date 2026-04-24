"""Soft dependency handling.

The paper's framework leans on torchdiffeq, torchsde, torch_geometric. Any of
these may be absent from the deployment environment (e.g. constrained
inference nodes). This module provides uniform availability flags and, where
reasonable, pure-torch fallbacks that keep the public API usable even when
the optional stack is incomplete.
"""
from __future__ import annotations

from importlib import import_module

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _try(module: str) -> bool:
    try:
        import_module(module)
        return True
    except Exception:
        return False


HAS_TORCHDIFFEQ = _try("torchdiffeq")
HAS_TORCHSDE = _try("torchsde")
HAS_PYG = _try("torch_geometric")
HAS_QUANTUS = _try("quantus")
HAS_NETCAL = _try("netcal")
HAS_SHAP = _try("shap")
HAS_CAPTUM = _try("captum")


def require(flag: bool, pkg: str) -> None:
    if not flag:
        raise RuntimeError(
            f"'{pkg}' is required for this feature. Install with `pip install {pkg}`."
        )
