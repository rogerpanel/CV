"""Apply spectral normalisation to every ``nn.Linear`` inside a module subtree.

We use ``torch.nn.utils.parametrizations.spectral_norm`` (the new
parametrisation-based API) which composes correctly with AMP, FSDP, and
``torch.compile``.
"""
from __future__ import annotations
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def enforce_spectral_norm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and not _is_already_sn(child):
            parent_name, _, leaf = name.rpartition(".")
            parent = module if parent_name == "" else module.get_submodule(parent_name)
            setattr(parent, leaf, spectral_norm(child))
    return module


def _is_already_sn(layer: nn.Linear) -> bool:
    return hasattr(layer, "parametrizations") and "weight" in layer.parametrizations
