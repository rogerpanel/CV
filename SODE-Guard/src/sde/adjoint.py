"""Stochastic-adjoint sensitivity wrapper.

We defer to ``torchsde.sdeint_adjoint`` when available (Li et al., AISTATS 2020),
falling back to the in-house Euler–Maruyama with checkpointed forward path if
torchsde is not installed (e.g. in offline CI runners).
"""
from __future__ import annotations
import torch
from torch import Tensor

try:
    import torchsde
    _HAS_TORCHSDE = True
except ImportError:                                          # pragma: no cover
    _HAS_TORCHSDE = False


def sdeint_adjoint(sde_module, x0: Tensor, ts: Tensor, **kwargs) -> Tensor:
    """Differentiable SDE integration via the stochastic adjoint.

    Parameters
    ----------
    sde_module : torch.nn.Module
        Must expose ``f(t, x)`` and ``g(t, x)`` methods consistent with
        ``torchsde`` conventions (noise_type='general', sde_type='ito').
    x0 : Tensor (batch, d)
    ts : Tensor (T,)  monotonically increasing time stamps
    """
    if _HAS_TORCHSDE:
        return torchsde.sdeint_adjoint(
            sde_module, x0, ts,
            method=kwargs.pop("method", "euler"),
            dt=kwargs.pop("dt", 0.05),
            adaptive=kwargs.pop("adaptive", False),
            **kwargs,
        )

    # Fallback: discrete-time forward, autograd through the EM step.
    from .integrator import EulerMaruyama, EMConfig
    em = EulerMaruyama(EMConfig(t0=float(ts[0]), t1=float(ts[-1]),
                                dt=kwargs.pop("dt", 0.05),
                                save_trajectory=True))
    _, traj = em(x0, sde_module.f, sde_module.g)
    return traj
