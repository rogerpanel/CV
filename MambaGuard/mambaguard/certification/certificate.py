"""Composed three-layer MambaGuard certificate.

Reference (paper Theorem — Composed Certificate):

    E[V_ach(π̂_T, ε)] ≥ V*  −  L_f · ε  −  B · sqrt(ln|A_D| / (2T))
                       ^^^^   ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       SSE    Lipschitz    Hedge average regret
                       value  robustness   (Cesa-Bianchi–Lugosi 2006)

The three subtracted gaps correspond directly to the three layers of the
certification stack:
    1. ``L_f · ε``                — robustness gap from the composed
                                    Lipschitz bound of the SSM+GATv2+head
                                    backbone (see :mod:`lipschitz_bounds`).
    2. ``B · sqrt(ln|A_D|/(2T))`` — online-learning gap from Hedge
                                    (see :mod:`hedge`).
    3. ``V*``                     — strong Stackelberg equilibrium value of
                                    the underlying security game
                                    (see :mod:`stackelberg`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ------------------------------------------------------------------ #
# Functional form                                                    #
# ------------------------------------------------------------------ #
def composed_certificate(
    V_star: float,
    L_f: float,
    epsilon: float,
    B: float,
    T: int,
    num_actions: int,
) -> float:
    """DEPRECATED: dimensionally inconsistent.

    This function subtracts ``L_f * epsilon`` (a logit-space Lipschitz
    quantity) from a utility-valued quantity ``V*``. Use
    :func:`composed_certificate_v2`, which replaces the Lipschitz gap by
    ``B * Pr[~C_eps] = B * (1 - CertAcc(eps))`` coming from a
    Cohen-style smoothing certificate and adds a Monte-Carlo failure
    term ``B * alpha_MC``.

    Kept for backward compatibility of the legacy driver.

        V* - L_f*eps - B*sqrt(ln|A_D|/(2T)).

    Parameters
    ----------
    V_star : float
        Strong Stackelberg equilibrium value (game-theoretic optimum).
    L_f : float
        Composed Lipschitz constant of MambaGuard (Proposition in paper).
    epsilon : float
        Worst-case ℓ₂ perturbation budget on inputs.
    B : float
        Per-round loss range used in Hedge (utilities bounded in [0, B]).
    T : int
        Online-defence horizon (number of Hedge rounds).
    num_actions : int
        Cardinality of the defender action set, |A_D|.
    """
    if L_f < 0:
        raise ValueError(f"L_f must be ≥ 0, got {L_f}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be ≥ 0, got {epsilon}")
    if B < 0:
        raise ValueError(f"B must be ≥ 0, got {B}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if num_actions < 2:
        raise ValueError(f"num_actions must be ≥ 2, got {num_actions}")
    robustness_gap = L_f * epsilon
    online_gap = B * math.sqrt(math.log(num_actions) / (2.0 * T))
    return V_star - robustness_gap - online_gap


# ------------------------------------------------------------------ #
# Dimensionally consistent v2                                        #
# ------------------------------------------------------------------ #
def composed_certificate_v2(
    V_star: float,
    cert_acc: float,
    B: float,
    T: int,
    num_actions: int,
    alpha_mc: float = 1e-3,
) -> float:
    """Dimensionally-consistent composed certificate.

        E[V_ach] >= V*  -  B * Pr[~C_eps]
                        -  B * sqrt(ln|A_D| / (2T))
                        -  B * alpha_MC

    where ``Pr[~C_eps] = 1 - cert_acc(eps)`` is the fraction of inputs
    for which the Cohen-style randomised-smoothing certificate fails to
    certify at radius epsilon, and ``alpha_MC`` is the Monte-Carlo
    failure probability of the smoothing procedure itself.

    Parameters
    ----------
    V_star : float
        Strong Stackelberg equilibrium value in utility units (same
        units as B).
    cert_acc : float
        CertAcc(epsilon): fraction of test points correctly certified at
        the chosen epsilon. Must lie in [0, 1].
    B : float
        Per-round utility range; losses/utilities are bounded in [0, B].
    T : int
        Online-defence horizon (Hedge rounds).
    num_actions : int
        Cardinality of the defender action set.
    alpha_mc : float
        Monte-Carlo failure probability of the smoothing certificate
        (default 1e-3, matching Cohen et al. 2019).
    """
    if not 0.0 <= cert_acc <= 1.0:
        raise ValueError(f"cert_acc must be in [0, 1], got {cert_acc}")
    if B < 0:
        raise ValueError(f"B must be >= 0, got {B}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if num_actions < 2:
        raise ValueError(f"num_actions must be >= 2, got {num_actions}")
    if not 0.0 <= alpha_mc <= 1.0:
        raise ValueError(f"alpha_mc must be in [0, 1], got {alpha_mc}")
    robustness_gap = B * (1.0 - cert_acc)
    online_gap = B * math.sqrt(math.log(num_actions) / (2.0 * T))
    mc_gap = B * alpha_mc
    return V_star - robustness_gap - online_gap - mc_gap


# ------------------------------------------------------------------ #
# Object form (with ablation support)                                #
# ------------------------------------------------------------------ #
@dataclass
class CompositeCertificate:
    """Object capturing every input to the composed certificate.

    Convenient for logging, ablation tables, and unit tests.
    """

    V_star: float
    L_f: float
    epsilon: float
    B: float
    T: int
    num_actions: int

    # --------------------------------------------------------------- #
    def __post_init__(self) -> None:
        if self.L_f < 0:
            raise ValueError(f"L_f must be ≥ 0, got {self.L_f}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be ≥ 0, got {self.epsilon}")
        if self.B < 0:
            raise ValueError(f"B must be ≥ 0, got {self.B}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.num_actions < 2:
            raise ValueError(f"num_actions must be ≥ 2, got {self.num_actions}")

    # --------------------------------------------------------------- #
    def robustness_gap(self) -> float:
        """L_f · ε — Lipschitz-robustness loss."""
        return self.L_f * self.epsilon

    def online_gap(self) -> float:
        """B · sqrt(ln|A_D|/(2T)) — Hedge average regret."""
        return self.B * math.sqrt(math.log(self.num_actions) / (2.0 * self.T))

    def gap_components(self) -> dict:
        """Return the three terms that compose the certificate gap.

        Useful for ablation tables (which layer is dominating the slack?).
        """
        return {
            "V_star": self.V_star,
            "robustness_gap": self.robustness_gap(),
            "online_gap": self.online_gap(),
            "total_gap": self.robustness_gap() + self.online_gap(),
        }

    def lower_bound(self) -> float:
        """Return V* − L_f·ε − B·sqrt(ln|A_D|/(2T))."""
        return self.V_star - self.robustness_gap() - self.online_gap()


# ------------------------------------------------------------------ #
# Pretty-printer                                                     #
# ------------------------------------------------------------------ #
def report(cert: CompositeCertificate) -> str:
    """Format *cert* as a human-readable multi-line report."""
    parts = cert.gap_components()
    lb = cert.lower_bound()
    lines = [
        "MambaGuard Composed Certificate",
        "===============================",
        f"  Stackelberg value      V*         = {cert.V_star:.6f}",
        f"  Composed Lipschitz     L_f        = {cert.L_f:.6f}",
        f"  Perturbation budget    epsilon    = {cert.epsilon:.6f}",
        f"  Per-round loss bound   B          = {cert.B:.6f}",
        f"  Horizon                T          = {cert.T}",
        f"  Defender actions       |A_D|      = {cert.num_actions}",
        "",
        f"  Robustness gap  L_f · ε                 = {parts['robustness_gap']:.6f}",
        f"  Online gap      B · sqrt(ln|A_D|/(2T))  = {parts['online_gap']:.6f}",
        f"  Total gap                              = {parts['total_gap']:.6f}",
        "",
        f"  Certified lower bound on E[V_ach]      = {lb:.6f}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Sanity check                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    cert = CompositeCertificate(
        V_star=0.85,
        L_f=2.3,
        epsilon=0.05,
        B=1.0,
        T=1000,
        num_actions=5,
    )
    print(report(cert))
    assert math.isclose(
        cert.lower_bound(),
        composed_certificate(
            V_star=0.85, L_f=2.3, epsilon=0.05, B=1.0, T=1000, num_actions=5
        ),
    )
    print("\n[OK] CompositeCertificate ≡ composed_certificate")
