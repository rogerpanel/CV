"""Gradient-guided problem-space attack.

White-box over the target model: compute the constrained feature-space
adversarial direction (reusing the paper's feature-space attack and the
mutable-feature mask), then map the sign of each feature change to a traffic
knob. The knob is realised, the flow re-extracted and re-scored. This is the
problem-space analogue of the feature-constrained attack in the TIFS/ICLR paper.

The mapping is deliberately simple and auditable: each knob is driven by the
adversarial gradient on the feature(s) it can physically move.
"""
from __future__ import annotations

import numpy as np

from transforms import Knobs

# which flow features each knob can physically change (CICFlowMeter names on the
# testbed corpus; gradients on these decide the knob setting)
KNOB_FEATURES = {
    "pad_frac":    ["fwd_pkt_len_mean", "totlen_fwd_pkts", "pkt_size_avg", "fwd_seg_size_avg"],
    "delay_ms":    ["flow_iat_mean", "fwd_iat_mean", "flow_iat_std", "fwd_iat_std"],
    "mss":         ["tot_fwd_pkts", "fwd_pkt_len_max", "subflow_fwd_pkts"],
    "chaff_ratio": ["flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s"],
    "rate_mbit":   ["flow_byts_s", "flow_pkts_s"],
}

# step sizes that take a +1 sigma adversarial push on the feature group to a knob
# move, each already inside its budget
STEP = {"pad_frac": 0.1, "delay_ms": 10.0, "mss": 1, "chaff_ratio": 0.2, "rate_mbit": -2.0}


def direction(model_grad: dict[str, float]) -> Knobs:
    """model_grad: feature name -> signed adversarial gradient (toward benign).
    Returns a budgeted knob setting that moves traffic along that direction."""
    k = Knobs()
    for knob, feats in KNOB_FEATURES.items():
        g = np.mean([model_grad.get(f, 0.0) for f in feats])
        if knob == "mss":
            # more, smaller packets -> lower MSS when the attack wants higher pkt count
            k.mss = 576 if g > 0 else 0
        elif knob == "rate_mbit":
            # low-and-slow when the attack wants lower byte/pkt rate
            k.rate_mbit = 5.0 if g < 0 else 0.0
        else:
            setattr(k, knob, max(0.0, np.sign(g) * STEP[knob]))
    return k.clamp()


def feature_gradient(score_fn, x0: np.ndarray, feature_names: list[str],
                     mutable_mask: np.ndarray, benign_idx: int,
                     steps: int = 20, eps: float = 0.3) -> dict[str, float]:
    """Finite-difference adversarial gradient of the benign-logit margin on the
    mutable features (no autograd needed, so it works against any score_fn,
    including the platform API). Returns feature -> averaged signed gradient."""
    x = x0.copy()
    grad = np.zeros_like(x)
    h = 1e-2
    for j in np.where(mutable_mask)[0]:
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        grad[j] = (score_fn(xp[None])[0, benign_idx] - score_fn(xm[None])[0, benign_idx]) / (2 * h)
    # one PGD-style normalised step within eps keeps it in range; we only need signs
    g = grad * mutable_mask
    return {name: float(g[i]) for i, name in enumerate(feature_names)}
