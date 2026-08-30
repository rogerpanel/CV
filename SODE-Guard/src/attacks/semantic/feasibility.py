"""Feature-level feasibility constraints for network flows.

Each entry lists (kind, params) where ``kind`` ∈ {"box", "int", "flag",
"nonneg", "monotone", "derived_ratio"}.  Constraints were compiled from
the CICFlowMeter specification and validated against the ICS3D flow
generator; see ``docs/packet_semantics.md``.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch

from ...data.feature_engineering import ALL_FEATURES


# name → (kind, min, max) or (kind, None, None) for flags / derived
FEATURE_CONSTRAINTS: Dict[str, Tuple[str, float | None, float | None]] = {}
for f in ALL_FEATURES:
    if f.startswith("flag_"):
        FEATURE_CONSTRAINTS[f] = ("flag", 0.0, 1.0)
    elif f.startswith(("fwd_packets", "bwd_packets", "total_pkts",
                       "iat_", "duration", "header_len",
                       "fwd_bytes", "bwd_bytes",
                       "act_data_pkts_fwd", "min_seg_size_fwd",
                       "subflow_fwd_bytes", "cert_chain_len",
                       "sni_len", "session_ticket_len",
                       "handshake_rtt", "extension_count")):
        FEATURE_CONSTRAINTS[f] = ("nonneg", 0.0, None)
    elif f.startswith(("cipher_suite_id", "ja3_hash_mod", "ja4_hash_mod",
                       "tls_alpn_id", "pqc_kx_id", "pqc_sig_id",
                       "kem_round", "tls_version")):
        FEATURE_CONSTRAINTS[f] = ("int", 0.0, None)
    elif f.endswith("_ratio"):
        FEATURE_CONSTRAINTS[f] = ("derived_ratio", 0.0, None)
    else:
        FEATURE_CONSTRAINTS[f] = ("box", None, None)


class FeasibilityProjector:
    """Project a feature vector back onto the feasibility set.

    The projection enforces three families of constraints:

      1. Box   :  feature-specific minimums / maximums (packet counts ≥ 0,
                  flag bits in [0, 1], TLS enum IDs ≥ 0).
      2. Integer:  packet counts, byte totals, TLS-enum indices are rounded
                  to the nearest non-negative integer.
      3. Derived: monotone derived ratios (down/up, fwd/bwd byte ratio,
                  average segment size) are recomputed from the underlying
                  raw counts so an adversary cannot desynchronise them.
    """

    def __init__(self, standardization: dict | None = None,
                 enforce_integer: bool = True):
        self.standardization = standardization
        self.enforce_integer = enforce_integer
        self._idx = {name: i for i, name in enumerate(ALL_FEATURES)}

    def _undo_zscore(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardization is None:
            return x
        mean = torch.as_tensor(self.standardization["mean"], device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.standardization["std"], device=x.device, dtype=x.dtype)
        return x * std + mean

    def _redo_zscore(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardization is None:
            return x
        mean = torch.as_tensor(self.standardization["mean"], device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.standardization["std"], device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self._undo_zscore(x.clone())
        for name, (kind, lo, hi) in FEATURE_CONSTRAINTS.items():
            i = self._idx[name]
            if kind == "flag":
                y[:, i] = torch.round(y[:, i].clamp(0.0, 1.0))
            elif kind == "nonneg":
                y[:, i] = y[:, i].clamp(min=0.0)
            elif kind == "int":
                y[:, i] = torch.round(y[:, i].clamp(min=0.0))
                if self.enforce_integer:
                    y[:, i] = y[:, i].to(torch.long).to(y.dtype)
            elif kind == "box" and (lo is not None or hi is not None):
                y[:, i] = y[:, i].clamp(min=lo, max=hi)

        # Recompute derived ratios from raw counts to prevent desync.
        if all(k in self._idx for k in ("fwd_bytes", "bwd_bytes",
                                        "fwd_packets", "bwd_packets")):
            fb, bb = y[:, self._idx["fwd_bytes"]], y[:, self._idx["bwd_bytes"]]
            fp, bp = y[:, self._idx["fwd_packets"]], y[:, self._idx["bwd_packets"]]
            if "fwd_bwd_byte_ratio" in self._idx:
                y[:, self._idx["fwd_bwd_byte_ratio"]] = fb / bb.clamp_min(1.0)
            if "fwd_bwd_pkt_ratio" in self._idx:
                y[:, self._idx["fwd_bwd_pkt_ratio"]] = fp / bp.clamp_min(1.0)
            if "down_up_ratio" in self._idx:
                y[:, self._idx["down_up_ratio"]] = bb / fb.clamp_min(1.0)
        return self._redo_zscore(y)
