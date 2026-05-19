"""83-dim flow feature extraction shared by SODE-Guard and RobustIDPS.

Group breakdown (matches Appendix B of the manuscript):

    flow stats       18   duration, total fwd/bwd bytes, packet counts, ...
    inter-arrival    14   min/max/mean/std/skew/kurtosis (fwd, bwd)
    flag indicators  12   PSH, URG, SYN, FIN, RST, ACK, CWR, ECE, NS, ...
    payload stats     9   bytes/sec, bytes/packet, header length, ...
    derived ratios   10   down/up ratio, avg segment size, ...
    TLS metadata     12   cipher suite, ja3 hash, ja4 hash, sni length, ...
    handshake info    8   pqc_kx, pqc_sig, alpn, version, resumption, ...
                       ----
                      83
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


FEATURE_GROUPS: dict[str, list[str]] = {
    "flow_stats": [
        "duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
        "fwd_pkts_per_s", "bwd_pkts_per_s", "fwd_bytes_per_s", "bwd_bytes_per_s",
        "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean", "fwd_pkt_len_std",
        "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean", "bwd_pkt_len_std",
        "total_pkts",
    ],
    "iat": [
        "iat_fwd_min", "iat_fwd_max", "iat_fwd_mean", "iat_fwd_std", "iat_fwd_skew",
        "iat_fwd_kurt", "iat_fwd_q25",
        "iat_bwd_min", "iat_bwd_max", "iat_bwd_mean", "iat_bwd_std", "iat_bwd_skew",
        "iat_bwd_kurt", "iat_bwd_q25",
    ],
    "flags": [
        "flag_psh", "flag_urg", "flag_syn", "flag_fin", "flag_rst", "flag_ack",
        "flag_cwr", "flag_ece", "flag_ns", "flag_psh_bwd", "flag_urg_bwd",
        "flag_syn_bwd",
    ],
    "payload": [
        "header_len_fwd", "header_len_bwd", "payload_bytes_per_s",
        "payload_bytes_per_pkt", "min_payload", "max_payload",
        "mean_payload", "std_payload", "entropy_payload",
    ],
    "ratios": [
        "down_up_ratio", "fwd_bwd_byte_ratio", "fwd_bwd_pkt_ratio",
        "avg_fwd_seg_size", "avg_bwd_seg_size", "init_win_fwd", "init_win_bwd",
        "act_data_pkts_fwd", "min_seg_size_fwd", "subflow_fwd_bytes",
    ],
    "tls": [
        "tls_version", "cipher_suite_id", "ja3_hash_mod", "ja4_hash_mod",
        "sni_len", "cert_chain_len", "tls_resumed", "tls_alpn_id",
        "tls_grease", "ech_present", "session_ticket_len", "ocsp_present",
    ],
    "handshake": [
        "pqc_kx_id", "pqc_sig_id", "kem_round", "hybrid_kx",
        "handshake_rtt", "renegotiated", "extension_count", "version_drift",
    ],
}
ALL_FEATURES: list[str] = [f for grp in FEATURE_GROUPS.values() for f in grp]
assert len(ALL_FEATURES) == 83, f"Expected 83 features, got {len(ALL_FEATURES)}"


@dataclass
class FlowFeatureExtractor:
    """Lightweight adapter that maps heterogeneous flow CSV/Parquet schemas
    onto the canonical 83-feature vector. Missing columns are filled with 0."""

    fill_value: float = 0.0

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        df = df.copy()
        for col in missing:
            df[col] = self.fill_value
        return df[ALL_FEATURES].to_numpy(dtype=np.float32, copy=True)


def standardize(x: np.ndarray, mean: np.ndarray | None = None,
                std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)
