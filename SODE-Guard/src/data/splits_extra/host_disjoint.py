"""Host-disjoint split (Reviewer 2 request).

Ensures no source IP or destination IP simultaneously appears in the
training and test partitions. The set of unique host identifiers is
first split by a hash bucket; each bucket is deterministically assigned
to one of {train, val, test} using the requested fractions.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


def _bucket_hash(h: str, salt: int) -> float:
    d = hashlib.blake2b(f"{salt}:{h}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "little") / (2 ** 64)


def host_disjoint_split(df: pd.DataFrame, *,
                        host_cols: tuple[str, ...] = ("src_ip", "dst_ip"),
                        train: float = 0.70, val: float = 0.15, test: float = 0.15,
                        seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hosts = set()
    for c in host_cols:
        if c in df.columns:
            hosts.update(df[c].astype(str).unique().tolist())
    if not hosts:
        raise ValueError(f"No host columns in DataFrame; expected any of {host_cols}")

    tr_h, va_h, te_h = set(), set(), set()
    for h in hosts:
        u = _bucket_hash(h, seed)
        if u < train:
            tr_h.add(h)
        elif u < train + val:
            va_h.add(h)
        else:
            te_h.add(h)

    def mask(rows, hs):
        m = np.zeros(len(rows), dtype=bool)
        for c in host_cols:
            if c in rows.columns:
                m |= rows[c].astype(str).isin(hs).values
        return rows[m]

    return mask(df, tr_h), mask(df, va_h), mask(df, te_h)
