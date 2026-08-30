"""Flow deduplication and cross-split leakage detection.

Reviewer 2 asked for stronger dataset provenance guarantees. This
module (a) fingerprints each flow, (b) drops near-duplicates within
a corpus, and (c) reports leakage between two partitions expressed as
the fraction of test-side fingerprints that also occur on the train
side.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


_QUANTIZE = np.float32


def _quantize_row(row: np.ndarray, precision: int = 4) -> bytes:
    q = np.round(row, decimals=precision).astype(_QUANTIZE)
    return q.tobytes()


def fingerprint_flows(X: np.ndarray, precision: int = 4) -> np.ndarray:
    """Deterministic per-flow fingerprint (BLAKE2b(quantised features))."""
    out = np.empty(X.shape[0], dtype=object)
    for i in range(X.shape[0]):
        out[i] = hashlib.blake2b(_quantize_row(X[i], precision), digest_size=16).hexdigest()
    return out


def drop_near_duplicates(df: pd.DataFrame, feat_cols: list[str],
                         precision: int = 4) -> tuple[pd.DataFrame, dict]:
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    fp = fingerprint_flows(X, precision=precision)
    df2 = df.copy(); df2["_fingerprint"] = fp
    before = len(df2)
    df2 = df2.drop_duplicates("_fingerprint")
    after = len(df2)
    return df2.drop(columns="_fingerprint"), {"before": before, "after": after,
                                              "removed": before - after,
                                              "unique_rate": after / max(before, 1)}


def leakage_report(df_train: pd.DataFrame, df_test: pd.DataFrame,
                   feat_cols: list[str], precision: int = 4) -> dict:
    train_fp = set(fingerprint_flows(df_train[feat_cols].to_numpy(np.float32), precision).tolist())
    test_fp = fingerprint_flows(df_test[feat_cols].to_numpy(np.float32), precision)
    hits = int(sum(1 for f in test_fp if f in train_fp))
    return {"test_size": int(len(test_fp)),
            "leakage_hits": hits,
            "leakage_rate": hits / max(len(test_fp), 1)}
