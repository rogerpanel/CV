"""Scenario- / capture-disjoint split.

Some IDS corpora (CIC-IDS2017, CIC-IDS2018, ICS3D) label each flow with
the *capture scenario* or *day-of-attack* it came from. Splitting by
scenario is stronger than temporal splitting because it guarantees that
even short-range temporal correlations cannot leak across the partition.
"""
from __future__ import annotations
import hashlib
import pandas as pd


def _hash01(x: str, seed: int) -> float:
    d = hashlib.blake2b(f"{seed}:{x}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "little") / (2 ** 64)


def scenario_disjoint_split(df: pd.DataFrame, *,
                            scenario_col: str = "scenario",
                            train: float = 0.70, val: float = 0.15, test: float = 0.15,
                            seed: int = 42):
    if scenario_col not in df.columns:
        raise KeyError(f"Missing column '{scenario_col}'")
    scenarios = df[scenario_col].astype(str).unique()
    tr, va, te = set(), set(), set()
    for s in scenarios:
        u = _hash01(s, seed)
        (tr if u < train else va if u < train + val else te).add(s)
    m_tr = df[scenario_col].astype(str).isin(tr)
    m_va = df[scenario_col].astype(str).isin(va)
    m_te = df[scenario_col].astype(str).isin(te)
    return df[m_tr], df[m_va], df[m_te]
