"""SIEM plugin adapter.

The paper reports deployment on three SIEMs: Splunk (financial),
QRadar (healthcare), and ArcSight (energy). Each exposes a different
alert schema, but they all share the same *outbound* needs — pull raw
events, normalise into a TemporalGraph, and push back verdicts.

This module provides a vendor-neutral adapter with pluggable translators.
The default `NetFlowTranslator` handles NetFlow-v2 JSON, which is what
all three SIEMs emit through their REST APIs today.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ct_explain.data.graph_builder import ContinuousTimeGraphBuilder, TemporalGraph


class BaseTranslator(ABC):
    @abstractmethod
    def normalise(self, raw_events: Iterable[dict]) -> list[dict]:
        """Return a stream of dicts with keys (src, dst, timestamp, features)."""


class NetFlowTranslator(BaseTranslator):
    """NetFlow-v2 JSON translator (works for Splunk / QRadar / ArcSight)."""

    feature_keys = (
        "flow_duration", "total_fwd_pkts", "total_bwd_pkts",
        "total_fwd_bytes", "total_bwd_bytes", "flow_pkts_s", "flow_bytes_s",
        "pkt_len_mean", "pkt_len_std", "iat_mean", "iat_std",
        "syn_flag_cnt", "ack_flag_cnt", "fin_flag_cnt", "rst_flag_cnt",
        "bytes_per_pkt", "pkts_per_second",
    )

    def normalise(self, raw_events: Iterable[dict]) -> list[dict]:
        out: list[dict] = []
        for e in raw_events:
            features = [float(e.get(k, 0.0)) for k in self.feature_keys]
            out.append({
                "src": str(e.get("src_ip", e.get("source", "unknown"))),
                "dst": str(e.get("dst_ip", e.get("destination", "unknown"))),
                "timestamp": float(e.get("timestamp", 0.0)),
                "features": np.asarray(features, dtype=np.float32),
                "label": e.get("label"),
            })
        return out


@dataclass
class SIEMPlugin:
    """Vendor-neutral SIEM ↔ CT-Explain bridge."""

    vendor: str = "generic"
    translator: BaseTranslator = None

    def __post_init__(self) -> None:
        if self.translator is None:
            self.translator = NetFlowTranslator()

    # ------------------------------------------------------------------ #
    def ingest(self, raw_events: Iterable[dict]) -> TemporalGraph:
        records = self.translator.normalise(raw_events)
        builder = ContinuousTimeGraphBuilder()
        builder.add_stream(records)
        return builder.build(metadata={"siem": self.vendor})

    # ------------------------------------------------------------------ #
    def emit(self, verdict: dict, url: str | None = None) -> dict:
        """Push a verdict back to the SIEM.

        If ``url`` is None, just returns the payload for testing. The
        `requests` dependency is imported lazily so unit tests run offline.
        """
        payload = {
            "vendor": self.vendor,
            **verdict,
        }
        if url is None:
            return payload
        import requests
        r = requests.post(url, json=payload, timeout=5)
        return {"status": r.status_code, "payload": payload}
