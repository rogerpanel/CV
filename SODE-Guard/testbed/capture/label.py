"""Label extracted flows from (attacker IP, run window).

Every scenario writes a window record {scenario, family, attacker_ip, t0, t1} to
the run manifest. A flow is labelled with a scenario's family iff one of its
endpoints is that scenario's attacker IP and its start time falls inside the
window; otherwise it is BenignTraffic. This is the same (source, window) labelling
the plan specifies, and it is deterministic and auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BENIGN = "BenignTraffic"


def label_flows(flows: pd.DataFrame, windows: list[dict],
                src_col="src_ip", dst_col="dst_ip", ts_col="timestamp") -> pd.Series:
    # flow start time in epoch seconds
    ts = pd.to_datetime(flows[ts_col], errors="coerce").astype("int64") / 1e9
    labels = pd.Series(BENIGN, index=flows.index)
    for w in windows:
        ip = w["attacker_ip"]
        in_win = (ts >= w["t0"]) & (ts <= w["t1"])
        is_atk = (flows[src_col] == ip) | (flows[dst_col] == ip)
        labels[in_win & is_atk] = w["family"]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flows_csv")
    ap.add_argument("manifest_json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    flows = pd.read_csv(a.flows_csv, low_memory=False)
    windows = json.loads(Path(a.manifest_json).read_text())["windows"]
    flows["label"] = label_flows(flows, windows)
    flows.to_csv(a.out, index=False)
    print(flows["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
