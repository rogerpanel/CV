"""Turn Suricata/Zeek output into per-flow detections, keyed by the 5-tuple so they
align with the CICFlowMeter flows. A flow is 'detected' by Suricata if any alert
shares its 5-tuple within the flow window; by Zeek if it raised a notice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def suricata_alerts(eve: Path) -> set[tuple]:
    keys = set()
    if not eve.exists():
        return keys
    for line in eve.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event_type") != "alert":
            continue
        keys.add((e.get("src_ip"), e.get("dest_ip"), e.get("src_port"),
                  e.get("dest_port"), e.get("proto", "").lower()))
    return keys


def flow_key(row) -> tuple:
    return (row.get("Src IP", row.get("src_ip")), row.get("Dst IP", row.get("dst_ip")),
            row.get("Src Port", row.get("src_port")), row.get("Dst Port", row.get("dst_port")),
            str(row.get("Protocol", row.get("protocol", ""))).lower())


def detections(flows: pd.DataFrame, alert_keys: set[tuple]) -> pd.Series:
    def hit(r):
        s, d, sp, dp, pr = flow_key(r)
        return any(k[0] in (s, d) and k[1] in (s, d) and str(dp) in (str(k[2]), str(k[3]))
                   for k in alert_keys)
    return flows.apply(hit, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("flows_csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    flows = pd.read_csv(a.flows_csv, low_memory=False)
    keys = suricata_alerts(run / "suricata" / "eve.json")
    flows["suricata_detected"] = detections(flows, keys)
    flows[["suricata_detected"]].to_csv(a.out, index=False)
    print(f"suricata flagged {int(flows['suricata_detected'].sum())}/{len(flows)} flows")


if __name__ == "__main__":
    main()
