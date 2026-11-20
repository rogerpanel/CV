"""Score testbed flows through the RobustIDPS.ai backend.

The platform's FastAPI backend classifies an uploaded flow CSV at POST
/api/predict (CIC-IoT-2023 / CICIDS2018 feature schemas). The testbed scores the
*same* realised flows through it, so the platform's built-in detectors are
evaluated on exactly the traffic the neural detectors and trees see (plan Q4).

The backend runs on the internal network of the testbed VM (or a read-only
scoring endpoint); the URL is passed in and is expected to be a private address.
Nothing from the production host is touched.

    python scoring/score_platform.py --url http://10.77.0.5:8000 \
        --flows runs/clean/flows.csv --out runs/clean/scores_platform
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def assert_internal(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname or ""
    try:
        addr = socket.getaddrinfo(host, None)[0][4][0]
        ip = ipaddress.ip_address(addr)
    except Exception as e:
        raise SystemExit(f"platform url {host!r}: {e}")
    if not (ip.is_private or ip.is_loopback):
        raise SystemExit(f"REFUSING to POST testbed traffic to public host {host} ({ip})")


def score(url: str, flows_csv: Path, batch: int = 5000):
    assert_internal(url)
    df = pd.read_csv(flows_csv, low_memory=False)
    preds, confidences = [], []
    for i in range(0, len(df), batch):
        chunk = df.iloc[i:i + batch]
        buf = chunk.to_csv(index=False).encode()
        r = requests.post(f"{url.rstrip('/')}/api/predict",
                          files={"file": ("flows.csv", buf, "text/csv")}, timeout=300)
        r.raise_for_status()
        rows = r.json().get("predictions", r.json().get("results", []))
        preds += [row.get("label_pred", row.get("prediction", "")) for row in rows]
        confidences += [row.get("confidence", row.get("score", float("nan"))) for row in rows]
    return np.array(preds), np.array(confidences, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="internal RobustIDPS.ai backend base URL")
    ap.add_argument("--flows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pred, conf = score(a.url, Path(a.flows))
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outd / "platform.npz", pred=pred, confidence=conf)
    (outd / "meta.json").write_text(json.dumps({"url": a.url, "n": int(len(pred))}, indent=2))
    print(f"platform scored {len(pred)} flows -> {outd/'platform.npz'}")


if __name__ == "__main__":
    main()
