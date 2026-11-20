"""Functionality oracle.

An evasive run only counts if its attack still achieves its goal. Each oracle
returns True iff the attack was functional, reading the artefacts the scenario
left in RUN_OUT. This keeps evasion honest: a transformation that breaks the
attack is not a successful evasion, it is a failed attack.

Mapping (from the plan):
  * brute force -> a valid credential is still found
  * exploit     -> the file is still read / the RCE marker is still written
  * scan        -> the same open ports are still reported
  * DoS         -> benign p95 latency still rises by >= 2x
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path(os.environ.get("RUN_OUT", "/out"))

# expected open ports per scanned target, filled in on the first clean run
EXPECTED_PORTS_FILE = OUT.parent / "expected_ports.json"


def scan_ports() -> bool:
    txt = (OUT / "nmap.txt")
    if not txt.exists():
        return False
    found = {int(l.split("/")[0]) for l in txt.read_text().splitlines()
             if "/tcp" in l and "open" in l}
    if EXPECTED_PORTS_FILE.exists():
        expected = set(json.loads(EXPECTED_PORTS_FILE.read_text()))
        return expected.issubset(found)
    return len(found) > 0


def nuclei_hit() -> bool:
    f = OUT / "nuclei.jsonl"
    return f.exists() and any(line.strip() for line in f.read_text().splitlines())


def credential_found() -> bool:
    f = OUT / "hydra.txt"
    return f.exists() and "login:" in f.read_text()


def file_read() -> bool:
    f = OUT / "grafana_leak.txt"
    return f.exists() and "root:" in f.read_text(errors="ignore")


def influx_listed() -> bool:
    f = OUT / "influx_db.txt"
    if not f.exists():
        return False
    try:
        return "series" in json.loads(f.read_text()).get("results", [{}])[0]
    except Exception:
        return "name" in f.read_text()


def rce_marker() -> bool:
    return (OUT / "rce_marker.txt").exists() and \
        (OUT / "rce_marker.txt").read_text().strip() == "RCE-OK"


def latency_up() -> bool:
    """DoS oracle: the benign latency probe (capture/latency_probe.py) must show a
    >= 2x rise in p95 during the attack window."""
    f = OUT / "latency.json"
    if not f.exists():
        return False
    d = json.loads(f.read_text())
    return d.get("p95_during", 0) >= 2.0 * d.get("p95_baseline", 1e9)


ORACLES = {
    "scan_ports": scan_ports, "nuclei_hit": nuclei_hit,
    "credential_found": credential_found, "file_read": file_read,
    "influx_listed": influx_listed, "rce_marker": rce_marker,
    "latency_up": latency_up,
}


def check(name: str) -> bool:
    if not name:
        return True
    return ORACLES[name]()
