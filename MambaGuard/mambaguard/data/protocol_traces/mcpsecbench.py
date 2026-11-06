"""MCPSecBench: a security benchmark and playground for MCP.

Source: Y. Yang et al., "MCPSecBench: A Systematic Security Benchmark and
Playground for Testing Model Context Protocols", arXiv:2508.13220.
Data:   https://github.com/AIS2Lab/MCPSecBench (MIT License).

What the repository provides
----------------------------
* ``data/data.json``: 11 attack prompts (keys ``attack``, ``prompt``,
  ``result``; ``result`` is the LLM-judge instruction, not an outcome).
* ``data/experiments.csv``: per-run outcomes, 15 runs each for Claude Desktop,
  OpenAI and Cursor, for 17 attacks (``1`` success, ``0`` failed, ``-1``
  detected, ``2`` blocked by protection, ``-`` not applicable).
* ``code/*.py``: the benign servers (``addserver.py``, ``download.py``) and
  the attacker's servers (``maliciousadd.py``, ``squatting.py``).

It records no MCP messages, so episodes are reconstructed: the host lists the
tools of every server that the playground configures (tool definitions parsed
statically from the server source), then the user sends the attack prompt.
The attacker's tool lists are labelled 1; so is the prompt for the four
attacks carried by the prompt itself. For the seven attacks whose prompt is
benign, a benign twin sends the same prompt with only the benign servers
present.

Six of the 17 attacks (schema inconsistencies, slash-command overlap,
vulnerable client, MCP rebinding, man-in-the-middle, configuration drift) are
implemented only as client/transport code and have no prompt; they are not
represented as episodes. Their per-run outcomes are still available through
:func:`load_mcpsecbench_outcomes`.

Server names are replaced by neutral identifiers (``mcp-server-0`` ...) and
the listing order is shuffled per episode, because the attacker's server in
the playground is literally named ``malicious``; the real names are kept in
``Episode.extras["server_names"]``.

With 11 attack and 7 benign episodes this is an evaluation set, not a training
set.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Any, Iterator

from ..canonicalize import ProtocolMessage
from .common import (
    Episode,
    EpisodeBuilder,
    extract_server,
    flatten,
    require_file,
    rpc_request,
    rpc_result,
    split_episodes,
)

BENCHMARK = "mcpsecbench"
BENIGN_SERVERS = ("addserver.py", "download.py")
MALICIOUS_SERVER = "maliciousadd.py"
SQUATTING_SERVER = "squatting.py"

PROMPT_BORNE = {"Prompt Injection", "Sandbox Escape", "Vulnerable Server", "Tool Misuse via Confused AI"}
NEEDS_SQUATTING_SERVER = {"Package Name Squatting(Server Name)"}

SURFACE = {
    "Prompt Injection": "user", "Tool Misuse via Confused AI": "user",
    "Schema Inconsistencies": "client", "Slash Command Overlap": "client", "Vulnerable Client": "client",
    "MCP Rebinding": "transport", "Man-in-the-Middle": "transport",
}

# data.json attack name -> experiments.csv column name
CSV_COLUMN = {
    "Tool Poisoning Attack": "Tool Poisoning Attack",
    "Tool Shadowing Attack": "Tool Shadowing Attack",
    "Package Name Squatting(Server Name)": "Package Name Squatting(server name)",
    "Package Name Squatting(Tool Name)": "Package Name Squatting(tools name)",
    "Data Exfiltration": "Data Exfiltration",
    "Prompt Injection": "Prompt Injection",
    "Rug Pull": "Rug Pull Attack",
    "Indirect Prompt Injection": "Indirect Prompt Injection",
    "Sandbox Escape": "Sandbox Escape",
    "Tool Misuse via Confused AI": "Tool/Service Misuse via “Confused AI”",
    "Vulnerable Server": "Vulnerable Server",
}
CODES = {"1": "success", "0": "failed", "-1": "detected", "2": "blocked"}


def load_mcpsecbench_outcomes(root: str) -> dict[str, dict[str, list[str]]]:
    """``{attack column: {provider: [outcome per run]}}`` from ``experiments.csv``."""
    path = os.path.join(root, "data", "experiments.csv")
    require_file(path, "Clone https://github.com/AIS2Lab/MCPSecBench and pass the clone as root.")
    out: dict[str, dict[str, list[str]]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        for row in reader:
            if not row or not row[0].strip():
                continue
            provider = row[0].strip()
            for col, val in zip(header[1:], row[1:]):
                out.setdefault(col, {}).setdefault(provider, []).append(CODES.get(val.strip(), "n/a"))
    return out


def _servers(root: str, files: tuple[str, ...]) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    out = []
    for f in files:
        name, tools, res = extract_server(os.path.join(root, "code", f))
        out.append((name or os.path.splitext(f)[0], tools, res))
    return out


def _episode(eid: str, prompt: str, servers: list, hot: set[str], prompt_label: int,
             attack_type: str, group: str, anonymise: bool = True, **extras: Any) -> Episode:
    b = EpisodeBuilder(eid, BENCHMARK)
    rid = 0
    if anonymise:
        servers = sorted(servers, key=lambda s: hashlib.sha256(f"{group}:{s[0]}".encode()).hexdigest())
    alias = {s[0]: (f"mcp-server-{i}" if anonymise else s[0]) for i, s in enumerate(servers)}
    extras["server_names"] = {alias[s[0]]: s[0] for s in servers}
    for real, tools, resources in servers:
        name = alias[real]
        rid += 1
        b.add("M_cap", "host", name, rpc_request("tools/list", None, rid))
        b.add("M_cap", name, "host", rpc_result({"tools": tools}, rid), label=int(real in hot))
        if resources:
            rid += 1
            b.add("M_cap", "host", name, rpc_request("resources/list", None, rid))
            b.add("M_cap", name, "host", rpc_result({"resources": resources}, rid), label=int(real in hot))
    b.add("M_comm", "user", "host", {"role": "user", "content": prompt}, label=prompt_label)
    return b.build(attack_type != "benign", attack_type, group, **extras)


def load_mcpsecbench_episodes(root: str, split: str = "all", include_benign_twins: bool = True,
                              anonymise_servers: bool = True) -> list[Episode]:
    """Load MCPSecBench as labelled episodes. ``root`` is the repository clone."""
    path = os.path.join(root, "data", "data.json")
    require_file(path, "Clone https://github.com/AIS2Lab/MCPSecBench and pass the clone as root.")
    cases = json.load(open(path, encoding="utf-8"))
    benign = _servers(root, BENIGN_SERVERS)
    malicious = _servers(root, (MALICIOUS_SERVER,))
    squatting = _servers(root, (SQUATTING_SERVER,))
    outcomes = load_mcpsecbench_outcomes(root)

    episodes: list[Episode] = []
    for i, case in enumerate(cases):
        attack, prompt = case["attack"], case["prompt"]
        servers = benign + malicious + (squatting if attack in NEEDS_SQUATTING_SERVER else [])
        hot = {s[0] for s in malicious} | ({s[0] for s in squatting} if attack in NEEDS_SQUATTING_SERVER else set())
        group = f"case{i:02d}"
        per_run = outcomes.get(CSV_COLUMN.get(attack, attack), {})
        rates = {p: (sum(o == "success" for o in runs) / sum(o != "n/a" for o in runs)) if any(o != "n/a" for o in runs) else None
                 for p, runs in per_run.items()}
        eid = f"{BENCHMARK}:{group}"
        episodes.append(_episode(eid, prompt, servers, hot, int(attack in PROMPT_BORNE), attack, group,
                                 anonymise_servers, surface=SURFACE.get(attack, "server"), outcomes=per_run, success_rate=rates))
        if include_benign_twins and attack not in PROMPT_BORNE:
            episodes.append(_episode(eid + ":benign", prompt, benign, set(), 0, "benign", group,
                                     anonymise_servers, twin_of=attack))
    return split_episodes(episodes, split)


def load_mcpsecbench(root: str, split: str = "all") -> Iterator[ProtocolMessage]:
    """Registry entry point. Defaults to ``all``: the set is too small to split."""
    yield from flatten(load_mcpsecbench_episodes(root, split))
