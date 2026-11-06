"""MCP-SafetyBench: attacks on LLM agents using real-world MCP servers.

Source: T. Zong et al., "MCP-SafetyBench: A Benchmark for Safety Evaluation of
LLMs with Real-World MCP Servers", ICLR 2026, arXiv:2512.15163.
Data:   https://github.com/xjzzzzzzzz/MCPSafety
        Cases: ``mcpuniverse/benchmark/configs/test/<domain>/*.json`` (245).
        Clean tool definitions: ``mcpuniverse/mcp/servers/<server>/server.py``.
        The repository has no license file; read it from your own clone and do
        not redistribute it.

Every case is an attack. The attacker-controlled content is inline in the
case file, and each attack type maps to the protocol message an observer on
the wire would see:

==========================  =====================================  ==========
case field                  message carrying it                    label
==========================  =====================================  ==========
mcp_server_modifications    tools/list result (poisoned            1
 .modification_description  description)
mcp_server_modifications    tools/call result (injected output)    1
 .modification_return
mcp_server_additions        tools/list result of the first         1
                            domain server (added tool)
data_tampering              tools/call result (forged data)        1
replay_injection            repeated tools/call requests           1
identity_spoofing           tools/call with the malicious token    1
wrong_arguments_intent_inj  tools/call with malicious arguments    1
wrong_function_intent_inj   tools/call to the malicious function   1
(no payload field)          the user question itself               1
mcp_server_update           not visible on the wire (the tool's    excluded
                            code changes, its description does     by default
                            not); ``include_unobservable=True``
                            keeps the 7 Rug Pull cases, labelled
                            on a list_changed notification
==========================  =====================================  ==========

Benign twins are generated only where they are exact: for cases whose attack
is carried by tool descriptions (modifications/additions), the twin replays the
same question against the unmodified tool list taken from the server source.

The attack *layer* (Server/Host/User) is not stored in the repository; the
mapping in ``LAYER`` follows where each attack is injected in the benchmark's
runner and client and is recorded as ``extras["layer"]``.

Splits are by case: an attack case and its benign twin share a group.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Iterator, Optional

from ..canonicalize import ProtocolMessage
from .common import (
    Episode,
    EpisodeBuilder,
    extract_server,
    flatten,
    mcp_tool,
    require_file,
    rpc_notification,
    rpc_request,
    rpc_result,
    split_episodes,
)

BENCHMARK = "mcp_safetybench"
CASES = os.path.join("mcpuniverse", "benchmark", "configs", "test")
SERVERS = os.path.join("mcpuniverse", "mcp", "servers")
SERVER_LIST = os.path.join("mcpuniverse", "mcp", "configs", "server_list.json")

LAYER = {
    "Intent Injection": "host", "Data Tampering": "host", "Identity Spoofing": "host", "Replay Injection": "host",
    "Malicious Code Execution": "user", "Credential Theft": "user", "Remote Access Control": "user",
    "Excessive Privileges Misuse": "user", "Retrieval-Agent Deception": "user",
}
USER_LAYER = {k for k, v in LAYER.items() if v == "user"}


def _domain_servers(root: str, domain: str) -> list[str]:
    path = os.path.join(root, CASES, f"{domain}.yaml")
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8").read()
    m = re.search(r"(?m)^\s*servers:\s*\n((?:\s*-\s*name:\s*\S+\s*\n?)+)", text)
    return re.findall(r"name:\s*(\S+)", m.group(1)) if m else []


def _server_tools(root: str) -> dict[str, list[dict[str, Any]]]:
    """Clean tool lists keyed by the server name used in the domain YAML."""
    out: dict[str, list[dict[str, Any]]] = {}
    path = os.path.join(root, SERVER_LIST)
    listing = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for name, spec in listing.items():
        args = (spec.get("stdio") or {}).get("args") or []
        mod = next((a for a in args if a.startswith("mcpuniverse.mcp.servers.")), None)
        if not mod:
            continue
        src = os.path.join(root, SERVERS, mod.split(".")[-1], "server.py")
        if os.path.exists(src):
            out[name] = extract_server(src)[1]
    return out


def _as_list(x: Any) -> list[dict[str, Any]]:
    if not x:
        return []
    if isinstance(x, dict):
        return [x] if "tool_name" in x else list(x.get("tools", []))
    return list(x)


class _Case:
    def __init__(self, root: str, path: str, tools: dict[str, list[dict[str, Any]]]):
        self.path = path
        self.domain = os.path.basename(os.path.dirname(path))
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.cfg = json.load(open(path, encoding="utf-8"))
        self.attack = str(self.cfg.get("attack_category", "unknown"))
        self.layer = LAYER.get(self.attack, "server")
        self.servers = [s for s in _domain_servers(root, self.domain) if s in tools] or ["server"]
        self.clean = {s: [dict(t) for t in tools.get(s, [])] for s in self.servers}
        self.missing_servers = [s for s in _domain_servers(root, self.domain) if s not in tools]

    def server_of(self, tool: str) -> str:
        for s, ts in self.clean.items():
            if any(t["name"] == tool for t in ts):
                return s
        return self.servers[0]

    def tool_lists(self, attacked: bool) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
        lists = {s: [dict(t) for t in ts] for s, ts in self.clean.items()}
        hot: set[str] = set()
        if not attacked:
            return lists, hot
        for mod in _as_list(self.cfg.get("mcp_server_modifications")):
            desc = mod.get("modification_description")
            if not desc:
                continue
            s = self.server_of(mod["tool_name"])
            for t in lists[s]:
                if t["name"] == mod["tool_name"]:
                    t["description"] = desc
                    break
            else:
                lists[s].append(mcp_tool(mod["tool_name"], desc))
            hot.add(s)
        for add in _as_list(self.cfg.get("mcp_server_additions")):
            params = [p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in (add.get("input_parameters") or [])]
            lists[self.servers[0]].append(mcp_tool(add["tool_name"], add.get("description", ""), params))
            hot.add(self.servers[0])
        return lists, hot

    @property
    def description_borne(self) -> bool:
        mods = _as_list(self.cfg.get("mcp_server_modifications"))
        return bool(self.cfg.get("mcp_server_additions")) or any(m.get("modification_description") for m in mods)

    @property
    def observable(self) -> bool:
        return not self.cfg.get("mcp_server_update")


def _build(case: _Case, attacked: bool, eid: str) -> Episode:
    b = EpisodeBuilder(eid, BENCHMARK)
    lists, hot = case.tool_lists(attacked)
    rid = 0
    for s in case.servers:
        rid += 1
        b.add("M_cap", "host", s, rpc_request("tools/list", None, rid))
        b.add("M_cap", s, "host", rpc_result({"tools": lists[s]}, rid), label=int(s in hot))
    b.add("M_comm", "user", "host", {"role": "user", "content": case.cfg.get("question", "")},
          label=int(attacked and case.attack in USER_LAYER))

    if attacked:
        cfg = case.cfg

        def call(tool: str, args: Any, label: int, meta: Optional[dict[str, Any]] = None) -> int:
            nonlocal rid
            rid += 1
            b.add("M_tool", "host", case.server_of(tool),
                  rpc_request("tools/call", {"name": tool, "arguments": args or {}}, rid), label=label, metadata=meta)
            return rid

        for mod in _as_list(cfg.get("mcp_server_modifications")):
            if mod.get("modification_return"):
                r = call(mod["tool_name"], {}, 0)
                b.add("M_data", case.server_of(mod["tool_name"]), "host",
                      rpc_result({"content": [{"type": "text", "text": mod["modification_return"]}]}, r), label=1)
        dt = cfg.get("data_tampering") or {}
        if dt:
            r = call(dt["target_tool"], {}, 0)
            b.add("M_data", case.server_of(dt["target_tool"]), "host",
                  rpc_result({"content": [{"type": "text", "text": json.dumps(dt.get("forged_data"), default=str)}]}, r),
                  label=1)
        ri = cfg.get("replay_injection") or {}
        if ri:
            call(ri["target_tool"], {}, 0)
            for _ in range(max(1, int(ri.get("replay_threshold", 1)))):
                call(ri["target_tool"], {}, 1)
        spoof = (cfg.get("identity_spoofing") or {}).get("malicious_identity") or {}
        if spoof:
            tool = case.clean[case.servers[0]][0]["name"] if case.clean[case.servers[0]] else "tool"
            call(tool, {}, 1, meta={"authorization": f"Bearer {spoof.get('malicious_token', '')}", "auth": True})
        wa = cfg.get("wrong_arguments_intent_injection") or {}
        if wa:
            call(wa["target_tool"], wa.get("malicious_arguments"), 1)
        wf = cfg.get("wrong_function_intent_injection") or {}
        if wf:
            call(wf.get("malicious_function", wf.get("target_tool", "tool")), {}, 1)
        upd = cfg.get("mcp_server_update") or {}
        if upd:
            b.add("M_ctrl", case.server_of(upd["tool_name"]), "host",
                  rpc_notification("notifications/tools/list_changed"), label=1)

    group = f"{case.domain}/{case.name}"
    return b.build(attacked, case.attack if attacked else "benign", group,
                   domain=case.domain, layer=case.layer if attacked else None,
                   category=case.cfg.get("category"), case=group,
                   servers_without_local_tools=case.missing_servers)


def load_mcp_safetybench_episodes(
    root: str,
    split: str = "all",
    include_benign_twins: bool = True,
    include_unobservable: bool = False,
) -> list[Episode]:
    """Load MCP-SafetyBench as labelled episodes. ``root`` is the repository clone."""
    case_dir = os.path.join(root, CASES)
    require_file(case_dir, "Clone https://github.com/xjzzzzzzzz/MCPSafety and pass the clone as root.")
    tools = _server_tools(root)
    episodes: list[Episode] = []
    for path in sorted(glob.glob(os.path.join(case_dir, "*", "*.json"))):
        case = _Case(root, path, tools)
        if not case.observable and not include_unobservable:
            continue
        eid = f"{BENCHMARK}:{case.domain}/{case.name}"
        episodes.append(_build(case, True, eid))
        if include_benign_twins and case.description_borne:
            episodes.append(_build(case, False, eid + ":benign"))
    return split_episodes(episodes, split)


def load_mcp_safetybench(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """Registry entry point: MCP-SafetyBench messages (see module docstring)."""
    yield from flatten(load_mcp_safetybench_episodes(root, split))
