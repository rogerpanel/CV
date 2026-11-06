"""MCPTox: tool-poisoning attacks on real-world MCP servers.

Source: Z. Wang et al., "MCPTox: A Benchmark for Tool Poisoning Attack on
Real-World MCP Servers", AAAI 2026, arXiv:2508.14925.
Data:   https://github.com/zhiqiangwang4/MCPTox-Benchmark (``response_all.json``).
        The repository has no license file; read it from your own clone and do
        not redistribute it.

Layout read here (``response_all.json``)::

    servers[<name>] = {server_id, category_id, tool_names, clean_system_promot,
                       clean_querys, malicious_instance: [
                         {wrong_data, poisoned_tool, metadata: {paradigm, "security risk"},
                          datas: [{id, query, response{model: str}, label{model: str},
                                   online_result{backend_model_responses[0], labeled_model_results[0]}}]}]}

Episode construction
--------------------
Attack episode (one per malicious instance; ``wrong_data == 0`` gives the
paper's 1,312)::

    host   -> server   tools/list request                 label 0
    server -> host     tools/list result, clean tools +   label 1
                       the poisoned tool
    user   -> host     the benign user query              label 0
   [host   -> server   tools/call parsed from one model's response, only with
                       ``with_model_call=<model>``; label 1 iff that model's
                       recorded outcome is "Success"]

Benign episode (one per distinct (server, query) pair, drawn from the attack
queries and ``clean_querys``): the same messages with the clean tool list only.
The attack and benign windows therefore differ only in the poisoned tool.

Splits are by server (45 groups), stratified by server category, so test
servers are never seen in training.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from typing import Any, Iterator, Optional

from ..canonicalize import ProtocolMessage
from .common import (
    Episode,
    EpisodeBuilder,
    flatten,
    mcp_tool,
    require_file,
    rpc_request,
    rpc_result,
    split_episodes,
)

BENCHMARK = "mcptox"
DATA_FILE = "response_all.json"
SUCCESS = "Success"


def parse_tool_blocks(text: str) -> list[dict[str, Any]]:
    """Parse ``Tool: x / Description: ... / Arguments: - a: ...`` blocks."""
    if text.count("\\n") > text.count("\n"):
        # About half of MCPTox's poisoned tools store line breaks as literal "\n".
        text = text.replace("\\n", "\n")
    tools: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    mode = ""
    for line in text.splitlines():
        if line.startswith("Tool: "):
            cur = {"name": line[len("Tool: "):].strip(), "description": [], "params": []}
            tools.append(cur)
            mode = ""
        elif cur is None:
            continue
        elif line.startswith("Description: ") and mode == "":
            cur["description"].append(line[len("Description: "):])
            mode = "desc"
        elif line.strip() == "Arguments:" or line.startswith("Arguments:"):
            mode = "args"
        elif mode == "desc":
            cur["description"].append(line)
        elif mode == "args":
            m = re.match(r"\s*-\s*([^:]+):", line)
            if m:
                cur["params"].append(m.group(1).strip())
            elif line.strip() == "":
                continue
            else:
                mode = "done"
    return [mcp_tool(t["name"], "\n".join(t["description"]).strip(), t["params"]) for t in tools]


def wrapper_style(description: str) -> str:
    if "<IMPORTANT>" in description:
        return "important_tag"
    if "ignore the previous" in description.lower():
        return "ignore_previous"
    return "plain"


def parse_model_call(response: str) -> Optional[dict[str, Any]]:
    """Extract ``{"tool": ..., "arguments": ...}`` from a recorded model response."""
    if not isinstance(response, str):
        return None
    s, e = response.find("{"), response.rfind("}")
    if s < 0 or e <= s:
        return None
    blob = response[s:e + 1]
    for parse in (json.loads, ast.literal_eval):
        try:
            obj = parse(blob)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return {"name": str(obj["tool"]), "arguments": obj.get("arguments", {})}
    return None


def _outcomes(datum: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    labels = dict(datum.get("label") or {})
    responses = dict(datum.get("response") or {})
    online = datum.get("online_result") or {}
    for d in (online.get("labeled_model_results") or [])[:1]:
        labels.update(d)
    for d in (online.get("backend_model_responses") or [])[:1]:
        responses.update(d)
    return labels, responses


def _position(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (n + 1)


def _episode(
    eid: str,
    server: str,
    clean: list[dict[str, Any]],
    query: str,
    group: str,
    poisoned: Optional[dict[str, Any]] = None,
    call: Optional[dict[str, Any]] = None,
    call_label: int = 0,
    attack_type: str = "benign",
    **extras: Any,
) -> Episode:
    tools = list(clean)
    if poisoned is not None:
        tools.insert(_position(eid, len(tools)), poisoned)
    b = EpisodeBuilder(eid, BENCHMARK)
    b.add("M_cap", "host", server, rpc_request("tools/list", None, 1))
    b.add("M_cap", server, "host", rpc_result({"tools": tools}, 1), label=int(poisoned is not None))
    b.add("M_comm", "user", "host", {"role": "user", "content": query})
    if call is not None:
        b.add("M_tool", "host", server, rpc_request("tools/call", call, 2), label=call_label)
    return b.build(poisoned is not None, attack_type, group, server=server, **extras)


def load_mcptox_episodes(
    root: str,
    split: str = "all",
    include_wrong_data: bool = False,
    include_benign: bool = True,
    with_model_call: Optional[str] = None,
    group_by: str = "server",
) -> list[Episode]:
    """Load MCPTox as labelled episodes. ``root`` is the repository clone."""
    path = os.path.join(root, DATA_FILE)
    require_file(path, "Clone https://github.com/zhiqiangwang4/MCPTox-Benchmark and pass the clone as root.")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    episodes: list[Episode] = []
    for server, s in data["servers"].items():
        clean = parse_tool_blocks(s.get("clean_system_promot", ""))
        category = str(s.get("category_id"))
        group = server if group_by == "server" else None
        queries: list[str] = []
        for i, inst in enumerate(s.get("malicious_instance", [])):
            if inst.get("wrong_data", 0) != 0 and not include_wrong_data:
                continue
            poisoned_list = parse_tool_blocks(inst.get("poisoned_tool", ""))
            if not poisoned_list:
                continue
            poisoned = poisoned_list[0]
            meta = inst.get("metadata") or {}
            for datum in inst.get("datas", [])[:1]:
                labels, responses = _outcomes(datum)
                judged = [v for v in labels.values() if v and v != "None"]
                call, call_label = None, 0
                if with_model_call is not None:
                    call = parse_model_call(responses.get(with_model_call, ""))
                    call_label = int(labels.get(with_model_call) == SUCCESS)
                eid = f"{BENCHMARK}:{server}:{i}"
                episodes.append(_episode(
                    eid, server, clean, datum.get("query", ""), group or eid, poisoned, call, call_label,
                    attack_type=str(meta.get("security risk", "unknown")),
                    category=category,
                    paradigm=meta.get("paradigm"),
                    wrapper=wrapper_style(poisoned["description"]),
                    poisoned_tool=poisoned["name"],
                    wrong_data=inst.get("wrong_data", 0),
                    outcomes=labels,
                    success_rate=(sum(v == SUCCESS for v in judged) / len(judged)) if judged else None,
                ))
                queries.append(datum.get("query", ""))
        if include_benign:
            seen: set[str] = set()
            for q in queries + list(s.get("clean_querys") or []):
                if not q or q in seen:
                    continue
                seen.add(q)
                eid = f"{BENCHMARK}:{server}:benign:{len(seen) - 1}"
                episodes.append(_episode(eid, server, clean, q, group or eid, category=category))

    return split_episodes(episodes, split, stratum=lambda e: e.extras.get("category", "?"))


def load_mcptox(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """Registry entry point: MCPTox messages (see module docstring)."""
    yield from flatten(load_mcptox_episodes(root, split))
