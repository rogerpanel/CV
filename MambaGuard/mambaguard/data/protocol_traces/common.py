"""Shared episode model, JSON-RPC builders, tool extraction and splits.

An *episode* is one detection window: a short, ordered list of protocol
messages exchanged around a single user task. Every benchmark loader returns
episodes; ``flatten`` turns them into :class:`ProtocolMessage` streams for the
registry API.

Label convention
----------------
* ``ProtocolMessage.label`` is 1 when the message carries attacker-controlled
  content (a poisoned tool description, an injected tool result, a forged
  Agent Card, ...), else 0.
* ``Episode.is_attack`` is 1 when any message in the window is attacker
  controlled. ``Episode.attack_type`` is the benchmark's own attack name, or
  ``"benign"``.
* Whether the attack *succeeded* against some LLM is a separate question; when a
  benchmark records it, it is kept in ``Episode.extras["outcomes"]``.

Timestamps
----------
None of these benchmarks record wall-clock times, so ``t_m`` is synthetic: it
orders messages inside an episode and keeps episodes disjoint. Do not use
``temporal_split`` on these data; use :func:`split_episodes`, which splits by
group (server, case, ...) so the same group never appears in two splits.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from ..canonicalize import ProtocolMessage

EPISODE_SPACING_S = 1000.0
SPLITS = ("train", "val", "test")


@dataclass
class Episode:
    """One labelled detection window."""

    episode_id: str
    benchmark: str
    messages: list[ProtocolMessage]
    is_attack: bool
    attack_type: str
    group: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> int:
        return int(self.is_attack)


class EpisodeBuilder:
    """Accumulate the messages of one episode with increasing timestamps."""

    def __init__(self, episode_id: str, benchmark: str, t0: float = 0.0) -> None:
        self.episode_id = episode_id
        self.benchmark = benchmark
        self.t = t0
        self.messages: list[ProtocolMessage] = []

    def add(
        self,
        tau: str,
        src: str,
        dst: str,
        payload: Any,
        label: int = 0,
        dt: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProtocolMessage:
        self.t += dt
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=False)
        msg = ProtocolMessage(
            tau=tau,
            src=src,
            dst=dst,
            payload=text,
            metadata=dict(metadata or {}),
            t_m=self.t,
            label=int(label),
            msg_id=f"{self.episode_id}#{len(self.messages)}",
            extras={"episode_id": self.episode_id, "benchmark": self.benchmark, "step": len(self.messages)},
        )
        self.messages.append(msg)
        return msg

    def build(self, is_attack: bool, attack_type: str, group: str, **extras: Any) -> Episode:
        for m in self.messages:
            m.extras.update({"attack_type": attack_type, "is_attack_episode": int(is_attack), "group": group})
        return Episode(self.episode_id, self.benchmark, self.messages, bool(is_attack), attack_type, group, extras)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 payloads (MCP and A2A both use JSON-RPC 2.0)
# ---------------------------------------------------------------------------
def rpc_request(method: str, params: Optional[dict[str, Any]], req_id: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        out["params"] = params
    return out


def rpc_result(result: Any, req_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_notification(method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        out["params"] = params
    return out


def mcp_tool(name: str, description: str, params: Optional[Sequence[str]] = None) -> dict[str, Any]:
    props = {p: {"type": "string"} for p in (params or [])}
    return {"name": name, "description": description, "inputSchema": {"type": "object", "properties": props}}


# ---------------------------------------------------------------------------
# Static extraction of MCP tool definitions from server source (never executed)
# ---------------------------------------------------------------------------
def _is_tool_decorator(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    return isinstance(target, ast.Attribute) and target.attr == "tool"


def _decorator_description(dec: ast.expr) -> Optional[str]:
    if isinstance(dec, ast.Call):
        for kw in dec.keywords:
            if kw.arg == "description" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
    return None


def extract_server(path: str) -> tuple[Optional[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a FastMCP-style server file with :mod:`ast`.

    Returns ``(server_name, tools, resources)``. The file is parsed, not run.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    server_name: Optional[str] = None
    tools: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None)) == "FastMCP"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            server_name = str(node.args[0].value)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = [a.arg for a in node.args.args if a.arg not in ("self", "ctx")]
        doc = (ast.get_docstring(node) or "").strip()
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if _is_tool_decorator(dec):
                tools.append(mcp_tool(node.name, _decorator_description(dec) or doc, params))
            elif isinstance(target, ast.Attribute) and target.attr == "resource":
                uri = dec.args[0].value if isinstance(dec, ast.Call) and dec.args and isinstance(dec.args[0], ast.Constant) else ""
                resources.append({"uri": str(uri), "name": node.name, "description": doc})
    return server_name, tools, resources


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def _h(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()


def assign_groups(
    episodes: Sequence[Episode],
    fracs: tuple[float, float] = (0.7, 0.15),
    stratum: Optional[Callable[[Episode], str]] = None,
    salt: str = "mambaguard",
) -> dict[str, str]:
    """Map every group to ``train``/``val``/``test``, deterministically.

    Groups are stratified (default stratum: the attack type of the group's
    first attack episode, else ``benign``) and ordered by a salted hash inside
    each stratum, so the assignment does not depend on file order.
    """
    stratum = stratum or (lambda e: e.attack_type)
    group_stratum: dict[str, str] = {}
    for e in episodes:
        s = stratum(e)
        if e.group not in group_stratum or (group_stratum[e.group] == "benign" and e.is_attack):
            group_stratum[e.group] = s
    by_stratum: dict[str, list[str]] = {}
    for g, s in group_stratum.items():
        by_stratum.setdefault(s, []).append(g)
    out: dict[str, str] = {}
    tr, va = fracs
    for s, groups in by_stratum.items():
        groups.sort(key=lambda g: _h(g, salt))
        n = len(groups)
        for i, g in enumerate(groups):
            q = (i + 0.5) / n
            out[g] = "train" if q < tr else ("val" if q < tr + va else "test")
    return out


def split_episodes(
    episodes: Sequence[Episode],
    split: str,
    fracs: tuple[float, float] = (0.7, 0.15),
    stratum: Optional[Callable[[Episode], str]] = None,
) -> list[Episode]:
    if split == "all":
        return list(episodes)
    if split not in SPLITS:
        raise ValueError(f"unknown split={split!r}; expected one of {SPLITS} or 'all'")
    assignment = assign_groups(episodes, fracs, stratum)
    return [e for e in episodes if assignment[e.group] == split]


def flatten(episodes: Iterable[Episode]) -> Iterator[ProtocolMessage]:
    """Yield the messages of all episodes with disjoint timestamps."""
    for k, ep in enumerate(episodes):
        for m in ep.messages:
            m.t_m = k * EPISODE_SPACING_S + m.t_m
            yield m


# ---------------------------------------------------------------------------
# JSONL cache
# ---------------------------------------------------------------------------
def episodes_to_jsonl(episodes: Iterable[Episode], path: str) -> int:
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps({
                "episode_id": e.episode_id, "benchmark": e.benchmark, "is_attack": int(e.is_attack),
                "attack_type": e.attack_type, "group": e.group, "extras": e.extras,
                "messages": [{"tau": m.tau, "src": m.src, "dst": m.dst, "payload": m.payload,
                              "t_m": m.t_m, "label": m.label, "metadata": m.metadata} for m in e.messages],
            }, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def episodes_from_jsonl(path: str) -> list[Episode]:
    out: list[Episode] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            b = EpisodeBuilder(r["episode_id"], r["benchmark"])
            for m in r["messages"]:
                b.t = m["t_m"]
                b.add(m["tau"], m["src"], m["dst"], m["payload"], m["label"], dt=0.0, metadata=m.get("metadata"))
            out.append(b.build(bool(r["is_attack"]), r["attack_type"], r["group"], **r.get("extras", {})))
    return out


def require_file(path: str, hint: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {hint}")


def summarise(episodes: Sequence[Episode]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for e in episodes:
        by_type[e.attack_type] = by_type.get(e.attack_type, 0) + 1
    return {
        "episodes": len(episodes),
        "attack_episodes": sum(e.is_attack for e in episodes),
        "benign_episodes": sum(not e.is_attack for e in episodes),
        "messages": sum(len(e.messages) for e in episodes),
        "attack_messages": sum(m.label == 1 for e in episodes for m in e.messages),
        "groups": len({e.group for e in episodes}),
        "attack_types": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
    }
