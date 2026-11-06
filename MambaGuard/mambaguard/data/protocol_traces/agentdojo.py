"""AgentDojo: recorded tool-calling agent runs with prompt injection.

Source: E. Debenedetti et al., "AgentDojo", NeurIPS 2024 Datasets & Benchmarks,
arXiv:2406.13352. Data: https://github.com/ethz-spylab/agentdojo (MIT License).

Unlike the MCP benchmarks, AgentDojo ships *recorded* agent traces under
``runs/<pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json``, so no
agent is re-run. Each file is one episode:

* ``messages``: system / user / assistant / tool turns. Assistant turns carry
  ``tool_calls`` (``function`` + ``args``); tool turns carry the ``content``
  returned to the agent.
* ``injections``: ``{placeholder: attacker_text}``, empty for benign runs.
* ``security``: ``True`` iff the attack succeeded. ``utility``: task solved.

Episode construction (one message per turn):

    user      -> agent   user task                                 label 0
    agent     -> <tool>  tools/call (function + args)              label 0
    <tool>    -> agent   tool result                               label 1 iff
                         it contains injected attacker text
    ...                  (further assistant/tool turns)

A tool result is labelled 1 when a distinctive slice of any injected string
appears in it after whitespace/escape normalisation (the benchmark reformats
injected text: blank-line collapse, YAML folding). This matcher labels a tool
result in every attacked run whose injection actually surfaced, and never fires
on benign runs (verified on 400 attacked + 97 benign runs).

By default only *observable* attacked runs are kept (the injection reached a
tool result), so every attack episode has at least one labelled message, and
benign runs come from the ``none/none.json`` files. Attacked runs whose
injection never surfaced (the attack failed before reaching the poisoned
resource) are available with ``observable_only=False`` and are then labelled on
the episode only.

Splits are grouped by ``(suite, user_task)`` so a user task never crosses
splits, and stratified by suite. One model's runs are loaded at a time
(``model=...``); the model is part of no label.
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
    flatten,
    require_file,
    rpc_request,
    rpc_result,
    split_episodes,
)

BENCHMARK = "agentdojo"
RUNS = "runs"
_WS = re.compile(r"\s+")


def _norm(content: Any) -> str:
    if isinstance(content, list):
        content = " ".join(b.get("content", "") if isinstance(b, dict) else str(b) for b in content)
    return _WS.sub(" ", str(content).replace("\\n", " ").replace("\\", " ")).strip().lower()


def _needle(injection: str) -> str:
    runs = re.findall(r"[A-Za-z0-9 ,.'\-]{20,}", injection)
    return _norm(max(runs, key=len) if runs else injection)[:80]


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(b.get("content", "") if isinstance(b, dict) else str(b) for b in content)
    return "" if content is None else str(content)


def available_models(root: str) -> list[str]:
    runs = os.path.join(root, RUNS)
    if not os.path.isdir(runs):
        return []
    return sorted(d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d)))


def _pick_model(root: str, model: Optional[str]) -> str:
    models = available_models(root)
    if not models:
        raise FileNotFoundError(
            f"{os.path.join(root, RUNS)} not found or empty. Clone "
            "https://github.com/ethz-spylab/agentdojo and pass the clone as root."
        )
    if model is not None:
        if model not in models:
            raise ValueError(f"model {model!r} not in runs/; available: {models}")
        return model
    # Deterministic default: a widely used model present in the standard release.
    for preferred in ("claude-3-5-sonnet-20241022", "gpt-4o-2024-05-13"):
        if preferred in models:
            return preferred
    return min((m for m in models if "-repeat_user_prompt" not in m), default=models[0])


def _episode(path: str, model: str) -> Optional[Episode]:
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    suite = run.get("suite_name", "unknown")
    user_task = run.get("user_task_id") or os.path.basename(os.path.dirname(os.path.dirname(path)))
    attack = run.get("attack_type")
    is_attack = bool(run.get("injections"))
    needles = [_needle(v) for v in (run.get("injections") or {}).values()]
    eid = f"{BENCHMARK}:{model}:{suite}:{user_task}:{attack or 'none'}:{run.get('injection_task_id') or 'none'}"

    b = EpisodeBuilder(eid, BENCHMARK)
    observed = False
    for m in run.get("messages", []):
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            b.add("M_comm", "user", "agent", {"role": "user", "content": _content_text(m.get("content"))})
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                b.add("M_tool", "agent", call.get("function", "tool"),
                      rpc_request("tools/call", {"name": call.get("function"), "arguments": call.get("args", {})}, call.get("id")))
            text = _content_text(m.get("content"))
            if text.strip() and not (m.get("tool_calls")):
                b.add("M_comm", "agent", "user", {"role": "assistant", "content": text})
        elif role == "tool":
            norm = _norm(m.get("content"))
            injected = is_attack and any(n and n in norm for n in needles)
            observed = observed or injected
            tool_name = (m.get("tool_call") or {}).get("function", "tool")
            b.add("M_data", tool_name, "agent",
                  rpc_result({"content": _content_text(m.get("content"))}, m.get("tool_call_id")),
                  label=int(injected),
                  metadata={"error": bool(m.get("error"))})
    if not b.messages:
        return None
    return b.build(is_attack, attack or "benign", group=f"{suite}:{user_task}",
                   suite=suite, user_task=user_task, injection_task=run.get("injection_task_id"),
                   attack_type_raw=attack, security=run.get("security"), utility=run.get("utility"),
                   model=model, observable=observed)


def load_agentdojo_episodes(
    root: str,
    split: str = "all",
    model: Optional[str] = None,
    attacks: Optional[list[str]] = None,
    include_benign: bool = True,
    observable_only: bool = True,
    include_injection_task_runs: bool = False,
) -> list[Episode]:
    """Load recorded AgentDojo runs for one model as labelled episodes."""
    model = _pick_model(root, model)
    base = os.path.join(root, RUNS, model)
    episodes: list[Episode] = []
    for path in sorted(glob.glob(os.path.join(base, "*", "*", "*", "*.json"))):
        parts = path.split(os.sep)
        task_dir, attack = parts[-3], parts[-2]
        if task_dir.startswith("injection_task_") and not include_injection_task_runs:
            continue  # the "user task" here is the attacker goal: not benign, not a normal episode
        benign = attack == "none"
        if benign and not include_benign:
            continue
        if not benign and attacks is not None and attack not in attacks:
            continue
        ep = _episode(path, model)
        if ep is None:
            continue
        if ep.is_attack and observable_only and not ep.extras["observable"]:
            continue
        episodes.append(ep)
    return split_episodes(episodes, split, stratum=lambda e: e.extras.get("suite", "?"))


def load_agentdojo(root: str, split: str = "train") -> Iterator[ProtocolMessage]:
    """Registry entry point: AgentDojo messages (see module docstring)."""
    yield from flatten(load_agentdojo_episodes(root, split))
