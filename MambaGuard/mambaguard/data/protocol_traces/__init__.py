"""Loaders for protocol-native agent-security benchmarks.

=================  ===================================================  ==========================
name               source                                               root argument
=================  ===================================================  ==========================
``mcpsecbench``    github.com/AIS2Lab/MCPSecBench                       repository clone
``mcp_safetybench`` github.com/xjzzzzzzzz/MCPSafety                     repository clone
``mcptox``         github.com/zhiqiangwang4/MCPTox-Benchmark            repository clone
``a2a_sandbox``    generated from the 11 findings of A2ABreak           A2ABreak clone, cache
                   (github.com/arlotfi79/A2ABreak)                      folder, or ``"builtin"``
=================  ===================================================  ==========================

Each module exposes ``load_<name>_episodes(root, split, ...)`` returning
:class:`Episode` objects, and ``load_<name>(root, split)`` yielding flattened
:class:`~mambaguard.data.canonicalize.ProtocolMessage` streams for
``DATASET_REGISTRY``. See ``reproducibility/PROTOCOL_BENCHMARKS.md`` for
download commands, pinned commits and expected counts.
"""
from __future__ import annotations

from .agentdojo import available_models, load_agentdojo, load_agentdojo_episodes
from .injecagent import load_injecagent, load_injecagent_episodes
from .a2a_sandbox import generate_a2a_sandbox, load_a2a_sandbox, load_a2a_sandbox_episodes, write_a2a_sandbox
from .common import Episode, episodes_from_jsonl, episodes_to_jsonl, flatten, split_episodes, summarise
from .mcp_safetybench import load_mcp_safetybench, load_mcp_safetybench_episodes
from .mcpsecbench import load_mcpsecbench, load_mcpsecbench_episodes, load_mcpsecbench_outcomes
from .mcptox import load_mcptox, load_mcptox_episodes

EPISODE_LOADERS = {
    "mcpsecbench": load_mcpsecbench_episodes,
    "mcp_safetybench": load_mcp_safetybench_episodes,
    "mcptox": load_mcptox_episodes,
    "a2a_sandbox": load_a2a_sandbox_episodes,
    "agentdojo": load_agentdojo_episodes,
    "injecagent": load_injecagent_episodes,
}

__all__ = [
    "load_injecagent_episodes",
    "load_injecagent",
    "load_agentdojo_episodes",
    "load_agentdojo",
    "available_models",
    "Episode",
    "EPISODE_LOADERS",
    "episodes_from_jsonl",
    "episodes_to_jsonl",
    "flatten",
    "generate_a2a_sandbox",
    "load_a2a_sandbox",
    "load_a2a_sandbox_episodes",
    "load_mcp_safetybench",
    "load_mcp_safetybench_episodes",
    "load_mcpsecbench",
    "load_mcpsecbench_episodes",
    "load_mcpsecbench_outcomes",
    "load_mcptox",
    "load_mcptox_episodes",
    "split_episodes",
    "summarise",
    "write_a2a_sandbox",
]
