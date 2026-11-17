"""Dynamic execution graph G_exec(t) for multi-agent SOC workflows.

Five node types: agent, action, tool, message, observation.
Typed temporal edges track dependencies:
    agent --executes--> action
    action --invokes--> tool
    tool  --returns-->  observation
    agent --sends-->    message
    message --received_by--> agent
    action --depends_on--> action          (dataflow / control flow)

This is a minimal, dependency-free graph container used by ConformalGuard's
cascade-risk computation. For integration with the main CT-TGNN encoder,
`to_temporal_graph()` converts it into the standard TemporalGraph
representation with a single shared feature schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch

from ct_explain.data.graph_builder import TemporalGraph


class NodeKind(str, Enum):
    AGENT = "agent"
    ACTION = "action"
    TOOL = "tool"
    MESSAGE = "message"
    OBSERVATION = "observation"


class EdgeKind(str, Enum):
    EXECUTES = "executes"
    INVOKES = "invokes"
    RETURNS = "returns"
    SENDS = "sends"
    RECEIVED_BY = "received_by"
    DEPENDS_ON = "depends_on"


@dataclass
class AgentAction:
    node_id: str
    agent_id: str
    timestamp: float
    payload: dict
    privilege_level: int = 0          # 0 = read-only, 3 = admin
    is_action: bool = True


@dataclass
class _Edge:
    src: str
    dst: str
    kind: EdgeKind
    timestamp: float


class DynamicExecutionGraph:
    """Mutable temporal DAG of an agent workflow."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._edges: list[_Edge] = []

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def add_node(
        self,
        node_id: str,
        kind: NodeKind,
        timestamp: float,
        attrs: Optional[dict] = None,
    ) -> None:
        self._nodes[node_id] = {
            "kind": kind,
            "timestamp": timestamp,
            "attrs": attrs or {},
        }

    def add_edge(
        self, src: str, dst: str, kind: EdgeKind, timestamp: float
    ) -> None:
        if src not in self._nodes or dst not in self._nodes:
            raise KeyError("Both endpoints must be registered as nodes.")
        self._edges.append(_Edge(src=src, dst=dst, kind=kind, timestamp=timestamp))

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def actions(self) -> list[AgentAction]:
        result: list[AgentAction] = []
        for nid, attrs in self._nodes.items():
            if attrs["kind"] == NodeKind.ACTION:
                result.append(
                    AgentAction(
                        node_id=nid,
                        agent_id=attrs["attrs"].get("agent_id", ""),
                        timestamp=attrs["timestamp"],
                        payload=attrs["attrs"],
                        privilege_level=int(
                            attrs["attrs"].get("privilege_level", 0)
                        ),
                    )
                )
        return sorted(result, key=lambda a: a.timestamp)

    def k_hop_neighbours(self, node_id: str, k: int) -> set[str]:
        """Undirected k-hop BFS."""
        if node_id not in self._nodes:
            return set()
        frontier = {node_id}
        visited = {node_id}
        for _ in range(k):
            new_front: set[str] = set()
            for e in self._edges:
                if e.src in frontier and e.dst not in visited:
                    new_front.add(e.dst)
                if e.dst in frontier and e.src not in visited:
                    new_front.add(e.src)
            visited |= new_front
            frontier = new_front
            if not frontier:
                break
        visited.discard(node_id)
        return visited

    def influence(self, src: str, dst: str) -> float:
        """Temporal-attention-derived influence score Inf(src → dst).

        Uses time decay × path count. Later, when a CT-TGNN encoder is
        attached (via `attach_encoder`), this is overridden to return the
        actual attention score. The default fallback keeps unit tests
        runnable without a model.
        """
        if src == dst:
            return 1.0
        matches = [
            e for e in self._edges
            if e.src == src and e.dst == dst
        ]
        if not matches:
            return 0.0
        dt = np.mean([abs(e.timestamp - self._nodes[dst]["timestamp"]) for e in matches])
        return float(np.exp(-0.01 * dt))

    # ------------------------------------------------------------------ #
    # Materialization
    # ------------------------------------------------------------------ #
    def to_temporal_graph(
        self, feature_fn=None, feature_dim: int = 16
    ) -> TemporalGraph:
        ordered_ids = list(self._nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(ordered_ids)}
        if feature_fn is None:
            def feature_fn(_n: str, attrs: dict) -> np.ndarray:
                v = np.zeros(feature_dim, dtype=np.float32)
                kind_idx = list(NodeKind).index(attrs["kind"])
                v[kind_idx] = 1.0
                v[-1] = float(attrs["attrs"].get("privilege_level", 0))
                return v

        node_feats = np.stack(
            [feature_fn(n, self._nodes[n]) for n in ordered_ids], axis=0
        )

        edge_index = np.zeros((2, len(self._edges)), dtype=np.int64)
        edge_feats = np.zeros((len(self._edges), feature_dim), dtype=np.float32)
        edge_times = np.zeros(len(self._edges), dtype=np.float32)
        for j, e in enumerate(self._edges):
            edge_index[0, j] = id_to_idx[e.dst]
            edge_index[1, j] = id_to_idx[e.src]
            edge_times[j] = e.timestamp
            kind_idx = list(EdgeKind).index(e.kind)
            edge_feats[j, kind_idx] = 1.0

        return TemporalGraph(
            node_ids=ordered_ids,
            node_features=torch.from_numpy(node_feats),
            edge_index=torch.from_numpy(edge_index),
            edge_features=torch.from_numpy(edge_feats),
            edge_times=torch.from_numpy(edge_times),
            metadata={"kind": "execution_graph"},
        )
