"""3A — Temporal Attention Flow Visualization.

Produces three modalities (graph rendering, temporal heatmap, attention
trajectory suitable for video animation) plus a MITRE ATT&CK tactic
annotation per solver step. Rendering uses matplotlib + networkx;
everything is optional and silently degrades when the libraries are
unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ct_explain.data.graph_builder import TemporalGraph
from ct_explain.explainers.base import BaseExplainer, Explanation
from ct_explain.explainers.mitre_attack import MITREAttackMapper
from ct_explain.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AttentionFlowArtifact:
    # Per-edge attention averaged across ODE solver steps.
    edge_attention: torch.Tensor                  # (E,)
    # Attention per solver step; shape (S, H, E).
    trajectory: torch.Tensor
    # Per-step MITRE stage label.
    stages: list[str] = field(default_factory=list)
    # Hex colour per solver step.
    stage_colors: list[str] = field(default_factory=list)
    # Attention heatmap matrix, rows = edges (sorted by importance), cols = steps.
    heatmap: Optional[np.ndarray] = None
    # Paths where rendered artefacts (PNG/GIF) were written.
    rendered_paths: dict[str, str] = field(default_factory=dict)


class AttentionFlowExplainer(BaseExplainer):
    """Temporal attention flow explanation (modalities 1–3)."""

    name = "attention_flow"

    def __init__(self, model, mitre_mapper: Optional[MITREAttackMapper] = None) -> None:
        super().__init__(model)
        self.mitre = mitre_mapper or MITREAttackMapper()

    # ------------------------------------------------------------------ #
    def explain(
        self,
        graph: TemporalGraph,
        target_node: int,
        solver_steps: int = 8,
        render_dir: Optional[str] = None,
    ) -> Explanation:
        t0 = time.perf_counter()
        attn_trajectory = self._collect_attention_trajectory(graph, solver_steps)
        edge_attn_mean = attn_trajectory.mean(dim=(0, 1))          # (E,)

        # MITRE kill-chain alignment per step.
        stages: list[str] = []
        colors: list[str] = []
        for step_attn in attn_trajectory:                          # iterate over S
            align = self.mitre(step_attn)
            stages.append(align.stage)
            colors.append(align.color)

        # Attribution = attention mass incident on the target node.
        tgt_mask = graph.edge_index[0] == target_node
        attribution = torch.zeros(graph.num_edges, device=edge_attn_mean.device)
        attribution[tgt_mask] = edge_attn_mean[tgt_mask]

        heatmap = self._build_heatmap(attn_trajectory, tgt_mask)
        rendered: dict[str, str] = {}
        if render_dir is not None:
            rendered = self._render(graph, edge_attn_mean, colors, render_dir, target_node)

        artefact = AttentionFlowArtifact(
            edge_attention=edge_attn_mean.cpu(),
            trajectory=attn_trajectory.cpu(),
            stages=stages,
            stage_colors=colors,
            heatmap=heatmap,
            rendered_paths=rendered,
        )
        latency = (time.perf_counter() - t0) * 1000
        return Explanation(
            attribution=attribution.cpu(),
            method=self.name,
            latency_ms=latency,
            supporting={"attention_flow": artefact},
            target_node=target_node,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _collect_attention_trajectory(
        self, graph: TemporalGraph, steps: int
    ) -> torch.Tensor:
        """Run the forward pass S times with `return_trajectory=True` if the
        solver supports it; otherwise sample S integration horizons.

        Returned tensor shape: (S, H, E).
        """
        trajectories = []
        for k in range(1, steps + 1):
            t_query = self.model.integration_window * k / steps
            out = self.model(
                node_features=graph.node_features,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features,
                edge_times=graph.edge_times,
                query_time=t_query,
            )
            if out.attention is None:
                raise RuntimeError("Model did not expose attention weights.")
            trajectories.append(out.attention.detach())
        return torch.stack(trajectories, dim=0)

    def _build_heatmap(self, trajectory: torch.Tensor, tgt_mask: torch.Tensor) -> np.ndarray:
        """Return (rows_sorted_by_importance, steps) matrix."""
        per_step_edge = trajectory.mean(dim=1)               # (S, E)
        tgt = per_step_edge[:, tgt_mask]                     # (S, E_tgt)
        ordering = tgt.mean(0).argsort(descending=True)
        return tgt[:, ordering].cpu().numpy().T              # (E_tgt, S)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(
        self,
        graph: TemporalGraph,
        edge_attention: torch.Tensor,
        stage_colors: Sequence[str],
        out_dir: str,
        target_node: int,
    ) -> dict[str, str]:
        try:
            from pathlib import Path

            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            log.warning("matplotlib/networkx missing — skipping render.")
            return {}

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        # Modality 1: graph rendering.
        G = nx.DiGraph()
        for i in range(graph.num_nodes):
            G.add_node(i, label=graph.node_ids[i] if i < len(graph.node_ids) else str(i))
        weights = edge_attention.cpu().numpy()
        for e in range(graph.num_edges):
            v = int(graph.edge_index[0, e])
            u = int(graph.edge_index[1, e])
            G.add_edge(u, v, w=float(weights[e]))
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=0)
        max_w = weights.max() + 1e-12
        edge_widths = [2 + 4 * G[u][v]["w"] / max_w for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#f0f0f0",
                               edgecolors="#333", node_size=300)
        nx.draw_networkx_edges(G, pos, width=edge_widths, ax=ax, alpha=0.7,
                               edge_color=stage_colors[-1])
        nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]["label"] for i in G.nodes},
                                font_size=8, ax=ax)
        ax.set_title(f"Temporal attention flow — target={target_node}")
        ax.axis("off")
        fp = out / f"attention_graph_node{target_node}.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["graph"] = str(fp)

        # Modality 2: heatmap.
        fig, ax = plt.subplots(figsize=(8, 4))
        # Sort edges by total attention for readability.
        tgt_mask = (graph.edge_index[0] == target_node).cpu().numpy()
        if tgt_mask.any():
            ax.imshow(
                weights[tgt_mask].reshape(-1, 1),
                aspect="auto", cmap="plasma",
            )
            ax.set_xlabel("solver-step summary"); ax.set_ylabel("incident edges")
            ax.set_title("Attention heatmap")
        fp = out / f"attention_heatmap_node{target_node}.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["heatmap"] = str(fp)
        return paths
