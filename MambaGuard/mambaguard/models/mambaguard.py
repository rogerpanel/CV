"""MambaGuard top-level model: SSM temporal backbone + temporal GATv2 + detection head."""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detection_head import DetectionHead
from .gatv2_temporal import TemporalGATv2Layer
from .selective_ssm import SelectiveSSMBlock


class RMSNorm(nn.Module):
    """Root-mean-square layer norm; cheaper than LayerNorm, no centring (Zhang & Sennrich 2019)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


@dataclass
class MambaGuardConfig:
    """Hyper-parameter container for :class:`MambaGuard`."""

    d_p: int = 384                  # canonicalised message embedding (MiniLM)
    d_mu: int = 32                  # metadata feature width
    d_model: int = 256
    d_state: int = 16
    expand: int = 2
    kernel_size: int = 4
    n_blocks: int = 4
    n_gat_layers: int = 2
    gat_heads: int = 4
    edge_dim: int = 16
    d_T: int = 64
    num_classes: int = 34
    dropout: float = 0.1
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    readout: str = "graph"          # "graph": per-agent SSM + temporal GATv2; "flat": window as one sequence
    backbone: str = "ssm"           # "ssm" (selective scan) or "transformer" (attention ablation)
    chunk_dim: int = 0              # >0: per-chunk payload features are scored by a small MLP and
    chunk_hidden: int = 128         #     max+mean pooled per message (multiple-instance pooling)


class MambaGuard(nn.Module):
    """Selective-SSM temporal encoder fused with a temporal GATv2 over the agent graph."""

    def __init__(self, cfg: Optional[MambaGuardConfig] = None) -> None:
        super().__init__()
        cfg = cfg or MambaGuardConfig()
        self.cfg = cfg

        self.chunk_mlp = (nn.Sequential(nn.Linear(cfg.chunk_dim, cfg.chunk_hidden), nn.GELU(),
                                        nn.Linear(cfg.chunk_hidden, cfg.chunk_hidden))
                          if cfg.chunk_dim > 0 else None)
        d_chunk_out = 2 * cfg.chunk_hidden if cfg.chunk_dim > 0 else 0
        self.input_proj = nn.Linear(cfg.d_p + d_chunk_out + cfg.d_mu, cfg.d_model)
        self.input_norm = RMSNorm(cfg.d_model)

        # Stack of 4 selective SSM blocks with residual + RMSNorm (pre-norm Mamba layout).
        if cfg.backbone not in ("ssm", "transformer"):
            raise ValueError(f"unknown backbone {cfg.backbone!r}")
        if cfg.readout not in ("graph", "flat"):
            raise ValueError(f"unknown readout {cfg.readout!r}")
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(cfg.d_model, nhead=4, dim_feedforward=2 * cfg.d_model,
                                           dropout=cfg.dropout, batch_first=True)
                if cfg.backbone == "transformer" else
                SelectiveSSMBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    expand=cfg.expand,
                    dt_min=cfg.dt_min,
                    dt_max=cfg.dt_max,
                    kernel_size=cfg.kernel_size,
                )
                for _ in range(cfg.n_blocks)
            ]
        )
        self.norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_blocks)])
        self.final_norm = RMSNorm(cfg.d_model)

        # Temporal GATv2 over per-agent pooled features z^a_t.
        self.gat_layers = nn.ModuleList()
        for i in range(cfg.n_gat_layers):
            self.gat_layers.append(
                TemporalGATv2Layer(
                    d_in=cfg.d_model,
                    d_out=cfg.d_model,
                    edge_dim=cfg.edge_dim,
                    heads=cfg.gat_heads,
                    d_T=cfg.d_T,
                    dropout=cfg.dropout,
                    lipschitz_norm=True,
                    concat=False,
                )
            )

        self.head = DetectionHead(
            d_in=cfg.d_model, num_classes=cfg.num_classes, dropout=cfg.dropout
        )

    # ------------------------------------------------------------------
    def _encode_agent(self, p: torch.Tensor, mu: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-agent sequence stack. ``p``: (B, L, d_p); ``mu``: (B, L, d_mu)."""
        x = torch.cat([p, mu], dim=-1)
        x = self.input_norm(self.input_proj(x))
        for blk, norm in zip(self.blocks, self.norms):
            if isinstance(blk, nn.TransformerEncoderLayer):
                L = x.shape[1]
                causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
                pad = None if mask is None else ~mask
                x = blk(norm(x), src_mask=causal, src_key_padding_mask=pad)
            else:
                x = x + blk(norm(x))       # pre-norm residual; causality preserved by blk
        return self.final_norm(x)

    @staticmethod
    def _masked_mean(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=1)
        m = mask.to(h.dtype).unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _forward_graph(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Per-window protocol graph: every endpoint is a node whose incident
        messages (in time order) are encoded by the sequence stack; every message
        is a typed, time-stamped edge for the temporal GATv2 layers."""
        pe, mu = batch["payload_emb"], batch["metadata"]
        mask, et, ts = batch["attention_mask"], batch["edge_type"], batch["timestamps"]
        src, dst, n_nodes = batch["src_idx"], batch["dst_idx"], batch["n_nodes"]
        B, T, _ = pe.shape
        dev = pe.device
        offsets = torch.cumsum(n_nodes, 0) - n_nodes
        V = int(n_nodes.sum().item())

        seqs: list[list[tuple[int, int]]] = [[] for _ in range(V)]
        e_src, e_dst, e_b, e_t = [], [], [], []
        lengths = mask.sum(dim=1).tolist()
        src_l, dst_l, off_l = src.tolist(), dst.tolist(), offsets.tolist()
        for b in range(B):
            for t in range(int(lengths[b])):
                u, v = off_l[b] + src_l[b][t], off_l[b] + dst_l[b][t]
                seqs[u].append((b, t))
                if v != u:
                    seqs[v].append((b, t))
                e_src.append(u); e_dst.append(v); e_b.append(b); e_t.append(t)
        Lmax = max(1, max(len(q) for q in seqs))
        gb = torch.zeros((V, Lmax), dtype=torch.long)
        gt = torch.zeros((V, Lmax), dtype=torch.long)
        nmask = torch.zeros((V, Lmax), dtype=torch.bool)
        for v, q in enumerate(seqs):
            if q:
                bb, tt = zip(*q)
                gb[v, :len(q)] = torch.tensor(bb)
                gt[v, :len(q)] = torch.tensor(tt)
                nmask[v, :len(q)] = True
        gb, gt, nmask = gb.to(dev), gt.to(dev), nmask.to(dev)
        h = self._encode_agent(pe[gb, gt], mu[gb, gt], nmask)       # (V, Lmax, d_model)
        z = self._masked_mean(h, nmask)                              # (V, d_model)

        if self.gat_layers and e_src:
            eb = torch.tensor(e_b, device=dev)
            etq = torch.tensor(e_t, device=dev)
            edge_index = torch.tensor([e_src, e_dst], dtype=torch.long, device=dev)
            onehot = F.one_hot(et[eb, etq], num_classes=5).to(pe.dtype)
            edge_attr = torch.cat([onehot, mu[eb, etq]], dim=-1)[:, : self.cfg.edge_dim]
            if edge_attr.shape[1] < self.cfg.edge_dim:
                edge_attr = F.pad(edge_attr, (0, self.cfg.edge_dim - edge_attr.shape[1]))
            t_end = (ts * mask).max(dim=1).values
            edge_time = (t_end[eb] - ts[eb, etq]).clamp(min=0.0)
            for layer in self.gat_layers:
                z = z + layer(z, edge_index, edge_attr=edge_attr, edge_time=edge_time)

        node_win = torch.repeat_interleave(torch.arange(B, device=dev), n_nodes.to(dev))
        mean = torch.zeros((B, z.shape[1]), device=dev, dtype=z.dtype).index_add_(0, node_win, z)
        mean = mean / n_nodes.to(dev).clamp(min=1).unsqueeze(-1).to(z.dtype)
        mx = torch.full((B, z.shape[1]), -1e4, device=dev, dtype=z.dtype).scatter_reduce(
            0, node_win.unsqueeze(-1).expand_as(z), z, reduce="amax", include_self=True)
        pooled = mean + mx
        return {"logits": self.head(pooled), "agent_features": z, "token_features": h}

    def forward(
        self,
        messages: Mapping[str, torch.Tensor],
        graph: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        ``messages`` keys (required):
            * ``p``  : (A, L, d_p) per-agent canonicalised message embeddings.
            * ``mu`` : (A, L, d_mu) per-agent metadata features.
        ``messages`` keys (optional):
            * ``mask`` : (A, L) boolean, True where positions are valid.

        ``graph`` keys (optional):
            * ``edge_index`` : (2, E)
            * ``edge_attr``  : (E, edge_dim)
            * ``edge_time``  : (E,)

        Returns dict with ``logits`` (A, num_classes) and ``agent_features``.
        """
        if self.chunk_mlp is not None and "chunk_emb" in messages:
            messages = dict(messages)
            messages["payload_emb"] = torch.cat(
                [messages["payload_emb"], self._pool_chunks(messages["chunk_emb"], messages["chunk_mask"])], dim=-1)
        if "p" not in messages and "payload_emb" in messages:
            # Batch produced by mambaguard.data.protocol_collate.
            if self.cfg.readout == "graph" and "src_idx" in messages:
                return self._forward_graph(messages)
            messages = {"p": messages["payload_emb"], "mu": messages["metadata"],
                        "mask": messages.get("attention_mask")}
        if graph is not None and not isinstance(graph, Mapping):
            graph = None
        p = messages["p"]
        mu = messages["mu"]
        if p.dim() != 3 or mu.dim() != 3:
            raise ValueError("messages['p'] and ['mu'] must be (A, L, *) tensors")

        mask = messages.get("mask")
        h = self._encode_agent(p, mu, mask)  # (A, L, d_model)
        z = self._masked_mean(h, mask)       # (A, d_model)

        if graph is not None and "edge_index" in graph:
            edge_index = graph["edge_index"]
            edge_attr = graph.get("edge_attr")
            edge_time = graph.get("edge_time")
            for layer in self.gat_layers:
                z = z + layer(z, edge_index, edge_attr=edge_attr, edge_time=edge_time)

        logits = self.head(z)
        return {"logits": logits, "agent_features": z, "token_features": h}

    def _pool_chunks(self, chunks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, K, d_c) chunk features -> (B, T, 2H): max and mean of per-chunk MLP scores.

        Both poolings are 1-Lipschitz, so the bound below stays finite; a single
        unusual chunk (a poisoned tool, an injected sentence) can set a max coordinate.
        """
        h = self.chunk_mlp(chunks)
        m = mask.unsqueeze(-1).to(h.dtype)
        any_valid = mask.any(dim=-1, keepdim=True).to(h.dtype)
        mx = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=2)
        mx = torch.where(any_valid > 0, mx, torch.zeros_like(mx))
        mean = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return torch.cat([mx, mean], dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """Composed bound ``L_f ≤ M_h · L_gat · max_block (γβ(1+κ_g))/(1−ρ)``."""
        # Worst Mamba block bound (since they are residually stacked the product
        # multiplies, but residual gives (1 + L_blk) per block).
        block_bound = 1.0
        for blk in self.blocks:
            if not hasattr(blk, "lipschitz_bound"):
                return float("inf")  # softmax attention is not globally Lipschitz
            lb = blk.lipschitz_bound()
            if lb == float("inf"):
                return float("inf")
            block_bound *= 1.0 + lb

        gat_bound = 1.0
        for layer in self.gat_layers:
            gat_bound *= 1.0 + layer.lipschitz_bound()

        # Detection head: two linear layers with GELU (1-Lipschitz) and LayerNorm
        # (1-Lipschitz w.r.t. its scaled output) — bound by product of weight norms.
        w1 = float(torch.linalg.matrix_norm(self.head.fc1.weight, ord=2).item())
        w2 = float(torch.linalg.matrix_norm(self.head.fc2.weight, ord=2).item())
        head_bound = w1 * w2

        # Input projection contributes once.
        win = float(torch.linalg.matrix_norm(self.input_proj.weight, ord=2).item())
        if self.chunk_mlp is not None:
            # [x, pool(MLP(c))]: GELU <= 1.13-Lipschitz, [max, mean] <= sqrt(2); joint input bound max(1, L_c)
            l1 = float(torch.linalg.matrix_norm(self.chunk_mlp[0].weight, ord=2).item())
            l2 = float(torch.linalg.matrix_norm(self.chunk_mlp[2].weight, ord=2).item())
            win *= max(1.0, math.sqrt(2.0) * 1.13 * l1 * l2)

        return win * block_bound * gat_bound * head_bound

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | MambaGuardConfig) -> "MambaGuard":
        """Build from a dataclass instance or a plain mapping of hyper-parameters."""
        if isinstance(cfg, MambaGuardConfig):
            return cls(cfg)
        return cls(MambaGuardConfig(**dict(cfg)))
