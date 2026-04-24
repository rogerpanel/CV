"""Supervised trainer for the CT-TGNN / SDE-TGNN detection backbone."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ct_explain.data.graph_builder import TemporalGraph
from ct_explain.training.losses import CTExplainLoss
from ct_explain.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_val: float = -1.0
    history: list[dict] = field(default_factory=list)


class CTExplainTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        loss_fn: Optional[CTExplainLoss] = None,
        device: str = "cpu",
        out_dir: str = "runs/latest",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn or CTExplainLoss()
        self.state = TrainerState()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sched: Optional[CosineAnnealingLR] = None

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_graphs: Iterable[TemporalGraph],
        val_graphs: Optional[Iterable[TemporalGraph]] = None,
        epochs: int = 10,
    ) -> TrainerState:
        train_list = list(train_graphs)
        val_list = list(val_graphs) if val_graphs else None
        self.sched = CosineAnnealingLR(self.opt, T_max=max(1, epochs))

        for ep in range(1, epochs + 1):
            self.state.epoch = ep
            train_metrics = self._run_epoch(train_list, train=True)
            val_metrics = (
                self._run_epoch(val_list, train=False) if val_list else {}
            )
            self.sched.step()
            row = {"epoch": ep, **train_metrics,
                   **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.state.history.append(row)
            log.info("[ep %d] %s", ep, row)

            score = val_metrics.get("accuracy", -row.get("loss", 0.0))
            if score > self.state.best_val:
                self.state.best_val = score
                self.save("best.pt")

        self.save("latest.pt")
        (self.out_dir / "history.json").write_text(json.dumps(self.state.history, indent=2))
        return self.state

    # ------------------------------------------------------------------ #
    def _run_epoch(self, graphs: list[TemporalGraph], train: bool) -> dict:
        self.model.train(train)
        total_loss = 0.0
        total_n = 0
        correct = 0
        t0 = time.time()
        for g in graphs:
            g = g.to(self.device)
            labels = g.node_labels
            if labels is None:
                continue
            mask = labels != -1
            if not mask.any():
                continue

            out = self.model(
                node_features=g.node_features,
                edge_index=g.edge_index,
                edge_features=g.edge_features,
                edge_times=g.edge_times,
            )
            logits = out.logits[mask]
            y = labels[mask]
            parts = self.loss_fn(logits, y)
            loss = parts["total"]

            if train:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                self.state.global_step += 1

            with torch.no_grad():
                pred = logits.argmax(-1)
                correct += int((pred == y).sum().item())
                total_n += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())

        metrics = {
            "loss": total_loss / max(total_n, 1),
            "accuracy": correct / max(total_n, 1),
            "elapsed_s": round(time.time() - t0, 2),
            "n_examples": total_n,
        }
        return metrics

    # ------------------------------------------------------------------ #
    def save(self, name: str = "latest.pt") -> Path:
        fp = self.out_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optim_state": self.opt.state_dict(),
                "state": self.state.__dict__,
            },
            fp,
        )
        return fp

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            self.opt.load_state_dict(ckpt["optim_state"])
