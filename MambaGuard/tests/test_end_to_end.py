"""End-to-end smoke test: build tiny model + run a few train steps."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_train_one_epoch_smoke(tiny_model_kwargs, tmp_output_dir: Path):
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.models import MambaGuard
        from mambaguard.training import MambaGuardTrainer
        from mambaguard.training.trainer import TrainerConfig
    except Exception as exc:
        pytest.skip(f"mambaguard runtime unavailable: {exc}")

    model = MambaGuard.from_config(tiny_model_kwargs)

    def _batch():
        d_p = tiny_model_kwargs["d_p"]
        d_mu = tiny_model_kwargs["d_mu"]
        edge_dim = tiny_model_kwargs["edge_dim"]
        # (A=agents, L=messages-per-agent, *)
        A, L = 4, 6
        return {
            "p": torch.randn(A, L, d_p),
            "mu": torch.randn(A, L, d_mu),
            "edge_index": torch.tensor(
                [[0, 1, 2], [1, 2, 3]], dtype=torch.long
            ),
            "edge_attr": torch.randn(3, edge_dim),
            "edge_time": torch.arange(3, dtype=torch.float32),
            "labels": torch.randint(0, tiny_model_kwargs["num_classes"], (A,)),
        }

    loader = [_batch() for _ in range(4)]

    cfg = TrainerConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        focal_gamma=0.0,
        focal_alpha=1.0,
        lipschitz_lambda=0.0,
        warmup_steps=0,
        amp_dtype="float32",
        num_classes=tiny_model_kwargs["num_classes"],
        out_dir=str(tmp_output_dir),
    ) if hasattr(__import__("mambaguard.training.trainer", fromlist=["TrainerConfig"]), "TrainerConfig") else None

    trainer = MambaGuardTrainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        cfg=cfg,
        device="cpu",
    )
    state = trainer.train()
    assert state is not None
