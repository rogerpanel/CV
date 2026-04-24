"""Console-script entry points. The individual CLIs live under `scripts/`."""
from __future__ import annotations


def train_cmd() -> None:
    from scripts.train import main
    main()


def evaluate_cmd() -> None:
    from scripts.evaluate import main
    main()


def explain_cmd() -> None:
    from scripts.explain import main
    main()


def certify_cmd() -> None:
    from scripts.calibrate import main
    main()
