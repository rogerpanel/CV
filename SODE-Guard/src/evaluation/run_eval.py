"""Entry-point ``python -m src.evaluation.run_eval``.

Loads a checkpoint, runs clean evaluation, adversarial sweep, anti-concentration
certification and latency benchmark, and writes ``eval.json`` next to the
checkpoint.
"""
from .latency import run_eval_cli

if __name__ == "__main__":
    run_eval_cli()
