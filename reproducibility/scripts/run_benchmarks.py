#!/usr/bin/env python3
"""
Standard Benchmark Evaluation
==============================
Run experiments on CIC-IDS2018, UNSW-NB15, and CIC-IoT-2023
for comparison with published baselines (Table IV).

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --max_samples 50000 --epochs 30
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import BenchmarkDataLoader, create_dataloaders
from src.data.preprocessing import Preprocessor
from src.models.framework import TABNODEPointProcessFramework
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation (CIC-IDS2018, UNSW-NB15, CIC-IoT-2023)"
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = BenchmarkDataLoader(data_dir=args.data_dir)
    benchmarks = {
        "CIC-IDS2018": loader.load_cicids2018,
        "UNSW-NB15": loader.load_unsw_nb15,
        "CIC-IoT-2023": loader.load_ciciot2023,
    }

    all_results = {}

    for name, load_fn in benchmarks.items():
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: {name}")
        print(f"{'=' * 60}")

        try:
            X, y = load_fn()
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

        preprocessor = Preprocessor()
        X, y = preprocessor.fit_transform(X, y)

        train_loader, val_loader, test_loader = create_dataloaders(
            X, y,
            batch_size=cfg["training"]["batch_size"],
            max_samples=args.max_samples,
        )

        model = TABNODEPointProcessFramework(
            input_dim=X.shape[1],
            hidden_dim=cfg["model"]["hidden_dim"],
            n_classes=preprocessor.n_classes,
            n_ode_blocks=cfg["model"]["n_ode_blocks"],
            time_constants=cfg["model"]["time_constants"],
            n_heads=cfg["transformer"]["n_heads"],
            n_transformer_layers=cfg["transformer"]["n_layers"],
            d_model=cfg["transformer"]["d_model"],
        )

        trainer = Trainer(
            model, device,
            lr=cfg["training"]["lr"],
            patience=cfg["training"]["patience"],
            checkpoint_dir=f"checkpoints/benchmarks/{name}",
        )

        trainer.train(train_loader, val_loader, epochs=args.epochs)

        evaluator = Evaluator(model, device)
        results = evaluator.evaluate_detection(test_loader)
        all_results[name] = results

    # Summary table
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY (Table IV)")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<20} {'Accuracy':>10} {'F1':>10} {'AUROC':>10}")
    print("-" * 50)
    for name, res in all_results.items():
        print(f"{name:<20} {res.get('accuracy', 0):>10.4f} "
              f"{res.get('f1_weighted', 0):>10.4f} "
              f"{res.get('auroc', 0):>10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
