#!/usr/bin/env python3
"""
Evaluation Script
=================
Load a trained model and run comprehensive evaluation.

Usage:
    python scripts/evaluate.py --dataset containers --checkpoint checkpoints/containers/best_model.pt
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import ICS3DDataLoader, BenchmarkDataLoader, create_dataloaders
from src.data.preprocessing import Preprocessor
from src.models.framework import TABNODEPointProcessFramework
from src.evaluation.metrics import Evaluator
from src.evaluation.visualization import Visualizer
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate TA-BN-ODE Framework")
    parser.add_argument("--dataset", type=str, default="containers")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load data
    from scripts.train import load_dataset
    X, y = load_dataset(args.dataset, args.data_dir)

    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(X, y)

    _, _, test_loader = create_dataloaders(
        X, y, batch_size=cfg["training"]["eval_batch_size"],
        max_samples=args.max_samples,
    )

    # Load model
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate_all(
        test_loader,
        class_names=preprocessor.class_names.tolist(),
    )

    # Visualize
    viz = Visualizer(save_dir=f"figures/{args.dataset}_eval")
    viz.plot_parameter_efficiency()


if __name__ == "__main__":
    main()
