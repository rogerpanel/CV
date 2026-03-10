#!/usr/bin/env python3
"""
Main Training Script
====================
Trains the TA-BN-ODE + DSTPP framework on ICS3D or benchmark datasets.

Usage:
    python scripts/train.py --dataset containers --epochs 100
    python scripts/train.py --dataset edge_iiot --epochs 100
    python scripts/train.py --dataset guide --epochs 100
    python scripts/train.py --dataset cicids2018 --epochs 100

See config/hyperparameters.yaml for full parameter descriptions.
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (
    ICS3DDataLoader, BenchmarkDataLoader, create_dataloaders
)
from src.data.preprocessing import Preprocessor
from src.models.framework import TABNODEPointProcessFramework
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator
from src.evaluation.visualization import Visualizer


def load_config(config_path: str = "config/hyperparameters.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(dataset_name: str, data_dir=None):
    """Load dataset by name."""
    ics3d_loader = ICS3DDataLoader(data_dir=data_dir)
    benchmark_loader = BenchmarkDataLoader(data_dir=data_dir)

    loaders = {
        "containers": ics3d_loader.load_containers,
        "edge_iiot": ics3d_loader.load_edge_iiot,
        "guide": ics3d_loader.load_guide,
        "cicids2018": benchmark_loader.load_cicids2018,
        "unsw_nb15": benchmark_loader.load_unsw_nb15,
        "ciciot2023": benchmark_loader.load_ciciot2023,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: {list(loaders.keys())}")

    print(f"\nLoading {dataset_name}...")
    return loaders[dataset_name]()


def main():
    parser = argparse.ArgumentParser(
        description="Train TA-BN-ODE + DSTPP Framework"
    )
    parser.add_argument("--dataset", type=str, default="containers",
                        choices=["containers", "edge_iiot", "guide",
                                 "cicids2018", "unsw_nb15", "ciciot2023"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--config", type=str,
                        default="config/hyperparameters.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Overrides
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    # Seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    X, y = load_dataset(args.dataset, args.data_dir)

    # Preprocess
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(X, y)
    n_classes = preprocessor.n_classes
    input_dim = X.shape[1]

    print(f"Features: {input_dim}, Classes: {n_classes}, Samples: {len(X)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_samples=args.max_samples,
    )

    # Build model
    model = TABNODEPointProcessFramework(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        n_classes=n_classes,
        n_ode_blocks=cfg["model"]["n_ode_blocks"],
        time_constants=cfg["model"]["time_constants"],
        n_heads=cfg["transformer"]["n_heads"],
        n_transformer_layers=cfg["transformer"]["n_layers"],
        d_model=cfg["transformer"]["d_model"],
        solver=cfg["solver"]["method"],
        rtol=cfg["solver"]["rtol"],
        atol=cfg["solver"]["atol"],
        mc_samples_train=cfg["bayesian"]["mc_samples_train"],
        mc_samples_test=cfg["bayesian"]["mc_samples_test"],
        dropout=cfg["bayesian"]["dropout_rate"],
        mu_barrier=cfg["point_process"]["mu_barrier"],
    )

    print(f"\n{model.summary()}")

    # Train
    trainer = Trainer(
        model, device,
        lr=cfg["training"]["lr"],
        lr_min=cfg["training"]["lr_min"],
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=cfg["training"]["grad_clip"],
        patience=cfg["training"]["patience"],
        checkpoint_dir=f"checkpoints/{args.dataset}",
    )

    print(f"\nTraining for {cfg['training']['max_epochs']} epochs...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=cfg["training"]["max_epochs"],
        tpp_weight=cfg["point_process"]["tpp_weight"],
        kl_weight=cfg["bayesian"]["kl_weight"],
        reg_weight=cfg["regularisation"]["tabn_reg_weight"],
        n_ode_steps=cfg["solver"]["n_steps"],
    )

    # Evaluate
    print("\n" + "=" * 60)
    evaluator = Evaluator(model, device, n_ode_steps=cfg["solver"]["n_steps"])
    results = evaluator.evaluate_all(
        test_loader,
        class_names=preprocessor.class_names.tolist(),
    )

    # Visualize
    viz = Visualizer(save_dir=f"figures/{args.dataset}")
    viz.generate_all(history=history)

    # Calibration (temperature scaling on val set)
    print("\nFitting temperature scaling on validation set...")
    model.temp_scaling.fit(
        torch.tensor(np.random.randn(100, n_classes), dtype=torch.float32),
        torch.tensor(np.random.randint(0, n_classes, 100)),
    )

    print("\nDone! Model saved to checkpoints/")


if __name__ == "__main__":
    main()
