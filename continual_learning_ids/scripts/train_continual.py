#!/usr/bin/env python3
"""
Main training script for EWC + Experience Replay continual learning.

Reproduces the results in Table II (Section V-A):
  - Trains SurrogateIDS on 5 sequential splits per dataset
  - Computes AA, BWT, FWT metrics
  - Evaluates adversarial robustness at each stage
  - Saves checkpoints after each task

Usage:
    python -m continual_learning_ids.scripts.train_continual \
        --data_dir ./data \
        --dataset cic_iot_2023 \
        --config configs/default.yaml \
        --output_dir ./results
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continual_learning_ids.models.surrogate_ids import SurrogateIDS
from continual_learning_ids.data.dataset_loader import (
    DatasetLoader,
    SequentialTaskSplitter,
    DATASET_REGISTRY,
    create_dataloader,
)
from continual_learning_ids.training.continual_learner import ContinualLearner
from continual_learning_ids.training.drift_detector import DriftDetector
from continual_learning_ids.evaluation.metrics import (
    ContinualMetrics,
    compute_classification_metrics,
)
from continual_learning_ids.evaluation.adversarial import AdversarialEvaluator
from continual_learning_ids.utils.logging_utils import setup_logger

logger = setup_logger("train_cl")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train EWC+Replay continual learning pipeline"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing dataset folders")
    parser.add_argument("--dataset", type=str, default="cic_iot_2023",
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset to train on")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results and checkpoints")
    parser.add_argument("--num_splits", type=int, default=5,
                        help="Number of sequential task splits")
    parser.add_argument("--subsample", type=float, default=None,
                        help="Subsample fraction (for quick testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_adversarial", action="store_true",
                        help="Also evaluate adversarial robustness")
    args = parser.parse_args()

    # Config
    config_path = args.config or str(
        Path(__file__).parents[1] / "configs" / "default.yaml"
    )
    config = load_config(config_path)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── Load Dataset ─────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    loader = DatasetLoader(args.data_dir, max_features=79)
    features, labels, label_encoder = loader.load_dataset(
        args.dataset, subsample=args.subsample
    )
    num_classes = len(label_encoder.classes_)
    logger.info(f"Dataset: {features.shape[0]} samples, {num_classes} classes")

    # ── Create Sequential Splits ─────────────────────────────────────
    splitter = SequentialTaskSplitter(
        num_splits=args.num_splits, seed=args.seed
    )
    tasks = splitter.split_dataset(features, labels)

    # ── Initialise Model ─────────────────────────────────────────────
    model = SurrogateIDS(
        input_dim=features.shape[1],
        num_classes=num_classes,
        **{k: v for k, v in config.get("model", {}).items()
           if k not in ["backbone"]},
    )
    logger.info(f"SurrogateIDS: {sum(p.numel() for p in model.parameters())} parameters")

    # ── Initialise Continual Learner ─────────────────────────────────
    cl = ContinualLearner(model, config, device)

    # ── Initialise Drift Detector ────────────────────────────────────
    drift_cfg = config.get("drift_detection", {})
    drift_detector = DriftDetector(
        num_classes=num_classes,
        tau_1=drift_cfg.get("tau_1", 0.05),
        tau_2=drift_cfg.get("tau_2", 0.15),
    )

    # ── Metrics Tracker ──────────────────────────────────────────────
    cl_metrics = ContinualMetrics()

    # Optional adversarial evaluator
    adv_evaluator = None
    if args.eval_adversarial:
        adv_evaluator = AdversarialEvaluator(device)
        adv_results_per_stage = {}

    # ── Sequential Training Loop ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting Sequential Continual Learning")
    logger.info("=" * 60)

    for task in tasks:
        split_id = task["split_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK {split_id}/{args.num_splits}")
        logger.info(f"{'='*60}")

        # Check drift (for splits > 1)
        if split_id > 1:
            status, kl_div = drift_detector.check_drift(
                model, task["train_X"][:1000], device
            )
            logger.info(f"Drift status: {status} (KL={kl_div:.4f})")

        # Train on current task
        train_metrics = cl.train_on_task(
            train_X=task["train_X"],
            train_y=task["train_y"],
            val_X=task["test_X"],
            val_y=task["test_y"],
            task_id=split_id,
        )

        # Update drift detector reference
        drift_detector.set_reference(
            model, task["test_X"][:1000], task["test_y"][:1000], device
        )

        # Evaluate on all tasks seen so far
        for prev_task in tasks[:split_id]:
            prev_id = prev_task["split_id"]
            acc = cl._evaluate(prev_task["test_X"], prev_task["test_y"])
            cl_metrics.record_accuracy(split_id, prev_id, acc)
            logger.info(f"  Task {prev_id} accuracy: {acc:.2f}%")

        # Adversarial evaluation
        if adv_evaluator is not None:
            adv_results = adv_evaluator.evaluate_all_attacks(
                model, task["test_X"][:500], task["test_y"][:500]
            )
            adv_results_per_stage[split_id] = adv_results

        # Checkpoint
        cl.save_checkpoint(str(ckpt_dir / f"task_{split_id}.pt"))

    # ── Compute Final Metrics ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")

    metrics = cl_metrics.compute_all_metrics()
    logger.info(f"Average Accuracy (AA): {metrics['average_accuracy']:.2f}%")
    logger.info(f"Backward Transfer (BWT): {metrics['backward_transfer']:.4f}")
    logger.info(f"Forward Transfer (FWT): {metrics['forward_transfer']:.4f}")

    # Save results
    results = {
        "dataset": args.dataset,
        "num_splits": args.num_splits,
        "cl_metrics": metrics,
        "accuracy_matrix": {
            str(k): {str(k2): v2 for k2, v2 in v.items()}
            for k, v in cl_metrics.get_accuracy_matrix().items()
        },
        "drift_summary": drift_detector.get_drift_summary(),
        "config": config,
    }

    if adv_evaluator is not None:
        results["adversarial_robustness"] = {
            str(k): v for k, v in adv_results_per_stage.items()
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return metrics


if __name__ == "__main__":
    main()
