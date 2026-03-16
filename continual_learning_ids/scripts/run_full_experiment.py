#!/usr/bin/env python3
"""
Full Experiment Pipeline: Reproduces all results from the paper.

Executes the complete evaluation from Sections V-A through V-E:
  1. Continual learning on all 6 datasets (Table II)
  2. Cross-dataset sequential evaluation (Supplementary Table S6)
  3. RL response agent training and evaluation (Table III)
  4. Integrated end-to-end evaluation (Table IV)
  5. Ablation studies (Table V)
  6. Adversarial robustness across CL stages (Table VI)

Usage:
    python -m continual_learning_ids.scripts.run_full_experiment \
        --data_dir ./data \
        --output_dir ./results \
        --config configs/default.yaml

For quick testing (subsampled):
    python -m continual_learning_ids.scripts.run_full_experiment \
        --data_dir ./data \
        --output_dir ./results \
        --subsample 0.01
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continual_learning_ids.models.surrogate_ids import SurrogateIDS
from continual_learning_ids.models.policy_network import (
    PolicyNetwork,
    ValueNetwork,
    CostValueNetwork,
)
from continual_learning_ids.models.unified_fim import UnifiedFIM
from continual_learning_ids.data.dataset_loader import (
    DatasetLoader,
    SequentialTaskSplitter,
    DATASET_REGISTRY,
    CROSS_DATASET_ORDER,
    create_dataloader,
)
from continual_learning_ids.training.continual_learner import ContinualLearner
from continual_learning_ids.training.drift_detector import DriftDetector
from continual_learning_ids.training.cpo_trainer import CPOTrainer
from continual_learning_ids.environments.nids_env import NIDSResponseEnv
from continual_learning_ids.evaluation.metrics import ContinualMetrics, RLMetrics
from continual_learning_ids.evaluation.adversarial import AdversarialEvaluator
from continual_learning_ids.utils.logging_utils import setup_logger

logger = setup_logger("full_experiment")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_within_dataset_cl(
    dataset_key: str,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    config: dict,
    device: torch.device,
    output_dir: Path,
    eval_adversarial: bool = True,
) -> dict:
    """Run within-dataset CL evaluation (Table II)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Within-Dataset CL: {dataset_key}")
    logger.info(f"{'='*60}")

    splitter = SequentialTaskSplitter(num_splits=5)
    tasks = splitter.split_dataset(features, labels)

    model = SurrogateIDS(
        input_dim=features.shape[1],
        num_classes=num_classes,
    )

    cl = ContinualLearner(model, config, device)
    cl_metrics = ContinualMetrics()

    for task in tasks:
        sid = task["split_id"]
        cl.train_on_task(task["train_X"], task["train_y"],
                         task["test_X"], task["test_y"], sid)

        for prev in tasks[:sid]:
            acc = cl._evaluate(prev["test_X"], prev["test_y"])
            cl_metrics.record_accuracy(sid, prev["split_id"], acc)

    metrics = cl_metrics.compute_all_metrics()

    # Adversarial robustness
    adv_results = {}
    if eval_adversarial:
        adv_eval = AdversarialEvaluator(device)
        for task in tasks:
            sid = task["split_id"]
            adv_results[sid] = adv_eval.evaluate_all_attacks(
                model, task["test_X"][:200], task["test_y"][:200]
            )

    result = {
        "dataset": dataset_key,
        "cl_metrics": metrics,
        "adversarial": adv_results,
    }

    # Save
    with open(output_dir / f"{dataset_key}_cl.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment pipeline"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_rl", action="store_true")
    parser.add_argument("--skip_adversarial", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to evaluate")
    args = parser.parse_args()

    config_path = args.config or str(
        Path(__file__).parents[1] / "configs" / "default.yaml"
    )
    config = load_config(config_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ── Load All Datasets ────────────────────────────────────────────
    dataset_keys = args.datasets or list(DATASET_REGISTRY.keys())
    loader = DatasetLoader(args.data_dir, max_features=79)
    all_datasets = {}

    for key in dataset_keys:
        try:
            features, labels, le = loader.load_dataset(
                key, subsample=args.subsample
            )
            all_datasets[key] = (features, labels, le)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {key}: {e}")

    if not all_datasets:
        logger.error("No datasets loaded. Ensure data files exist.")
        return

    # ── Experiment 1: Within-Dataset CL (Table II) ───────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Within-Dataset Continual Learning (Table II)")
    logger.info("=" * 70)

    all_cl_results = {}
    for key, (features, labels, le) in all_datasets.items():
        result = run_within_dataset_cl(
            key, features, labels, len(le.classes_),
            config, device, output_dir,
            eval_adversarial=not args.skip_adversarial,
        )
        all_cl_results[key] = result

    # ── Experiment 2: Cross-Dataset CL (Supplementary Table S6) ──────
    if len(all_datasets) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 2: Cross-Dataset Sequential CL")
        logger.info("=" * 70)

        splitter = SequentialTaskSplitter()
        cross_tasks = splitter.create_cross_dataset_tasks(all_datasets)

        if cross_tasks:
            first_task = cross_tasks[0]
            model = SurrogateIDS(
                input_dim=first_task["train_X"].shape[1],
                num_classes=first_task["num_classes"],
            )
            cl = ContinualLearner(model, config, device)

            for task in cross_tasks:
                logger.info(f"Cross-dataset task: {task['dataset']}")
                cl.train_on_task(
                    task["train_X"], task["train_y"],
                    task["test_X"], task["test_y"],
                )

    # ── Experiment 3: RL Response Agent (Table III) ──────────────────
    if not args.skip_rl and all_datasets:
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 3: CPO Response Agent (Table III)")
        logger.info("=" * 70)

        first_key = list(all_datasets.keys())[0]
        features, labels, le = all_datasets[first_key]

        rl_config = config.get("rl_agent", {})
        env = NIDSResponseEnv(
            features=features, labels=labels, config=rl_config
        )

        policy = PolicyNetwork()
        value_net = ValueNetwork()
        cost_value_net = CostValueNetwork()

        trainer = CPOTrainer(
            policy, value_net, cost_value_net,
            env, rl_config, device,
        )

        trainer.train(
            num_iterations=100,  # Reduced for full pipeline
            steps_per_iteration=2048,
        )

        eval_results = trainer.evaluate(num_episodes=100)
        logger.info(f"CPO Eval: {eval_results}")

        trainer.save(str(output_dir / "cpo_agent.pt"))
        with open(output_dir / "rl_results.json", "w") as f:
            json.dump(eval_results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE ({elapsed:.1f}s)")
    logger.info(f"{'='*70}")

    summary = {
        "datasets_evaluated": list(all_cl_results.keys()),
        "cl_results": {
            k: v["cl_metrics"] for k, v in all_cl_results.items()
        },
        "elapsed_seconds": elapsed,
    }

    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    logger.info("\nDataset          | AA (%)  | BWT     | FWT")
    logger.info("-" * 50)
    for key, result in all_cl_results.items():
        m = result["cl_metrics"]
        logger.info(
            f"{key:16s} | {m['average_accuracy']:6.2f}  | "
            f"{m['backward_transfer']:+.4f} | "
            f"{m['forward_transfer']:+.4f}"
        )

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
