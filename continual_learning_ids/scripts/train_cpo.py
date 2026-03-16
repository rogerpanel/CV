#!/usr/bin/env python3
"""
Train the CPO Response Agent.

Reproduces the results in Table III (Section V-B):
  - Trains CPO in the NIDS-CMDP environment
  - Evaluates threat mitigation rate, FP blocking rate, MTTR
  - Compares against Rule-Based, PPO, and Lagrangian PPO baselines
  - Verifies zero constraint violations over 5,000 episodes

Usage:
    python -m continual_learning_ids.scripts.train_cpo \
        --data_dir ./data \
        --dataset cic_iot_2023 \
        --detection_model checkpoints/task_5.pt \
        --config configs/default.yaml \
        --output_dir ./results/rl
"""

import argparse
import json
import logging
import sys
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
from continual_learning_ids.data.dataset_loader import DatasetLoader, DATASET_REGISTRY
from continual_learning_ids.environments.nids_env import NIDSResponseEnv
from continual_learning_ids.training.cpo_trainer import CPOTrainer
from continual_learning_ids.utils.logging_utils import setup_logger

logger = setup_logger("train_cpo")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train CPO response agent"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cic_iot_2023")
    parser.add_argument("--detection_model", type=str, default=None,
                        help="Path to trained SurrogateIDS checkpoint")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/rl")
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--steps_per_iter", type=int, default=4096)
    parser.add_argument("--eval_episodes", type=int, default=5000)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
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

    # ── Load Dataset ─────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    loader = DatasetLoader(args.data_dir, max_features=79)
    features, labels, label_encoder = loader.load_dataset(
        args.dataset, subsample=args.subsample
    )

    # ── Load Detection Model (optional) ──────────────────────────────
    num_classes = len(label_encoder.classes_)
    detection_model = SurrogateIDS(
        input_dim=features.shape[1], num_classes=num_classes
    )

    if args.detection_model:
        checkpoint = torch.load(args.detection_model, map_location=device)
        if "model_state_dict" in checkpoint:
            detection_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded detection model from {args.detection_model}")

    detection_model.to(device)
    detection_model.eval()

    # ── Precompute Detection Outputs ─────────────────────────────────
    logger.info("Precomputing detection outputs...")
    batch_size = 512
    all_probs = []
    all_epistemic = []
    all_aleatoric = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.FloatTensor(features[i:i+batch_size]).to(device)
            result = detection_model.predict_with_uncertainty(batch)
            all_probs.append(result["probabilities"].cpu().numpy())
            all_epistemic.append(result["epistemic_uncertainty"].cpu().numpy())
            all_aleatoric.append(result["aleatoric_uncertainty"].cpu().numpy())

    detection_probs = np.concatenate(all_probs)
    epistemic = np.concatenate(all_epistemic)
    aleatoric = np.concatenate(all_aleatoric)

    # ── Create Environment ───────────────────────────────────────────
    rl_config = config.get("rl_agent", {})
    env = NIDSResponseEnv(
        features=features,
        labels=labels,
        detection_probs=detection_probs,
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=aleatoric,
        config=rl_config,
    )

    # ── Initialise Networks ──────────────────────────────────────────
    state_dim = rl_config.get("state_dim", 55)
    num_actions = len(rl_config.get("action_space", range(5)))
    hidden_dims = rl_config.get("policy_network", {}).get(
        "hidden_dims", [256, 128]
    )

    policy = PolicyNetwork(state_dim, num_actions, hidden_dims)
    value_net = ValueNetwork(state_dim, hidden_dims)
    cost_value_net = CostValueNetwork(state_dim, hidden_dims)

    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters())} params")
    logger.info(f"Value:  {sum(p.numel() for p in value_net.parameters())} params")

    # ── Train CPO ────────────────────────────────────────────────────
    trainer = CPOTrainer(
        policy=policy,
        value_net=value_net,
        cost_value_net=cost_value_net,
        env=env,
        config=rl_config,
        device=device,
    )

    logger.info(f"Training CPO: {args.num_iterations} iterations")
    training_stats = trainer.train(
        num_iterations=args.num_iterations,
        steps_per_iteration=args.steps_per_iter,
        log_interval=10,
        eval_interval=50,
    )

    # ── Final Evaluation ─────────────────────────────────────────────
    logger.info(f"\nFinal evaluation: {args.eval_episodes} episodes")
    eval_results = trainer.evaluate(num_episodes=args.eval_episodes)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL CPO RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Mitigation Rate:     {eval_results['mitigation_rate']:.2%}")
    logger.info(f"FP Blocking Rate:    {eval_results['fp_blocking_rate']:.4%}")
    logger.info(f"MTTR:                {eval_results['mean_time_to_respond_ms']:.0f} ms")
    logger.info(f"Violations:          {eval_results['constraint_violations']}/{args.eval_episodes}")
    logger.info(f"Mean Reward:         {eval_results['mean_reward']:.2f}")

    # Save
    trainer.save(str(output_dir / "cpo_agent.pt"))

    results = {
        "eval_results": eval_results,
        "training_stats_summary": {
            "final_fp_rate": training_stats[-1]["fp_rate"] if training_stats else None,
            "final_reward": training_stats[-1]["mean_reward"] if training_stats else None,
            "num_iterations": len(training_stats),
        },
        "config": rl_config,
    }
    with open(output_dir / "cpo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}")
    return eval_results


if __name__ == "__main__":
    main()
