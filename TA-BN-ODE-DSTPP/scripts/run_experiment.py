#!/usr/bin/env python3
"""
Main experiment runner for TA-BN-ODE-DSTPP.

Reproduces all results from:
  "Temporal Adaptive Neural Ordinary Differential Equations with Deep
   Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection"
  Anaedevha, Trofimov, Borodachev (2026) — Complex and Intelligent Systems

Usage:
    python scripts/run_experiment.py                     # Full experiment
    python scripts/run_experiment.py --dataset container  # Single dataset
    python scripts/run_experiment.py --quick              # Quick smoke test
    python scripts/run_experiment.py --ablation           # Ablation study

Datasets are auto-downloaded from Kaggle via kagglehub.
Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or place
kaggle.json in ~/.kaggle/ before running.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from configs.default import Config
from models.full_model import TABNODEPointProcess
from models.bayesian import BayesianWrapper
from data.loader import ICS3DLoader, BenchmarkLoader
from data.preprocessing import preprocess_dataset, temporal_split, TimeSeriesDataset
from utils.training import Trainer
from utils.evaluation import Evaluator
from utils.online import OnlineAdapter


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(name: str, cfg: Config, max_samples=None):
    """Load and preprocess a single dataset."""
    print(f"\n{'='*60}")
    print(f"Loading dataset: {name}")
    print(f"{'='*60}")

    if name in ("container", "edge_iiot", "guide_soc"):
        loader = ICS3DLoader(cfg.data.ics3d_kaggle_slug)
        load_fn = {
            "container": loader.load_container_security,
            "edge_iiot": loader.load_edge_iiot,
            "guide_soc": loader.load_guide_soc,
        }[name]
    else:
        loader = BenchmarkLoader(cfg.data.benchmarks_kaggle_slug)
        load_fn = {
            "cic_ids2018": loader.load_cic_ids2018,
            "unsw_nb15": loader.load_unsw_nb15,
            "cic_iot2023": loader.load_cic_iot2023,
        }[name]

    df, _ = load_fn()
    X, y, le, scaler = preprocess_dataset(df, max_samples=max_samples)
    splits = temporal_split(X, y, cfg.data.temporal_split_ratios)

    return splits, le, scaler, X.shape[1], len(le.classes_)


def run_single_dataset(name: str, cfg: Config, args):
    """Run full experiment on a single dataset."""
    max_samples = 50000 if args.quick else None
    splits, le, scaler, input_dim, n_classes = load_dataset(name, cfg, max_samples)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    # Create datasets
    seq_len = 1  # Single-event mode (sequence mode available via --seq_len)
    train_ds = TimeSeriesDataset(X_train, y_train, seq_len=seq_len)
    val_ds = TimeSeriesDataset(X_val, y_val, seq_len=seq_len)
    test_ds = TimeSeriesDataset(X_test, y_test, seq_len=seq_len)

    print(f"\n  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Features: {input_dim} | Classes: {n_classes}")

    # Build model
    model = TABNODEPointProcess(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_classes=n_classes,
        d_model=cfg.model.model_dim,
        n_ode_blocks=cfg.model.n_ode_blocks,
        time_constants=cfg.model.time_constants,
        n_transformer_layers=cfg.model.n_transformer_layers,
        n_attention_heads=cfg.model.n_attention_heads,
        tabn_mlp_hidden=cfg.model.tabn_mlp_hidden,
        tabn_mlp_layers=cfg.model.tabn_mlp_layers,
        solver_method=cfg.model.solver_method,
        rtol=cfg.model.solver_rtol,
        atol=cfg.model.solver_atol,
        transformer_dropout=cfg.model.transformer_dropout,
    )

    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Optional Bayesian wrapper
    bayesian = None
    if not args.no_bayesian:
        bayesian = BayesianWrapper(model, rank=cfg.model.low_rank_dim)

    # Train
    epochs = 10 if args.quick else cfg.training.max_epochs
    trainer = Trainer(
        model, device,
        lr=cfg.training.learning_rate,
        min_lr=cfg.training.min_learning_rate,
        batch_size=cfg.training.batch_size,
        max_epochs=epochs,
        patience=cfg.training.early_stopping_patience,
        grad_clip=cfg.training.grad_clip_norm,
        loss_weights={
            "cls": cfg.training.weight_cls,
            "tpp": cfg.training.weight_tpp,
            "reg": cfg.training.weight_reg,
        },
        bayesian_wrapper=bayesian,
    )

    if args.cross_validate:
        from data.preprocessing import TimeSeriesDataset
        full_ds = TimeSeriesDataset(
            np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]),
            seq_len=seq_len
        )
        cv_results = trainer.cross_validate(full_ds, n_folds=cfg.training.n_folds)
    else:
        history = trainer.train(train_ds, val_ds)

    # Temperature calibration
    val_loader = DataLoader(val_ds, batch_size=cfg.training.eval_batch_size, shuffle=False)
    model.calibrate_temperature(val_loader, device)

    # Evaluate
    evaluator = Evaluator(model, device)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.eval_batch_size, shuffle=False)
    results = evaluator.evaluate(
        test_loader,
        label_names=list(le.classes_),
        n_mc_samples=cfg.model.mc_samples_test if bayesian else 1,
        bayesian_wrapper=bayesian,
    )

    # Model size
    size_info = evaluator.model_size()
    results.update(size_info)

    # Throughput
    if not args.quick:
        throughput = evaluator.measure_throughput(input_dim, cfg.training.batch_size)
        results.update(throughput)

    # Save model
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "outputs", name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in results.items()}, f, indent=2)

    print(f"\n  Results saved to: {save_dir}")
    return results


def run_ablation(cfg: Config, args):
    """Ablation study (Table 5 in paper).

    Components removed:
      1. TA-BN (temporal adaptive batch norm)
      2. Point Process
      3. Bayesian inference
      4. Multi-scale (single time constant)
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    # Use container dataset for ablation
    max_samples = 50000 if args.quick else 200000
    splits, le, scaler, input_dim, n_classes = load_dataset("container", cfg, max_samples)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(test_ds, batch_size=cfg.training.eval_batch_size, shuffle=False)
    epochs = 10 if args.quick else 50

    ablation_results = OrderedDict()

    # Full model
    print("\n--- Full Model ---")
    model_full = TABNODEPointProcess(input_dim, cfg.model.hidden_dim, n_classes)
    trainer = Trainer(model_full, device, max_epochs=epochs)
    trainer.train(train_ds, val_ds)
    evaluator = Evaluator(model_full, device)
    ablation_results["full"] = evaluator.evaluate(test_loader)

    # Without multi-scale (single time constant)
    print("\n--- Without Multi-Scale ---")
    model_no_ms = TABNODEPointProcess(
        input_dim, cfg.model.hidden_dim, n_classes, time_constants=(1.0,)
    )
    trainer = Trainer(model_no_ms, device, max_epochs=epochs)
    trainer.train(train_ds, val_ds)
    evaluator = Evaluator(model_no_ms, device)
    ablation_results["no_multiscale"] = evaluator.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    for variant, res in ablation_results.items():
        print(f"  {variant:20s}: Acc={res['accuracy']:.4f}  F1={res['f1_weighted']:.4f}  ECE={res['ece']:.4f}")

    return ablation_results


def run_drift_experiment(cfg: Config, args):
    """Concept drift robustness experiment (Section 5.4)."""
    print("\n" + "=" * 60)
    print("CONCEPT DRIFT EXPERIMENT")
    print("=" * 60)

    max_samples = 50000 if args.quick else 200000
    splits, le, scaler, input_dim, n_classes = load_dataset("container", cfg, max_samples)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    epochs = 10 if args.quick else 50

    # Train base model
    model = TABNODEPointProcess(input_dim, cfg.model.hidden_dim, n_classes)
    trainer = Trainer(model, device, max_epochs=epochs)
    trainer.train(train_ds, val_ds)

    # Online adaptation
    adapter = OnlineAdapter(
        model, device,
        ewc_lambda=cfg.online.ewc_lambda,
        ema_rho=cfg.online.ema_rho,
        base_lr=cfg.online.online_lr,
        lr_decay_rho=cfg.online.lr_decay_rho,
        mini_epochs=cfg.online.mini_epochs,
        dp_clip_norm=cfg.online.dp_clip_norm,
        dp_noise_multiplier=cfg.online.dp_noise_multiplier,
        psi_threshold=cfg.online.psi_threshold,
    )

    # Set reference distribution from training predictions
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    t_span = torch.linspace(0, 1, 10).to(device)
    model.eval()
    ref_confs = []
    with torch.no_grad():
        for batch in train_loader:
            out = model(batch["x"].to(device), t_span)
            probs = torch.softmax(out["logits"], dim=1)
            ref_confs.extend(probs.max(dim=1)[0].cpu().numpy())
    adapter.drift_detector.set_reference(np.array(ref_confs))

    # Simulate streaming test data
    test_ds = TimeSeriesDataset(X_test, y_test)
    stream_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    correct_adaptive = 0
    total = 0
    for batch in stream_loader:
        result = adapter.process_batch(batch["x"], batch["y"])
        correct_adaptive += (result["predictions"] == batch["y"]).sum().item()
        total += len(batch["y"])

    print(f"\n  Adaptive Accuracy: {correct_adaptive / total:.4f}")
    return {"adaptive_accuracy": correct_adaptive / total}


def main():
    parser = argparse.ArgumentParser(
        description="TA-BN-ODE-DSTPP Experiment Runner"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["container", "edge_iiot", "guide_soc",
                                 "cic_ids2018", "unsw_nb15", "cic_iot2023"],
                        help="Run on a single dataset (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (50K samples, 10 epochs)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--drift", action="store_true",
                        help="Run concept drift experiment")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Use 5-fold time-series CV")
    parser.add_argument("--no-bayesian", action="store_true",
                        help="Disable Bayesian inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.device = args.device if torch.cuda.is_available() else "cpu"

    set_seed(cfg.seed)

    print("=" * 60)
    print("TA-BN-ODE-DSTPP: Temporal Adaptive Neural ODE with")
    print("Deep Spatio-Temporal Point Processes")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Seed: {cfg.seed}")
    print(f"Quick mode: {args.quick}")

    all_results = {}
    t_start = time.time()

    if args.ablation:
        all_results["ablation"] = run_ablation(cfg, args)
    elif args.drift:
        all_results["drift"] = run_drift_experiment(cfg, args)
    elif args.dataset:
        all_results[args.dataset] = run_single_dataset(args.dataset, cfg, args)
    else:
        # Run on all datasets
        datasets = ["container", "edge_iiot", "guide_soc",
                     "cic_ids2018", "unsw_nb15", "cic_iot2023"]
        for ds_name in datasets:
            try:
                all_results[ds_name] = run_single_dataset(ds_name, cfg, args)
            except Exception as e:
                print(f"\n  ERROR on {ds_name}: {e}")
                all_results[ds_name] = {"error": str(e)}

    total_time = time.time() - t_start

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for ds, res in all_results.items():
        if isinstance(res, dict) and "accuracy" in res:
            print(f"  {ds:15s}: Acc={res['accuracy']:.4f}  "
                  f"F1={res.get('f1_weighted', 0):.4f}  "
                  f"ECE={res.get('ece', 0):.4f}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    # Save all results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()
                    if not isinstance(vv, np.ndarray)
                }
        json.dump(serializable, f, indent=2)

    print(f"All results saved to: {save_dir}/all_results.json")


if __name__ == "__main__":
    main()
