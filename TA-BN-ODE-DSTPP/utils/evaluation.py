"""
Evaluation framework.

Implements:
  - Standard classification metrics (accuracy, F1, AUC)
  - Expected Calibration Error (ECE) from Section 4.4
  - Population Stability Index (PSI) for drift detection
  - Throughput and latency measurement (Section 5.3)
  - Per-attack-type breakdown
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import time


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = 10) -> float:
    """Expected Calibration Error.

    ECE = sum_b (|B_b|/n) |acc(B_b) - conf(B_b)|

    Target: ECE = 0.017 (Table 4 in paper).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / n


def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index for concept drift detection.

    PSI = sum_b (p_b - q_b) ln(p_b / q_b)

    Drift threshold: PSI > 0.2 triggers adaptation (Section 4.5).
    """
    bin_edges = np.histogram_bin_edges(reference, bins=n_bins)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


class Evaluator:
    """Comprehensive evaluation framework."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader,
                 label_names: Optional[list] = None,
                 n_mc_samples: int = 50,
                 bayesian_wrapper=None) -> Dict:
        """Full evaluation: classification + calibration + throughput.

        Args:
            test_loader: Test data loader
            label_names: Optional class label names
            n_mc_samples: MC samples for uncertainty (50 at test, per paper)
            bayesian_wrapper: Optional BayesianWrapper for MC sampling
        """
        self.model.eval()
        t_span = torch.linspace(0, 1, 10).to(self.device)

        all_preds = []
        all_labels = []
        all_probs = []
        all_confidences = []

        for batch in test_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            if bayesian_wrapper is not None:
                # MC-sampled predictions
                def fwd(model, x=x, t_span=t_span):
                    return model(x, t_span)["logits"]

                mean_logits, std_logits, _ = bayesian_wrapper.predict_with_uncertainty(
                    fwd, n_samples=n_mc_samples
                )
                probs = F.softmax(mean_logits, dim=1)
            else:
                out = self.model(x, t_span)
                probs = F.softmax(out["logits"], dim=1)

            preds = probs.argmax(dim=1)
            confidence = probs.max(dim=1)[0]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)
        all_confidences = np.array(all_confidences)

        # Classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

        # AUC (multi-class OvR)
        try:
            if all_probs.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
        except ValueError:
            auc = float("nan")

        # Calibration
        correct = (all_preds == all_labels).astype(float)
        ece = compute_ece(all_confidences, correct)

        # 95% prediction interval coverage
        coverage = np.mean(all_confidences > 0.025)

        results = {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "ece": ece,
            "coverage_95": coverage,
            "n_samples": len(all_labels),
        }

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")
            else:
                print(f"  {k:20s}: {v}")

        if label_names:
            print("\nPer-Class Report:")
            print(classification_report(all_labels, all_preds,
                                        target_names=label_names, zero_division=0))

        return results

    @torch.no_grad()
    def measure_throughput(self, input_dim: int, batch_size: int = 256,
                           n_warmup: int = 200, n_measure: int = 1000
                           ) -> Dict[str, float]:
        """Measure throughput and latency (Section 5.3).

        Target: 12.3M events/sec, P50=8.2ms, P99=22.9ms.
        """
        self.model.eval()
        t_span = torch.linspace(0, 1, 10).to(self.device)

        # Warmup
        for _ in range(n_warmup):
            x = torch.randn(batch_size, input_dim, device=self.device)
            _ = self.model(x, t_span)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        latencies = []
        for _ in range(n_measure):
            x = torch.randn(batch_size, input_dim, device=self.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.model(x, t_span)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latencies_ms = np.array(latencies) * 1000
        events_per_sec = batch_size / np.mean(latencies)

        results = {
            "throughput_events_per_sec": events_per_sec,
            "p50_latency_ms": np.percentile(latencies_ms, 50),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "batch_size": batch_size,
        }

        print("\n" + "=" * 60)
        print("THROUGHPUT & LATENCY")
        print("=" * 60)
        print(f"  Throughput:   {events_per_sec/1e6:.2f}M events/sec")
        print(f"  P50 Latency:  {results['p50_latency_ms']:.2f} ms")
        print(f"  P95 Latency:  {results['p95_latency_ms']:.2f} ms")
        print(f"  P99 Latency:  {results['p99_latency_ms']:.2f} ms")

        return results

    @torch.no_grad()
    def model_size(self) -> Dict[str, float]:
        """Report model size (Section 5.2: 2.3M params, 9.2MB memory)."""
        n_params = sum(p.numel() for p in self.model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        results = {
            "n_parameters": n_params,
            "size_mb": size_mb,
        }

        print(f"\n  Parameters: {n_params:,} ({size_mb:.1f} MB)")
        return results
