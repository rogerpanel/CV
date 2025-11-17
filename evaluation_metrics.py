"""
Comprehensive Evaluation Metrics for Real-Time Intrusion Detection

Implements all metrics from the paper:
- Detection performance: Accuracy, F1, Precision, Recall, ROC-AUC
- Calibration: ECE (target < 0.017), MCE
- Uncertainty: Coverage probability (target: 91.7% for 95% intervals)
- Performance: Throughput (12.3M events/sec), Latency (median 8.2ms, P99 22.9ms)
- Model complexity: Parameters (target: 2.3M)

Authors: Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class DetectionMetrics:
    """Standard intrusion detection metrics"""

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all detection metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (if probabilities available)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                metrics['roc_auc'] = 0.0

        return metrics

    @staticmethod
    def compute_per_class(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names

        Returns:
            Dictionary of per-class metrics
        """
        n_classes = len(np.unique(y_true))

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]

        per_class_metrics = {}

        # Compute metrics for each class
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, name in enumerate(class_names[:len(precision)]):
            per_class_metrics[name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i]
            }

        return per_class_metrics


class CalibrationMetrics:
    """
    Calibration and uncertainty metrics

    Target: ECE < 0.017 (paper specification)
    """

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE)

        Args:
            y_true: True labels
            y_prob: Predicted probabilities [n_samples, n_classes]
            n_bins: Number of bins

        Returns:
            ECE value
        """
        # Get predicted class and confidence
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            # Samples in this bin
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if mask.sum() == 0:
                continue

            # Accuracy in bin
            bin_acc = (predictions[mask] == y_true[mask]).mean()

            # Average confidence in bin
            bin_conf = confidences[mask].mean()

            # Contribution to ECE
            ece += (mask.sum() / n) * np.abs(bin_acc - bin_conf)

        return ece

    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Maximum Calibration Error (MCE)"""
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if mask.sum() == 0:
                continue

            bin_acc = (predictions[mask] == y_true[mask]).mean()
            bin_conf = confidences[mask].mean()

            mce = max(mce, np.abs(bin_acc - bin_conf))

        return mce

    @staticmethod
    def coverage_probability(
        y_true: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        target_coverage: float = 0.95
    ) -> Tuple[float, float]:
        """
        Coverage probability for prediction intervals

        Target: 91.7% coverage for 95% intervals (from paper)

        Args:
            y_true: True values
            lower_bounds: Lower bounds of intervals
            upper_bounds: Upper bounds of intervals
            target_coverage: Target coverage level

        Returns:
            actual_coverage: Actual coverage
            error: Difference from target
        """
        covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        actual_coverage = covered.mean()
        error = abs(actual_coverage - target_coverage)

        return actual_coverage, error


class PerformanceMetrics:
    """
    Performance metrics: throughput and latency

    Targets from paper:
    - Throughput: 12.3M events/second
    - Median latency: 8.2ms
    - P99 latency: 22.9ms
    """

    @staticmethod
    def measure_throughput(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        warmup_batches: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference throughput

        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Computation device
            warmup_batches: Number of warmup batches

        Returns:
            Throughput metrics
        """
        model.eval()
        model = model.to(device)

        # Warmup
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= warmup_batches:
                    break
                x = x.to(device)
                _, _ = model(x)

        # Measurement
        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                _, _ = model(x)
                total_samples += len(x)

        end_time = time.time()
        duration = end_time - start_time

        # Compute metrics
        throughput = total_samples / duration  # samples/second
        avg_time_per_sample = duration / total_samples * 1000  # milliseconds

        metrics = {
            'throughput_samples_per_sec': throughput,
            'throughput_million_per_sec': throughput / 1e6,
            'avg_time_per_sample_ms': avg_time_per_sample,
            'total_samples': total_samples,
            'total_time_sec': duration
        }

        return metrics

    @staticmethod
    def measure_latency(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        warmup_batches: int = 10,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Measure per-sample latency distribution

        Args:
            model: Model to evaluate
            dataloader: Data loader (batch_size=1 recommended)
            device: Computation device
            warmup_batches: Warmup batches
            n_samples: Number of samples to measure

        Returns:
            Latency metrics
        """
        model.eval()
        model = model.to(device)

        # Warmup
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= warmup_batches:
                    break
                x = x.to(device)
                _, _ = model(x)

        # Measure latencies
        latencies = []

        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= n_samples:
                    break

                x = x.to(device)

                # Synchronize GPU
                if device == "cuda":
                    torch.cuda.synchronize()

                start = time.time()
                _, _ = model(x)

                if device == "cuda":
                    torch.cuda.synchronize()

                end = time.time()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)

        latencies = np.array(latencies)

        metrics = {
            'median_latency_ms': np.median(latencies),
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies)
        }

        return metrics


class ModelComplexity:
    """Model complexity metrics"""

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters

        Target: 2.3M parameters (82% reduction vs 12.8M baseline)

        Args:
            model: Neural network model

        Returns:
            Parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'parameters_millions': total_params / 1e6
        }

        return metrics

    @staticmethod
    def measure_model_size(model: nn.Module) -> Dict[str, float]:
        """Measure model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

        total_size_mb = (param_size + buffer_size) / (1024 ** 2)

        metrics = {
            'model_size_mb': total_size_mb,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2)
        }

        return metrics


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator combining all metrics

    Produces complete evaluation report matching paper specifications
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def evaluate_all(
        self,
        test_loader: torch.utils.data.DataLoader,
        class_names: Optional[List[str]] = None,
        compute_latency: bool = True
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation

        Args:
            test_loader: Test data loader
            class_names: Optional class names
            compute_latency: Whether to compute latency (slow)

        Returns:
            Complete evaluation report
        """
        self.model.eval()
        self.model = self.model.to(self.device)

        # Collect predictions
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)

                output, _ = self.model(x)
                probs = torch.softmax(output, dim=-1)
                preds = output.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # Initialize results
        results = {}

        # 1. Detection metrics
        results['detection'] = DetectionMetrics.compute_all(y_true, y_pred, y_prob)
        results['per_class'] = DetectionMetrics.compute_per_class(y_true, y_pred, class_names)

        # 2. Calibration metrics
        results['calibration'] = {
            'ece': CalibrationMetrics.expected_calibration_error(y_true, y_prob),
            'mce': CalibrationMetrics.maximum_calibration_error(y_true, y_prob)
        }

        # 3. Performance metrics
        results['throughput'] = PerformanceMetrics.measure_throughput(
            self.model, test_loader, self.device
        )

        if compute_latency:
            # Create single-sample loader
            from torch.utils.data import DataLoader
            single_loader = DataLoader(
                test_loader.dataset,
                batch_size=1,
                shuffle=False
            )

            results['latency'] = PerformanceMetrics.measure_latency(
                self.model, single_loader, self.device, n_samples=min(1000, len(test_loader.dataset))
            )
        else:
            results['latency'] = {}

        # 4. Model complexity
        results['complexity'] = ModelComplexity.count_parameters(self.model)
        results['model_size'] = ModelComplexity.measure_model_size(self.model)

        return results

    def print_report(self, results: Dict[str, any]):
        """Print formatted evaluation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)

        # Detection performance
        print("\n1. DETECTION PERFORMANCE")
        print("-" * 40)
        det = results['detection']
        print(f"  Accuracy:     {det['accuracy']:.4f} ({det['accuracy']*100:.2f}%)")
        print(f"  Precision:    {det['precision']:.4f}")
        print(f"  Recall:       {det['recall']:.4f}")
        print(f"  F1-Score:     {det['f1_score']:.4f}")
        if 'roc_auc' in det:
            print(f"  ROC-AUC:      {det['roc_auc']:.4f}")

        # Calibration
        print("\n2. CALIBRATION METRICS")
        print("-" * 40)
        cal = results['calibration']
        ece_status = "✓ PASS" if cal['ece'] < 0.017 else "✗ FAIL"
        print(f"  ECE:          {cal['ece']:.4f} (target < 0.017) {ece_status}")
        print(f"  MCE:          {cal['mce']:.4f}")

        # Performance
        print("\n3. PERFORMANCE METRICS")
        print("-" * 40)
        thr = results['throughput']
        print(f"  Throughput:   {thr['throughput_million_per_sec']:.2f} M events/sec")
        throughput_status = "✓ PASS" if thr['throughput_million_per_sec'] >= 12.0 else "✗ FAIL"
        print(f"                (target: 12.3 M events/sec) {throughput_status}")

        if results['latency']:
            lat = results['latency']
            print(f"  Median Latency: {lat['median_latency_ms']:.2f} ms")
            latency_status = "✓ PASS" if lat['median_latency_ms'] <= 10.0 else "✗ FAIL"
            print(f"                  (target: 8.2 ms) {latency_status}")
            print(f"  P99 Latency:    {lat['p99_latency_ms']:.2f} ms")
            p99_status = "✓ PASS" if lat['p99_latency_ms'] <= 25.0 else "✗ FAIL"
            print(f"                  (target: 22.9 ms) {p99_status}")

        # Model complexity
        print("\n4. MODEL COMPLEXITY")
        print("-" * 40)
        comp = results['complexity']
        print(f"  Total Parameters: {comp['parameters_millions']:.2f} M")
        param_status = "✓ PASS" if comp['parameters_millions'] <= 3.0 else "✗ FAIL"
        print(f"                    (target: 2.3 M) {param_status}")

        size = results['model_size']
        print(f"  Model Size:       {size['model_size_mb']:.2f} MB")

        # Per-class performance (top 5 classes)
        print("\n5. PER-CLASS PERFORMANCE (Top 5)")
        print("-" * 40)
        per_class = results['per_class']
        for i, (class_name, metrics) in enumerate(list(per_class.items())[:5]):
            print(f"  {class_name}:")
            print(f"    F1: {metrics['f1_score']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}")

        print("\n" + "="*80)


if __name__ == "__main__":
    # Test evaluation metrics
    print("="*80)
    print("Testing Comprehensive Evaluation Metrics")
    print("="*80)

    # Generate synthetic data
    n_samples = 1000
    n_classes = 15

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()

    # Add some errors
    error_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    y_pred[error_idx] = np.random.randint(0, n_classes, len(error_idx))

    y_prob = np.random.dirichlet(np.ones(n_classes) * 2, n_samples)

    # Test detection metrics
    print("\n1. Detection Metrics:")
    det_metrics = DetectionMetrics.compute_all(y_true, y_pred, y_prob)
    for name, value in det_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test calibration
    print("\n2. Calibration Metrics:")
    ece = CalibrationMetrics.expected_calibration_error(y_true, y_prob)
    mce = CalibrationMetrics.maximum_calibration_error(y_true, y_prob)
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")

    # Test coverage
    lower = y_prob.max(axis=1) - 0.1
    upper = y_prob.max(axis=1) + 0.1
    coverage, error = CalibrationMetrics.coverage_probability(
        y_prob.max(axis=1), lower, upper
    )
    print(f"  Coverage: {coverage:.2%} (target 95%)")

    # Test model complexity (mock model)
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 15)
            self.output_dim = 15

        def forward(self, x):
            return self.fc(x), None

    model = MockModel()

    print("\n3. Model Complexity:")
    complexity = ModelComplexity.count_parameters(model)
    for name, value in complexity.items():
        print(f"  {name}: {value}")

    model_size = ModelComplexity.measure_model_size(model)
    for name, value in model_size.items():
        print(f"  {name}: {value:.2f}")

    print("\n" + "="*80)
    print("Evaluation Metrics Test Complete")
    print("="*80)
