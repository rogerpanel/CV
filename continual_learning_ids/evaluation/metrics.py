"""
Continual Learning and RL Evaluation Metrics.

Implements the metrics from Section III and V:
  - Average Accuracy (AA)
  - Backward Transfer (BWT)
  - Forward Transfer (FWT)
  - Per-class accuracy retention
  - RL: Threat Mitigation Rate, FP Blocking Rate, MTTR, Constraint Violations
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import logging

logger = logging.getLogger(__name__)


class ContinualMetrics:
    """
    Compute continual learning evaluation metrics.

    Tracks the accuracy matrix a_{T,t} where a_{T,t} is the accuracy
    on task t after training through task T.

    From Section III (Equations 1-3):
      AA  = (1/T) * Σ_{t=1}^{T} a_{T,t}
      BWT = (1/(T-1)) * Σ_{t=1}^{T-1} (a_{T,t} - a_{t,t})
      FWT = (1/(T-1)) * Σ_{t=2}^{T} (a_{t,t} - a_tilde_t)
    """

    def __init__(self):
        self.accuracy_matrix: Dict[int, Dict[int, float]] = {}
        self.random_baselines: Dict[int, float] = {}

    def record_accuracy(
        self,
        training_task: int,
        eval_task: int,
        accuracy: float,
    ) -> None:
        """
        Record a_{T,t}: accuracy on eval_task after training through training_task.
        """
        if training_task not in self.accuracy_matrix:
            self.accuracy_matrix[training_task] = {}
        self.accuracy_matrix[training_task][eval_task] = accuracy

    def set_random_baseline(self, task: int, accuracy: float) -> None:
        """Set a_tilde_t: accuracy of randomly initialised model on task t."""
        self.random_baselines[task] = accuracy

    def compute_average_accuracy(self) -> float:
        """
        AA = (1/T) * Σ_{t=1}^{T} a_{T,t}
        """
        if not self.accuracy_matrix:
            return 0.0

        T = max(self.accuracy_matrix.keys())
        final_accs = self.accuracy_matrix.get(T, {})
        if not final_accs:
            return 0.0

        return np.mean(list(final_accs.values()))

    def compute_backward_transfer(self) -> float:
        """
        BWT = (1/(T-1)) * Σ_{t=1}^{T-1} (a_{T,t} - a_{t,t})

        Negative values indicate forgetting.
        """
        if len(self.accuracy_matrix) < 2:
            return 0.0

        T = max(self.accuracy_matrix.keys())
        bwt_values = []

        for t in range(1, T):
            if (
                T in self.accuracy_matrix
                and t in self.accuracy_matrix[T]
                and t in self.accuracy_matrix
                and t in self.accuracy_matrix[t]
            ):
                a_Tt = self.accuracy_matrix[T][t]
                a_tt = self.accuracy_matrix[t][t]
                bwt_values.append(a_Tt - a_tt)

        return np.mean(bwt_values) if bwt_values else 0.0

    def compute_forward_transfer(self) -> float:
        """
        FWT = (1/(T-1)) * Σ_{t=2}^{T} (a_{t,t} - a_tilde_t)
        """
        if len(self.accuracy_matrix) < 2:
            return 0.0

        T = max(self.accuracy_matrix.keys())
        fwt_values = []

        for t in range(2, T + 1):
            if (
                t in self.accuracy_matrix
                and t in self.accuracy_matrix[t]
                and t in self.random_baselines
            ):
                a_tt = self.accuracy_matrix[t][t]
                a_tilde = self.random_baselines[t]
                fwt_values.append(a_tt - a_tilde)

        return np.mean(fwt_values) if fwt_values else 0.0

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all CL metrics."""
        return {
            "average_accuracy": self.compute_average_accuracy(),
            "backward_transfer": self.compute_backward_transfer(),
            "forward_transfer": self.compute_forward_transfer(),
        }

    def get_accuracy_matrix(self) -> Dict:
        """Return the full accuracy matrix for reporting."""
        return dict(self.accuracy_matrix)

    def compute_per_class_retention(
        self,
        initial_per_class: Dict[int, float],
        final_per_class: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Compute per-class accuracy retention: a_{T,k} / a_{1,k}

        From Supplementary Material Table S2.
        """
        retention = {}
        for cls in initial_per_class:
            if cls in final_per_class and initial_per_class[cls] > 0:
                retention[cls] = final_per_class[cls] / initial_per_class[cls]
        return retention


class RLMetrics:
    """
    RL response agent evaluation metrics.

    From Section V-B, Table III:
      - Threat Mitigation Rate (%)
      - False-Positive Blocking Rate (%)
      - Mean Time To Respond (ms)
      - Safety Constraint Violations (count)
      - Cumulative Discounted Reward
    """

    def __init__(self):
        self.episode_stats: List[Dict] = []

    def record_episode(self, stats: Dict) -> None:
        """Record metrics for a single evaluation episode."""
        self.episode_stats.append(stats)

    def compute_summary(self) -> Dict[str, float]:
        """Compute aggregate metrics over all recorded episodes."""
        if not self.episode_stats:
            return {}

        n = len(self.episode_stats)

        mitigation_rates = [
            s.get("mitigation_rate", 0) for s in self.episode_stats
        ]
        fp_rates = [
            s.get("fp_blocking_rate", 0) for s in self.episode_stats
        ]
        rewards = [s.get("mean_reward", 0) for s in self.episode_stats]
        violations = sum(
            1 for s in self.episode_stats if s.get("constraint_violated", False)
        )

        return {
            "mitigation_rate": np.mean(mitigation_rates),
            "fp_blocking_rate": np.mean(fp_rates),
            "mean_reward": np.mean(rewards),
            "constraint_violations": violations,
            "total_episodes": n,
            "violation_rate": violations / n,
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Returns accuracy, per-class precision/recall/F1, and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
    }
