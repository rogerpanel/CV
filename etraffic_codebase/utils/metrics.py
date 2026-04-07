"""
Comprehensive evaluation metrics for encrypted traffic intrusion detection

Implements all metrics reported in the paper:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- False Positive Rate (FPR)
- Matthews Correlation Coefficient (MCC)
- Per-class metrics for multi-class classification

Reference: Paper Section 4.3 - Evaluation Metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import warnings


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted',
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, num_classes)
        average: Averaging strategy ('micro', 'macro', 'weighted')
        class_names: Names of classes

    Returns:
        Dictionary of all computed metrics
    """
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2

            if num_classes == 2:
                y_score = y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob
                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
                metrics['pr_auc'] = average_precision_score(y_true, y_score)
            else:
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics['roc_auc'] = roc_auc_score(
                        y_true_bin, y_prob, average=average, multi_class='ovr'
                    )
                    metrics['pr_auc'] = average_precision_score(
                        y_true_bin, y_prob, average=average
                    )
        except Exception as e:
            print(f"Warning: Could not compute AUC metrics: {e}")

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1."""
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]

    per_class = {}
    for idx, class_id in enumerate(unique_classes):
        name = class_names[idx] if idx < len(class_names) else f"Class_{class_id}"
        binary_true = (y_true == class_id).astype(int)
        binary_pred = (y_pred == class_id).astype(int)

        per_class[name] = {
            'precision': precision_score(binary_true, binary_pred, zero_division=0),
            'recall': recall_score(binary_true, binary_pred, zero_division=0),
            'f1_score': f1_score(binary_true, binary_pred, zero_division=0),
            'support': int(np.sum(binary_true))
        }

    return per_class


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True, save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
        y_score = y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    else:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def print_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """Print detailed classification report."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 80)


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as LaTeX table."""
    lines = [
        "\\begin{table}[h]", "\\centering",
        "\\begin{tabular}{lc}", "\\toprule",
        "Metric & Value \\\\", "\\midrule",
    ]
    for name, value in metrics.items():
        formatted_name = name.replace('_', ' ').title()
        if isinstance(value, float) and value < 1.0:
            formatted_value = f"{value * 100:.2f}\\%"
        else:
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"{formatted_name} & {formatted_value} \\\\")

    lines.extend([
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Performance Metrics}", "\\label{tab:metrics}",
        "\\end{table}"
    ])
    return "\n".join(lines)
