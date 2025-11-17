"""
Comprehensive evaluation metrics for encrypted traffic intrusion detection

This module implements all metrics reported in the paper:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- False Positive Rate (FPR)
- Matthews Correlation Coefficient (MCC)
- Per-class metrics for multi-class classification

All implementations follow scikit-learn conventions and are optimized
for class-imbalanced datasets common in network intrusion detection.
"""

import numpy as np
import torch
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted',
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, num_classes) - optional for AUC metrics
        average: Averaging strategy for multi-class ('micro', 'macro', 'weighted')
        class_names: Names of classes for detailed reporting

    Returns:
        Dictionary containing all computed metrics

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = compute_all_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Matthews Correlation Coefficient (robust for imbalanced datasets)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix based metrics
    cm = confusion_matrix(y_true, y_pred)

    if len(cm) == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)

        # False Positive Rate (critical for operational deployment)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # True Positive Rate (Recall)
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC metrics (require probabilities)
    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2

            if num_classes == 2:
                # Binary classification - use probability of positive class
                if len(y_prob.shape) > 1:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob

                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
                metrics['pr_auc'] = average_precision_score(y_true, y_score)

            else:
                # Multi-class classification
                y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics['roc_auc'] = roc_auc_score(
                        y_true_binarized, y_prob,
                        average=average, multi_class='ovr'
                    )
                    metrics['pr_auc'] = average_precision_score(
                        y_true_binarized, y_prob, average=average
                    )

        except Exception as e:
            print(f"Warning: Could not compute AUC metrics: {e}")

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1-score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)

    Returns:
        Dictionary mapping class names to their metrics

    Example:
        >>> per_class = compute_per_class_metrics(y_true, y_pred, ['Benign', 'Malware'])
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]

    per_class = {}

    for idx, class_id in enumerate(unique_classes):
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{class_id}"

        # Binary mask for current class
        binary_true = (y_true == class_id).astype(int)
        binary_pred = (y_pred == class_id).astype(int)

        per_class[class_name] = {
            'precision': precision_score(binary_true, binary_pred, zero_division=0),
            'recall': recall_score(binary_true, binary_pred, zero_division=0),
            'f1_score': f1_score(binary_true, binary_pred, zero_division=0),
            'support': int(np.sum(binary_true))
        }

    return per_class


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix with high-quality visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes for labels
        normalize: If True, normalize confusion matrix
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_confusion_matrix(y_true, y_pred, ['Benign', 'Malware'])
        >>> plt.show()
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn for better visualization
    sns.heatmap(
        cm, annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curve for binary or multi-class classification.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities (N, num_classes)
        class_names: Names of classes
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
        # Binary classification
        if len(y_prob.shape) > 1:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')

    else:
        # Multi-class classification
        y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_binarized.shape[1]

        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.4f})')

    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 80)


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as a LaTeX-ready table.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string for LaTeX table

    Example:
        >>> print(format_metrics_table(metrics))
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")

    for metric_name, value in metrics.items():
        # Format metric name (convert underscores to spaces, capitalize)
        formatted_name = metric_name.replace('_', ' ').title()

        # Format value based on type
        if isinstance(value, float):
            if value < 1.0:
                formatted_value = f"{value * 100:.2f}\\%"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        lines.append(f"{formatted_name} & {formatted_value} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Performance Metrics}")
    lines.append("\\label{tab:metrics}")
    lines.append("\\end{table}")

    return "\n".join(lines)
