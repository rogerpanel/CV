"""
Visualization utilities for encrypted traffic IDS

Implements visualization functions for:
- Training curves (loss, accuracy)
- Attention maps from transformer models
- Feature importance plots
- Architecture comparison plots (matching paper figures)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch


# Publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

COLORS = {
    'layer1': '#2980B9',
    'layer2': '#8E44AD',
    'layer3': '#27AE60',
    'layer4': '#E67E22',
    'layer5': '#E74C3C',
    'fedcolor': '#F39C12',
}


def plot_training_curves(
    train_losses: List[float], val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Plot training and validation curves."""
    epochs = range(1, len(train_losses) + 1)

    if train_accs is not None and val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(epochs, train_losses, 'o-', color=COLORS['layer1'],
                 label='Train', linewidth=2, markersize=3)
        ax1.plot(epochs, val_losses, 's-', color=COLORS['layer5'],
                 label='Val', linewidth=2, markersize=3)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_accs, 'o-', color=COLORS['layer1'],
                 label='Train', linewidth=2, markersize=3)
        ax2.plot(epochs, val_accs, 's-', color=COLORS['layer5'],
                 label='Val', linewidth=2, markersize=3)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, train_losses, 'o-', color=COLORS['layer1'],
                 label='Train', linewidth=2, markersize=3)
        ax1.plot(epochs, val_losses, 's-', color=COLORS['layer5'],
                 label='Val', linewidth=2, markersize=3)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_attention_maps(
    attention_weights: torch.Tensor,
    sequence_length: int, num_heads: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Visualize multi-head attention weights."""
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    if len(attention_weights.shape) == 3:
        num_heads = attention_weights.shape[0]
    else:
        attention_weights = attention_weights.reshape(
            num_heads, sequence_length, sequence_length
        )

    rows = int(np.ceil(num_heads / 4))
    cols = min(4, num_heads)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx in range(num_heads):
        r, c = idx // cols, idx % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        im = ax.imshow(attention_weights[idx], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {idx + 1}', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(num_heads, rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis('off')

    plt.suptitle('Multi-Head Attention Weights', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    datasets: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot model comparison bar chart (matching paper figures)."""
    if datasets is None:
        datasets = list(results.keys())

    model_names = list(results[datasets[0]].keys())
    x = np.arange(len(model_names))
    width = 0.8 / len(datasets)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [COLORS['layer1'], COLORS['layer4'], COLORS['layer3'],
              COLORS['layer2'], COLORS['layer5']]

    for idx, dataset in enumerate(datasets):
        values = [results[dataset].get(model, 0) for model in model_names]
        offset = (idx - len(datasets) / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=dataset,
                      color=colors[idx % len(colors)], edgecolor='black', linewidth=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontweight='bold')
    ax.set_title(f'Model Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_privacy_tradeoff(
    epsilon_values: List[float], accuracies: List[float],
    baseline_accuracy: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot privacy-accuracy tradeoff for federated learning."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(epsilon_values, accuracies, 'o-', color=COLORS['layer1'],
                label='FL with DP', linewidth=2, markersize=8)
    ax.axhline(y=baseline_accuracy, color=COLORS['layer5'], linestyle='--',
               linewidth=2, label='Centralized (no privacy)')

    ax.set_xlabel('Privacy Budget (epsilon)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Privacy-Performance Tradeoff', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_feature_importance(
    feature_names: List[str], importance_values: np.ndarray,
    top_k: int = 20, save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot feature importance (from SHAP values)."""
    sorted_idx = np.argsort(importance_values)[-top_k:]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_idx))
    ax.barh(y_pos, importance_values[sorted_idx],
            color=COLORS['layer3'], edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Top {top_k} Feature Importance', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
