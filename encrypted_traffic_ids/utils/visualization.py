"""
Visualization utilities for encrypted traffic IDS

This module implements visualization functions for:
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


# Set publication-quality plotting defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette matching paper
COLORS = {
    'layer1': '#2980B9',  # Blue
    'layer2': '#8E44AD',  # Purple
    'layer3': '#27AE60',  # Green
    'layer4': '#E67E22',  # Orange
    'layer5': '#E74C3C',  # Red
    'fedcolor': '#F39C12',  # Yellow-orange
}


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        train_losses: Training loss values per epoch
        val_losses: Validation loss values per epoch
        train_accs: Training accuracy values per epoch (optional)
        val_accs: Validation accuracy values per epoch (optional)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    epochs = range(1, len(train_losses) + 1)

    if train_accs is not None and val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Loss plot
        ax1.plot(epochs, train_losses, 'o-', color=COLORS['layer1'],
                 label='Training Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, val_losses, 's-', color=COLORS['layer5'],
                 label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, train_accs, 'o-', color=COLORS['layer1'],
                 label='Training Accuracy', linewidth=2, markersize=4)
        ax2.plot(epochs, val_accs, 's-', color=COLORS['layer5'],
                 label='Validation Accuracy', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Training and Validation Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(epochs, train_losses, 'o-', color=COLORS['layer1'],
                 label='Training Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, val_losses, 's-', color=COLORS['layer5'],
                 label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    return fig


def plot_attention_maps(
    attention_weights: torch.Tensor,
    sequence_length: int,
    num_heads: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize attention weights from transformer models.

    Args:
        attention_weights: Attention weight tensor (num_heads, seq_len, seq_len)
        sequence_length: Length of input sequence
        num_heads: Number of attention heads
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Ensure correct shape
    if len(attention_weights.shape) == 3:
        num_heads = attention_weights.shape[0]
    else:
        attention_weights = attention_weights.reshape(num_heads, sequence_length, sequence_length)

    # Create subplots for each attention head
    rows = int(np.ceil(num_heads / 4))
    cols = min(4, num_heads)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx in range(num_heads):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # Plot attention heatmap
        im = ax.imshow(attention_weights[idx], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {idx + 1}', fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Remove empty subplots
    total_subplots = rows * cols
    for idx in range(num_heads, total_subplots):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.suptitle('Multi-Head Attention Weights', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention maps saved to: {save_path}")

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    datasets: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot model comparison bar chart (matching paper figures).

    Args:
        results: Nested dict {dataset: {model: metric_value}}
        metric: Metric to plot ('accuracy', 'f1_score', etc.)
        datasets: List of dataset names to include
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object

    Example:
        >>> results = {
        ...     'BoT-IoT': {'CNN': 93.4, 'LSTM': 94.7, 'CNN-LSTM': 99.87},
        ...     'CICIDS': {'CNN': 91.2, 'LSTM': 93.1, 'CNN-LSTM': 98.42}
        ... }
        >>> plot_model_comparison(results, metric='accuracy')
    """
    if datasets is None:
        datasets = list(results.keys())

    # Extract model names (assuming all datasets have same models)
    model_names = list(results[datasets[0]].keys())

    # Prepare data for plotting
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

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}' if height < 10 else f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model Architecture', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=min([min(results[d].values()) for d in datasets]) - 2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")

    return fig


def plot_privacy_tradeoff(
    epsilon_values: List[float],
    accuracies: List[float],
    baseline_accuracy: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot privacy-accuracy tradeoff (federated learning).

    Args:
        epsilon_values: Privacy budget (epsilon) values
        accuracies: Accuracies corresponding to each epsilon
        baseline_accuracy: Centralized (no privacy) baseline accuracy
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot FL with DP curve
    ax.semilogx(epsilon_values, accuracies, 'o-', color=COLORS['layer1'],
                label='FL with DP', linewidth=2, markersize=8)

    # Plot baseline
    ax.axhline(y=baseline_accuracy, color=COLORS['layer5'], linestyle='--',
               linewidth=2, label='Centralized (no privacy)')

    # Annotate epsilon=1.0 point
    epsilon_1_idx = np.argmin(np.abs(np.array(epsilon_values) - 1.0))
    ax.annotate(f'ε=1 (strong privacy)\n{accuracies[epsilon_1_idx]:.1f}%',
                xy=(epsilon_values[epsilon_1_idx], accuracies[epsilon_1_idx]),
                xytext=(epsilon_values[epsilon_1_idx] * 0.3, accuracies[epsilon_1_idx] - 3),
                arrowprops=dict(arrowstyle='->', color=COLORS['layer1']),
                fontsize=9, color=COLORS['layer1'], fontweight='bold')

    ax.set_xlabel('Privacy Budget ε', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Privacy-Performance Tradeoff in Federated Learning',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(bottom=min(accuracies) - 2, top=100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Privacy tradeoff plot saved to: {save_path}")

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance (e.g., from SHAP values).

    Args:
        feature_names: Names of features
        importance_values: Importance scores for each feature
        top_k: Number of top features to display
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Sort by importance
    sorted_idx = np.argsort(importance_values)[-top_k:]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_idx))
    ax.barh(y_pos, importance_values[sorted_idx], color=COLORS['layer3'],
            edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importance', fontweight='bold', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    return fig
