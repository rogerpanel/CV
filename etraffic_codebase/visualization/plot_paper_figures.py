"""
Generate all figures for the encrypted traffic IDS paper

Reproduces all visualization figures presented in the paper:
- Model comparison across architectures
- Privacy-performance tradeoff
- Encrypted performance comparison
- Ablation study
- Dataset performance
- Inference time comparison
- Class distribution

Reference: Paper Section 4 - Experimental Results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_model_comparison, plot_privacy_tradeoff, COLORS

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_figure_model_comparison():
    """Figure 2: Performance comparison of deep learning architectures."""
    results = {
        'CICIDS2017 (Encrypted)': {
            'CNN': 93.4, 'LSTM': 94.7, 'CNN-LSTM': 99.87,
            'Transformer': 98.94, 'GNN': 96.8, 'Ensemble': 99.92
        },
        'NSL-KDD': {
            'CNN': 91.2, 'LSTM': 93.1, 'CNN-LSTM': 98.42,
            'Transformer': 97.41, 'GNN': 95.3, 'Ensemble': 98.96
        }
    }
    plot_model_comparison(results, save_path='./outputs/figure_model_comparison.pdf')
    plt.close()
    print("Generated: figure_model_comparison.pdf")


def plot_figure_encrypted_performance():
    """Performance across encrypted traffic datasets."""
    results = {
        'BoT-IoT Encrypted': {
            'CNN': 93.4, 'LSTM': 94.7, 'CNN-LSTM': 99.9,
            'Transformer': 98.9, 'GNN': 96.8, 'Ensemble': 99.9
        },
        'CICIDS HTTPS': {
            'CNN': 91.2, 'LSTM': 93.1, 'CNN-LSTM': 98.42,
            'Transformer': 97.41, 'GNN': 95.3, 'Ensemble': 98.96
        },
        'ISCX-VPN': {
            'CNN': 92.7, 'LSTM': 93.9, 'CNN-LSTM': 97.8,
            'Transformer': 96.8, 'GNN': 94.2, 'Ensemble': 98.4
        }
    }
    plot_model_comparison(results, save_path='./outputs/figure_encrypted_performance.pdf')
    plt.close()
    print("Generated: figure_encrypted_performance.pdf")


def plot_figure_privacy_tradeoff():
    """Figure 3: Privacy-performance tradeoff."""
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accuracies = [89.2, 92.1, 94.5, 96.3, 98.1, 99.1]
    plot_privacy_tradeoff(epsilon_values, accuracies, 99.4,
                          save_path='./outputs/figure_privacy_tradeoff.pdf')
    plt.close()
    print("Generated: figure_privacy_tradeoff.pdf")


def plot_figure_ablation_study():
    """Ablation study: component contributions."""
    components = ['Spatial\nOnly', 'Temporal\nOnly', 'Fusion\n(Simple)',
                  'Fusion\n(Attention)', 'Full\nModel']
    accuracies = [93.4, 94.7, 98.2, 98.9, 99.87]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(components, accuracies, color=COLORS['layer1'],
                  edgecolor='black', linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f'{h:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Contributions',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([90, 100])
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./outputs/figure_ablation_study.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_ablation_study.pdf")


def plot_figure_dataset_performance():
    """Comprehensive dataset performance comparison."""
    datasets = ['CICIDS2017', 'CICIDS2018', 'UNSW-NB15', 'BoT-IoT',
                'ISCX-VPN', 'Edge-IIoT', 'CIC-IoT-2023']
    accuracies = [98.42, 97.41, 96.8, 99.87, 97.8, 94.5, 99.2]
    f1_scores = [98.59, 97.52, 96.5, 99.87, 97.6, 94.2, 99.1]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, accuracies, width, label='Accuracy',
           color=COLORS['layer1'], edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, f1_scores, width, label='F1-Score',
           color=COLORS['layer4'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid CNN-LSTM Performance Across Datasets',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([90, 101])
    plt.tight_layout()
    plt.savefig('./outputs/figure_dataset_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_dataset_performance.pdf")


def plot_figure_inference_time():
    """Inference time comparison across models."""
    models = ['CNN', 'LSTM', 'CNN-LSTM', 'Transformer', 'GNN', 'Ensemble']
    times = [0.8, 1.5, 2.3, 1.8, 3.2, 5.1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=COLORS['layer3'],
                  edgecolor='black', linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f'{h:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax.set_title('Real-time Processing Performance', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5,
               label='5ms threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./outputs/figure_inference_time.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_inference_time.pdf")


def plot_figure_class_distribution():
    """Class distribution visualization for the IIS3D / HTTPS Traffic dataset."""
    classes = ['W (Website)', 'D (Download)', 'P (Video)', 'U (Upload)',
               'M (Music)', 'L (Live Video)']
    samples = [80789, 20393, 12553, 10862, 10701, 10373]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, samples, color=COLORS['layer4'],
                   edgecolor='black', linewidth=1)
    for i, (bar, sample) in enumerate(zip(bars, samples)):
        w = bar.get_width()
        pct = sample / sum(samples) * 100
        ax.text(w, i, f' {sample:,} ({pct:.1f}%)',
                va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('IIS3D Dataset Class Distribution (145,671 flows)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./outputs/figure_class_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_class_distribution.pdf")


def generate_all_figures():
    """Generate all figures for the paper."""
    os.makedirs('./outputs', exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING PAPER FIGURES")
    print("=" * 80 + "\n")

    plot_figure_model_comparison()
    plot_figure_encrypted_performance()
    plot_figure_privacy_tradeoff()
    plot_figure_ablation_study()
    plot_figure_dataset_performance()
    plot_figure_inference_time()
    plot_figure_class_distribution()

    print("\n" + "=" * 80)
    print("All figures generated successfully!")
    print("Figures saved to: ./outputs/")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    generate_all_figures()
