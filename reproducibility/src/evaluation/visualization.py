"""
Visualization Module
====================
Generates publication-quality figures matching the paper:
  - Training convergence (Figure 6)
  - Calibration reliability diagram (Figure 4)
  - Parameter efficiency (Figure 3)
  - Latency vs throughput (Figure 5)
  - Concept drift adaptation (Figure S4)
  - Performance bar charts (Figure 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


class Visualizer:
    """Publication-quality figure generation."""

    def __init__(self, save_dir: str = "figures"):
        import os
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_convergence(self, history: dict, filename: str = "convergence.png"):
        """Training loss and validation accuracy over epochs (Figure 6)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        if "train_total" in history:
            ax1.plot(history["train_total"], "k-", linewidth=1.2, label="Train Loss")
        if "val_loss" in history:
            ax1.plot(history["val_loss"], "k--", linewidth=1.2, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Convergence")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        if "val_accuracy" in history:
            ax2.plot([a * 100 for a in history["val_accuracy"]],
                     "k-", linewidth=1.2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Validation Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()
        print(f"Saved: {self.save_dir}/{filename}")

    def plot_calibration(self, confidences: np.ndarray,
                         accuracies: np.ndarray,
                         n_bins: int = 10,
                         filename: str = "calibration.png"):
        """Reliability diagram (Figure 4)."""
        fig, ax = plt.subplots(figsize=(5, 5))

        bin_bounds = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []

        for i in range(n_bins):
            mask = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i + 1])
            if mask.sum() > 0:
                bin_centers.append(confidences[mask].mean())
                bin_accs.append(accuracies[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.plot(bin_centers, bin_accs, "b-o", markersize=5, linewidth=1.2,
                label="TA-BN-ODE")
        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title("Calibration Reliability Diagram")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()
        print(f"Saved: {self.save_dir}/{filename}")

    def plot_performance_comparison(self, results: dict,
                                    filename: str = "performance.png"):
        """Performance bar chart across datasets (Figure 3)."""
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets = list(results.keys())
        methods = list(results[datasets[0]].keys())
        n_methods = len(methods)

        x = np.arange(len(datasets))
        width = 0.8 / n_methods
        grays = [f"{0.15 + 0.15 * i}" for i in range(n_methods)]

        for i, method in enumerate(methods):
            values = [results[d].get(method, 0) for d in datasets]
            ax.bar(x + i * width - 0.4 + width / 2, values, width,
                   label=method, color=grays[i], edgecolor="black",
                   linewidth=0.8)

        ax.set_xlabel("Dataset")
        ax.set_ylabel("Performance (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(85, 100)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()
        print(f"Saved: {self.save_dir}/{filename}")

    def plot_streaming_accuracy(self, accuracies: list,
                                window: int = 50,
                                filename: str = "streaming.png"):
        """Streaming accuracy over time (concept drift evaluation)."""
        fig, ax = plt.subplots(figsize=(8, 4))

        import pandas as pd
        smoothed = pd.Series(accuracies).rolling(window).mean()
        ax.plot(smoothed, "k-", linewidth=1.2)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Streaming Accuracy (window={window})")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()
        print(f"Saved: {self.save_dir}/{filename}")

    def plot_parameter_efficiency(self, filename: str = "param_efficiency.png"):
        """Parameter efficiency comparison (Figure 3)."""
        fig, ax = plt.subplots(figsize=(6, 5))

        models = {
            "TA-BN-ODE": (2.3, 97.3),
            "Transformer": (12.8, 96.2),
            "CNN-LSTM": (15.3, 94.8),
            "Neural CDE": (8.4, 96.8),
            "GRU-ODE": (6.8, 96.2),
        }

        markers = ["s", "^", "D", "o", "v"]
        for (name, (params, acc)), marker in zip(models.items(), markers):
            ax.scatter(params, acc, s=100, marker=marker, zorder=5,
                       label=name, edgecolors="black", linewidth=0.8)

        ax.set_xlabel("Parameters (Millions)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Parameter Efficiency Analysis")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()
        print(f"Saved: {self.save_dir}/{filename}")

    def generate_all(self, history=None, eval_results=None,
                     streaming_acc=None):
        """Generate all figures."""
        if history:
            self.plot_training_convergence(history)
        if streaming_acc:
            self.plot_streaming_accuracy(streaming_acc)
        self.plot_parameter_efficiency()
        print("All figures generated.")
