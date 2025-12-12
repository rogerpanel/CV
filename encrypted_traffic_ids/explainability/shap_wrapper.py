"""
SHAP Explainability Wrapper for Encrypted Traffic Models

Provides interpretable explanations for encrypted traffic intrusion detection decisions.
Computes Shapley values to identify which encrypted traffic features (packet sizes,
inter-arrival times, flow statistics) contribute most to attack classification.

Key Features:
- KernelSHAP for deep learning models (model-agnostic)
- TreeSHAP for tree-based models (exact, efficient)
- Feature importance ranking
- Visualization tools

References:
    Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions
    Paper Section 3.6 - Explainability via SHAP
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import shap
except ImportError:
    print("Warning: SHAP library not installed. Install with: pip install shap")
    shap = None


class SHAPExplainer:
    """
    SHAP-based explainer for encrypted traffic IDS models.

    Provides interpretable explanations for model predictions on encrypted traffic
    without accessing payload contents.
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = 'kernel'
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained PyTorch model
            background_data: Background dataset for SHAP (typically training subset)
            feature_names: Names of features for visualization
            explainer_type: Type of SHAP explainer ('kernel', 'deep', 'gradient')
        """
        if shap is None:
            raise ImportError("SHAP library required. Install with: pip install shap")

        self.model = model
        self.model.eval()
        self.explainer_type = explainer_type

        # Default feature names for encrypted traffic
        if feature_names is None:
            feature_names = [
                'Packet Size', 'Inter-Arrival Time', 'Direction',
                'Fwd Packets', 'Bwd Packets', 'Flow Duration',
                'Fwd IAT Mean', 'Bwd IAT Mean'
            ]
        self.feature_names = feature_names

        # Prepare background data
        if isinstance(background_data, torch.Tensor):
            background_data = background_data.cpu().numpy()

        # Handle sequence data (flatten for SHAP if needed)
        if len(background_data.shape) == 3:
            # (batch, seq_len, features) -> use mean over sequence
            self.background_data = background_data.mean(axis=1)
            self.is_sequential = True
        else:
            self.background_data = background_data
            self.is_sequential = False

        # Create prediction function
        self.predict_fn = self._create_predict_function()

        # Initialize SHAP explainer
        self.explainer = self._create_explainer()

    def _create_predict_function(self) -> Callable:
        """
        Create prediction function compatible with SHAP.

        Returns:
            Prediction function
        """
        def predict(x):
            """Predict probabilities for SHAP."""
            # Handle input shape
            if isinstance(x, np.ndarray):
                x_tensor = torch.FloatTensor(x)
            else:
                x_tensor = x

            # Move to same device as model
            device = next(self.model.parameters()).device
            x_tensor = x_tensor.to(device)

            # If model expects sequences and input is flat, reshape
            if self.is_sequential and len(x_tensor.shape) == 2:
                # Expand: (batch, features) -> (batch, 1, features)
                x_tensor = x_tensor.unsqueeze(1)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probs = torch.softmax(outputs, dim=1)

            return probs.cpu().numpy()

        return predict

    def _create_explainer(self):
        """
        Create SHAP explainer based on type.

        Returns:
            SHAP explainer instance
        """
        if self.explainer_type == 'kernel':
            # KernelSHAP: Model-agnostic, works for any model
            explainer = shap.KernelExplainer(
                self.predict_fn,
                self.background_data,
                link='identity'
            )
        elif self.explainer_type == 'deep':
            # DeepSHAP: For deep learning models
            background_tensor = torch.FloatTensor(self.background_data)
            device = next(self.model.parameters()).device
            background_tensor = background_tensor.to(device)

            if self.is_sequential:
                background_tensor = background_tensor.unsqueeze(1)

            explainer = shap.DeepExplainer(self.model, background_tensor)
        elif self.explainer_type == 'gradient':
            # GradientSHAP: Uses gradients for explanation
            background_tensor = torch.FloatTensor(self.background_data)
            device = next(self.model.parameters()).device
            background_tensor = background_tensor.to(device)

            if self.is_sequential:
                background_tensor = background_tensor.unsqueeze(1)

            explainer = shap.GradientExplainer(self.model, background_tensor)
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")

        return explainer

    def explain(
        self,
        instances: np.ndarray,
        nsamples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for instances.

        Args:
            instances: Input instances to explain (n_instances, n_features)
            nsamples: Number of samples for KernelSHAP (ignored for other types)

        Returns:
            SHAP values (n_instances, n_features, n_classes)
        """
        # Handle shape
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()

        if self.is_sequential and len(instances.shape) == 3:
            instances_flat = instances.mean(axis=1)
        else:
            instances_flat = instances

        # Compute SHAP values
        if self.explainer_type == 'kernel':
            shap_values = self.explainer.shap_values(instances_flat, nsamples=nsamples)
        else:
            instances_tensor = torch.FloatTensor(instances_flat)
            device = next(self.model.parameters()).device
            instances_tensor = instances_tensor.to(device)

            if self.is_sequential:
                instances_tensor = instances_tensor.unsqueeze(1)

            shap_values = self.explainer.shap_values(instances_tensor)

            # Convert to numpy
            if isinstance(shap_values, list):
                shap_values = [sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv
                              for sv in shap_values]
            elif isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().numpy()

        return shap_values

    def get_feature_importance(
        self,
        instances: np.ndarray,
        class_idx: int = 1,
        nsamples: int = 100
    ) -> Dict[str, float]:
        """
        Get feature importance ranking for a specific class.

        Args:
            instances: Input instances
            class_idx: Class index to explain (default 1 = attack class)
            nsamples: Number of SHAP samples

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Compute SHAP values
        shap_values = self.explain(instances, nsamples=nsamples)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            class_shap = shap_values[class_idx]
        else:
            class_shap = shap_values[:, :, class_idx]

        # Compute mean absolute SHAP value per feature
        importance = np.abs(class_shap).mean(axis=0)

        # Create feature importance dict
        feature_importance = {
            name: score for name, score in zip(self.feature_names, importance)
        }

        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return feature_importance

    def plot_summary(
        self,
        instances: np.ndarray,
        class_idx: int = 1,
        nsamples: int = 100,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP summary plot.

        Args:
            instances: Input instances
            class_idx: Class to explain
            nsamples: Number of SHAP samples
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        # Compute SHAP values
        shap_values = self.explain(instances, nsamples=nsamples)

        # Prepare data
        if isinstance(shap_values, list):
            class_shap = shap_values[class_idx]
        else:
            class_shap = shap_values[:, :, class_idx]

        # Handle shape for plotting
        if self.is_sequential and len(instances.shape) == 3:
            instances_flat = instances.mean(axis=1)
        else:
            instances_flat = instances

        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            class_shap,
            instances_flat,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_force(
        self,
        instance_idx: int,
        instances: np.ndarray,
        class_idx: int = 1,
        nsamples: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP force plot for a single instance.

        Args:
            instance_idx: Index of instance to explain
            instances: Input instances
            class_idx: Class to explain
            nsamples: Number of SHAP samples
            save_path: Path to save figure
        """
        # Compute SHAP values
        shap_values = self.explain(instances, nsamples=nsamples)

        # Get single instance
        if isinstance(shap_values, list):
            instance_shap = shap_values[class_idx][instance_idx]
        else:
            instance_shap = shap_values[instance_idx, :, class_idx]

        # Prepare instance data
        if self.is_sequential and len(instances.shape) == 3:
            instance_data = instances[instance_idx].mean(axis=0)
        else:
            instance_data = instances[instance_idx]

        # Get base value (expected value)
        base_value = self.predict_fn(self.background_data).mean(axis=0)[class_idx]

        # Create force plot
        shap.force_plot(
            base_value,
            instance_shap,
            instance_data,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def explain_prediction(
    model: nn.Module,
    instance: np.ndarray,
    background_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_idx: int = 1,
    nsamples: int = 100
) -> Dict[str, float]:
    """
    Explain a single prediction using SHAP.

    Args:
        model: Trained model
        instance: Single instance to explain
        background_data: Background dataset
        feature_names: Feature names
        class_idx: Class to explain
        nsamples: Number of SHAP samples

    Returns:
        Feature importance dictionary
    """
    explainer = SHAPExplainer(
        model,
        background_data,
        feature_names=feature_names,
        explainer_type='kernel'
    )

    # Add batch dimension if needed
    if len(instance.shape) == 1:
        instance = instance.reshape(1, -1)

    importance = explainer.get_feature_importance(instance, class_idx, nsamples)

    return importance


def plot_shap_summary(
    model: nn.Module,
    instances: np.ndarray,
    background_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_idx: int = 1,
    nsamples: int = 100,
    save_path: Optional[str] = None
):
    """
    Create SHAP summary plot.

    Args:
        model: Trained model
        instances: Instances to explain
        background_data: Background dataset
        feature_names: Feature names
        class_idx: Class to explain
        nsamples: Number of SHAP samples
        save_path: Path to save figure
    """
    explainer = SHAPExplainer(
        model,
        background_data,
        feature_names=feature_names,
        explainer_type='kernel'
    )

    explainer.plot_summary(instances, class_idx, nsamples, save_path=save_path)
