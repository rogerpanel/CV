"""
SHAP Explainability for Encrypted Traffic Intrusion Detection

This module provides interpretability for encrypted traffic IDS decisions using
SHapley Additive exPlanations (SHAP). Identifies which traffic features (packet sizes,
inter-arrival times, etc.) contribute most to attack detection without accessing
payload contents.

Components:
- KernelSHAP wrapper for deep learning models
- TreeSHAP wrapper for tree-based ensembles
- Visualization utilities for SHAP values

References:
    Paper Section 3.6 - Explainability via SHAP
    Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions
"""

from .shap_wrapper import SHAPExplainer, explain_prediction, plot_shap_summary

__all__ = [
    'SHAPExplainer',
    'explain_prediction',
    'plot_shap_summary'
]
