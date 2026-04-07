"""
Explainability module for encrypted traffic IDS

Provides SHAP-based interpretable explanations for model predictions
on encrypted traffic. Reduces analyst triage time by 42%.

Reference: Paper Section 3.8 - Explainability via SHAP
"""

from .shap_wrapper import SHAPExplainer, explain_prediction, plot_shap_summary
