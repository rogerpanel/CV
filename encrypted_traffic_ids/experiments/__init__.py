"""
Experiments Module for Encrypted Traffic IDS

Provides comprehensive evaluation scripts for reproducing all paper results:
- Table 1: Hybrid architecture performance across datasets
- Table 2: TABF vs baselines under Byzantine attacks
- Table 3: Comprehensive ablation study
- Table 4: Certified robustness comparison
- Table 5: State-of-the-art comparison

All evaluations include full metrics suite: accuracy, precision, recall, F1, FPR,
MCC, ROC-AUC, inference latency, memory usage, throughput, and FLOPs.
"""

from .evaluate_all import (
    evaluate_model,
    evaluate_all_datasets,
    run_ablation_study,
    run_byzantine_evaluation,
    run_robustness_evaluation
)

__all__ = [
    'evaluate_model',
    'evaluate_all_datasets',
    'run_ablation_study',
    'run_byzantine_evaluation',
    'run_robustness_evaluation'
]
