"""
etraffic_codebase: Encrypted Traffic Intrusion Detection System
================================================================

Official implementation for:
"Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving
 Encrypted Traffic Intrusion Detection"

Submitted to: Engineering Applications of Artificial Intelligence (EAAI)

Authors:
    Roger Nick Anaedevha, Alexander G. Trofimov, Yuri V. Borodachev
    National Research Nuclear University MEPhI, Moscow, Russia

Modules:
    models          - Neural network architectures (CNN-LSTM, Transformer, GNN, Ensemble)
    data            - Data loading and preprocessing for 10 benchmark datasets
    training        - Training pipeline with FocalLoss and early stopping
    federated       - Federated learning with TABF Byzantine filtering
    adversarial     - Protocol-aware certified robustness (Theorem 1)
    self_supervised - InfoNCE contrastive pretraining
    few_shot        - Prototypical Networks and MAML meta-learning
    explainability  - SHAP-based model interpretability
    experiments     - Comprehensive evaluation and paper results reproduction
    visualization   - Paper figure generation
    utils           - Configuration, metrics, reproducibility utilities
"""

__version__ = '1.0.0'
__author__ = 'Roger Nick Anaedevha'
__email__ = 'ar006@campus.mephi.ru'
