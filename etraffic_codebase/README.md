# Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection

**Official implementation of the research paper submitted to Engineering Applications of Artificial Intelligence (EAAI).**

> Roger Nick Anaedevha, Alexander Gennadevich Trofimov, and Yuri Vladimirovich Borodachev.
> "Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection."
> National Research Nuclear University MEPhI, Moscow, Russia.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.0](https://img.shields.io/badge/pytorch-2.1.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository contains the complete implementation of our encrypted traffic intrusion detection framework that achieves **97.8--99.9% detection accuracy** with false positive rates ≤0.2% across nine heterogeneous benchmarks. The system analyzes encrypted network traffic without requiring payload access by combining four complementary neural network branches:

- **Hybrid CNN-BiLSTM** for spatial-temporal feature learning (Section 3.1)
- **Transformer with Efficient Channel Attention** (TransECA-Net) for long-range dependencies (Section 3.2)
- **Graph Neural Networks** for coordinated multi-flow attack detection (Section 3.3)
- **Ensemble aggregation** achieving 99.92% accuracy on CICIDS2017 (Section 3.4)

Additional contributions:
- **Protocol-aware certified robustness** enlarging robust radii by 58% over standard methods (Theorem 1)
- **Traffic-Aware Byzantine Filtering (TABF)** maintaining >95% accuracy under 40% adversaries (Algorithm 1)
- **Self-supervised InfoNCE pretraining** improving few-shot detection by 7.3 pp (Section 3.6)
- **SHAP explainability** reducing analyst triage time by 42% (Section 3.8)

## Key Results

| Dataset | Model | Accuracy | F1-Score | FPR | Latency |
|---------|-------|----------|----------|-----|---------|
| **BoT-IoT Encrypted** | CNN-LSTM | **99.87%** | 99.87% | 0.13% | 2.3ms |
| **CICIDS2017 HTTPS** | CNN-LSTM | **98.42%** | 98.59% | 1.32% | 2.3ms |
| **ISCX-VPN** | TransECA-Net | **98.94%** | 98.91% | 1.06% | 1.8ms |
| **Edge-IIoT (FL)** | FedAvg+DP | **94.5%** | 94.2% | 0.98% | - |
| **Ensemble** | All Models | **99.92%** | 99.90% | 0.08% | 5.1ms |

Six-month pilot deployment on ~10,000 devices: 28/37 confirmed threats, 0.18% operational FPR.

## Quick Start

### Installation

```bash
cd etraffic_codebase
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from models import HybridCNNLSTM
from data import create_dataloaders
from training import Trainer
from utils import load_config, set_seed

# Reproducibility
set_seed(42)

# Load configuration
config = load_config('configs/config.yaml')

# Initialize model (3,847,234 parameters)
model = HybridCNNLSTM(
    input_dim=88,           # 88 features from IIS3D dataset
    num_classes=6,          # 6 HTTPS traffic categories
    cnn_channels=[64, 128, 256, 512],
    lstm_hidden_dim=256,
    use_depthwise_separable=True,
    use_attention_fusion=True
)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=128, use_weighted_sampling=True
)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, train_loader, val_loader, config.training.__dict__, device)
history = trainer.train(num_epochs=100)
```

## Datasets

### Primary Dataset: IIS3D / HTTPS Traffic Classification
- **DOI:** [10.34740/kaggle/dsv/12479689](https://doi.org/10.34740/kaggle/dsv/12479689)
- **Size:** 145,671 HTTPS flow records, 88 numerical features
- **Classes:** W (Website, 55.5%), D (Download, 14.0%), P (Video, 8.6%), U (Upload, 7.5%), M (Music, 7.4%), L (Live Video, 7.1%)
- **Split:** 70% train / 15% validation / 15% test (stratified)

### Additional Benchmarks (Section 4.1)
| Dataset | Records | Features | Classes | Reference |
|---------|---------|----------|---------|-----------|
| CICIDS2017 | 2.8M | 78 | 15 | Sharafaldin et al. (2018) |
| CICIDS2018 | 16M | 80 | 14 | CSE-CIC-IDS2018 |
| UNSW-NB15 | 2.5M | 49 | 10 | Moustafa & Slay (2015) |
| BoT-IoT | 72M | 46 | 5 | Koroniotis et al. (2019) |
| ISCX-VPN | 260K | 28 | 14 | Drapper Gil et al. (2016) |
| CESNET-TLS | - | - | 180 | Luxemburk et al. (2024) |
| VisQUIC | 100K | - | 44K+ | Shahla et al. (2024) |
| Edge-IIoTset | - | 61 | 14 | Ferrag et al. (2022) |
| CIC-IoT-2023 | - | - | - | CIC (2023) |

## Architecture

```
Input (batch_size, seq_len, 88_features)
    |
    +---> Spatial Pathway (Multi-Scale CNN)
    |     +---> Parallel Conv1D (k=3,5,7,9) with Depthwise Separable
    |     +---> BatchNorm + ReLU + Dropout
    |     +---> Global Avg+Max Pooling
    |
    +---> Temporal Pathway (Bidirectional LSTM)
    |     +---> 2-layer BiLSTM (hidden=256)
    |     +---> Final hidden state concatenation
    |
    +---> Attention Fusion
    |     +---> Learned weights for spatial + temporal
    |
    +---> Classification Head
          +---> FC(512) -> FC(256) -> FC(num_classes)
```

Total parameters: **3,847,234**

## Training

### Standard Training
```bash
python -m training.train --config configs/config.yaml --model cnn_lstm --dataset IIS3D --epochs 100
```

### Federated Learning with TABF
```bash
python -m training.train --config configs/config.yaml --federated --num-clients 10 --num-rounds 20 --epsilon 1.0
```

### Self-Supervised Pretraining
```python
from self_supervised import ContrastiveEncoder, ContrastivePretrainer, TrafficAugmentation
from models import HybridCNNLSTM

backbone = HybridCNNLSTM(input_dim=88, num_classes=6)
encoder = ContrastiveEncoder(backbone, feature_dim=512, projection_dim=128)
pretrainer = ContrastivePretrainer(encoder, TrafficAugmentation(), temperature=0.07)
pretrainer.pretrain(unlabeled_loader, num_epochs=100)
```

### Generate Paper Figures
```bash
python visualization/plot_paper_figures.py
```

## Reproducing Paper Results

### Table 1: Hybrid Architecture Performance
```python
from experiments import evaluate_model
from models import HybridCNNLSTM

model = HybridCNNLSTM(input_dim=88, num_classes=6)
metrics = evaluate_model(model, test_loader, device)
# Expected: 99.87% accuracy, 0.13% FPR on BoT-IoT
```

### Table 3: Federated Learning with Byzantine Resilience
```python
from experiments import run_byzantine_evaluation
results = run_byzantine_evaluation(model, datasets, device)
# TABF maintains >95% accuracy with 40% compromised clients
```

### Theorem 1: Certified Robustness
```python
from experiments import run_robustness_evaluation
results = run_robustness_evaluation(model, test_loader, device)
# Protocol-aware radius ~1.58x standard randomized smoothing
```

## Project Structure

```
etraffic_codebase/
+-- configs/                    # Configuration files
|   +-- config.yaml             # Master configuration
+-- data/                       # Data loading and preprocessing
|   +-- preprocessing.py        # Feature extraction (88 features)
|   +-- dataset.py              # PyTorch Dataset classes
|   +-- dataset_loaders.py      # Per-dataset loaders (10 datasets)
|   +-- loaders.py              # DataLoader creation
+-- models/                     # Neural network architectures
|   +-- base.py                 # Base model class
|   +-- cnn_lstm.py             # Hybrid CNN-BiLSTM (Section 3.1)
|   +-- transformer.py          # TransECA-Net, FlowTransformer (Section 3.2)
|   +-- gnn.py                  # GraphSAGE, GAT (Section 3.3)
|   +-- ensemble.py             # Ensemble classifier (Section 3.4)
+-- federated/                  # Federated learning (Section 3.5)
|   +-- fedavg.py               # FedAvg algorithm
|   +-- tabf.py                 # Traffic-Aware Byzantine Filtering (Algorithm 1)
|   +-- differential_privacy.py # (epsilon,delta)-DP mechanisms
|   +-- aggregation.py          # Aggregation strategies
+-- adversarial/                # Adversarial robustness (Section 3.4)
|   +-- protocol_aware_robustness.py  # Theorem 1, Definition 1
+-- self_supervised/            # Self-supervised pretraining (Section 3.6)
|   +-- contrastive.py          # InfoNCE contrastive learning
+-- few_shot/                   # Few-shot learning (Section 3.7)
|   +-- prototypical.py         # Prototypical Networks
|   +-- maml.py                 # MAML meta-learning
+-- training/                   # Training pipeline (Section 4.2)
|   +-- train.py                # Trainer with FocalLoss, early stopping
+-- experiments/                # Evaluation scripts (Section 4)
|   +-- evaluate_all.py         # All paper results reproduction
+-- explainability/             # SHAP explainability (Section 3.8)
|   +-- shap_wrapper.py         # SHAP feature importance
+-- visualization/              # Paper figures
|   +-- plot_paper_figures.py   # All paper figures
+-- utils/                      # Utilities
|   +-- config_loader.py        # YAML configuration
|   +-- metrics.py              # Evaluation metrics
|   +-- reproducibility.py      # Seed setting
|   +-- visualization.py        # Plotting utilities
+-- manuscripts/                # Paper LaTeX sources
+-- outputs/                    # Generated figures
+-- requirements.txt            # Python dependencies
+-- README.md                   # This file
```

## Evaluation Metrics

All metrics follow Q1 journal standards:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and weighted averages
- **ROC-AUC, PR-AUC**: Area under curves
- **FPR**: False Positive Rate (critical for deployment)
- **MCC**: Matthews Correlation Coefficient (robust to imbalance)
- **Inference Latency**: Real-time processing capability

## Privacy-Preserving Features

### Federated Learning
- Decentralized training across distributed monitoring nodes
- No centralization of sensitive traffic data
- Gradient Similarity Aggregation (35% communication reduction)

### Differential Privacy
- (epsilon, delta)-DP with Gaussian noise mechanism
- sigma = sqrt(2 * ln(1.25/delta)) / epsilon
- Configurable privacy-utility tradeoff

### CKKS Homomorphic Encryption
- Encrypted parameter aggregation (~12% overhead)
- Ring dimension N = 2^13

## Citation

```bibtex
@article{anaedevha2025encrypted,
  title={Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving
         Encrypted Traffic Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich
          and Borodachev, Yuri Vladimirovich},
  journal={Engineering Applications of Artificial Intelligence},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- National Research Nuclear University MEPhI for computational resources
- Grant for research centers in Artificial Intelligence, Ministry of Economic Development of the Russian Federation
- UNSW Canberra, UNB, CESNET for providing benchmark datasets

## Contact

- **Roger Nick Anaedevha**: ar006@campus.mephi.ru
- **Manuscript Repository**: [rogerpanel/eTraffic-models](https://github.com/rogerpanel/eTraffic-models)
- **Dataset DOI**: [10.34740/kaggle/dsv/12479689](https://doi.org/10.34740/kaggle/dsv/12479689)
