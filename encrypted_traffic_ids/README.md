# Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection

**Official implementation of the research paper:**
> Roger Nick Anaedevha, Alexander Gennadevich Trofimov, and Yuri Vladimirovich Borodachev. "Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection," IEEE Transactions on Information Forensics and Security, 2025.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.0](https://img.shields.io/badge/pytorch-2.1.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains the complete implementation of our encrypted traffic intrusion detection system that achieves **99.87% accuracy** on BoT-IoT encrypted sessions without accessing payload contents. The system combines:

- **Hybrid CNN-LSTM** for spatial-temporal feature learning
- **Transformer architectures** (TransECA-Net, FlowTransformer) with self-attention
- **Graph Neural Networks** for topology-aware detection
- **Federated Learning** with differential privacy (94.5-99.2% accuracy)
- **Few-Shot Learning** for zero-day attack detection
- **SHAP explainability** for interpretable predictions

## 🔥 Key Results

| Dataset | Model | Accuracy | F1-Score | FPR | Latency |
|---------|-------|----------|----------|-----|---------|
| **BoT-IoT Encrypted** | CNN-LSTM | **99.87%** | 99.87% | 0.13% | 2.3ms |
| **CICIDS2017 HTTPS** | CNN-LSTM | **98.42%** | 98.59% | 1.32% | 2.3ms |
| **ISCX-VPN** | TransECA-Net | **98.94%** | 98.91% | 1.06% | 1.8ms |
| **Edge-IIoT (FL)** | FedAvg | **94.5%** | 94.2% | 0.98% | - |
| **Ensemble** | All Models | **99.92%** | 99.90% | 0.08% | 5.1ms |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rogerpanel/encrypted_traffic_ids.git
cd encrypted_traffic_ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from models import HybridCNNLSTM
from data import FlowFeatureExtractor, create_dataloaders
from training import Trainer
from utils import load_config, set_seed

# Set random seed for reproducibility
set_seed(42)

# Load configuration
config = load_config('configs/config.yaml')

# Initialize model
model = HybridCNNLSTM(
    input_dim=8,  # Number of features per packet
    num_classes=2,  # Binary classification
    cnn_channels=[64, 128, 256, 512],
    lstm_hidden_dim=256,
    use_depthwise_separable=True
)

# Create data loaders (assuming you have preprocessed data)
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=128
)

# Initialize trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, train_loader, val_loader, config, device)

# Train model
history = trainer.train(num_epochs=100)
```

## 📊 Datasets

Our experiments use the following encrypted traffic datasets:

### 1. **ISCX-VPN-NonVPN-2016**
- **Size:** 14 application categories with VPN/non-VPN traffic
- **Citation:** Drapper Gil et al., "Characterization of Encrypted and VPN Traffic Using Time-Related Features," ICISSP 2016
- **Download:** https://www.unb.ca/cic/datasets/vpn.html

### 2. **CESNET-TLS-Year22**
- **Size:** 180 web service labels, year-long capture (2022)
- **Citation:** Jan Luxemburk et al., "CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from backbone lines," Scientific Data, 2024
- **DOI:** 10.1038/s41597-024-03927-4

### 3. **VisQUIC**
- **Size:** 100,000 labeled QUIC traces from 44,000+ websites
- **Citation:** Robert J. Shahla et al., "Exploring QUIC Dynamics: A Large-Scale Dataset for Encrypted Traffic Analysis," arXiv:2410.03728, 2024
- **Download:** https://github.com/robshahla/VisQUIC

### 4. **Edge-IIoTset**
- **Size:** 10+ IoT devices, 14 attack types, 61 features
- **Citation:** Mohamed Amine Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications," IEEE Access, 2022
- **DOI:** 10.1109/ACCESS.2022.3165809

### 5. **BoT-IoT**
- **Size:** 72M+ records (69.3 GB pcap, 16.7 GB CSV)
- **Citation:** Nickolaos Koroniotis et al., "Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset," Future Generation Computer Systems, 2019
- **DOI:** 10.1016/j.future.2018.10.045

## 🏗️ Architecture

### Hybrid CNN-LSTM Model

```
Input (batch_size, seq_len, num_features)
    │
    ├─→ Spatial Pathway (CNN)
    │   ├─→ Multi-scale Convolutions (3x3, 5x5, 7x7, 9x9)
    │   ├─→ Depthwise Separable Conv (67% complexity reduction)
    │   ├─→ Batch Normalization + ReLU
    │   └─→ Global Pooling (Avg + Max)
    │
    ├─→ Temporal Pathway (Bi-LSTM)
    │   ├─→ Bidirectional LSTM (2 layers, 256 hidden)
    │   └─→ Final hidden state concatenation
    │
    ├─→ Attention Fusion
    │   └─→ Learned attention weights for spatial + temporal
    │
    └─→ Classification Head
        ├─→ FC(512) + ReLU + Dropout
        ├─→ FC(256) + ReLU + Dropout
        └─→ FC(num_classes)
```

## 📈 Training

### Standard Training

```bash
# Train CNN-LSTM on encrypted traffic
python -m training.train \
    --config configs/config.yaml \
    --model cnn_lstm \
    --dataset BoT-IoT \
    --epochs 100 \
    --batch-size 128

# Train Transformer model
python -m training.train \
    --config configs/config.yaml \
    --model transformer \
    --dataset ISCX-VPN \
    --epochs 100
```

### Federated Learning

```bash
# Train with federated learning and differential privacy
python -m training.federated_train \
    --config configs/config.yaml \
    --num-clients 10 \
    --num-rounds 20 \
    --epsilon 1.0  # Privacy budget
```

### Generate Paper Figures

```bash
# Generate all figures from the paper
python visualization/plot_paper_figures.py
```

## 🔬 Reproducing Paper Results

### Table 1: Hybrid Architecture Performance

```python
from experiments import run_hybrid_experiment

results = run_hybrid_experiment(
    dataset='BoT-IoT',
    model_config={
        'cnn_channels': [64, 128, 256, 512],
        'lstm_hidden_dim': 256,
        'use_attention_fusion': True
    }
)
# Expected: 99.87% accuracy, 0.13% FPR
```

### Table 2: Transformer Evaluation

```python
from experiments import run_transformer_experiment

results = run_transformer_experiment(
    dataset='ISCX-VPN',
    model='TransECA-Net',
    use_eca=True
)
# Expected: 98.94% accuracy
```

### Table 3: Federated Learning Results

```python
from experiments import run_federated_experiment

results = run_federated_experiment(
    dataset='Edge-IIoT',
    num_clients=10,
    epsilon=1.0,
    use_differential_privacy=True
)
# Expected: 94.5% accuracy with ε=1.0
```

## 📁 Project Structure

```
encrypted_traffic_ids/
├── configs/                    # Configuration files
│   └── config.yaml
├── data/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── preprocessing.py        # Feature extraction
│   ├── dataset.py              # PyTorch datasets
│   └── loaders.py              # Data loaders
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── base.py                 # Base model class
│   ├── cnn_lstm.py             # Hybrid CNN-LSTM
│   ├── transformer.py          # TransECA-Net, FlowTransformer
│   ├── gnn.py                  # GraphSAGE, GAT
│   └── ensemble.py             # Ensemble classifier
├── federated/                  # Federated learning
│   ├── __init__.py
│   ├── fedavg.py               # FedAvg algorithm
│   ├── differential_privacy.py # DP mechanisms
│   └── aggregation.py          # Aggregation strategies
├── training/                   # Training scripts
│   └── train.py                # Main training loop
├── evaluation/                 # Evaluation utilities
├── visualization/              # Plotting and visualization
│   └── plot_paper_figures.py   # Generate paper figures
├── utils/                      # Utility functions
│   ├── metrics.py              # Evaluation metrics
│   ├── visualization.py        # Plotting utilities
│   ├── config_loader.py        # Configuration loader
│   └── reproducibility.py      # Seed setting
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Configuration

All experiments are configured through `configs/config.yaml`. Key parameters:

```yaml
# Model Architecture
model:
  cnn_lstm:
    cnn_channels: [64, 128, 256, 512]
    lstm_hidden_dim: 256
    use_depthwise_separable: true
    use_attention_fusion: true

# Training
training:
  learning_rate: 0.001
  batch_size: 128
  num_epochs: 100
  early_stopping_patience: 10
  loss_function: 'focal_loss'  # or 'cross_entropy'

# Federated Learning
federated:
  num_clients: 10
  num_rounds: 20
  epsilon: 1.0  # Privacy budget
  use_differential_privacy: true
```

## 📊 Evaluation Metrics

The codebase computes comprehensive metrics for Q1 journal standards:

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and weighted averages
- **ROC-AUC, PR-AUC**: Area under ROC and precision-recall curves
- **FPR (False Positive Rate)**: Critical for operational deployment
- **MCC (Matthews Correlation Coefficient)**: Robust to class imbalance
- **Inference Latency**: Real-time processing capability

## 🔐 Privacy-Preserving Features

### Federated Learning
- Decentralized training across multiple clients
- No centralization of sensitive encrypted traffic data
- Gradient similarity aggregation (35% communication reduction)

### Differential Privacy
- (ε, δ)-differential privacy guarantees
- Gaussian noise mechanism
- Privacy budget tracking
- Configurable privacy-utility tradeoff

## 🎯 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{anaedevha2025encrypted,
  title={Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and Borodachev, Yuri Vladimirovich},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **National Research Nuclear University MEPhI** for computational resources
- **UNSW Canberra, UNB, CESNET** for providing datasets
- **IEEE Access** for supporting open science

## 📧 Contact

- **Roger Nick Anaedevha**: ar006@campus.mephi.ru
- **Project Link**: https://github.com/rogerpanel/encrypted_traffic_ids

## 🔗 Related Work

- [TransECA-Net](https://doi.org/10.xxxx) - Transformer with Efficient Channel Attention
- [E-GRACL](https://doi.org/10.xxxx) - GraphSAGE for encrypted traffic
- [NIDS-FGPA](https://doi.org/10.xxxx) - Federated learning with Paillier encryption

---

**⭐ Star this repository if you find it helpful!**
