# Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection

**Complete Implementation - IEEE TNNLS Submission**

> Roger Nick Anaedevha, Alexander Gennadevich Trofimov, and Yuri Vladimirovich Borodachev. "Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection," IEEE Transactions on Neural Networks and Learning Systems, 2025.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.0](https://img.shields.io/badge/pytorch-2.1.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains the **complete implementation** of our encrypted traffic intrusion detection system achieving **97.8-99.9% accuracy** with **FPR ≤ 0.2%** on encrypted sessions without accessing payload contents.

### 🎯 Key Innovations

1. **Hybrid CNN-BiLSTM-Transformer-GNN Architecture** (99.87% accuracy on BoT-IoT)
2. **Protocol-Admissible Perturbations** with certified robustness (Theorem 1: ~58% improvement)
3. **Traffic-Aware Byzantine Filtering (TABF)** maintaining >95% accuracy with 40% Byzantine clients
4. **Federated Learning** with (ε, δ)-differential privacy guarantees
5. **Few-Shot Meta-Learning** for zero-day attack detection (93-98.5% on 5-way 5-shot)
6. **SHAP Explainability** for interpretable encrypted traffic decisions

### 🔥 Key Results

| Dataset | Model | Accuracy | F1-Score | FPR | Latency | Certified Radius |
|---------|-------|----------|----------|-----|---------|------------------|
| **BoT-IoT Encrypted** | CNN-LSTM | **99.87%** | 99.87% | 0.13% | 2.3ms | 0.189 |
| **CICIDS2017 HTTPS** | CNN-LSTM | **98.42%** | 98.59% | 1.32% | 2.3ms | 0.176 |
| **ISCX-VPN** | TransECA-Net | **98.94%** | 98.91% | 1.06% | 1.8ms | 0.182 |
| **Edge-IIoT (FL)** | TABF | **94.5%** | 94.2% | 0.98% | - | - |
| **Ensemble** | All Models | **99.92%** | 99.90% | 0.08% | 5.1ms | 0.195 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rogerpanel/encrypted_traffic_ids.git
cd encrypted_traffic_ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Datasets Setup

Download the datasets and organize as follows:

```
data/
├── CICIDS2017/
├── CICIDS2018/
├── UNSW-NB15/
├── ISCX-VPN-2016/
├── CESNET-TLS-Year22/
├── VisQUIC/
├── CIC-IoT-2023/
├── Edge-IIoTset/
├── BoT-IoT/
└── IIS3D/
```

See [datasets_bibliography.bib](datasets_bibliography.bib) for download links and citations.

### Basic Usage

```python
import torch
from models import HybridCNNLSTM
from data.dataset_loaders import DatasetFactory
from training.train import Trainer
from utils.reproducibility import set_seed
from utils.config_loader import load_config

# Set random seed for reproducibility
set_seed(42)

# Load configuration
config = load_config('configs/config.yaml')

# Load dataset
train_dataset, val_dataset, test_dataset = DatasetFactory.load_dataset(
    'BoTIoT',
    './data/BoT-IoT'
)

# Create data loaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridCNNLSTM(
    input_dim=8,
    num_classes=2,
    cnn_channels=[64, 128, 256, 512],
    lstm_hidden_dim=256,
    use_depthwise_separable=True,
    use_attention_fusion=True
).to(device)

# Train
trainer = Trainer(model, train_loader, val_loader, config, device)
history = trainer.train(num_epochs=100)

# Evaluate
from experiments import evaluate_model
metrics = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
```

## 📊 Reproducing Paper Results

### Table 2: TABF vs Baselines Under Byzantine Attacks

```python
from experiments import run_byzantine_evaluation
from models import HybridCNNLSTM

# Load model and datasets
model = HybridCNNLSTM.from_checkpoint('checkpoints/best_model.pth')
datasets = load_all_datasets(config)

# Run Byzantine evaluation
results = run_byzantine_evaluation(
    model,
    datasets,
    device,
    byzantine_ratios=[0.0, 0.1, 0.2, 0.3, 0.4]
)

# Expected: TABF maintains 95.3% with 40% Byzantine vs FedAvg's 68.2%
```

### Table 3: Comprehensive Ablation Study

```python
from experiments import run_ablation_study

# Run ablation study
results = run_ablation_study(
    base_config=config['model'],
    dataset=(train_dataset, val_dataset, test_dataset),
    device=device
)

# Configurations tested:
# - Spatial only (CNN)
# - Temporal only (LSTM)
# - CNN + LSTM (simple concat)
# - CNN + LSTM + Attention
# - Full model with depthwise separable convolutions
```

### Table 4: Certified Robustness Evaluation

```python
from experiments import run_robustness_evaluation
from adversarial import RandomizedSmoothing

# Evaluate certified robustness
results = run_robustness_evaluation(
    model,
    test_loader,
    device,
    epsilon_values=[0.01, 0.05, 0.1, 0.2]
)

# Expected: Enhanced radius = r_std * √(1 + (1-ρ)/ρ) ≈ 1.58× improvement
```

### Few-Shot Learning Evaluation

```python
from few_shot import PrototypicalNetwork, PrototypicalTrainer

# Initialize Prototypical Network
proto_net = PrototypicalNetwork(
    input_dim=8,
    embedding_dim=128,
    hidden_dims=[256, 512, 256]
).to(device)

# Train
trainer = PrototypicalTrainer(
    proto_net,
    device,
    n_way=5,
    k_shot=5,
    n_query=15
)

history = trainer.train(train_dataset, val_dataset, n_episodes=10000)
# Expected: 93-98.5% on 5-way 5-shot encrypted traffic classification
```

### SHAP Explainability

```python
from explainability import SHAPExplainer, plot_shap_summary

# Create explainer
explainer = SHAPExplainer(
    model,
    background_data=X_train[:100],  # Representative background
    feature_names=['Packet Size', 'IAT', 'Direction', 'Fwd Packets',
                  'Bwd Packets', 'Flow Duration', 'Fwd IAT', 'Bwd IAT']
)

# Explain predictions
feature_importance = explainer.get_feature_importance(X_test[:100], class_idx=1)
print("Top features for attack detection:")
for feature, importance in list(feature_importance.items())[:5]:
    print(f"  {feature}: {importance:.4f}")

# Generate summary plot
plot_shap_summary(model, X_test[:100], X_train[:100], save_path='shap_summary.pdf')
```

## 🏗️ Architecture Details

### Hybrid CNN-BiLSTM Model

```
Input (batch_size, seq_len, num_features)
    │
    ├─→ Spatial Pathway (CNN)
    │   ├─→ Multi-scale Convolutions (3×3, 5×5, 7×7, 9×9)
    │   ├─→ Depthwise Separable Conv (67% complexity ↓)
    │   ├─→ Batch Normalization + ReLU
    │   └─→ Global Pooling (Avg + Max)
    │
    ├─→ Temporal Pathway (Bi-LSTM)
    │   ├─→ Bidirectional LSTM (2 layers, 256 hidden)
    │   └─→ Final hidden state concatenation
    │
    ├─→ Attention Fusion
    │   └─→ Learned attention: α·spatial + (1-α)·temporal
    │
    └─→ Classification Head
        ├─→ FC(512) + ReLU + Dropout(0.5)
        ├─→ FC(256) + ReLU + Dropout(0.3)
        └─→ FC(num_classes)
```

**Complexity:** 67% reduction with depthwise separable convolutions
**Parameters:** ~2.3M trainable parameters
**Inference:** 2.3ms per sample on NVIDIA RTX 3090

### Protocol-Admissible Perturbations

**Definition 1:** For encrypted traffic x, perturbation δ is protocol-admissible if:
1. Packet sizes: 40 ≤ size(x + δ) ≤ 1500 bytes
2. Inter-arrival times: IAT(x + δ) ≥ 0
3. Direction flags: dir(x + δ) ∈ {0, 1}

**Theorem 1 (Enhanced Certified Radius):**

Given ρ = fraction of ℓ₂ ball satisfying protocol constraints, the enhanced certified radius is:

```
r_enhanced = r_std × √(1 + β(ρ))
```

where β(ρ) = (1 - ρ)/ρ and r_std is the standard randomized smoothing radius.

For TLS 1.3 traffic with ρ ≈ 0.42: **~58% improvement** over standard smoothing.

### Traffic-Aware Byzantine Filtering (TABF)

**Algorithm 1:** TABF Aggregation

```
Input: Global model θ, client updates {θ_m}, validation set D_val
Output: Aggregated parameters θ_agg

1. For each client m:
   - Compute IAT distribution: p_IAT^m
   - Compute TLS feature correlations: Corr(F_TLS^m)

2. Score clients:
   s_m = α · KL(p_val || p_m) + (1-α) · ||Corr_val - Corr_m||_F

3. Filter: Keep clients with s_m ≤ percentile(s, 75%)

4. Aggregate: θ_agg = median({θ_m : m ∈ trusted})
```

**Results:** Maintains 95.3% accuracy with 40% Byzantine clients (vs 68.2% for FedAvg)

## 📈 Training

### Standard Training

```bash
# Train CNN-LSTM on BoT-IoT
python -m training.train \
    --config configs/config.yaml \
    --model cnn_lstm \
    --dataset BoTIoT \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001
```

### Federated Learning with TABF

```bash
# Train with TABF aggregation
python -m training.federated_train \
    --config configs/config.yaml \
    --num-clients 10 \
    --num-rounds 20 \
    --aggregation tabf \
    --tabf-alpha 0.5 \
    --tabf-percentile 75
```

### Adversarial Training

```bash
# Train with protocol-aware adversarial examples
python -m training.adversarial_train \
    --config configs/config.yaml \
    --epsilon 0.1 \
    --attack-type protocol_pgd \
    --epochs 100
```

## 📁 Project Structure

```
encrypted_traffic_ids/
├── adversarial/                    # Adversarial robustness
│   ├── protocol_aware_robustness.py  # Protocol-admissible perturbations
│   └── __init__.py
├── configs/                        # Configuration files
│   └── config.yaml
├── data/                           # Data loading and preprocessing
│   ├── dataset.py                  # PyTorch datasets
│   ├── dataset_loaders.py          # Loaders for 9+ datasets
│   ├── loaders.py                  # Data loaders
│   ├── preprocessing.py            # Feature extraction
│   └── __init__.py
├── explainability/                 # SHAP interpretability
│   ├── shap_wrapper.py             # SHAP explainer
│   └── __init__.py
├── experiments/                    # Evaluation scripts
│   ├── evaluate_all.py             # Comprehensive evaluation
│   └── __init__.py
├── federated/                      # Federated learning
│   ├── aggregation.py              # Aggregation strategies
│   ├── differential_privacy.py     # DP mechanisms
│   ├── fedavg.py                   # FedAvg algorithm
│   ├── tabf.py                     # TABF algorithm
│   └── __init__.py
├── few_shot/                       # Few-shot learning
│   ├── maml.py                     # MAML implementation
│   ├── prototypical.py             # Prototypical Networks
│   └── __init__.py
├── models/                         # Model architectures
│   ├── base.py                     # Base model class
│   ├── cnn_lstm.py                 # Hybrid CNN-LSTM
│   ├── ensemble.py                 # Ensemble classifier
│   ├── gnn.py                      # GraphSAGE, GAT
│   ├── transformer.py              # TransECA-Net, FlowTransformer
│   └── __init__.py
├── training/                       # Training scripts
│   ├── train.py                    # Main training loop
│   └── __init__.py
├── utils/                          # Utilities
│   ├── config_loader.py            # Config loader
│   ├── metrics.py                  # Evaluation metrics
│   ├── reproducibility.py          # Seed setting
│   └── visualization.py            # Plotting
├── visualization/                  # Paper figures
│   └── plot_paper_figures.py       # Generate all figures
├── datasets_bibliography.bib       # Dataset citations
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── README_COMPLETE.md              # Extended documentation
```

## 🔧 Hyperparameters

All hyperparameters from the paper (see `configs/config.yaml`):

### Model Architecture
- CNN channels: [64, 128, 256, 512]
- LSTM hidden dim: 256
- Depthwise separable: True
- Attention fusion: True
- Dropout: 0.5 (FC1), 0.3 (FC2)

### Training
- Learning rate: 0.001 (Adam optimizer)
- Batch size: 128
- Epochs: 100
- Early stopping patience: 10
- LR scheduler: Exponential (γ = 0.95)
- Loss function: Focal Loss (α=0.25, γ=2.0)
- Gradient clipping: 5.0

### Federated Learning
- Clients: 10
- Communication rounds: 20
- Local epochs: 5
- Privacy budget ε: 1.0
- DP noise σ: 0.01
- TABF alpha: 0.5
- TABF percentile: 75

### Few-Shot Learning
- N-way: 5
- K-shot: 5
- Query samples: 15
- Embedding dim: 128
- Meta-learning rate: 0.001
- Inner learning rate (MAML): 0.01

### Robustness
- Smoothing σ: 0.1
- Certification samples: 10000
- Protocol ρ: 0.42 (TLS 1.3)

## 📊 Evaluation Metrics

Complete metrics computed by `experiments/evaluate_all.py`:

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and weighted
- **ROC-AUC, PR-AUC**: Area under curves
- **FPR**: False positive rate (critical for IDS)
- **MCC**: Matthews correlation coefficient
- **Inference Latency**: Mean ± std (ms per sample)
- **Throughput**: Samples per second
- **Memory**: Model parameters and GPU memory
- **FLOPs**: Floating-point operations
- **Certified Radius**: For robustness evaluation

## 🎯 Citation

If you use this code, please cite:

```bibtex
@article{anaedevha2025encrypted,
  title={Hybrid Spatial-Temporal Deep Learning for Privacy-Preserving Encrypted Traffic Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and Borodachev, Yuri Vladimirovich},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE}
}
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **National Research Nuclear University MEPhI** for computational resources
- **UNSW Canberra, UNB, CESNET** for datasets
- **IEEE TNNLS** for supporting rigorous research

## 📧 Contact

- **Roger Nick Anaedevha**: ar006@campus.mephi.ru
- **Project**: https://github.com/rogerpanel/encrypted_traffic_ids

---

**⭐ Star this repository if you find it helpful!**

**Code Availability:** This implementation is available at commit `<COMMIT_HASH_HERE>` for reproducibility.
