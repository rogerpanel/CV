# Temporal Adaptive Neural ODEs for Real-Time Network Intrusion Detection

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxxx)
[![IEEE](https://img.shields.io/badge/IEEE-TNNLS-blue.svg)](https://ieeexplore.ieee.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Temporal Adaptive Neural Ordinary Differential Equations with Deep Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection"**

**Authors:** Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev

**Submitted to:** IEEE Transactions on Neural Networks and Learning Systems (November 2025)

---

## 🎯 Key Contributions

1. **TA-BN-ODE Architecture**: Temporal Adaptive Batch Normalization Neural ODEs resolving incompatibility between discrete normalization and continuous dynamics
2. **Deep Spatio-Temporal Point Processes**: Transformer-based intensity modeling with logarithmic barrier optimization (O(n³) → O(n^(3/2)) complexity reduction)
3. **Bayesian Inference with Calibration**: Structured variational inference achieving 91.7% coverage and ECE = 0.017
4. **LLM Integration**: Llama-3.1-8B-Instruct for zero-shot detection (87.6% F1 on novel attacks)
5. **Online Learning Under Drift**: PSI-based drift detection with Elastic Weight Consolidation (98.8% accuracy over 50 days)

---

## 📊 Results

### Detection Performance (ICS3D Datasets - 18.9M records)

| Dataset | Accuracy | F1-Score | Latency (P50/P99) |
|---------|----------|----------|-------------------|
| Container Security (697K) | **99.4%** | 99.3% | 8.1ms / 22.3ms |
| Edge-IIoT (4M) | **98.6%** | 98.5% | 8.2ms / 22.9ms |
| GUIDE SOC (1M) | 92.9% | **92.7%** | 8.5ms / 23.1ms |

### Standard Benchmarks

| Dataset | Accuracy | Parameters | Throughput |
|---------|----------|------------|------------|
| CIC-IDS2018 | **97.8%** | 2.3M | 12.5M events/sec |
| UNSW-NB15 | **96.3%** | 2.3M | 12.1M events/sec |
| CIC-IoT-2023 | **98.2%** | 2.3M | 12.4M events/sec |

### Calibration Metrics

- **Expected Calibration Error (ECE):** 0.017 (vs. baseline 0.089)
- **Coverage Probability:** 91.7% for 95% prediction intervals
- **Temperature Scaling:** Optimized on validation set

### Performance

- **Throughput:** 12.3 million events/second (batch size 256)
- **Median Latency:** 8.2ms (P99: 22.9ms)
- **Model Size:** 2.3M parameters (82% reduction vs. 12.8M baseline)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rogerpanel/CV.git
cd CV

# Create conda environment
conda create -n ta-bn-ode python=3.9
conda activate ta-bn-ode

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchdiffeq
pip install numpy pandas scikit-learn matplotlib seaborn
pip install transformers  # For LLM integration (optional)
pip install pyyaml tqdm
```

### Training

```bash
# Train on Edge-IIoT dataset
python train_ta_bn_ode.py --config config.yaml --dataset Edge-IIoT

# Train with LLM integration (requires Llama-3.1-8B-Instruct)
python train_ta_bn_ode.py --config config.yaml --enable-llm

# Train with online learning
python train_ta_bn_ode.py --config config.yaml --online-learning

# Quick test with small sample
python train_ta_bn_ode.py --config config.yaml --sample-size 10000
```

### Evaluation

```bash
# Comprehensive evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset Edge-IIoT

# Measure throughput and latency
python evaluate.py --checkpoint checkpoints/best_model.pt --benchmark
```

### Inference

```python
import torch
from ta_bn_ode import create_ta_bn_ode_model

# Load model
model = create_ta_bn_ode_model(input_dim=64, num_classes=15)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()

# Predict
x = torch.randn(1, 64)  # Network features
output, hidden = model(x)
prediction = output.argmax(dim=-1)

# With uncertainty
from bayesian_inference import BayesianNeuralODEPP
bayesian_model = BayesianNeuralODEPP(model, n_train=10000)
mean_pred, std_pred = bayesian_model(x, n_samples=10, return_uncertainty=True)
```

---

## 📁 Project Structure

```
CV/
├── ta_bn_ode.py                 # TA-BN-ODE architecture
├── point_process.py             # Spatio-temporal point processes
├── bayesian_inference.py        # Bayesian inference & calibration
├── llm_integration.py           # LLM zero-shot detection
├── online_learning.py           # Online learning with drift detection
├── evaluation_metrics.py        # Comprehensive evaluation metrics
├── dataset_loaders.py           # Dataset loaders (ICS3D, benchmarks)
├── train_ta_bn_ode.py          # Main training script
├── config.yaml                  # Configuration file
├── README_TABN_ODE.md          # This file
├── node_v10ca.tex              # Paper (main)
├── node_v10caapp.tex           # Paper (supplementary)
└── checkpoints/                 # Model checkpoints
```

---

## 🔬 Technical Details

### TA-BN-ODE Architecture

```python
from ta_bn_ode import TA_BN_ODE

model = TA_BN_ODE(
    input_dim=64,
    hidden_dim=256,
    output_dim=15,
    n_ode_blocks=2,
    ode_layers=2,
    solver='dopri5',  # Adaptive Runge-Kutta (Dormand-Prince)
    rtol=1e-3,
    atol=1e-4
)

# Multi-scale time constants (microseconds to hours)
time_constants = [1e-6, 1e-3, 1.0, 3600.0]
```

### Point Process

```python
from point_process import MarkedTemporalPointProcess

point_process = MarkedTemporalPointProcess(
    n_types=15,  # Attack types
    d_model=256,
    nhead=8,
    num_layers=4
)

# Compute conditional intensity
lambda_t = point_process.compute_intensity(t, event_times, event_types)

# Log-likelihood
log_lik = point_process.log_likelihood(events, T)
```

### Online Learning

```python
from online_learning import OnlineAdaptiveSystem

online_system = OnlineAdaptiveSystem(
    model=model,
    ema_rate=0.98,  # Exponential moving average
    ewc_weight=5e-3,  # Elastic Weight Consolidation
    psi_threshold=0.2  # Population Stability Index
)

# Online update
metrics = online_system.update_online(x, y, X_batch)

# Drift detected: PSI > 0.2
if metrics['drift_detected']:
    print(f"Drift detected! PSI: {metrics['psi']:.4f}")
```

---

## 📈 Hyperparameters (From Paper Supplementary)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hidden dimension | 256 | Model dimension |
| ODE blocks | 2 | Number of ODE blocks |
| Transformer layers | 4 | Point process layers |
| Attention heads | 8 | Multi-head attention |
| Batch size | 256 | Training batch size |
| Learning rate | 1e-3 → 1e-5 | Cosine annealing |
| EMA rate (ρ) | 0.98 | Online learning |
| EWC weight (η) | 5×10⁻³ | Catastrophic forgetting |
| PSI threshold | 0.2 | Drift detection |

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{anaedevha2025tabnnode,
  title={Temporal Adaptive Neural Ordinary Differential Equations with Deep Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and Borodachev, Yuri Vladimirovich},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  note={Submitted}
}
```

---

## 📝 Key Algorithms

### Algorithm 1: TA-BN-ODE Forward Pass

1. Encode inputs: `h₀ = Encoder(x)`
2. For each ODE block:
   - Integrate: `h(t) = ODESolve(f_θ, h₀, [0,T], dopri5)`
   - Apply TA-BN: `h = TA-BN(h, t)`
3. Decode: `y = Decoder(h(T))`

### Algorithm 2: Online Update with EWC and DP-SGD

1. Monitor drift: `PSI = Σ(p_current - p_baseline) × log(p_current/p_baseline)`
2. If `PSI > 0.2`: Trigger adaptation
3. Compute loss: `L = L_task + (η/2) Σ F_i(θ_i - θ*_i)²`
4. Apply DP-SGD: Clip gradients and add Gaussian noise
5. Update EMA: `θ_ema ← ρ θ_ema + (1-ρ) θ`

---

## 🔧 Advanced Usage

### Custom Dataset

```python
from dataset_loaders import create_dataloaders
import numpy as np

# Your data
X = np.load('my_network_features.npy')  # [n_samples, n_features]
y = np.load('my_labels.npy')  # [n_samples]

# Create dataloaders
dataloaders = create_dataloaders(X, y, batch_size=256)

# Train
from train_ta_bn_ode import train_model
model = train_model(dataloaders, config)
```

### Temperature Scaling Calibration

```python
from bayesian_inference import TemperatureScaling

temp_scaling = TemperatureScaling(num_classes=15)

# Fit on validation set
temp_scaling.fit(val_logits, val_labels)

# Apply to test set
calibrated_probs = temp_scaling(test_logits)
```

### Zero-Shot Detection with LLM

```python
from llm_integration import LLMTemporalReasoning

llm = LLMTemporalReasoning(model_name="meta-llama/Llama-3.1-8B-Instruct")

# Zero-shot detection
results = llm.zero_shot_detect(
    feature_vector=x,
    feature_names=feature_names,
    attack_stages=['reconnaissance', 'exfiltration']
)

print(f"Detected: {results['overall']['attack_type']}")
print(f"Confidence: {results['overall']['confidence']:.2%}")
```

---

## 📊 Datasets

### ICS3D (Integrated Cloud Security 3Datasets)

- **Container Security**: 697,289 samples, 99.4% accuracy
- **Edge-IIoT**: 4,000,000 samples, 98.6% accuracy
- **GUIDE SOC**: 1,000,000 samples, 92.7% F1-score

### Standard Benchmarks

- **CIC-IDS2018**: 97.8% accuracy
- **UNSW-NB15**: 96.3% accuracy
- **CIC-IoT-2023**: 98.2% accuracy

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

- **Roger Nick Anaedevha**: [rogernickanaedevha@gmail.com](mailto:rogernickanaedevha@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- IEEE Transactions on Neural Networks and Learning Systems
- ICS3D dataset providers
- PyTorch and torchdiffeq teams
- Meta AI for Llama-3.1-8B-Instruct

---

## 📚 References

See `node_v10ca.tex` and `node_v10caapp.tex` for complete references.

---

**Note:** This codebase implements the paper submitted to IEEE TNNLS in November 2025. For questions about the paper, please refer to the LaTeX sources or contact the authors.
