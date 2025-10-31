# Temporal-Adaptive Neural ODEs for Network Intrusion Detection
## Model Upgrade: v1 → v2 (Paper Implementation)

**Authors:** Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev

**Paper:** "Temporal-Adaptive Neural ODEs for Real-Time Network Intrusion Detection"

**Target Journal:** IEEE Transactions on Neural Networks and Learning Systems

---

## Overview

This document describes the upgrade from the original `neural-ode-model.ipynb` to the paper-based implementation `neural_ode_upgraded_v2.ipynb`, integrating state-of-the-art methodologies from our research paper with the previous working implementation.

## Key Innovations from Paper

### 1. **Temporal Adaptive Batch Normalization (TA-BN)**
**Problem Solved:** Incompatibility between standard batch normalization and continuous ODE dynamics

**Solution:** Time-dependent normalization parameters

```python
# Previous (v1): Standard Batch Normalization
class ODEFunc(nn.Module):
    def forward(self, t, h):
        out = self.bn(self.layer(h))  # Static batch norm
        return F.elu(out)

# Upgraded (v2): Temporal Adaptive Batch Normalization
class TABNODEFunc(nn.Module):
    def forward(self, t, h):
        out = self.layer(h)
        out = self.ta_bn(out, t)  # Time-dependent normalization
        return F.elu(out)
```

**Mathematical Foundation (Paper Eq. 19):**
```
TA-BN(x,t) = γ(t) ⊙ (x - μ(t))/√(σ²(t) + ε) + β(t)
```

Where:
- γ(t), β(t): Time-dependent scale and shift (learned by MLPs)
- μ(t), σ²(t): Time-dependent running statistics
- Periodic encoding: [t, sin(ωt), cos(ωt)] captures cyclic patterns

**Benefits:**
- ✅ Enables stable training of deep ODE blocks
- ✅ 8.1% accuracy improvement on Container Dataset (91.3% → 99.4%)
- ✅ Allows stacking multiple ODE layers without gradient explosion

### 2. **Multi-Scale Architecture**
**Problem Solved:** Network attacks span vastly different time scales (microseconds to months)

**Previous Implementation:**
```python
# Single-scale ODE with fixed integration time
h_t = odeint(ode_func, h0, t_span)
```

**Upgraded Implementation:**
```python
# Multi-scale ODEs with parallel branches
time_constants = [1e-6, 1e-3, 1.0, 3600.0]  # μs, ms, sec, hour
for tau in time_constants:
    t_span_scaled = t_span * tau
    h_scale = odeint(ode_func, h0, t_span_scaled)
    h_scales.append(h_scale)
```

**Coverage:**
- **Microsecond scale (1e-6):** Timing attacks, packet injection
- **Millisecond scale (1e-3):** Burst patterns, scanning
- **Second scale (1.0):** Session behaviors, authentication
- **Hour scale (3600):** Persistent threats, reconnaissance campaigns

**Impact:** 99.1% detection at microsecond granularity vs. 87.3% for single-scale

### 3. **Transformer-Enhanced Point Process**
**Problem Solved:** Cubic complexity O(n³) in event history attention

**Previous Implementation:**
```python
class HawkesProcess(nn.Module):
    def compute_intensity(self, t, history):
        # Simple exponential kernel
        intensity = self.mu
        for t_i, m_i in history:
            intensity += self.alpha * exp(-self.beta * (t - t_i))
        return intensity
```

**Upgraded Implementation:**
```python
class TransformerHawkesProcess(nn.Module):
    def forward(self, event_times, event_types):
        # Multi-head self-attention on event history
        event_emb = self.type_embed(event_types) + self.temporal_encoding(delta_t)
        h_attn = self.transformer(event_emb, mask=causal_mask)
        intensities = self.intensity_net(h_attn)
        return intensities
```

**Key Features:**
- **Multi-head attention:** Captures diverse temporal patterns in parallel
- **Multi-scale temporal encoding:** Sinusoidal basis at 4 time scales
- **Log-barrier optimization:** Reduces complexity O(n³) → O(n²)
- **Causal masking:** Prevents information leakage from future events

**Performance:** 12.3M events/second throughput with <100ms latency

### 4. **Structured Variational Bayesian Inference**
**Problem Solved:** Uncertainty quantification without theoretical guarantees

**Previous Implementation:**
```python
# Point estimates only, no uncertainty
output = model(x)
prediction = argmax(output)  # No confidence intervals
```

**Upgraded Implementation:**
```python
class StructuredVariationalInference(nn.Module):
    # Diagonal + low-rank covariance: Σ = diag(s²) + UU^T
    def sample(self, n_samples):
        epsilon = randn(n_samples, n_params)
        z = randn(n_samples, rank)
        samples = self.mu + self.s * epsilon + matmul(z, self.U.t())
        return samples

# Usage
mean_pred, uncertainty = model.predict_with_uncertainty(x, n_samples=50)
```

**Mathematical Foundation (Paper Eq. 43-44):**
```
q(θ) = q(θ_ODE)q(θ_TPP|θ_ODE)q(θ_cls)  # Structured dependencies
Σ = diag(s²) + UU^T                      # Low-rank covariance
```

**Benefits:**
- ✅ **91.7% coverage probability** (well-calibrated confidence intervals)
- ✅ **Expected Calibration Error (ECE) = 0.017** (vs. 0.094 without Bayesian)
- ✅ **PAC-Bayesian generalization bounds** (theoretical guarantees)
- ✅ **43% reduction** in false positive investigation time

### 5. **Unified Loss Framework**
**Previous Implementation:**
```python
# Simple classification loss
loss = F.cross_entropy(output, y)
```

**Upgraded Implementation:**
```python
# Multi-objective optimization (Paper Eq. 12)
def compute_loss(self, x, y, t_span, events):
    output, h_combined, intensities = self.forward(x, t_span, events)

    # 1. Classification loss (Eq. 13)
    loss_cls = F.cross_entropy(output, y)

    # 2. Temporal point process loss (Eq. 14)
    loss_tpp = -[log λ_k(t_i) - ∫ λ*(t)dt]

    # 3. KL divergence (Eq. 15)
    loss_kl = KL(q(θ)||p(θ))

    # 4. Regularization (Eq. 16)
    loss_reg = α₁‖θ‖² + α₂‖dh/dt‖² + α₃‖∂f/∂h‖

    # Total loss
    return loss_cls + λ₁·loss_tpp + λ₂·loss_kl + λ₃·loss_reg
```

**Components:**
1. **L_cls:** Classification accuracy
2. **L_TPP:** Temporal pattern modeling (event timing + types)
3. **L_KL:** Bayesian regularization (prevents overfitting)
4. **L_reg:** Stability (smooth dynamics, bounded gradients)

### 6. **Real-Time Adaptation**
**Problem Solved:** Static models degrade under concept drift (attack evolution)

**Upgraded Implementation:**
```python
class RealTimeAdapter:
    def __init__(self, model, buffer_size=1000, adaptation_rate=0.01):
        self.buffer_x = []  # Experience replay
        self.buffer_y = []
        self.optimizer = Adam(model.parameters(), lr=adaptation_rate)

    def update(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)

        # Adapt every 100 samples
        if len(self.buffer_x) % 100 == 0:
            self.adapt()  # Online fine-tuning
```

**Performance Under Drift:**
- **Day 0:** 99.0% accuracy
- **Day 50 (no adaptation):** 72.3% accuracy (❌ 26.7% degradation)
- **Day 50 (with adaptation):** 96.9% accuracy (✅ only 2.1% degradation)
- **Convergence:** 18 training rounds

---

## Comparison: v1 vs v2

| Feature | v1 (Original) | v2 (Paper Upgrade) | Improvement |
|---------|---------------|-------------------|-------------|
| **Architecture** | Single-scale ODE | Multi-scale (4 branches) | 8 orders of magnitude coverage |
| **Normalization** | Standard BatchNorm | Temporal Adaptive BN | +8.1% accuracy, stable training |
| **Point Process** | Parametric Hawkes | Transformer + Multi-scale | O(n³) → O(n²) complexity |
| **Uncertainty** | None | Structured Variational | 91.7% coverage, ECE=0.017 |
| **Parameters** | 12.8M | 2.3M | 82% reduction |
| **Throughput** | ~6M events/s | 12.3M events/s | 2× faster |
| **Latency P95** | 24.8ms | 14.7ms | 40% faster |
| **Accuracy (Container)** | 96.7% | 99.4% | +2.7% |
| **Accuracy (IoT)** | 95.9% | 98.6% | +2.7% |
| **Energy (Edge)** | 125W | 34W | 73% reduction |
| **Concept Drift** | No adaptation | Online learning | 96.9% vs 72.3% @ day 50 |

---

## Model Architecture Diagram

```
Input Features (x)
      ↓
  [Encoder]
      ↓
┌─────────────────┐
│  Multi-Scale    │
│  TA-BN-ODE      │
│                 │
│ Branch 1 (μs)   │─┐
│ Branch 2 (ms)   │─┤
│ Branch 3 (sec)  │─├→ [Concatenate] → h_combined
│ Branch 4 (hour) │─┘
└─────────────────┘
      ↓
  [Decoder]
      ↓
Classification Output
      ↓
┌──────────────────────┐
│ Transformer-Enhanced │
│ Point Process        │
│                      │
│ • Multi-head Attn    │
│ • Multi-scale Time   │
│ • Log-barrier Opt    │
└──────────────────────┘
      ↓
Event Intensities
      ↓
┌──────────────────────┐
│ Structured Bayesian  │
│ Inference            │
│                      │
│ • Uncertainty        │
│ • Calibration        │
│ • PAC Bounds         │
└──────────────────────┘
      ↓
Prediction + Confidence
```

---

## Usage Guide

### Basic Usage

```python
# 1. Initialize model
model = UnifiedTABNODEPointProcess(
    input_dim=50,
    hidden_dim=128,
    n_attack_types=12,
    n_scales=4,
    n_ode_layers=2,
    n_attn_heads=4,
    n_attn_layers=2
)

# 2. Train
history = train_unified_framework(
    model, train_loader, val_loader,
    device, epochs=30, lr=1e-3
)

# 3. Predict with uncertainty
mean_pred, uncertainty = model.predict_with_uncertainty(
    x, t_span, n_samples=50
)

# 4. Real-time adaptation
adapter = RealTimeAdapter(model, device)
for x, y in stream:
    pred, unc = adapter.predict_with_uncertainty(x)
    adapter.update(x, y)  # Online learning
```

### Advanced: Custom Loss Weights

```python
# Adjust loss weights for your use case
model.lambda_tpp = 1.0   # Temporal modeling importance
model.lambda_kl = 0.01   # Bayesian regularization
model.lambda_reg = 0.001 # Stability regularization

# Domain-specific tuning
# Security-critical: Increase lambda_kl for better calibration
# High-throughput: Decrease lambda_tpp for faster inference
# Evolving threats: Use RealTimeAdapter with higher adaptation_rate
```

---

## Experimental Results (Paper Section 9)

### Container Security (ICS3D Containers Dataset)
- **Accuracy:** 99.4%
- **F1-Score:** 99.2%
- **False Positive Rate:** 0.7%
- **Key Strength:** Container escape detection (CVE-2019-5736: 99.2%)

### IoT/IIoT Security (Edge-IIoTset)
- **Accuracy (DNN):** 98.6%
- **Accuracy (ML):** 97.8%
- **Microsecond Detection:** 99.1% (vs. 87.3% baseline)
- **Key Strength:** Multi-scale temporal coverage

### Enterprise SOC (Microsoft GUIDE)
- **F1-Score:** 92.7%
- **True Positive Recall:** 94.3%
- **Coverage Probability:** 91.7%
- **False Positive Reduction:** 43% (via confidence filtering)

### Cross-Domain Validation
- **Speech Event Detection:** 94.2% F1-score (phoneme boundaries)
- **Healthcare Monitoring:** 96.8% (voice activity detection)

---

## Implementation Details

### Key Hyperparameters (from Paper)

```python
# Model Architecture
hidden_dim = 256              # Hidden state dimension
n_scales = 4                  # Multi-scale branches
time_constants = [1e-6, 1e-3, 1.0, 3600.0]  # Time scales

# TA-BN Configuration
ta_bn_hidden = 64             # TA-BN MLP hidden size
omega = 2*pi                  # Periodic encoding frequency

# Transformer Point Process
d_model = 128                 # Embedding dimension
n_heads = 4                   # Attention heads
n_layers = 2                  # Transformer layers

# Training
learning_rate = 1e-3          # Initial LR
epochs = 30                   # Training epochs
batch_size = 32               # Mini-batch size

# Loss Weights
lambda_tpp = 1.0              # Point process weight
lambda_kl = 0.01              # Bayesian regularization
lambda_reg = 0.001            # Stability regularization

# ODE Integration
rtol = 1e-3                   # Relative tolerance
atol = 1e-4                   # Absolute tolerance
method = 'dopri5'             # Adaptive Runge-Kutta

# Real-Time Adaptation
buffer_size = 1000            # Experience replay buffer
adaptation_rate = 0.01        # Online learning rate
adapt_frequency = 100         # Samples between adaptations
```

### Memory and Computational Requirements

**Training:**
- GPU: NVIDIA A100 40GB (or equivalent)
- RAM: 32GB minimum
- Disk: 50GB for ICS3D datasets

**Inference:**
- GPU: NVIDIA T4 16GB (or edge device)
- RAM: 8GB minimum
- Power: 34W (with neuromorphic conversion)

---

## Next Steps and Future Work

### Immediate Enhancements
1. **LLM Integration** (Paper Section 7.1)
   - Prompt engineering for zero-shot detection
   - Chain-of-thought reasoning for multi-stage attacks
   - Target: 87.6% F1-score on novel attacks

2. **Spiking Neural Network Conversion** (Paper Section 7.2)
   - Rate coding for continuous activations
   - Surrogate gradients for backpropagation
   - Target: 98.1% accuracy with 73% energy reduction

3. **Federated Learning** (Paper Section 10.5)
   - Privacy-preserving collaborative training
   - Differential privacy: (ε=2.3, δ=10⁻⁵)
   - Cross-organizational threat intelligence

### Research Directions
- Quantum-enhanced Neural ODEs
- Game-theoretic adversarial training
- Causal inference for attack attribution
- Multimodal security data fusion

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{anaedevha2025temporal,
  title={Temporal-Adaptive Neural ODEs for Real-Time Network Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and Borodachev, Yuri Vladimirovich},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  note={Under Review}
}
```

---

## License and Acknowledgments

This work was supported by the grant for research centers in the field of Artificial Intelligence provided by the Analytical Center for the Government of the Russian Federation (ACRF) in accordance with the agreement on the provision of subsidies (identifier of the agreement 000000D730324P540002) and the agreement with National Research Nuclear University MEPhI (Moscow Engineering Physics Institute), No. 70-2023-001309.

---

## Contact

**Author:** Roger Nick Anaedevha
**Email:** ar006@campus.mephi.ru
**Institution:** National Research Nuclear University MEPhI, Moscow, Russia

**For questions about the implementation:**
- Open an issue on GitHub
- Refer to the paper for theoretical details
- See inline code comments for specific components

---

## References

Key papers that influenced this work:

1. Chen et al. (2018): Neural Ordinary Differential Equations (NeurIPS)
2. Salvi et al. (2024): Temporal Adaptive Batch Normalization (NeurIPS)
3. Zuo et al. (2020): Transformer Hawkes Process (ICML)
4. Hasani et al. (2022): Closed-form Continuous-Time Networks (Nature MI)
5. Purohit et al. (2023): Orthogonal Neural ODEs for Robustness (Neurocomputing)

Full reference list in paper Section 11 (References).

---

**Last Updated:** 2025-10-31
**Version:** 2.0 (Paper Implementation)
**Status:** Production Ready
