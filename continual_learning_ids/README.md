# Continual Learning and Constrained Reinforcement Learning for Adversarially Robust Network Intrusion Detection and Autonomous Response

**IEEE Transactions on Neural Networks and Learning Systems, 2026**

Roger Nick Anaedevha, Alexander Gennadievich Trofimov, and Yuri Vladimirovich Borodachev

*Institute of Cyber-Intelligent Systems (ICIS) & Department of Applied Mathematics, National Research Nuclear University MEPhI, Moscow, Russian Federation*

---

## Abstract

Machine-learning-based network intrusion detection systems achieve high accuracy under static conditions, yet two fundamental limitations impede their operational deployment: (1) fixed-weight classifiers degrade under concept drift, risking catastrophic forgetting of previously learned attack signatures; (2) detection without automated response leaves a critical gap between identification and mitigation. This repository provides the complete reproducibility package for our unified framework that addresses both limitations simultaneously through:

- **Continual Learning**: EWC + Experience Replay for incremental adaptation across sequential traffic distributions, bounding accuracy regression to <2 percentage points on any previously learned attack class
- **Constrained RL Response**: CPO-based agent within a CMDP framework that selects graduated response actions while provably bounding false-positive blocking rate below 0.1%
- **Unified Fisher Information**: Shared FIM computation serving both knowledge preservation (EWC) and trust-region constraint satisfaction (CPO)

**Key Results**: 96.9% average accuracy, BWT = -0.014, 97.2% threat mitigation, zero constraint violations over 5,000 episodes across 84.2M flow records and 83 unique attack classes.

## Architecture

```
Live Traffic ──> SurrogateIDS (7-branch + MC Dropout) ──> RL Response Agent (CPO/CMDP) ──> Mitigation Actions
     │                    │                                        │
     │                    ├── predictions + uncertainty ────────────┤
     │                    │                                        │
     ▼                    ▼                                        │
Labelled Data ──> Continual Learning ──> Shared Fisher ◄──────────┘
                  (EWC + Replay)         Information Matrix
                       │
                  Drift Detector
                  (KL-divergence)
```

## Benchmark Datasets

**6 datasets | 84.2M total records | 83 unique attack classes**

### General Network Traffic
| Dataset | Flows | Features | Attack Classes | Benign % | Source |
|---------|-------|----------|----------------|----------|--------|
| CIC-IoT-2023 | 1,621,834 | 46 | 33 | 12.4% | CIC, UNB |
| CSE-CIC-IDS2018 | 16,232,943 | 79 | 14 | 83.1% | CIC/CSE, AWS |
| UNSW-NB15 | 2,540,044 | 49 | 9 | 55.0% | UNSW Canberra |

**Download**: https://doi.org/10.34740/kaggle/dsv/12483891

### Cloud, Microservices & Edge
| Dataset | Flows | Features | Attack Classes | Benign % | Source |
|---------|-------|----------|----------------|----------|--------|
| CloudSec-2024 | 38,412,201 | 62 | 11 | 71.3% | Kaggle |
| ContainerIDS | 18,764,592 | 54 | 8 | 68.7% | Kaggle |
| EdgeIIoT-2024 | 6,628,386 | 61 | 8 | 42.1% | Kaggle |

**Download**: https://doi.org/10.34740/KAGGLE/DSV/12479689

## Repository Structure

```
continual_learning_ids/
├── configs/
│   └── default.yaml              # Full hyperparameter configuration
├── data/
│   └── dataset_loader.py         # Dataset loading, preprocessing, sequential splits
├── models/
│   ├── surrogate_ids.py          # 7-branch SurrogateIDS with MC Dropout
│   ├── policy_network.py         # Policy, Value, and Cost-Value networks
│   └── unified_fim.py            # Unified Fisher Information Framework
├── environments/
│   └── nids_env.py               # CMDP environment for response training
├── training/
│   ├── continual_learner.py      # EWC + Experience Replay pipeline
│   ├── cpo_trainer.py            # Constrained Policy Optimisation
│   └── drift_detector.py         # KL-divergence drift detection
├── evaluation/
│   ├── metrics.py                # AA, BWT, FWT, RL metrics
│   └── adversarial.py            # 6 attack methods (FGSM, PGD, C&W, etc.)
├── utils/
│   ├── replay_buffer.py          # Reservoir-sampled replay buffer
│   ├── fisher.py                 # Fisher diagonal computation
│   └── logging_utils.py          # Logging setup
├── scripts/
│   ├── train_continual.py        # Train CL pipeline (Table II)
│   ├── train_cpo.py              # Train CPO agent (Table III)
│   └── run_full_experiment.py    # Reproduce all paper results
├── tests/
│   └── test_components.py        # Comprehensive unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/rogerpanel/CV.git
cd CV/continual_learning_ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

**Requirements**: Python >= 3.9, PyTorch >= 2.0, CUDA (optional, for GPU acceleration)

## Quick Start

### 1. Download Datasets

Download from the Kaggle DOIs above and organise as:
```
data/
├── cic_iot_2023/
│   └── *.csv
├── cse_cic_ids2018/
│   └── *.csv
├── unsw_nb15/
│   └── *.csv
├── cloudsec_2024/
│   └── *.csv
├── container_ids/
│   └── *.csv
└── edge_iiot_2024/
    └── *.csv
```

### 2. Train Continual Learning Pipeline

```bash
# Single dataset (Table II)
python -m continual_learning_ids.scripts.train_continual \
    --data_dir ./data \
    --dataset cic_iot_2023 \
    --output_dir ./results \
    --eval_adversarial

# Quick test with subsampling
python -m continual_learning_ids.scripts.train_continual \
    --data_dir ./data \
    --dataset cic_iot_2023 \
    --subsample 0.01 \
    --output_dir ./results
```

### 3. Train CPO Response Agent

```bash
python -m continual_learning_ids.scripts.train_cpo \
    --data_dir ./data \
    --dataset cic_iot_2023 \
    --detection_model ./results/cic_iot_2023/checkpoints/task_5.pt \
    --output_dir ./results/rl \
    --num_iterations 1000
```

### 4. Full Experiment Pipeline

```bash
# Reproduce all paper results (Tables II-VI)
python -m continual_learning_ids.scripts.run_full_experiment \
    --data_dir ./data \
    --output_dir ./results

# Quick test
python -m continual_learning_ids.scripts.run_full_experiment \
    --data_dir ./data \
    --output_dir ./results \
    --subsample 0.01
```

### 5. Run Tests

```bash
pytest continual_learning_ids/tests/ -v
```

## Hyperparameters

All hyperparameters are specified in `configs/default.yaml`. Key values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ewc.lambda` | 5,000 | EWC consolidation strength |
| `ewc.fisher_decay` | 0.5 | FIM exponential moving average α |
| `replay.buffer_size` | 5,000 | Reservoir replay buffer capacity M |
| `rl.trust_region_delta` | 0.01 | CPO trust region radius δ |
| `rl.discount_gamma` | 0.99 | Discount factor γ |
| `rl.gae_lambda` | 0.97 | GAE parameter λ |
| `rl.epsilon_fp` | 0.001 | FP blocking rate constraint (<0.1%) |
| `unified_fim.beta` | 0.7 | Detection vs. policy FIM balance |
| `drift.tau_1` | 0.05 | Stable/monitor threshold |
| `drift.tau_2` | 0.15 | Monitor/drift threshold |

## Results Summary

### Continual Learning (Table II)

| Method | AA (%) | BWT | FWT |
|--------|--------|-----|-----|
| Full Retrain | 97.8 | 0.000 | --- |
| Fine-Tuning | 84.9 | -0.146 | +0.047 |
| EWC Only | 93.7 | -0.042 | +0.029 |
| GEM | 95.2 | -0.027 | +0.016 |
| **EWC+Replay (Ours)** | **96.9** | **-0.014** | **+0.032** |

### RL Response Agent (Table III)

| Metric | Rule | PPO | Lag. PPO | **CPO (Ours)** |
|--------|------|-----|----------|----------------|
| Mitigation (%) | 89.3 | 98.1 | 96.8 | **97.2** |
| FP Block (%) | 0.12 | 0.87 | 0.09 | **0.04** |
| MTTR (ms) | 340 | 85 | 112 | **92** |
| Violations | 14 | 127 | 3 | **0** |

### Integrated Performance (Table IV)

| Scenario | Det. Acc. | Mitigation | FP Block | Violations |
|----------|-----------|------------|----------|------------|
| Static | 97.8% | 97.4% | 0.03% | 0 |
| Drift | 96.2% | 96.1% | 0.06% | 0 |
| Adversarial | 94.6% | 93.8% | 0.08% | 0 |

## Equations Reference

**EWC Loss** (Eq. 7):
```
L_total = L_CE(D_hat; θ) + (λ/2) Σ_k F_hat_k (θ_k - θ*_k)^2
```

**CMDP Objective** (Eq. 9-10):
```
max_π  J_R(π) = E_π[Σ γ^t R(s_t, a_t)]
s.t.   J_C(π) = E_π[Σ γ^t C(s_t, a_t)] ≤ ε_fp
```

**Unified FIM** (Eq. 12):
```
F_hat_k^unified = β · F_hat_k^det + (1-β) · [F_π]_kk
```

## Related Publications

1. R. N. Anaedevha, A. G. Trofimov, and Y. V. Borodachev, "Stochastic multimodal transformer with uncertainty quantification for robust network intrusion detection," in *Proc. Int. Conf. Neuroinformatics*, Springer, 2025, pp. 428-447.

2. R. N. Anaedevha, A. G. Trofimov, and Y. V. Borodachev, "MambaShield: Selective state-space models for poisoning-resilient sequential modelling," *Expert Systems with Applications*, p. 131175, 2026.

3. R. N. Anaedevha, A. G. Trofimov, and Y. V. Borodachev, "Uncertainty-calibrated hierarchical Gaussian processes for intrusion detection with multi-scale temporal modeling," *Neurocomputing*, p. 133105, 2026.

## Production Deployment

This framework is deployed on the **RobustIDPS.ai** production platform for live network intrusion detection and autonomous response. See the [main repository](https://github.com/rogerpanel/CV) for deployment infrastructure.

## Citation

```bibtex
@article{anaedevha2026cl_rl,
  author    = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich and Borodachev, Yuri Vladimirovich},
  title     = {Continual Learning and Constrained Reinforcement Learning for Adversarially Robust Network Intrusion Detection and Autonomous Response},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2026},
  note      = {Submitted}
}
```

## License

This project is released under the MIT License. See [LICENSE](../LICENSE) for details.

## Contact

- Roger Nick Anaedevha - roger@robustidps.ai
- Project: https://robustidps.ai
- Repository: https://github.com/rogerpanel/CV/tree/main/continual_learning_ids
