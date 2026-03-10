# TA-BN-ODE-DSTPP

**Temporal Adaptive Neural Ordinary Differential Equations with Deep Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection**

Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev

*National Research Nuclear University MEPhI, Moscow*

Published in **Complex and Intelligent Systems** (Q1)

## Overview

This repository contains the complete reproducibility codebase for the TA-BN-ODE-DSTPP framework, which integrates:

1. **Temporal Adaptive Batch Normalization Neural ODEs (TA-BN-ODE)** — Continuous-depth neural networks with time-dependent normalization that captures diurnal traffic patterns across eight orders of magnitude (microseconds to hours)
2. **Deep Spatio-Temporal Point Processes (DSTPP)** — Transformer-based intensity functions with Hawkes cross-excitation for modeling discrete attack event dynamics
3. **Structured Variational Bayesian Inference** — Low-rank Gaussian posterior for calibrated uncertainty quantification (ECE = 0.017)
4. **Online Adaptation** — EWC-based concept drift detection and adaptation via Population Stability Index

## Key Results

| Dataset | Accuracy | F1 (weighted) | ECE |
|---------|----------|---------------|-----|
| Container Security | 99.4% | 99.3% | 0.017 |
| Edge-IIoTset | 98.6% | 98.5% | 0.019 |
| GUIDE SOC | 92.7% | 92.1% | 0.023 |
| CIC-IDS2018 | 97.8% | 97.6% | 0.015 |
| UNSW-NB15 | 96.3% | 96.0% | 0.021 |
| CIC-IoT-2023 | 98.2% | 98.0% | 0.018 |

- **Parameters:** 2.3M (82% reduction vs transformer baseline)
- **Throughput:** 12.3M events/sec | **P50 latency:** 8.2ms | **P99:** 22.9ms
- **Memory:** 9.2MB (vs 51.2MB transformer)

## Datasets

- **ICS3D (Integrated Cloud Security 3Datasets):** DOI [10.34740/kaggle/dsv/12483891](https://doi.org/10.34740/kaggle/dsv/12483891)
- **Standard Benchmarks:** DOI [10.34740/KAGGLE/DSV/12479689](https://doi.org/10.34740/KAGGLE/DSV/12479689)

Datasets are automatically downloaded via `kagglehub`. Configure Kaggle credentials:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```
Or place `kaggle.json` in `~/.kaggle/`.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: PyTorch >= 2.0, torchdiffeq, scikit-learn, kagglehub, pyro-ppl, transformers (for LLM integration).

## Repository Structure

```
TA-BN-ODE-DSTPP/
├── configs/
│   └── default.py              # All hyperparameters (Table S1 in supplementary)
├── models/
│   ├── ta_bn_ode.py            # TA-BN, multi-scale ODE (Eq. 1, 4, 5, 7)
│   ├── dstpp.py                # Transformer intensity + Hawkes (Eq. 8, 11)
│   ├── bayesian.py             # Low-rank variational posterior (Eq. 12)
│   └── full_model.py           # End-to-end model (Algorithm 1, Eq. 3)
├── data/
│   ├── loader.py               # ICS3D + benchmark dataset loaders
│   └── preprocessing.py        # z-score, temporal split, imputation
├── utils/
│   ├── training.py             # Trainer with 5-fold TS-CV, early stopping
│   ├── evaluation.py           # Metrics: accuracy, F1, ECE, throughput
│   ├── online.py               # EWC + PSI drift detection (Algorithm S2)
│   └── llm_integration.py      # LLM zero-shot analysis (Section 4.6)
├── scripts/
│   ├── run_experiment.py       # Main CLI experiment runner
│   └── run_ablation.py         # Ablation study (Table 5)
├── notebooks/
│   └── reproduce_results.ipynb # Interactive reproducibility notebook
├── requirements.txt
└── README.md
```

## Quick Start

### Full Experiment (All 6 Datasets)
```bash
python scripts/run_experiment.py
```

### Single Dataset
```bash
python scripts/run_experiment.py --dataset container
python scripts/run_experiment.py --dataset cic_ids2018
```

### Quick Smoke Test (50K samples, 10 epochs)
```bash
python scripts/run_experiment.py --quick
```

### 5-Fold Time-Series Cross-Validation
```bash
python scripts/run_experiment.py --dataset container --cross-validate
```

### Ablation Study
```bash
python scripts/run_experiment.py --ablation
```

### Concept Drift Experiment
```bash
python scripts/run_experiment.py --drift
```

### Interactive Notebook
```bash
jupyter notebook notebooks/reproduce_results.ipynb
```

## Reproducibility Checklist (Section S11)

- [x] Random seed: 42 (NumPy, PyTorch, CUDA)
- [x] Deterministic cuDNN (`torch.backends.cudnn.deterministic = True`)
- [x] All hyperparameters in `configs/default.py` match Supplementary Table S1
- [x] Temporal 70/15/15 split (no shuffling)
- [x] ODE solver: Dormand-Prince (dopri5), rtol=1e-3, atol=1e-4
- [x] Adjoint method for O(1) memory gradients
- [x] Adam optimizer, lr=1e-3 → 1e-5 (cosine annealing)
- [x] Gradient clipping: max norm 1.0
- [x] Early stopping: patience 10 epochs
- [x] Temperature calibration on validation set
- [x] Bayesian: 10 MC samples (train), 50 MC samples (test)
- [x] Online: EWC λ=1e-2, EMA ρ=0.02, R=18 mini-epochs

## Hardware

Paper experiments: NVIDIA P100 80GB GPU. The codebase automatically falls back to CPU if no GPU is available.

## Citation

If you use this code, please cite:

```bibtex
@article{anaedevha2026temporal,
  title={Temporal Adaptive Neural Ordinary Differential Equations with Deep
         Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich
          and Borodachev, Yuri Vladimirovich},
  journal={Complex and Intelligent Systems},
  year={2026}
}
```

## License

This code is provided for academic research and reproducibility purposes.
