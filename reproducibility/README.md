# TA-BN-ODE + DSTPP: Reproducibility Package

**Paper:** *Temporal Adaptive Neural Ordinary Differential Equations with Deep Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection*

**Journal:** Complex and Intelligent Systems (Q1)

**Authors:** Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev

**Affiliation:** National Research Nuclear University MEPhI (Moscow Engineering Physics Institute)

---

## Overview

This repository provides the complete implementation for reproducing all results in the paper. The framework integrates five key components:

| Component | Module | Paper Section |
|-----------|--------|---------------|
| **TA-BN-ODE** — Temporal Adaptive Batch Normalization Neural ODEs | `src/models/tabn_ode.py` | Section IV |
| **DSTPP** — Deep Spatio-Temporal Point Processes with log-barrier | `src/models/point_process.py` | Section V |
| **Bayesian Inference** — Structured variational with PAC-Bayes bounds | `src/models/bayesian.py` | Section VI |
| **LLM Integration** — Zero-shot detection via temporal prompting | `src/models/llm_integration.py` | Section VII |
| **Online Adaptation** — EWC + PSI drift detection | `src/training/online_adapter.py` | Section VIII-F |

## Datasets

### ICS3D — Integrated Cloud Security 3Datasets (18.9M records)
- **DOI:** [10.34740/kaggle/dsv/12483891](https://doi.org/10.34740/kaggle/dsv/12483891)
- Container Security (697,289 flows from Kubernetes clusters)
- Edge-IIoTset (4M records from 7-layer IoT testbed)
- GUIDE SOC (1M incidents from 6,100 organisations, MITRE ATT&CK)

### Standard Benchmarks
- **DOI:** [10.34740/KAGGLE/DSV/12479689](https://doi.org/10.34740/KAGGLE/DSV/12479689)
- CIC-IDS2018 (16.2M records)
- UNSW-NB15 (257,673 records)
- CIC-IoT-2023

Datasets are downloaded automatically via `kagglehub` on first run.

## Installation

```bash
# Option 1: pip
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml
conda activate tabn-ode
```

**Required:** PyTorch >= 2.0, torchdiffeq >= 0.2.3

## Quick Start

```bash
# Train on Container Security dataset
python scripts/train.py --dataset containers --epochs 100

# Train on Edge-IIoT
python scripts/train.py --dataset edge_iiot --epochs 100

# Run standard benchmark evaluation (Table IV)
python scripts/run_benchmarks.py

# Evaluate a saved checkpoint
python scripts/evaluate.py --dataset containers \
    --checkpoint checkpoints/containers/best_model.pt
```

For a quick test with limited data:
```bash
python scripts/train.py --dataset containers --max_samples 5000 --epochs 10
```

## Project Structure

```
reproducibility/
├── README.md
├── requirements.txt
├── environment.yml
├── config/
│   └── hyperparameters.yaml       # Table S1 (full hyperparameters)
├── src/
│   ├── models/
│   │   ├── tabn_ode.py            # TA-BN-ODE architecture (Sec. IV)
│   │   ├── point_process.py       # DSTPP + log-barrier (Sec. V)
│   │   ├── bayesian.py            # Structured VI + calibration (Sec. VI)
│   │   ├── llm_integration.py     # LLM zero-shot detection (Sec. VII)
│   │   └── framework.py           # Complete integrated model (Fig. 2)
│   ├── data/
│   │   ├── loader.py              # ICS3D + benchmark loading
│   │   └── preprocessing.py       # Feature engineering (Sec. VIII-C)
│   ├── training/
│   │   ├── trainer.py             # Training loop + early stopping
│   │   └── online_adapter.py      # EWC + PSI adaptation (Alg. S2)
│   └── evaluation/
│       ├── metrics.py             # Detection, calibration, throughput
│       └── visualization.py       # Publication-quality figures
├── scripts/
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation script
│   └── run_benchmarks.py          # Standard benchmark evaluation
└── notebooks/
    └── neural-ode-model-v2.ipynb  # Interactive notebook
```

## Key Hyperparameters (Table S1)

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 256 |
| ODE blocks | 2 |
| Time constants | {10⁻⁶, 10⁻³, 1, 3600} s |
| Activation | ELU |
| Transformer layers/heads | 4 / 8 (d_model=512) |
| ODE solver | Dopri5, rtol=10⁻³, atol=10⁻⁴ |
| Optimiser | Adam, LR=10⁻³→10⁻⁵ (cosine) |
| Batch size | 256 (train), 1024 (eval) |
| MC samples | 10 (train), 50 (test) |
| Calibration | Temperature scaling on val set |
| Online learning | EMA ρ=0.98, EWC η=5×10⁻³ |

## Main Results

### ICS3D (Table III)
| Dataset | Accuracy | F1 |
|---------|----------|-----|
| Containers | 99.4% ± 0.2 | 99.2% |
| Edge-IIoT | 98.6% ± 0.3 | 98.3% |
| GUIDE (SOC) | — | 92.7% ± 0.8 |

### Standard Benchmarks (Table IV)
| Dataset | Accuracy |
|---------|----------|
| CIC-IDS2018 | 97.8% |
| UNSW-NB15 | 96.3% |
| CIC-IoT-2023 | 98.2% |

### Model Efficiency
- **Parameters:** 2.3M (82% reduction vs. transformer baseline)
- **Throughput:** 12.3M events/sec
- **P50 Latency:** 8.2ms
- **ECE:** 0.017

## Citation

If you use this code, please cite:

```bibtex
@article{anaedevha2026tabnodepp,
  title={Temporal Adaptive Neural Ordinary Differential Equations with Deep
         Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and
          Borodachev, Yuri Vladimirovich},
  journal={Complex and Intelligent Systems},
  year={2026},
  publisher={Springer}
}
```

## License

This code is provided for academic reproducibility purposes.
