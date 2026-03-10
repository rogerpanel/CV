"""
Default configuration for TA-BN-ODE-DSTPP model.

Reference: "Temporal Adaptive Neural Ordinary Differential Equations with
Deep Spatio-Temporal Point Processes for Real-Time Network Intrusion Detection"
Anaedevha, Trofimov, Borodachev (2026) -- Complex and Intelligent Systems

All hyperparameters match Section 5 and Supplementary Table S1.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    # TA-BN-ODE architecture (Section 4.1)
    hidden_dim: int = 256
    model_dim: int = 512
    n_ode_blocks: int = 2
    time_constants: Tuple[float, ...] = (1e-6, 1e-3, 1.0, 3600.0)
    activation: str = "elu"
    tabn_mlp_hidden: int = 64
    tabn_mlp_layers: int = 2

    # DSTPP transformer (Section 4.2)
    n_transformer_layers: int = 4
    n_attention_heads: int = 8
    transformer_dropout: float = 0.1

    # Bayesian inference (Section 4.3)
    low_rank_dim: int = 10  # r << d_b
    mc_samples_train: int = 10
    mc_samples_test: int = 50

    # ODE solver
    solver_method: str = "dopri5"
    solver_rtol: float = 1e-3
    solver_atol: float = 1e-4


@dataclass
class TrainingConfig:
    # Optimizer (Section 5.1)
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    batch_size: int = 256
    eval_batch_size: int = 1024
    max_epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0

    # Cross-validation
    n_folds: int = 5

    # Loss weights
    weight_cls: float = 1.0
    weight_tpp: float = 1.0
    weight_elbo: float = 1.0
    weight_reg: float = 1e-4


@dataclass
class OnlineConfig:
    # Concept drift detection (Section 4.5)
    psi_threshold: float = 0.2
    n_psi_bins: int = 10

    # EWC online adaptation (Algorithm S2)
    ewc_lambda: float = 1e-2
    ema_rho: float = 0.02
    online_lr: float = 1e-3
    lr_decay_rho: float = 0.02
    online_mini_epochs: int = 18

    # Differential privacy
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.0  # Set >0 to enable DP-SGD


@dataclass
class LLMConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 256


@dataclass
class DataConfig:
    # ICS3D datasets (DOI: 10.34740/kaggle/dsv/12483891)
    ics3d_kaggle_slug: str = "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d"
    # Standard benchmarks (DOI: 10.34740/KAGGLE/DSV/12479689)
    benchmarks_kaggle_slug: str = "rogernickanaedevha/integrated-cloud-security-3datasets-benchmarks"

    # Preprocessing
    temporal_split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    random_seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda"
    seed: int = 42
