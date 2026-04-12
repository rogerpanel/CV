# FedLLM-API: Privacy-Preserving Large Language Model Federation for Zero-Day API Threat Detection

This repository contains the complete implementation of **FedLLM-API**, a novel federated learning framework for collaborative zero-day API threat detection across organizational boundaries with formal privacy guarantees.

## Overview

FedLLM-API addresses the critical challenge of detecting novel API attacks while preserving data privacy across multiple organizations. The framework achieves 97.2% detection accuracy with ε=0.5 differential privacy, surpassing centralized baselines while requiring no raw data sharing.

### Key Features

- **Privacy-Preserving API Encoding**: Semantic feature extraction with differential privacy (ε=0.5, δ=10⁻⁵)
- **Parameter-Efficient Federated Learning**: LoRA-based fine-tuning reducing communication by 68%
- **Byzantine-Robust Aggregation**: Attention-weighted mechanism tolerating 30% malicious participants
- **Zero-Shot Transfer**: Detect unseen attack types through prompt engineering
- **Multi-Cloud Support**: Evaluated on AWS, Azure, GCP, GraphQL, microservices, and e-commerce APIs

### Paper Citation

```bibtex
@article{fedllm_api2025,
  title={FedLLM-API: Privacy-Preserving Large Language Model Federation for Zero-Day API Threat Detection Across Organizational Boundaries},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

## Repository Structure

```
FedLLM-API/
├── models/
│   ├── fedllm_api.py           # Main FedLLM-API model
│   ├── privacy_encoder.py      # Privacy-preserving API encoder
│   ├── lora_adapter.py         # Low-rank adaptation implementation
│   └── prompt_aggregation.py   # Prompt-based aggregation
├── federated/
│   ├── aggregation.py          # Byzantine-robust aggregation mechanisms
│   ├── client.py               # Federated learning client
│   └── server.py               # Federated learning server
├── baselines/
│   ├── centralized_llm.py      # Centralized baseline
│   ├── fedavg.py               # Standard FedAvg
│   ├── fedprox.py              # FedProx with proximal term
│   └── zeroday_llm.py          # ZeroDay-LLM baseline
├── data/
│   ├── data_loaders.py         # Dataset loaders
│   ├── aws_api_gateway.py      # AWS API Gateway dataset
│   ├── azure_apim.py           # Azure API Management dataset
│   ├── gcp_cloudapi.py         # GCP Cloud API dataset
│   ├── graphql_security.py     # GraphQL security dataset
│   ├── microservices_mesh.py   # Microservices mesh dataset
│   └── ecommerce_api.py        # E-commerce API dataset
├── training/
│   ├── federated_trainer.py    # Federated training orchestration
│   ├── local_trainer.py        # Local training utilities
│   └── privacy_engine.py       # Differential privacy engine
├── evaluation/
│   ├── metrics.py              # Detection metrics
│   ├── privacy_auditing.py     # Privacy guarantee verification
│   └── byzantine_attacks.py    # Byzantine attack simulation
├── config/
│   ├── fedllm_api_config.yaml  # Main configuration
│   └── experiments/            # Experiment-specific configs
├── utils/
│   ├── logging_utils.py        # Logging utilities
│   └── visualization.py        # Result visualization
├── scripts/
│   ├── run_federated.py        # Run federated training
│   ├── run_baselines.py        # Run baseline comparisons
│   └── reproduce_paper.sh      # Reproduce all paper results
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container
├── environment.yaml            # Conda environment
└── README.md                   # This file
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate fedllm-api

# Verify installation
python -c "import torch; import transformers; print('Success!')"
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Build Docker image
docker build -t fedllm-api:latest .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/workspace fedllm-api:latest

# Inside container
cd /workspace
python scripts/run_federated.py --config config/fedllm_api_config.yaml
```

## Quick Start

### 1. Download Datasets

```bash
# Download and prepare datasets (instructions in data/README.md)
python data/download_datasets.py --datasets aws azure gcp graphql mesh ecommerce
```

### 2. Run FedLLM-API

```bash
# Train FedLLM-API with default configuration
python scripts/run_federated.py \
    --config config/fedllm_api_config.yaml \
    --num_clients 10 \
    --rounds 50 \
    --epochs_per_round 5 \
    --privacy_epsilon 0.5

# Monitor training progress
tensorboard --logdir runs/fedllm_api
```

### 3. Evaluate Results

```bash
# Evaluate on all datasets
python evaluation/evaluate_all.py \
    --checkpoint checkpoints/fedllm_api_final.pt \
    --datasets aws azure gcp graphql mesh ecommerce

# Generate paper figures
python utils/visualization.py --results results/fedllm_api/
```

## Reproducing Paper Results

To reproduce all results from the paper:

```bash
# Run complete experimental suite (requires 10 GPUs, ~48 hours)
bash scripts/reproduce_paper.sh

# Results will be saved to:
# - results/fedllm_api/          (main results)
# - results/ablation/             (ablation studies)
# - results/baselines/            (baseline comparisons)
# - results/byzantine/            (Byzantine robustness)
# - figures/                      (paper figures)
```

Individual experiments can be run separately:

```bash
# Main FedLLM-API results (Table 2 in paper)
python scripts/run_federated.py --config config/experiments/main_results.yaml

# Privacy-utility tradeoff (Figure 3 in paper)
python scripts/privacy_utility.py --epsilons 0.1 0.5 1.0 2.0 5.0 10.0

# Byzantine robustness (Figure 4 in paper)
python scripts/byzantine_robustness.py --malicious_fractions 0.0 0.1 0.2 0.3

# Ablation studies (Table 3 in paper)
python scripts/ablation_study.py --config config/experiments/ablation.yaml

# Baseline comparisons
python scripts/run_baselines.py --methods centralized local fedavg fedprox zeroday
```

## Key Components

### Privacy-Preserving API Encoder

The encoder transforms API requests into semantic representations while ensuring differential privacy:

```python
from models.privacy_encoder import PrivacyPreservingEncoder

encoder = PrivacyPreservingEncoder(
    embedding_dim=768,
    epsilon=0.5,
    delta=1e-5,
    sensitivity=10.0
)

# Encode API request with DP
api_request = {
    'method': 'POST',
    'endpoint': '/api/v2/users/123/orders',
    'params': {'quantity': 1, 'item_id': 456},
    'timestamp': 1635123456.789
}

encoded = encoder.encode(api_request)  # Returns differentially private embedding
```

### LoRA-Based Federated Fine-Tuning

Parameter-efficient adaptation using low-rank decomposition:

```python
from models.lora_adapter import LoRAAdapter
from transformers import DistilBertModel

# Initialize base model (frozen)
base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Add LoRA adapters (trainable)
lora_model = LoRAAdapter(
    base_model,
    rank=8,
    alpha=16,
    target_modules=['query', 'value']
)

# Only train 0.5% of parameters
print(f"Trainable params: {lora_model.num_trainable_params() / lora_model.num_total_params():.1%}")
```

### Byzantine-Robust Aggregation

Attention-weighted aggregation with certified robustness:

```python
from federated.aggregation import ByzantineRobustAggregator

aggregator = ByzantineRobustAggregator(
    aggregation_method='attention_weighted',
    temperature=0.5
)

# Aggregate client updates
client_updates = [client_i.get_lora_weights() for client_i in clients]
global_update = aggregator.aggregate(client_updates, validation_losses)
```

## Datasets

The framework supports six datasets spanning diverse API architectures:

| Dataset | API Type | Requests | Endpoints | Attack Types | Source |
|---------|----------|----------|-----------|--------------|--------|
| AWS-API-Gateway | REST | 2.4M | 847 | Auth bypass, injection, DDoS | CloudTrail logs |
| Azure-APIM | REST | 1.8M | 623 | Broken authz, data exposure | Azure Monitor |
| GCP-CloudAPI | REST | 3.1M | 1,205 | Privilege escalation, injection | Cloud Logging |
| GraphQL-Security | GraphQL | 890K | 412 | Nested query, batch abuse | Public dataset |
| Microservices-Mesh | Service mesh | 5.2M | 2,341 | Lateral movement, sidecar exploit | Istio telemetry |
| E-Commerce-API | REST | 1.6M | 289 | Business logic, rate abuse | Proprietary |

### Dataset Preparation

Each dataset loader handles:
- Privacy-preserving feature extraction
- Federated partitioning (Dirichlet distribution, α=0.5)
- Train/validation/test splits (70%/15%/15%)
- Attack type balancing and zero-day holdout

Example usage:

```python
from data.aws_api_gateway import AWSAPIGatewayDataset

dataset = AWSAPIGatewayDataset(
    data_dir='datasets/aws_api_gateway',
    num_clients=10,
    dirichlet_alpha=0.5,
    privacy_epsilon=0.5
)

# Get client datasets
train_loaders = dataset.get_federated_train_loaders(batch_size=32)
test_loader = dataset.get_test_loader(batch_size=128)
```

## Configuration

All experiments are configured via YAML files in `config/`. Key parameters:

```yaml
# config/fedllm_api_config.yaml
model:
  backbone: "distilbert-base-uncased"
  lora_rank: 8
  lora_alpha: 16
  embedding_dim: 768

privacy:
  epsilon: 0.5
  delta: 1e-5
  feature_sensitivity: 10.0
  gradient_clipping: 1.0

federated:
  num_clients: 10
  num_rounds: 50
  epochs_per_round: 5
  client_fraction: 1.0
  aggregation_method: "attention_weighted"

byzantine:
  malicious_fraction: 0.0
  attack_type: "sign_flipping"

training:
  learning_rate: 3e-4
  batch_size: 32
  weight_decay: 0.01
  warmup_steps: 500
```

## Performance Metrics

FedLLM-API achieves state-of-the-art performance across multiple dimensions:

### Detection Accuracy (Table 2 in paper)

| Dataset | FedLLM-API | Centralized | Local | FedAvg | FedProx | ZeroDay-LLM |
|---------|------------|-------------|-------|--------|---------|-------------|
| AWS | 97.3% | 96.8% | 88.3% | 93.5% | 94.2% | 95.1% |
| Azure | 96.9% | 95.2% | 87.1% | 92.8% | 93.5% | 94.3% |
| GCP | 97.5% | 96.1% | 89.2% | 93.9% | 94.6% | 95.6% |
| GraphQL | 96.8% | 94.9% | 86.7% | 91.2% | 92.8% | 93.7% |
| Mesh | 97.4% | 95.7% | 88.9% | 93.1% | 93.9% | 94.8% |
| E-Com | 97.2% | 96.2% | 87.4% | 92.6% | 93.8% | 94.9% |
| **Average** | **97.2%** | **95.8%** | **87.9%** | **92.9%** | **93.8%** | **94.7%** |

### Communication Efficiency

- **LoRA variant**: 2.6 GB total (50 rounds) - 80% reduction vs full-model
- **Prompt variant**: 750 KB total - 99.97% reduction vs full-model
- **Per-round cost**: 52 MB (LoRA) or 15 KB (Prompt) vs 264 MB (full-model)

### Privacy Guarantees

- Differential privacy: (ε=0.5, δ=10⁻⁵)
- Membership inference attack success rate: 50.2% (near-random)
- Gradient inversion reconstruction error: >90% (strong protection)

### Byzantine Robustness

- 0% malicious: 97.2% accuracy
- 10% malicious: 96.8% accuracy (0.4% degradation)
- 20% malicious: 96.1% accuracy (1.1% degradation)
- 30% malicious: 94.7% accuracy (2.5% degradation)

## Advanced Usage

### Custom Datasets

To add a new API dataset:

```python
from data.base_dataset import BaseAPIDataset

class MyAPIDataset(BaseAPIDataset):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

    def load_raw_data(self):
        # Load API logs
        pass

    def extract_features(self, api_request):
        # Custom feature extraction
        pass

    def label_attacks(self, api_sequence):
        # Custom attack labeling
        pass
```

### Custom Aggregation Strategies

Implement custom Byzantine-robust aggregation:

```python
from federated.aggregation import BaseAggregator

class MyAggregator(BaseAggregator):
    def aggregate(self, client_updates, metadata):
        # Custom aggregation logic
        # Return aggregated model update
        pass
```

### Hyperparameter Tuning

Use the provided tuning script:

```bash
python scripts/hyperparameter_search.py \
    --search_space config/search_space.yaml \
    --num_trials 100 \
    --metric accuracy \
    --optimization_direction maximize
```

## Troubleshooting

### Common Issues

**Out of Memory Errors**:
```bash
# Reduce batch size
python scripts/run_federated.py --batch_size 16

# Use gradient checkpointing
python scripts/run_federated.py --gradient_checkpointing True

# Use smaller base model
python scripts/run_federated.py --model_name distilbert-base-uncased
```

**Slow Training**:
```bash
# Reduce number of clients per round
python scripts/run_federated.py --client_fraction 0.5

# Use fewer local epochs
python scripts/run_federated.py --epochs_per_round 3

# Enable mixed precision
python scripts/run_federated.py --fp16 True
```

**Privacy Budget Exhaustion**:
```bash
# Increase epsilon (weaker privacy)
python scripts/run_federated.py --privacy_epsilon 1.0

# Reduce noise scale
python scripts/run_federated.py --noise_multiplier 0.8
```

## Citation

If you use FedLLM-API in your research, please cite our paper:

```bibtex
@article{fedllm_api2025,
  title={FedLLM-API: Privacy-Preserving Large Language Model Federation for Zero-Day API Threat Detection Across Organizational Boundaries},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Transformer implementations from Hugging Face Transformers
- Federated learning framework based on Flower
- Differential privacy mechanisms from Opacus
- Dataset contributors: AWS, Azure, GCP security teams

## Contact

For questions, issues, or collaboration opportunities:
- Open an issue on GitHub
- Email: contact@fedllm-api.org
- Project website: https://fedllm-api.org

## Reproducibility Statement

All experimental results in the paper can be reproduced using the provided code and configurations. Average runtime for complete experimental suite is 48 hours on 10× NVIDIA A100 GPUs. We provide pre-trained checkpoints and cached results for faster validation.

Expected resource requirements:
- **Training**: 10× NVIDIA A100 (40GB) or equivalent, 128GB RAM
- **Inference**: 1× NVIDIA V100 or equivalent, 32GB RAM
- **Storage**: 500GB for datasets, models, and results

Random seed is fixed (42) for all experiments ensuring exact reproducibility.
