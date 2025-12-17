# TripleE-TGNN: Triple-Embedding Temporal Graph Neural Networks for Multi-Granularity Microservices Security

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/pyg-2.3.1-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation** of the paper:

> **TripleE-TGNN: Triple-Embedding Temporal Graph Neural Networks for Multi-Granularity Microservices Security**
> Roger Nick Anaedevha, Alexander Gennadevich Trofimov, Yuri Vladimirovich Borodachev
> *Submitted to IEEE Transactions on Information Forensics and Security*

## Overview

TripleE-TGNN is a novel intrusion detection system for microservices architectures that jointly learns hierarchical representations at three complementary granularities:

1. **Service-Level**: Aggregate behavioral patterns through temporal analysis of service mesh telemetry
2. **Trace-Level**: Distributed request flows across service dependencies using attention-weighted path encodings
3. **Node-Level**: Individual pod dynamics and resource consumption patterns

By integrating these multi-granularity embeddings through a temporal heterogeneous graph neural network, TripleE-TGNN achieves **96.8% detection accuracy** on complex microservices applications, outperforming single-granularity baselines by **8.3%** and temporal graph attention networks by **5.2%**.

### Key Results

| Dataset | Services | Accuracy | Precision | Recall | F1-Score |
|---------|----------|----------|-----------|--------|----------|
| **Train-Ticket** | 41 | 96.8% | 96.4% | 97.2% | 96.8% |
| **Sock-Shop** | 14 | 95.3% | 94.9% | 95.7% | 95.3% |
| **Online-Boutique** | 11 | 94.7% | 94.2% | 95.1% | 94.6% |
| **DeathStarBench-Social** | 27 | 95.1% | 94.8% | 95.4% | 95.1% |
| **DeathStarBench-Hotel** | 17 | 94.5% | 94.1% | 94.9% | 94.5% |

**Robustness under Concept Drift**: 94.7% accuracy maintained during topology changes

## Features

### Triple-Embedding Architecture

- **Service-Level Encoder**: Graph Attention Network + GRU for aggregate service behaviors
- **Trace-Level Encoder**: Time-aware path encoding for distributed request flows
- **Node-Level Encoder**: Temporal Graph Conv + LSTM for pod-level dynamics
- **Heterogeneous Temporal GNN**: Cross-granularity integration with dual attention
- **Adaptive Fusion**: Granularity-level attention for dynamic weighting

### Multi-Granularity Attack Detection

| Attack Type | Primary Granularity | Accuracy |
|-------------|---------------------|----------|
| DDoS | Service | 98.2% |
| Resource Exhaustion | Service | 97.9% |
| Privilege Escalation | Trace | 97.4% |
| API Abuse | Trace | 95.8% |
| Authentication Bypass | Trace | 96.3% |
| Container Breakout | Node | 94.7% |
| Crypto Mining | Node | 95.3% |
| Lateral Movement | Node | 96.1% |

### Comprehensive Datasets

Five realistic microservices datasets with diverse attack scenarios:

1. **Train-Ticket** (41 services): E-commerce train booking with 2.4M requests, 12 attack types
2. **Sock-Shop** (14 services): E-commerce demo with 840K requests, 8 attack scenarios
3. **Online-Boutique** (11 services): Google Cloud native demo with 1.6M requests
4. **DeathStarBench** (13-27 services): Social network, hotel reservation, media services
5. **UNSW-NB15/NSL-KDD** (adapted): Traditional datasets mapped to microservices topologies

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/TripleE-TGNN.git
cd TripleE-TGNN

# Create conda environment
conda env create -f environment.yaml
conda activate triplee-tgnn

# Verify PyTorch and PyG
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### Option 2: pip Installation

```bash
# Create virtual environment
python3.10 -m venv triplee-env
source triplee-env/bin/activate  # On Windows: triplee-env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.3.1
```

### Option 3: Docker Container

```bash
# Build image
docker build -t triplee-tgnn:latest .

# Run with GPU support
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -p 6006:6006 \
  triplee-tgnn:latest

# Inside container
cd /workspace/TripleE-TGNN
python training/train_triplee.py --config config/triplee_config.yaml
```

## Quick Start

### 1. Download Datasets

```bash
# Train-Ticket (41 services, 2.4M requests)
python data/download_train_ticket.py --output datasets/train_ticket/

# Sock-Shop (14 services, 840K requests)
python data/download_sock_shop.py --output datasets/sock_shop/

# Online-Boutique (11 services, 1.6M requests)
python data/download_boutique.py --output datasets/boutique/

# Verify datasets
python data/verify_datasets.py
```

### 2. Train TripleE-TGNN

```bash
# Train on Train-Ticket dataset
python training/train_triplee.py \
  --config config/triplee_config.yaml \
  --dataset train_ticket \
  --output checkpoints/triplee_trainticket/

# Monitor with TensorBoard
tensorboard --logdir runs/triplee_trainticket/ --port 6006
```

**Expected output:**
```
Epoch 1/50: Loss=0.312, Acc=89.4%, Service=90.1%, Trace=91.2%, Node=87.8%
Epoch 10/50: Loss=0.098, Acc=94.8%, Service=95.2%, Trace=95.9%, Node=93.7%
Epoch 50/50: Loss=0.041, Acc=96.8%, Service=97.1%, Trace=97.4%, Node=95.9%
Granularity Weights: Service=0.35, Trace=0.42, Node=0.23
✓ Best model saved to checkpoints/triplee_trainticket/best_model.pth
```

### 3. Evaluate on Test Set

```bash
# Evaluate on benign test data
python evaluation/evaluate_triplee.py \
  --checkpoint checkpoints/triplee_trainticket/best_model.pth \
  --dataset train_ticket \
  --split test

# Evaluate by attack type
python evaluation/evaluate_by_attack.py \
  --checkpoint checkpoints/triplee_trainticket/best_model.pth \
  --output results/attack_breakdown.json
```

### 4. Real-Time Detection

```bash
# Deploy for real-time detection (requires Kubernetes cluster)
python inference/realtime_detector.py \
  --checkpoint checkpoints/triplee_trainticket/best_model.pth \
  --k8s-config ~/.kube/config \
  --namespace production \
  --alert-threshold 0.7

# Process offline trace data
python inference/process_traces.py \
  --checkpoint checkpoints/triplee_trainticket/best_model.pth \
  --traces jaeger_traces.json \
  --output detections.json
```

## Architecture Details

### Service-Level Embedding

```python
Service-Level Encoder:
  Input: Service dependency graph + temporal features

  Graph Attention Network:
    Layers: 3
    Hidden dimensions: [128, 256, 512]
    Attention heads: 8

  Temporal Encoder:
    Type: GRU
    Hidden size: 256
    Layers: 2

  Features (per service):
    - Request rate, error rate, latency percentiles
    - CPU/memory utilization
    - Dependency fanout/fanin
    - Service mesh metrics (mTLS, retries, timeouts)

  Output: 512-dimensional service embeddings
```

### Trace-Level Embedding

```python
Trace-Level Encoder:
  Input: Distributed traces with span graphs

  Time-Aware Graph Convolution:
    Layers: 3
    Hidden dimensions: [128, 256, 512]
    Time decay: φ(Δt) = √(1/(c·Δt + 1))

  Attention-Weighted Readout:
    Type: Global attention pooling
    Query dimension: 512

  Features (per span):
    - Span duration, service name, operation
    - HTTP status, error flags
    - Parent-child relationships
    - Resource consumption

  Output: 512-dimensional trace embeddings
```

### Node-Level Embedding

```python
Node-Level Encoder:
  Input: Pod interaction graph + resource metrics

  Temporal Graph Convolution:
    Layers: 3
    Hidden dimensions: [128, 256, 512]
    Aggregation: Mean pooling

  Temporal Encoder:
    Type: LSTM
    Hidden size: 256
    Layers: 2

  Features (per pod):
    - CPU/memory/network/disk usage
    - Process count, system calls
    - Network connections
    - Container runtime metrics

  Output: 512-dimensional pod embeddings
```

### Heterogeneous Temporal GNN

```python
Heterogeneous Integration:
  Node types: Services, Traces, Pods
  Edge types:
    - service-call (service → service)
    - trace-span (trace → service)
    - pod-deploy (pod → service)
    - pod-comm (pod → pod)

  Dual Attention:
    1. Type-specific attention: α_ij^τ for edge type τ
    2. Granularity-level attention: w_g for granularity g

  Cross-Granularity Fusion:
    h_fused = Σ_g w_g · h_g
    where w_g = softmax(q_g^T tanh(W_g h_g))

  Output: Unified anomaly score
```

## Datasets

### Train-Ticket (41 Services)

**Description**: Realistic e-commerce application for train ticket booking with complex service dependencies.

**Services**: user-service, auth-service, route-service, order-service, payment-service, seat-service, config-service, station-service, train-service, travel-service, preserve-service, contacts-service, assurance-service, food-service, consign-service, security-service, inside-payment-service, execute-service, cancel-service, rebook-service, and 21 more.

**Attack Scenarios** (12 types):
- SQL Injection on order-service database
- Authentication bypass via JWT manipulation
- Privilege escalation through auth-service
- Resource exhaustion (CPU bombs on payment-service)
- DDoS flooding on route-service
- API abuse (excessive seat queries)
- Data exfiltration from user-service
- Configuration tampering in config-service
- Container breakout on travel-service pods
- Cryptocurrency mining in worker pods
- Lateral movement through pod network
- Service mesh policy violations

**Statistics**:
```
Total Requests: 2,400,000
Duration: 96 hours
Benign Traffic: 92%
Attack Traffic: 8%
Pods: 150 (replicas for high-traffic services)
Avg Response Time: 127ms
```

### Sock-Shop (14 Services)

**Description**: E-commerce microservices demo selling socks.

**Services**: front-end, catalogue, cart, orders, payment, user, shipping, queue-master, rabbitmq, carts-db, catalogue-db, orders-db, user-db, session-db

**Attack Scenarios** (8 types):
- Catalog injection attacks
- Payment fraud via price manipulation
- Cart tampering
- User account hijacking
- DDoS on front-end
- Database extraction
- Session fixation
- Message queue poisoning

**Statistics**:
```
Total Requests: 840,000
Duration: 48 hours
Benign Traffic: 89%
Attack Traffic: 11%
Pods: 56
Avg Response Time: 89ms
```

### Online-Boutique (11 Services)

**Description**: Google's cloud-native microservices demo for an e-commerce platform.

**Services**: frontend, productcatalog, recommendation, cart, checkout, currency, payment, email, shipping, ad, redis-cart

**Attack Scenarios** (6 types):
- Checkout manipulation
- Recommendation poisoning
- Currency conversion attacks
- Cart race conditions
- Ad service abuse
- Email service exploitation

**Statistics**:
```
Total Requests: 1,600,000
Duration: 72 hours
Benign Traffic: 94%
Attack Traffic: 6%
Pods: 44
Avg Response Time: 73ms
```

### DeathStarBench Suite

**Social Network** (27 services): User timeline, posts, followers, media, URLs, mentions, compose, home timeline, user mentions, etc.

**Hotel Reservation** (17 services): Frontend, search, geo, profile, rate, recommendation, reservation, user, authentication, etc.

**Media Service** (13 services): Nginx, compose, review, plot, cast, video, page, user, rating, etc.

**Statistics**:
```
Total Requests: 5,000,000+
Duration: 120 hours
Complex dependency graphs
Realistic workload patterns
```

## Reproducing Paper Results

### Table 3: Overall Detection Performance

```bash
# Run full evaluation on all datasets
python experiments/reproduce_table3.py \
  --config config/triplee_config.yaml \
  --output results/table3_overall_performance.csv

# Expected results:
# Train-Ticket: Acc=96.8%, Prec=96.4%, Rec=97.2%, F1=96.8%
# Sock-Shop: Acc=95.3%, Prec=94.9%, Rec=95.7%, F1=95.3%
# Online-Boutique: Acc=94.7%, Prec=94.2%, Rec=95.1%, F1=94.6%
```

### Table 4: Attack Type Performance

```bash
# Reproduce attack-specific results
python experiments/reproduce_table4.py \
  --dataset train_ticket \
  --output results/table4_attack_performance.csv

# Expected accuracies:
# DDoS: 98.2%, Resource Exhaustion: 97.9%
# Privilege Escalation: 97.4%, API Abuse: 95.8%
# Container Breakout: 94.7%, Crypto Mining: 95.3%
```

### Table 5: Ablation Studies

```bash
# Reproduce granularity ablation
python experiments/reproduce_table5.py \
  --ablations service_only,trace_only,node_only,full \
  --output results/table5_ablation.csv

# Expected results:
# Service-Only: 88.5%
# Trace-Only: 92.2%
# Node-Only: 85.6%
# Full TripleE-TGNN: 96.8%
```

### Figure 2: Temporal Analysis

```bash
# Reproduce accuracy over time plot
python experiments/reproduce_figure2.py \
  --dataset train_ticket \
  --with-topology-change \
  --output results/figure2_temporal_accuracy.png

# Shows TripleE-TGNN maintains 96.2±1.4% accuracy
# and adapts to topology changes (95.8% vs TGAT's 87.3%)
```

## API Documentation

### Training

```python
from models.triplee_tgnn import TripleETGNN
from training.trainer import TripleETrainer
from data.microservices_datasets import get_microservices_dataloaders

# Initialize model
model = TripleETGNN(
    service_hidden_dims=[128, 256, 512],
    trace_hidden_dims=[128, 256, 512],
    node_hidden_dims=[128, 256, 512],
    gru_hidden_size=256,
    lstm_hidden_size=256,
    fusion_method='adaptive'
)

# Load datasets
train_loader, val_loader, test_loader = get_microservices_dataloaders(
    dataset='train_ticket',
    data_dir='datasets/train_ticket/',
    batch_size=32,
    num_workers=4
)

# Create trainer
trainer = TripleETrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=3e-4,
    focal_loss_gamma=2.0
)

# Train
trainer.train(epochs=50, checkpoint_dir='checkpoints/experiment1/')
```

### Inference

```python
from inference.detector import TripleEDetector

# Load trained model
detector = TripleEDetector(
    checkpoint='checkpoints/triplee_trainticket/best_model.pth',
    device='cuda'
)

# Process microservices telemetry
service_graph = load_service_graph(timestamp)
traces = load_distributed_traces(timestamp)
pod_metrics = load_pod_metrics(timestamp)

# Detect anomalies
result = detector.predict(service_graph, traces, pod_metrics)

print(f"Anomaly Score: {result['score']:.3f}")
print(f"Is Anomalous: {result['is_anomaly']}")
print(f"Granularity Weights: Service={result['service_weight']:.3f}, "
      f"Trace={result['trace_weight']:.3f}, Node={result['node_weight']:.3f}")
print(f"Top Suspicious Services: {result['suspicious_services'][:5]}")
print(f"Anomalous Traces: {len(result['anomalous_traces'])}")
```

### Granularity Analysis

```python
# Analyze which granularity contributes most to detection
analysis = detector.analyze_granularity_contribution(
    service_graph, traces, pod_metrics
)

print("Granularity Contributions:")
print(f"  Service-Level: {analysis['service_contribution']:.3f}")
print(f"  Trace-Level: {analysis['trace_contribution']:.3f}")
print(f"  Node-Level: {analysis['node_contribution']:.3f}")

# Visualize attention weights
detector.visualize_attention(
    service_graph, traces, pod_metrics,
    output='attention_weights.png'
)
```

## Performance Benchmarks

### Inference Latency

| Dataset | Services | Pods | Avg Latency (ms) | P95 Latency (ms) |
|---------|----------|------|------------------|------------------|
| Online-Boutique | 11 | 44 | 21 | 35 |
| Sock-Shop | 14 | 56 | 28 | 44 |
| DeathStarBench-Hotel | 17 | 68 | 35 | 53 |
| DeathStarBench-Social | 27 | 108 | 42 | 68 |
| Train-Ticket | 41 | 150 | 47 | 71 |

### Training Time

| Dataset | Requests | Training Time (GPU) | Training Time (CPU) |
|---------|----------|---------------------|---------------------|
| Online-Boutique | 1.6M | 2.1 hours | 18 hours |
| Sock-Shop | 840K | 1.4 hours | 12 hours |
| Train-Ticket | 2.4M | 3.2 hours | 27 hours |
| DeathStarBench (all) | 5.0M | 6.8 hours | 58 hours |

### Memory Consumption

| Component | Training (MB) | Inference (MB) |
|-----------|---------------|----------------|
| Model Parameters | 124 | 124 |
| Service Graphs | 287 | 45 |
| Trace Graphs | 543 | 78 |
| Pod Graphs | 198 | 32 |
| GPU Memory (A100) | 4,850 | 1,230 |
| **Total** | **5,152** | **1,379** |

## Baselines

We provide implementations of comparison methods:

```bash
# Temporal Graph Attention Network (TGAT)
python baselines/train_tgat.py --config config/baselines/tgat.yaml

# Dynamic Self-Attention (DySAT)
python baselines/train_dysat.py --config config/baselines/dysat.yaml

# CONTINUUM (APT Detection)
python baselines/train_continuum.py --config config/baselines/continuum.yaml

# Static GAT
python baselines/train_gat.py --config config/baselines/gat.yaml

# GraphSAGE
python baselines/train_graphsage.py --config config/baselines/graphsage.yaml
```

## Troubleshooting

### Issue: Kubernetes metrics collection fails

**Solution**: Ensure proper RBAC permissions for metrics scraping:

```bash
# Create service account with required permissions
kubectl apply -f k8s/metrics-collector-rbac.yaml

# Verify metrics-server is running
kubectl get deployment metrics-server -n kube-system

# Test metrics access
kubectl top nodes
kubectl top pods --all-namespaces
```

### Issue: Trace collection incomplete

**Solution**: Ensure Jaeger/Zipkin is properly configured:

```bash
# Deploy Jaeger operator
kubectl apply -f k8s/jaeger-operator.yaml

# Configure sampling rate (use 1.0 for full capture during attacks)
kubectl set env deployment/my-service JAEGER_SAMPLER_TYPE=const
kubectl set env deployment/my-service JAEGER_SAMPLER_PARAM=1.0

# Verify trace collection
curl http://jaeger-query:16686/api/traces?service=my-service
```

### Issue: OOM during training on large datasets

**Solution**: Enable gradient checkpointing and reduce batch size:

```bash
python training/train_triplee.py \
  --batch-size 16 \
  --gradient-checkpointing \
  --accumulation-steps 2
```

### Issue: Low accuracy on custom microservices

**Solution**: Fine-tune on domain-specific data:

```bash
# Fine-tune pretrained model
python training/finetune_triplee.py \
  --pretrained checkpoints/triplee_trainticket/best_model.pth \
  --dataset custom_microservices \
  --epochs 20 \
  --learning-rate 1e-5
```

## Citation

If you use TripleE-TGNN in your research, please cite:

```bibtex
@article{anaedevha2025triplee,
  title={TripleE-TGNN: Triple-Embedding Temporal Graph Neural Networks for Multi-Granularity Microservices Security},
  author={Anaedevha, Roger Nick and Trofimov, Alexander Gennadevich and Borodachev, Yuri Vladimirovich},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Train-Ticket team for the microservices benchmark
- Google Cloud for Online-Boutique demo
- DeathStarBench authors for realistic workload suites
- Istio and Jaeger communities for service mesh and tracing tools
- PyTorch Geometric team for graph neural network framework

## Contact

For questions or collaborations:
- Roger Nick Anaedevha: roger.anaedevha@cs.msu.ru
- Issues: https://github.com/your-org/TripleE-TGNN/issues
- Pull Requests: https://github.com/your-org/TripleE-TGNN/pulls

---

**Note**: This codebase is provided for reproducibility and research purposes. For production deployment in critical microservices environments, additional security hardening, testing, and validation are recommended.
