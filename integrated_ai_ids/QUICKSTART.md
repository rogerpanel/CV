# Quick Start Guide - Integrated AI-IDS

Get up and running in 5 minutes!

## Installation

```bash
# Install from PyPI
pip install integrated-ai-ids

# OR install from source
git clone https://github.com/rogerpanel/CV.git
cd CV/integrated_ai_ids
pip install -e .
```

## Basic Usage

### 1. Python API

```python
from integrated_ai_ids import UnifiedIDS

# Initialize with all models
ids = UnifiedIDS(
    models=['neural_ode', 'optimal_transport', 'encrypted_traffic'],
    confidence_threshold=0.85
)

# Detect threats
import torch
sample_traffic = torch.randn(1, 64)  # Your feature vector
result = ids(sample_traffic)

print(f"Malicious: {result.is_malicious}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Attack Type: {result.attack_type}")
print(f"Explanation: {result.explanation}")
```

### 2. REST API

**Start Server:**
```bash
python -m integrated_ai_ids.api.rest_server --port 8000
```

**Make Requests:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "X-API-Key: demo-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.50",
    "src_port": 54321,
    "dst_port": 443,
    "protocol": "tcp",
    "packets": 150,
    "bytes": 45000,
    "duration": 2.5
  }'
```

### 3. Docker Deployment

```bash
# Run with Docker
docker run -d \
  --name ai-ids \
  --gpus all \
  -p 8000:8000 \
  integrated-ai-ids:latest

# Or with Docker Compose
cd deployment/docker
docker-compose up -d
```

### 4. Suricata Integration

```bash
# Install plugin
sudo python -m integrated_ai_ids.plugins.install suricata

# Start monitoring
sudo systemctl start suricata-ai-ids

# View enhanced alerts
tail -f /var/log/suricata/ai-enhanced.json
```

## Configuration

Edit `/etc/integrated-ai-ids/config.yaml`:

```yaml
models:
  - neural_ode
  - optimal_transport
  - encrypted_traffic

confidence_threshold: 0.85
enable_uncertainty: true
enable_explanation: true
```

## Next Steps

- 📚 Read [Integration Guide](docs/INTEGRATION_GUIDE.md) for SOC integration
- 🔧 See [Configuration](configs/model_config.yaml) for advanced settings
- 📊 Check [API Documentation](http://localhost:8000/docs) after starting server
- 🐛 Report issues at https://github.com/rogerpanel/CV/issues

## Performance Tips

- **GPU Acceleration**: Install with `pip install integrated-ai-ids[cuda]`
- **Batch Processing**: Use `/detect/batch` endpoint for multiple flows
- **Model Selection**: Disable unused models to reduce memory
- **Caching**: Enable feature caching for repeated patterns

## Support

- Email: ar006@campus.mephi.ru
- GitHub: https://github.com/rogerpanel/CV
- Documentation: https://integrated-ai-ids.readthedocs.io
