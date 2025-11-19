# Integrated AI-IDS: SOC Integration Guide

## Complete Step-by-Step Integration Guide

This guide provides comprehensive instructions for integrating the Integrated AI-IDS into production Security Operations Center (SOC) environments.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Integration Methods](#integration-methods)
   - [Method 1: Suricata Plugin Integration](#method-1-suricata-plugin-integration)
   - [Method 2: Snort Plugin Integration](#method-2-snort-plugin-integration)
   - [Method 3: Zeek/Bro Integration](#method-3-zeekbro-integration)
   - [Method 4: Standalone Deployment](#method-4-standalone-deployment)
   - [Method 5: SIEM Integration](#method-5-siem-integration)
5. [Configuration](#configuration)
6. [Production Deployment](#production-deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Network Traffic                              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │                               │
  ┌────▼────┐                    ┌────▼────┐
  │ Suricata│                    │  Snort  │
  │   or    │                    │   or    │
  │  Zeek   │                    │ Custom  │
  └────┬────┘                    └────┬────┘
       │                               │
       │  EVE JSON / Unified2 / Logs   │
       │                               │
       └───────────────┬───────────────┘
                       │
              ┌────────▼────────┐
              │  AI-IDS Plugin  │
              │   / Connector   │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼────┐               ┌─────▼──────┐
    │ Unified │               │  REST API  │
    │ AI-IDS  │◄──────────────┤  Service   │
    │  Model  │               └─────┬──────┘
    └────┬────┘                     │
         │                          │
         │  Enhanced Alerts         │
         │                          │
    ┌────▼──────────────────────────▼────┐
    │         Alert Management           │
    │    (Correlation, Prioritization)   │
    └────┬───────────────────────────────┘
         │
    ┌────▼────┐     ┌──────────┐     ┌────────┐
    │  SIEM   │     │ Incident │     │  SOC   │
    │ (Splunk)│     │ Response │     │ Team   │
    └─────────┘     └──────────┘     └────────┘
```

---

## Prerequisites

### Hardware Requirements

**Minimum (Testing):**
- CPU: 4 cores
- RAM: 8 GB
- GPU: Optional (CPU-only mode supported)
- Storage: 50 GB

**Recommended (Production):**
- CPU: 16+ cores (Intel Xeon or AMD EPYC)
- RAM: 32+ GB
- GPU: NVIDIA T4 or A100 (16+ GB VRAM)
- Storage: 500 GB SSD
- Network: 10 Gbps NIC

### Software Requirements

- **Operating System**: Ubuntu 20.04+ / RHEL 8+ / CentOS 8+
- **Python**: 3.8 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.21+ (for orchestrated deployment)
- **IDS**: Suricata 6.0+ or Snort 2.9.8+ or Zeek 4.0+

### Network Requirements

- Access to network traffic (SPAN port, TAP, or inline)
- Outbound HTTPS for model updates (optional)
- Internal network for API communication

---

## Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev git build-essential \
    libssl-dev libffi-dev python3-setuptools
```

**RHEL/CentOS:**
```bash
sudo yum install -y python3-pip python3-devel git gcc openssl-devel \
    libffi-devel python3-setuptools
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv /opt/integrated-ai-ids/venv
source /opt/integrated-ai-ids/venv/bin/activate

# Install package
pip install --upgrade pip
pip install integrated-ai-ids

# OR install from source
git clone https://github.com/rogerpanel/CV.git
cd CV/integrated_ai_ids
pip install -r requirements.txt
pip install -e .
```

### Step 3: Download Pre-trained Models

```bash
# Download models
python -m integrated_ai_ids.utils.download_models --output /opt/integrated-ai-ids/models

# Verify installation
python -m integrated_ai_ids.utils.verify_installation
```

---

## Integration Methods

### Method 1: Suricata Plugin Integration

Suricata is a high-performance IDS/IPS with excellent JSON logging (EVE format).

#### Step 1.1: Install Suricata

```bash
# Ubuntu
sudo add-apt-repository ppa:oisf/suricata-stable
sudo apt-get update
sudo apt-get install suricata

# RHEL/CentOS
sudo yum install epel-release
sudo yum install suricata

# Verify
suricata --build-info
```

#### Step 1.2: Configure Suricata for AI-IDS

Edit `/etc/suricata/suricata.yaml`:

```yaml
# Enable EVE JSON logging
outputs:
  - eve-log:
      enabled: yes
      filetype: regular
      filename: /var/log/suricata/eve.json
      types:
        - alert:
            metadata: yes
            tagged-packets: yes
        - http:
            extended: yes
        - dns:
            query: yes
            answer: yes
        - tls:
            extended: yes
            session-resumption: yes
        - files:
            force-magic: yes
        - flow
        - netflow

# Configure for high performance
af-packet:
  - interface: eth0
    threads: 4
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
    use-mmap: yes
    ring-size: 2048
```

#### Step 1.3: Install AI-IDS Plugin

```bash
# Install plugin
sudo python -m integrated_ai_ids.plugins.install suricata

# This creates:
# - /etc/suricata/ai-ids.yaml (configuration)
# - /usr/local/bin/suricata-ai-ids (plugin binary)
# - /var/log/suricata/ai-enhanced.json (output)
```

#### Step 1.4: Configure Plugin

Edit `/etc/suricata/ai-ids.yaml`:

```yaml
# AI-IDS Configuration for Suricata
ai_ids:
  # Model configuration
  models:
    - neural_ode
    - optimal_transport
    - encrypted_traffic
    - graph

  # Detection settings
  confidence_threshold: 0.85
  enable_uncertainty: true
  enable_explanation: true

  # Input/Output
  eve_log_path: /var/log/suricata/eve.json
  output_path: /var/log/suricata/ai-enhanced.json

  # Rule generation
  enable_rule_generation: true
  rule_output_path: /etc/suricata/rules/ai-generated.rules

  # Performance
  batch_size: 32
  num_workers: 4

  # Alert correlation
  correlation_window: 60  # seconds
  enable_attack_chain_detection: true
```

#### Step 1.5: Start Services

```bash
# Start Suricata
sudo systemctl start suricata
sudo systemctl enable suricata

# Start AI-IDS plugin
sudo systemctl start suricata-ai-ids
sudo systemctl enable suricata-ai-ids

# Verify
sudo systemctl status suricata
sudo systemctl status suricata-ai-ids

# Monitor logs
sudo tail -f /var/log/suricata/ai-enhanced.json
```

#### Step 1.6: Reload AI-Generated Rules

```bash
# Add AI rules to Suricata config
echo "include /etc/suricata/rules/ai-generated.rules" | \
    sudo tee -a /etc/suricata/suricata.yaml

# Reload Suricata (no downtime)
sudo suricatasc -c reload-rules
```

---

### Method 2: Snort Plugin Integration

Snort is a widely-deployed IDS with Unified2 binary output.

#### Step 2.1: Install Snort

```bash
# Install dependencies
sudo apt-get install -y libpcap-dev libpcre3-dev libdumbnet-dev \
    bison flex zlib1g-dev liblzma-dev openssl libssl-dev

# Download and install Snort
wget https://www.snort.org/downloads/snort/snort-2.9.20.tar.gz
tar xzf snort-2.9.20.tar.gz
cd snort-2.9.20
./configure --enable-sourcefire
make
sudo make install

# Update libraries
sudo ldconfig

# Create snort user and directories
sudo groupadd snort
sudo useradd snort -r -s /sbin/nologin -c SNORT_IDS -g snort
sudo mkdir -p /etc/snort/rules
sudo mkdir /var/log/snort
sudo mkdir /usr/local/lib/snort_dynamicrules
```

#### Step 2.2: Configure Snort

Edit `/etc/snort/snort.conf`:

```conf
# Configure network variables
ipvar HOME_NET 192.168.1.0/24
ipvar EXTERNAL_NET !$HOME_NET

# Configure output to Unified2
output unified2: filename snort.log, limit 128, nostamp

# Enable detection
preprocessor normalize_tcp: ips ecn stream
preprocessor normalize_icmp4
preprocessor normalize_ip4
preprocessor normalize_tcp: ips ecn stream
```

#### Step 2.3: Install AI-IDS Plugin

```bash
# Install Snort plugin
sudo python -m integrated_ai_ids.plugins.install snort

# Configure
sudo vi /etc/snort/ai-ids.yaml
```

#### Step 2.4: Start Services

```bash
# Start Snort in daemon mode
sudo snort -D -c /etc/snort/snort.conf -i eth0 -l /var/log/snort

# Start AI-IDS processor
sudo systemctl start snort-ai-ids
sudo systemctl enable snort-ai-ids

# Monitor
tail -f /var/log/snort/ai-enhanced.log
```

---

### Method 3: Zeek/Bro Integration

Zeek provides rich protocol analysis and structured logging.

#### Step 3.1: Install Zeek

```bash
# Ubuntu
sudo apt-get install cmake make gcc g++ flex bison libpcap-dev \
    libssl-dev python3 python3-dev swig zlib1g-dev

# Install Zeek
wget https://download.zeek.org/zeek-4.2.1.tar.gz
tar xzf zeek-4.2.1.tar.gz
cd zeek-4.2.1
./configure
make
sudo make install

# Add to PATH
export PATH=/usr/local/zeek/bin:$PATH
```

#### Step 3.2: Configure Zeek

Edit `/usr/local/zeek/etc/node.cfg`:

```ini
[zeek]
type=standalone
host=localhost
interface=eth0
```

Edit `/usr/local/zeek/share/zeek/site/local.zeek`:

```zeek
# Enable JSON logging
@load policy/tuning/json-logs.zeek

# Load protocols
@load protocols/http
@load protocols/dns
@load protocols/ssl
@load protocols/ssh

# Enable file analysis
@load frameworks/files/hash-all-files
```

#### Step 3.3: Install AI-IDS Integration

```bash
# Install Zeek plugin
sudo python -m integrated_ai_ids.plugins.install zeek

# This creates Zeek script
sudo cp /opt/integrated-ai-ids/plugins/zeek/ai-ids.zeek \
    /usr/local/zeek/share/zeek/site/

# Load in local.zeek
echo "@load ./ai-ids.zeek" | sudo tee -a \
    /usr/local/zeek/share/zeek/site/local.zeek
```

#### Step 3.4: Start Services

```bash
# Deploy Zeek configuration
sudo zeekctl deploy

# Start AI-IDS processor
sudo systemctl start zeek-ai-ids
sudo systemctl enable zeek-ai-ids

# Monitor logs
tail -f /usr/local/zeek/logs/current/ai-enhanced.log
```

---

### Method 4: Standalone Deployment

Deploy AI-IDS as standalone service processing PCAP files or network streams.

#### Step 4.1: Configure Standalone Mode

```bash
# Create configuration
cat > /etc/integrated-ai-ids/config.yaml << EOF
# Standalone Configuration
mode: standalone

# Input sources
input:
  type: pcap  # or 'interface' for live capture
  path: /data/captures  # directory for PCAP files
  # interface: eth0  # for live capture
  bpf_filter: "tcp or udp"  # Berkeley Packet Filter

# Processing
processing:
  batch_size: 128
  num_workers: 8
  enable_multiprocessing: true

# Models
models:
  - neural_ode
  - optimal_transport
  - encrypted_traffic
  - federated
  - graph
  - llm

# Output
output:
  type: json  # json, csv, or syslog
  path: /var/log/ai-ids/detections.json
  syslog_server: 192.168.1.100:514  # if type=syslog

# Performance
performance:
  device: cuda  # cpu or cuda
  mixed_precision: true
  optimize_inference: true
EOF
```

#### Step 4.2: Run Standalone

```bash
# Process PCAP files
python -m integrated_ai_ids.standalone \
    --config /etc/integrated-ai-ids/config.yaml \
    --input /data/captures/*.pcap \
    --output /var/log/ai-ids/results.json

# Live capture
sudo python -m integrated_ai_ids.standalone \
    --config /etc/integrated-ai-ids/config.yaml \
    --interface eth0 \
    --daemon
```

---

### Method 5: SIEM Integration

Integrate with Security Information and Event Management systems.

#### Step 5.1: Splunk Integration

**Install Splunk App:**
```bash
# Create Splunk app
sudo mkdir -p /opt/splunk/etc/apps/integrated-ai-ids

# Install app
sudo python -m integrated_ai_ids.siem.install splunk \
    --splunk-home /opt/splunk

# Configure inputs
cat > /opt/splunk/etc/apps/integrated-ai-ids/default/inputs.conf << EOF
[monitor:///var/log/suricata/ai-enhanced.json]
disabled = false
sourcetype = ai_ids:json
index = security

[script://./bin/ai_ids_api.py]
disabled = false
interval = 60
sourcetype = ai_ids:api
index = security
EOF

# Restart Splunk
sudo /opt/splunk/bin/splunk restart
```

**Create Splunk Dashboard:**
```spl
# Search query
index=security sourcetype="ai_ids:json"
| eval severity_numeric=case(
    severity="critical", 4,
    severity="high", 3,
    severity="medium", 2,
    severity="low", 1
  )
| timechart span=5m avg(confidence) as avg_confidence,
    count(eval(is_malicious="true")) as threats

# Alert configuration
| search is_malicious="true" confidence>0.9
| sendemail to="soc@company.com"
    subject="Critical AI-IDS Alert"
```

#### Step 5.2: ELK Stack Integration

**Configure Logstash:**
```ruby
# /etc/logstash/conf.d/ai-ids.conf
input {
  file {
    path => "/var/log/suricata/ai-enhanced.json"
    codec => "json"
    type => "ai_ids"
  }
}

filter {
  if [type] == "ai_ids" {
    mutate {
      add_field => { "[@metadata][index]" => "ai-ids-%{+YYYY.MM.dd}" }
    }

    # Enrich with GeoIP
    geoip {
      source => "src_ip"
      target => "src_geo"
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "%{[@metadata][index]}"
  }
}
```

**Create Kibana Dashboard:**
```bash
# Import AI-IDS dashboard
curl -X POST "localhost:5601/api/saved_objects/_import" \
  -H "kbn-xsrf: true" \
  --form file=@/opt/integrated-ai-ids/dashboards/kibana-dashboard.ndjson
```

---

## Configuration

### Model Configuration

Edit `/etc/integrated-ai-ids/models.yaml`:

```yaml
# Neural ODE Configuration
neural_ode:
  hidden_dims: [128, 256, 256, 128]
  ode_solver: dopri5
  rtol: 1e-3
  atol: 1e-4
  enable_point_process: true

# Optimal Transport Configuration
optimal_transport:
  epsilon: 0.85
  delta: 1e-5
  enable_privacy: true
  sinkhorn_iterations: 100

# Encrypted Traffic Configuration
encrypted_traffic:
  cnn_filters: [64, 128, 256]
  lstm_hidden: 128
  transformer_layers: 6
  attention_heads: 8

# Graph Model Configuration
graph:
  num_relations: 5
  pooling_ratio: 0.2
  attention_heads: 4

# LLM Configuration
llm:
  model_name: gpt-4
  api_key: ${OPENAI_API_KEY}
  enable_chain_of_thought: true
  temperature: 0.7
```

### Performance Tuning

Edit `/etc/integrated-ai-ids/performance.yaml`:

```yaml
# Hardware acceleration
device: cuda  # cpu, cuda, or tpu
mixed_precision: true
compile_model: true  # torch.compile()

# Batching
batch_size: 128
max_batch_delay_ms: 50

# Multi-processing
num_workers: 8
prefetch_factor: 2

# Caching
enable_feature_cache: true
cache_size_mb: 1024

# Model optimization
quantization: int8  # fp16, int8, or none
pruning: structured
pruning_ratio: 0.3
```

---

## Production Deployment

### Docker Deployment

**Create Docker Image:**
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git

# Install AI-IDS
COPY . /app
WORKDIR /app
RUN pip3 install -e .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["python3", "-m", "integrated_ai_ids.api.rest_server"]
```

**Build and Run:**
```bash
# Build
docker build -t integrated-ai-ids:latest .

# Run
docker run -d \
    --name ai-ids \
    --gpus all \
    -p 8000:8000 \
    -v /var/log/suricata:/data/logs \
    -v /etc/integrated-ai-ids:/config \
    integrated-ai-ids:latest
```

### Kubernetes Deployment

**Create Kubernetes manifests:**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: integrated-ai-ids
  namespace: security
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-ids
  template:
    metadata:
      labels:
        app: ai-ids
    spec:
      containers:
      - name: ai-ids
        image: integrated-ai-ids:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_PATH
          value: /models
        volumeMounts:
        - name: models
          mountPath: /models
        - name: config
          mountPath: /config
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ai-ids-models
      - name: config
        configMap:
          name: ai-ids-config
---
apiVersion: v1
kind: Service
metadata:
  name: ai-ids-service
  namespace: security
spec:
  selector:
    app: ai-ids
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**
```bash
# Create namespace
kubectl create namespace security

# Apply manifests
kubectl apply -f k8s-deployment.yaml

# Verify
kubectl get pods -n security
kubectl logs -n security -l app=ai-ids
```

---

## Monitoring & Maintenance

### Prometheus Metrics

Configure Prometheus to scrape AI-IDS metrics:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ai-ids'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import pre-built dashboard:
```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/import \
    -H "Content-Type: application/json" \
    -d @/opt/integrated-ai-ids/dashboards/grafana-dashboard.json
```

### Log Rotation

Configure logrotate:
```bash
# /etc/logrotate.d/ai-ids
/var/log/ai-ids/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 root adm
    sharedscripts
    postrotate
        systemctl reload ai-ids
    endscript
}
```

### Model Updates

```bash
# Download latest models
python -m integrated_ai_ids.utils.update_models

# Reload without downtime
curl -X POST http://localhost:8000/model/update \
    -H "X-API-Key: your-api-key" \
    -d '{"checkpoint_path": "/models/latest.pth"}'
```

---

## Troubleshooting

### Common Issues

**Issue: High Memory Usage**
```bash
# Solution: Reduce batch size
sed -i 's/batch_size: 128/batch_size: 32/' /etc/integrated-ai-ids/config.yaml
systemctl restart ai-ids
```

**Issue: GPU Out of Memory**
```bash
# Solution: Enable mixed precision
sed -i 's/mixed_precision: false/mixed_precision: true/' \
    /etc/integrated-ai-ids/performance.yaml
```

**Issue: High Latency**
```bash
# Solution: Enable model compilation
python -m integrated_ai_ids.utils.optimize_model \
    --input /models/model.pth \
    --output /models/model-optimized.pth \
    --quantize int8
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m integrated_ai_ids.api.rest_server --debug

# Check logs
tail -f /var/log/ai-ids/debug.log
```

### Support

- Documentation: https://github.com/rogerpanel/CV/integrated_ai_ids/docs
- Issues: https://github.com/rogerpanel/CV/issues
- Email: ar006@campus.mephi.ru

---

## Next Steps

1. ✅ Complete integration following this guide
2. ✅ Test with sample traffic
3. ✅ Tune performance parameters
4. ✅ Configure monitoring and alerting
5. ✅ Integrate with SOC workflows
6. ✅ Train SOC team on AI-enhanced alerts
7. ✅ Plan for model updates and maintenance

**Congratulations! You now have a production-ready AI-powered IDS integrated into your SOC environment.**
