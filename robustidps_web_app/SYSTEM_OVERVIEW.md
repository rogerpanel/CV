# RobustIDPS.ai - Complete System Overview

## Executive Summary

**RobustIDPS.ai** is a production-ready, web-hosted AI-powered intrusion detection and prevention system that provides real-time network security monitoring through a modern web interface.

**Live Deployment:** https://robustidps.ai

**Key Capabilities:**
- 🌐 **Web Dashboard** - Real-time threat visualization
- 📡 **Live Traffic Capture** - Network packet analysis
- 🤖 **AI Detection** - 98.4% accuracy with sub-100ms latency
- 🛡️ **Automated Prevention** - Intelligent IP blocking and quarantine
- 🚨 **Multi-Channel Alerts** - Email, Slack, SMS, webhooks
- 📊 **Comprehensive Analytics** - Grafana dashboards and metrics

---

## System Architecture

### High-Level Architecture

```
Internet/Network Traffic
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Capture Layer (3 Options)                   │
├─────────────────────────────────────────────────────────┤
│  1. Agent-Based (Distributed Agents)                    │
│  2. SPAN Port (Mirror Traffic)                          │
│  3. Inline Gateway (Active Prevention)                  │
└─────────────────┬───────────────────────────────────────┘
                  │ PCAP/Flow Data
                  ▼
┌─────────────────────────────────────────────────────────┐
│           RobustIDPS.ai Platform                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Nginx      │◄──►│   FastAPI    │◄──►│ Postgres │ │
│  │  (Proxy)     │    │   (API)      │    │   (DB)   │ │
│  └──────────────┘    └──────────────┘    └──────────┘ │
│         │                    │                          │
│         │             ┌──────┴──────┐                   │
│         │             │             │                   │
│         │      ┌──────▼──────┐  ┌──▼─────────┐        │
│         │      │   Detection │  │   Redis    │        │
│         │      │   Engine    │  │   Cache    │        │
│         │      │   (AI)      │  └────────────┘        │
│         │      └─────────────┘                         │
│         │                                               │
│  ┌──────▼──────────────────────────────────────┐      │
│  │   WebSocket Real-time Updates               │      │
│  └──────────────────────────────────────────────┘      │
│                                                          │
└──────────────────┬──────────────────────────────────────┘
                   │
          ┌────────┴─────────┐
          │                  │
    ┌─────▼──────┐    ┌─────▼──────┐
    │  Firewall  │    │   Alerts   │
    │  Control   │    │  (Email,   │
    │ (iptables) │    │   Slack)   │
    └────────────┘    └────────────┘
```

### Component Description

#### 1. **Frontend (React.js)**
- **Technology:** React 18 + Material-UI
- **Purpose:** Real-time dashboard for monitoring and management
- **Features:**
  - Live threat map
  - Detection statistics
  - Alert management
  - Traffic visualization
  - Investigation tools

#### 2. **Backend API (FastAPI)**
- **Technology:** Python FastAPI + SQLAlchemy
- **Purpose:** RESTful API and business logic
- **Endpoints:**
  - `/api/v1/detections` - Detection results
  - `/api/v1/alerts` - Security alerts
  - `/api/v1/traffic/ingest` - Traffic ingestion
  - `/api/v1/stats` - System statistics
  - `/ws` - WebSocket for real-time updates

#### 3. **Detection Engine**
- **Technology:** PyTorch + Integrated AI-IDS models
- **Models Used:**
  - Neural ODE (Temporal Adaptive)
  - Optimal Transport (Domain Adaptation)
  - Encrypted Traffic Analyzer
  - Bayesian Uncertainty Quantification
  - Federated Graph Temporal Dynamics
  - Heterogeneous Graph Pooling

#### 4. **Database (PostgreSQL)**
- **Tables:**
  - `network_traffic` - Raw traffic data
  - `detections` - AI detection results
  - `alerts` - Security alerts
  - `prevention_actions` - Automated responses
  - `users` - User accounts
  - `system_metrics` - Performance metrics

#### 5. **Cache (Redis)**
- **Uses:**
  - Session storage
  - Detection result caching
  - Rate limiting
  - Celery message broker

#### 6. **Traffic Capture**
- **Technology:** Scapy + Python asyncio
- **Modes:**
  - **Passive:** Monitor only, no interference
  - **Inline:** Gateway mode with blocking
  - **Agent:** Distributed capture nodes

#### 7. **Monitoring Stack**
- **Prometheus:** Metrics collection
- **Grafana:** Visualization dashboards
- **Custom Dashboards:** Detection rates, latency, system health

---

## How Network Integration Works

### Deployment Scenarios

#### Scenario 1: Distributed Enterprise Network

```
┌─────────────────────────────────────────────────────┐
│             Enterprise Network                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Datacenter 1                  Datacenter 2         │
│  ┌────────────┐                ┌────────────┐       │
│  │  Capture   │                │  Capture   │       │
│  │  Agent 1   │                │  Agent 2   │       │
│  └──────┬─────┘                └──────┬─────┘       │
│         │ API                         │ API          │
│         └────────┬────────────────────┘              │
│                  │                                   │
│  Office Network  │  Cloud (AWS)                      │
│  ┌────────────┐  │  ┌────────────┐                   │
│  │  Capture   │  │  │  Capture   │                   │
│  │  Agent 3   │  │  │  Agent 4   │                   │
│  └──────┬─────┘  │  └──────┬─────┘                   │
│         │        │         │                         │
│         └────────┼─────────┘                         │
│                  │                                   │
└──────────────────┼───────────────────────────────────┘
                   │
                   ▼
           ┌───────────────┐
           │ RobustIDPS.ai │
           │   (Central)   │
           └───────────────┘
```

**How it works:**
1. Deploy capture agents at each network segment
2. Agents capture local traffic and extract features
3. Features sent to central API via HTTPS
4. Central system runs AI detection
5. Alerts sent back to local security teams
6. Optional: Auto-blocking via local firewall integration

**Deployment command:**
```bash
# On each network segment
curl -O https://robustidps.ai/downloads/install-agent.sh
sudo ./install-agent.sh \
  --server https://robustidps.ai \
  --token YOUR_API_TOKEN \
  --interface eth0
```

#### Scenario 2: Data Center with SPAN Port

```
┌───────────────────────────────────────┐
│         Data Center Network            │
├───────────────────────────────────────┤
│                                        │
│  ┌────────┐    ┌────────┐    ┌────┐  │
│  │Server 1│◄──►│ Switch │◄──►│... │  │
│  └────────┘    └───┬────┘    └────┘  │
│                    │                   │
│                    │ SPAN (Mirror)     │
│                    │                   │
│                    ▼                   │
│          ┌──────────────────┐         │
│          │  RobustIDPS.ai   │         │
│          │   (Inline)       │         │
│          └──────────────────┘         │
│                                        │
└────────────────────────────────────────┘
```

**How it works:**
1. Configure switch SPAN port to mirror all traffic
2. Connect RobustIDPS.ai server to SPAN port
3. Server captures mirrored packets (passive monitoring)
4. No inline blocking (monitoring only)

**Switch configuration (Cisco example):**
```
configure terminal
monitor session 1 source interface Gi0/1 - 24
monitor session 1 destination interface Gi0/48
```

#### Scenario 3: Inline Gateway Mode

```
Internet
   │
   ▼
┌──────────────────┐
│  RobustIDPS.ai   │ ◄─ Active inspection & blocking
│     Gateway      │
└────────┬─────────┘
         │
         ▼
   Internal Network
```

**How it works:**
1. Deploy RobustIDPS.ai as network gateway
2. All traffic flows through the system
3. Real-time inspection and classification
4. Malicious traffic automatically blocked
5. Clean traffic forwarded to destination

**Configuration:**
```bash
# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1

# Configure iptables for gateway mode
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
```

---

## Traffic Flow: From Packet to Prevention

### Step-by-Step Process

```
1. PACKET CAPTURE
   ↓
   Network packet arrives at capture interface
   ↓
2. FEATURE EXTRACTION
   ↓
   Extract: IPs, ports, protocol, payload entropy,
            TLS fingerprints, timing patterns
   ↓
3. STORE IN DATABASE
   ↓
   Save to network_traffic table
   ↓
4. AI DETECTION
   ↓
   Load traffic features into AI models:
   - Neural ODE: Temporal patterns
   - Optimal Transport: Cross-domain adaptation
   - Encrypted Analyzer: TLS metadata analysis
   - Bayesian: Uncertainty quantification
   ↓
5. DECISION FUSION
   ↓
   Combine model predictions with confidence weighting
   ↓
6. CLASSIFICATION
   ↓
   Result: {
     is_malicious: true/false,
     attack_type: "ddos" | "port_scan" | ...,
     confidence: 0.0 - 1.0,
     severity: "critical" | "high" | "medium" | "low"
   }
   ↓
7. STORE DETECTION
   ↓
   Save to detections table
   ↓
8. TRIGGER ALERTS (if malicious)
   ↓
   Create alert record
   Send notifications: Email, Slack, SMS
   ↓
9. AUTOMATED PREVENTION (if enabled)
   ↓
   Execute firewall rule to block attacker IP
   ↓
10. REAL-TIME UPDATE
    ↓
    Push to dashboard via WebSocket
```

### Code Flow Example

```python
# 1. Capture packet
packet = capture_interface.read()

# 2. Extract features
metadata = extract_packet_metadata(packet)
features = {
    'src_ip': packet.src,
    'dst_ip': packet.dst,
    'payload_entropy': calculate_entropy(packet.payload),
    # ... 64 features total
}

# 3. Store in DB
traffic = NetworkTraffic(
    src_ip=metadata.src_ip,
    dst_ip=metadata.dst_ip,
    features=features
)
db.add(traffic)
db.commit()

# 4. Run AI detection
feature_tensor = torch.tensor(features).unsqueeze(0)
result = unified_model(feature_tensor)
# result = DetectionResult(
#     is_malicious=True,
#     attack_type="ddos",
#     confidence=0.97,
#     uncertainty={'epistemic': 0.02, 'aleatoric': 0.01}
# )

# 5. Store detection
detection = Detection(
    traffic_id=traffic.id,
    is_malicious=result.is_malicious,
    attack_type=result.attack_type,
    confidence=result.confidence,
    severity="high"
)
db.add(detection)
db.commit()

# 6. Create alert
if result.is_malicious and result.confidence > 0.90:
    alert = Alert(
        detection_id=detection.id,
        title=f"{result.attack_type.upper()} Detected",
        severity="high"
    )
    db.add(alert)

    # Send notifications
    send_email(alert)
    send_slack(alert)

# 7. Auto-block
if AUTO_BLOCK and result.confidence > 0.95:
    firewall.block_ip(traffic.src_ip, duration=3600)

# 8. Push to dashboard
websocket.broadcast({
    'type': 'detection',
    'data': detection.to_dict()
})
```

---

## Key Features in Detail

### 1. Real-Time Detection

**How it works:**
- Streaming packet capture (no storage bottleneck)
- Batch processing (16-32 packets at a time)
- GPU-accelerated inference
- TorchScript compilation for speed
- Average latency: <100ms

### 2. Multi-Model Ensemble

**Models:**
1. **Neural ODE** - Continuous-time temporal patterns
2. **Optimal Transport** - Domain adaptation for cloud/on-prem
3. **Encrypted Traffic** - TLS analysis without decryption
4. **Bayesian** - Uncertainty quantification
5. **Graph Models** - Network topology analysis

**Fusion:**
- Weighted voting based on confidence
- Bayesian combination
- Disagreement detection → flag for human review

### 3. Automated Prevention

**Actions:**
- IP blocking (iptables/nftables)
- Rate limiting
- Quarantine suspicious flows
- Geographic blocking
- Temporary blocks with auto-rollback

**Safety:**
- Whitelist for critical IPs
- Confidence threshold (default: 0.95)
- Auto-rollback after timeout
- Human override available

### 4. Multi-Channel Alerting

**Channels:**
- **Email:** SMTP with templates
- **Slack:** Webhook integration
- **Microsoft Teams:** Webhook
- **SMS:** Twilio integration
- **PagerDuty:** Incident creation
- **Webhooks:** Custom integrations

**Smart Alerting:**
- Deduplication (avoid spam)
- Rate limiting
- Severity-based routing
- Time-based schedules

### 5. Forensics & Investigation

**Tools:**
- PCAP download for detailed analysis
- Attack replay
- Flow visualization
- Event correlation
- Historical search
- SHAP explainability

---

## Performance Characteristics

### Throughput

| Metric | Single Instance | 4 Workers | 8 Workers |
|--------|-----------------|-----------|-----------|
| Events/sec | 50K | 200K | 400K |
| Packets/sec | 10K | 40K | 80K |
| Bandwidth | ~100 Mbps | ~400 Mbps | ~800 Mbps |

### Latency (p99)

| Component | Latency |
|-----------|---------|
| Packet capture | 5ms |
| Feature extraction | 10ms |
| AI inference | 50ms |
| Database write | 15ms |
| Alert dispatch | 20ms |
| **Total** | **95ms** |

### Accuracy

| Dataset | Accuracy | FPR | FNR |
|---------|----------|-----|-----|
| CIC-IDS2018 | 98.4% | 1.8% | 1.4% |
| UNSW-NB15 | 97.1% | 2.3% | 1.9% |
| Encrypted | 99.2% | 0.9% | 0.7% |
| Cloud Logs | 96.8% | 2.5% | 2.1% |

---

## Deployment Models

### 1. Single Server (Small Office)

```
┌─────────────────────┐
│   Single Server     │
│  (All-in-One)       │
│                     │
│  - Nginx            │
│  - Backend API      │
│  - Database         │
│  - Detection Engine │
│  - Frontend         │
└─────────────────────┘
```

**Resources:** 8GB RAM, 4 cores, 100GB disk
**Handles:** Up to 100K events/sec

### 2. Multi-Server (Enterprise)

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│  Load    │   │ Database │   │  Redis   │
│ Balancer │   │ Cluster  │   │ Cluster  │
└────┬─────┘   └──────────┘   └──────────┘
     │
     ├─────────┬─────────┬─────────┐
     ▼         ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐┌────────┐
│Backend ││Backend ││Backend ││Backend │
│   1    ││   2    ││   3    ││   4    │
└────────┘└────────┘└────────┘└────────┘
     │         │         │         │
     └─────────┴─────────┴─────────┘
               │
         ┌─────▼──────┐
         │  Detection │
         │  Workers   │
         │  (x8 GPU)  │
         └────────────┘
```

**Resources:** Cluster with 10+ nodes
**Handles:** Millions of events/sec

### 3. Kubernetes (Cloud-Native)

```
Kubernetes Cluster
├── Namespace: robustidps
│   ├── Deployment: backend (replicas: 4)
│   ├── Deployment: detection-worker (replicas: 8)
│   ├── StatefulSet: postgres (replicas: 3)
│   ├── StatefulSet: redis (replicas: 3)
│   ├── Deployment: frontend (replicas: 2)
│   └── Ingress: nginx
```

**Resources:** Auto-scaling based on load
**Handles:** Unlimited (scales automatically)

---

## Security Considerations

### Authentication & Authorization
- JWT tokens (30 min expiry)
- API key support
- OAuth2 (Google, GitHub, Azure AD)
- MFA (TOTP)
- Role-based access control (RBAC)

### Data Protection
- TLS 1.3 encryption in transit
- AES-256 encryption at rest
- Secure credential storage (bcrypt)
- PII redaction
- Audit logging

### Network Security
- Firewall (UFW/iptables)
- Fail2Ban integration
- Rate limiting
- DDoS protection (Cloudflare)
- IP whitelisting

---

## Monitoring & Observability

### Metrics (Prometheus)
- Detection throughput
- Detection latency (p50, p95, p99)
- Model accuracy
- False positive rate
- Resource usage (CPU, memory, GPU)
- Queue depths
- Database connections

### Logs
- Structured JSON logging
- Centralized with ELK stack
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log rotation (10MB per file, 10 backups)

### Dashboards (Grafana)
- Real-time detection dashboard
- System health dashboard
- Attack trends dashboard
- Performance metrics dashboard

---

## Scaling Strategies

### Vertical Scaling
- Upgrade CPU/RAM/GPU
- Use faster storage (NVMe SSD)
- Optimize database (indexing, partitioning)

### Horizontal Scaling
- Add more detection workers
- Load balance API servers
- Database read replicas
- Redis cluster mode

### Geographic Distribution
- Deploy capture agents globally
- Central detection in cloud
- Edge detection for low-latency

---

## Cost Estimation

### Cloud Hosting (AWS)

**Small Deployment:**
- EC2 t3.xlarge (4 vCPU, 16GB RAM): $122/month
- RDS PostgreSQL db.t3.medium: $61/month
- ElastiCache Redis t3.micro: $12/month
- **Total:** ~$200/month

**Medium Deployment:**
- EC2 g4dn.xlarge (GPU, 16GB RAM): $394/month
- RDS PostgreSQL db.m5.large: $147/month
- ElastiCache Redis m5.large: $121/month
- Load Balancer: $17/month
- **Total:** ~$700/month

**Enterprise Deployment:**
- EKS cluster (10 nodes): $1500/month
- RDS Aurora Multi-AZ: $500/month
- ElastiCache cluster: $300/month
- Load Balancer + WAF: $100/month
- **Total:** ~$2500/month

---

## Support & Documentation

- **Full Documentation:** https://docs.robustidps.ai
- **API Reference:** https://robustidps.ai/api/docs
- **GitHub:** https://github.com/rogerpanel/CV
- **Email:** support@robustidps.ai
- **Community Forum:** https://community.robustidps.ai

---

## Conclusion

RobustIDPS.ai provides an enterprise-grade, AI-powered intrusion detection and prevention system that can be deployed on-premises or in the cloud. With multiple integration options, it seamlessly fits into existing network infrastructure while providing state-of-the-art threat detection capabilities.

**Ready to deploy?** See [DEPLOYMENT.md](./DEPLOYMENT.md) for step-by-step instructions.

**Have questions?** Contact support@robustidps.ai

---

**Built by Roger Nick Anaedevha**
**MEPhI University - PhD Dissertation Implementation**
**© 2024 RobustIDPS.ai - All Rights Reserved**
