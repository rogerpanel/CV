# RobustIDPS.ai - Production Web Application

**Advanced AI-Powered Intrusion Detection & Prevention System**

Live deployment at: https://robustidps.ai

## Overview

RobustIDPS.ai is a production-grade web application that provides:
- 🌐 **Web-based Dashboard** - Real-time monitoring and analytics
- 🚨 **Live Traffic Analysis** - Continuous network monitoring
- 🛡️ **Automated Prevention** - Intelligent threat blocking
- 📊 **Comprehensive Alerts** - Multi-channel notification system
- 🔍 **Forensics & Investigation** - Detailed attack analysis
- 🎯 **Multi-tenant Support** - Enterprise-ready architecture

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RobustIDPS.ai Platform                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │   Frontend   │◄────►│   Backend    │◄────►│    DB    │  │
│  │  (React.js)  │      │  (FastAPI)   │      │(Postgres)│  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
│         ▲                      ▲                            │
│         │                      │                            │
│         │              ┌───────┴────────┐                   │
│         │              │                │                   │
│         │      ┌───────▼──────┐  ┌─────▼──────┐           │
│         │      │ Detection    │  │   Redis    │           │
│         │      │ Engine       │  │   Cache    │           │
│         │      └──────────────┘  └────────────┘           │
│         │                                                   │
│  ┌──────┴─────────────────────────────────────────┐       │
│  │          WebSocket Real-time Updates            │       │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │
              ┌───────────┴───────────┐
              │                       │
      ┌───────▼──────┐        ┌──────▼───────┐
      │ Capture      │        │  Firewall    │
      │ Agents       │        │  Control     │
      │ (Network)    │        │  (iptables)  │
      └──────────────┘        └──────────────┘
```

### Traffic Capture Integration

**3 Deployment Options:**

1. **Agent-Based (Recommended)**
   - Deploy lightweight agents on network segments
   - Agents capture and forward traffic to API
   - Supports distributed deployment

2. **Network TAP/SPAN**
   - Mirror traffic from switch SPAN port
   - Direct PCAP ingestion
   - Passive monitoring

3. **Inline Prevention**
   - Deploy as network gateway
   - Active blocking capabilities
   - Real-time prevention

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Domain name pointing to your server
- SSL certificate (Let's Encrypt recommended)
- Minimum 8GB RAM, 4 CPU cores

### Installation

```bash
# Clone repository
git clone https://github.com/rogerpanel/CV.git
cd CV/robustidps_web_app

# Configure environment
cp .env.example .env
nano .env  # Edit with your settings

# Start application
docker-compose up -d

# Initialize database
docker-compose exec backend alembic upgrade head

# Create admin user
docker-compose exec backend python scripts/create_admin.py
```

### Access Dashboard

Navigate to: https://robustidps.ai

Default credentials:
- Username: `admin@robustidps.ai`
- Password: (set during initialization)

## Network Integration

### Option 1: Deploy Capture Agent

```bash
# On network segment to monitor
curl -O https://robustidps.ai/downloads/capture-agent.sh
chmod +x capture-agent.sh
./capture-agent.sh install --server https://robustidps.ai --token YOUR_API_TOKEN
```

### Option 2: Configure SPAN Port

```bash
# On Cisco switch
configure terminal
monitor session 1 source interface Gi0/1 - 24
monitor session 1 destination interface Gi0/48

# On server with Gi0/48 connected
docker-compose exec capture-agent tcpdump -i eth0 -w - | nc robustidps.ai 5000
```

### Option 3: Inline Gateway Mode

```bash
# Configure as network gateway
./scripts/configure_gateway.sh --interface eth0 --lan eth1
```

## Features

### Dashboard
- Real-time threat map
- Live traffic statistics
- Attack timeline
- Top threats & sources
- System health monitoring

### Detection
- 13 attack types classification
- 98.4% accuracy
- Sub-100ms detection latency
- Confidence scoring
- Explainable AI insights

### Prevention
- Automatic IP blocking
- Firewall rule generation
- Quarantine suspicious traffic
- Rate limiting
- Geographic blocking

### Alerts
- Email notifications
- Slack/Teams integration
- SMS alerts (Twilio)
- Webhook support
- Custom alert rules

### Forensics
- PCAP download
- Attack replay
- Flow visualization
- Event correlation
- Investigation tools

## Technology Stack

**Frontend:**
- React.js 18
- Material-UI (MUI)
- Redux for state management
- Chart.js & D3.js for visualization
- WebSocket for real-time updates

**Backend:**
- FastAPI (Python 3.10+)
- PostgreSQL 15
- Redis for caching
- Celery for async tasks
- SQLAlchemy ORM

**AI/ML:**
- PyTorch 2.0
- All dissertation models integrated
- TorchScript compilation
- ONNX export for edge deployment

**Infrastructure:**
- Docker & Docker Compose
- Kubernetes manifests
- nginx reverse proxy
- Let's Encrypt SSL
- Prometheus & Grafana monitoring

## API Documentation

### REST API

Once deployed, access interactive API docs at:
- Swagger UI: https://robustidps.ai/docs
- ReDoc: https://robustidps.ai/redoc

### WebSocket API

Connect to real-time updates:
```javascript
const ws = new WebSocket('wss://robustidps.ai/ws');
ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('New alert:', alert);
};
```

## Deployment

### Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions.

**Quick steps:**
1. Configure DNS for robustidps.ai
2. Setup SSL certificates
3. Configure environment variables
4. Deploy with Docker Compose or Kubernetes
5. Configure traffic capture
6. Setup monitoring

### Scaling

**Horizontal Scaling:**
```bash
# Scale detection workers
docker-compose up -d --scale detection-worker=5

# Scale capture agents
kubectl scale deployment capture-agent --replicas=10
```

**Load Balancing:**
- nginx for HTTP/WebSocket
- HAProxy for TCP traffic
- Kubernetes ingress

## Monitoring

### System Health

- Prometheus metrics: https://robustidps.ai/metrics
- Grafana dashboards: https://robustidps.ai/grafana
- Health endpoint: https://robustidps.ai/health

### Key Metrics

- Detection throughput (events/sec)
- Detection latency (p50, p95, p99)
- Alert rate
- False positive rate
- System resource usage

## Security

### Authentication
- JWT-based authentication
- OAuth2 integration (Google, GitHub, Azure AD)
- Multi-factor authentication (TOTP)
- Role-based access control (RBAC)

### Data Protection
- Encryption at rest (AES-256)
- TLS 1.3 in transit
- Secure credential storage
- Audit logging

### Compliance
- GDPR compliant
- SOC 2 Type II ready
- HIPAA compatible
- ISO 27001 aligned

## Configuration

### Environment Variables

```bash
# Application
APP_NAME=RobustIDPS.ai
APP_ENV=production
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/robustidps

# Redis
REDIS_URL=redis://redis:6379/0

# AI Models
MODEL_CHECKPOINT_PATH=/models/unified_ids.pt
ENABLE_GPU=true
BATCH_SIZE=32

# Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@robustidps.ai
SMTP_PASSWORD=your-password

SLACK_WEBHOOK_URL=https://hooks.slack.com/...
TWILIO_ACCOUNT_SID=ACxxxx
TWILIO_AUTH_TOKEN=xxxx

# Network
CAPTURE_INTERFACE=eth0
ENABLE_FIREWALL=true
```

### Custom Models

Upload your own trained models:
```bash
docker cp custom_model.pt robustidps-backend:/models/
docker-compose restart backend
```

## Troubleshooting

### Common Issues

**Traffic not being captured:**
```bash
# Check agent status
docker-compose exec capture-agent systemctl status capture-agent

# Verify network interface
docker-compose exec capture-agent ip link show

# Check connectivity
curl https://robustidps.ai/api/health
```

**High false positive rate:**
```bash
# Adjust detection threshold
# In dashboard: Settings → Detection → Confidence Threshold (increase to 0.90)

# Or via API
curl -X POST https://robustidps.ai/api/settings/detection \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"confidence_threshold": 0.90}'
```

**Performance issues:**
```bash
# Check resource usage
docker stats

# Scale workers
docker-compose up -d --scale detection-worker=8

# Enable caching
# Set ENABLE_REDIS_CACHE=true in .env
```

## Support

- Documentation: https://docs.robustidps.ai
- Issues: https://github.com/rogerpanel/CV/issues
- Email: support@robustidps.ai
- Community: https://community.robustidps.ai

## License

MIT License - See [LICENSE](../LICENSE) for details

## Citation

If you use RobustIDPS.ai in your research, please cite:

```bibtex
@phdthesis{anaedevha2024robustidps,
  title={Advanced AI-Powered Intrusion Detection Systems: Neural ODEs, Optimal Transport, and Federated Learning},
  author={Anaedevha, Roger Nick},
  year={2024},
  school={National Research Nuclear University MEPhI}
}
```

---

**Built with ❤️ by Roger Nick Anaedevha**
**MEPhI University | PhD Dissertation Implementation**
