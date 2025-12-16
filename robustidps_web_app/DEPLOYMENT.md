# RobustIDPS.ai Deployment Guide

Complete deployment guide for production hosting at https://robustidps.ai

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Server Setup](#server-setup)
3. [DNS Configuration](#dns-configuration)
4. [SSL Certificate](#ssl-certificate)
5. [Application Deployment](#application-deployment)
6. [Network Integration](#network-integration)
7. [Monitoring Setup](#monitoring-setup)
8. [Security Hardening](#security-hardening)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Server Requirements

**Minimum (Development/Small Network)**
- OS: Ubuntu 22.04 LTS or later
- CPU: 4 cores (Intel/AMD)
- RAM: 8 GB
- Storage: 100 GB SSD
- Network: 1 Gbps

**Recommended (Production)**
- OS: Ubuntu 22.04 LTS
- CPU: 16 cores (Intel/AMD)
- RAM: 32 GB
- GPU: NVIDIA (CUDA compatible, 8GB+ VRAM)
- Storage: 500 GB NVMe SSD
- Network: 10 Gbps

**Enterprise (High Traffic)**
- Kubernetes cluster with 3+ nodes
- Load balancer
- High-availability database
- Distributed storage

### Software Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify
docker --version
docker-compose --version
nvidia-smi  # Verify GPU
```

---

## Server Setup

### 1. Clone Repository

```bash
# Create application directory
sudo mkdir -p /opt/robustidps
sudo chown $USER:$USER /opt/robustidps
cd /opt/robustidps

# Clone repository
git clone https://github.com/rogerpanel/CV.git .
cd robustidps_web_app
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Critical settings to configure:**

```bash
# Application
SECRET_KEY=<generate-random-32-char-string>
JWT_SECRET_KEY=<generate-random-32-char-string>
APP_ENV=production
DEBUG=false

# Domain
DOMAIN=robustidps.ai
FRONTEND_URL=https://robustidps.ai
BACKEND_URL=https://robustidps.ai/api

# Database
POSTGRES_PASSWORD=<strong-random-password>

# Email Alerts
SMTP_USER=alerts@robustidps.ai
SMTP_PASSWORD=<email-app-password>
ALERT_TO_EMAILS=security@robustidps.ai

# Network Capture
CAPTURE_INTERFACE=eth0  # Your network interface
```

**Generate secure keys:**

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Create Directories

```bash
# Create required directories
mkdir -p {logs,data/pcaps,data/geoip,models,backups}
chmod 755 logs data backups
```

### 4. Download Model Checkpoints

```bash
# If you have pre-trained models
# Copy from integrated_ai_ids
cp ../integrated_ai_ids/checkpoints/unified_ids.pt ./models/

# Or download from your model repository
# wget https://your-model-repo.com/unified_ids.pt -O ./models/unified_ids.pt
```

### 5. Download GeoIP Database (Optional)

```bash
# Download MaxMind GeoLite2 database
# Requires free account: https://www.maxmind.com/en/geolite2/signup

# After getting license key:
mkdir -p data/geoip
cd data/geoip
wget "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&license_key=YOUR_LICENSE_KEY&suffix=tar.gz" -O GeoLite2-City.tar.gz
tar -xzf GeoLite2-City.tar.gz
mv GeoLite2-City_*/GeoLite2-City.mmdb .
cd ../..
```

---

## DNS Configuration

### Configure DNS Records

Point your domain to server IP:

```
Type    Name                Value                 TTL
A       robustidps.ai       YOUR_SERVER_IP        300
A       www.robustidps.ai   YOUR_SERVER_IP        300
CNAME   api.robustidps.ai   robustidps.ai         300
CNAME   ws.robustidps.ai    robustidps.ai         300
```

**Using Cloudflare (Recommended):**

1. Add domain to Cloudflare
2. Update nameservers at registrar
3. Add A records as above
4. Enable "Proxy status" (orange cloud)
5. SSL/TLS mode: Full (strict)

**Verify DNS:**

```bash
nslookup robustidps.ai
dig robustidps.ai +short
```

---

## SSL Certificate

### Option 1: Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Stop nginx if running
sudo systemctl stop nginx

# Obtain certificate
sudo certbot certonly --standalone -d robustidps.ai -d www.robustidps.ai

# Certificates will be at:
# /etc/letsencrypt/live/robustidps.ai/fullchain.pem
# /etc/letsencrypt/live/robustidps.ai/privkey.pem

# Copy to nginx directory
sudo mkdir -p nginx/ssl
sudo cp /etc/letsencrypt/live/robustidps.ai/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/robustidps.ai/privkey.pem nginx/ssl/
sudo chown -R $USER:$USER nginx/ssl

# Auto-renewal
sudo certbot renew --dry-run
```

### Option 2: Cloudflare Origin Certificate

1. Go to Cloudflare Dashboard → SSL/TLS → Origin Server
2. Create Certificate (15 year validity)
3. Save certificate and private key
4. Place in `nginx/ssl/`:

```bash
nano nginx/ssl/fullchain.pem  # Paste certificate
nano nginx/ssl/privkey.pem    # Paste private key
chmod 600 nginx/ssl/*.pem
```

---

## Application Deployment

### 1. Build and Start Services

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 2. Initialize Database

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Create admin user
docker-compose exec backend python scripts/create_admin.py \
  --email admin@robustidps.ai \
  --password "SecurePassword123!" \
  --full-name "Admin User"

# Verify
docker-compose exec postgres psql -U robustidps -d robustidps_db -c "\dt"
```

### 3. Verify Services

```bash
# Check backend health
curl http://localhost:8000/health

# Check API
curl http://localhost:8000/api/v1/

# Check frontend
curl http://localhost:3000

# Check metrics
curl http://localhost:9090/metrics
```

### 4. Access Dashboard

Open browser: https://robustidps.ai

Login with admin credentials created above.

---

## Network Integration

### Option 1: Deploy Capture Agent on Network Segment

**On remote network segment:**

```bash
# Download capture agent installer
curl -O https://robustidps.ai/downloads/install-agent.sh
chmod +x install-agent.sh

# Install agent
sudo ./install-agent.sh \
  --server https://robustidps.ai \
  --token YOUR_API_TOKEN \
  --interface eth0

# Check status
sudo systemctl status robustidps-agent

# View logs
sudo journalctl -u robustidps-agent -f
```

### Option 2: Configure SPAN Port

**On Cisco switch:**

```
configure terminal
monitor session 1 source interface Gi0/1 - 24
monitor session 1 destination interface Gi0/48
end
write memory
```

**On server connected to Gi0/48:**

```bash
# Configure capture on mirrored interface
docker-compose exec capture-agent python -m app.services.traffic_capture \
  --interface eth1 \
  --filter "not port 22"  # Exclude SSH
```

### Option 3: Inline Gateway Mode

**Configure as network gateway:**

```bash
# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf

# Configure iptables
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
sudo iptables -A FORWARD -i eth0 -o eth1 -m state --state RELATED,ESTABLISHED -j ACCEPT

# Save rules
sudo apt install iptables-persistent -y
sudo netfilter-persistent save

# Configure capture on both interfaces
docker-compose exec capture-agent python -m app.services.traffic_capture \
  --interface eth0 \
  --interface eth1
```

---

## Monitoring Setup

### 1. Prometheus Configuration

Edit `prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'robustidps-backend'
    static_configs:
      - targets: ['backend:8000']

  - job_name: 'robustidps-postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'robustidps-redis'
    static_configs:
      - targets: ['redis:6379']
```

### 2. Grafana Dashboards

Access Grafana: https://robustidps.ai/grafana

Default credentials: admin / admin (change immediately)

**Import dashboards:**
1. Go to Dashboards → Import
2. Upload `grafana/dashboards/robustidps-main.json`
3. Configure data source: Prometheus

### 3. Set Up Alerts

Configure alerting:

```bash
# Edit alert rules
nano prometheus/alert_rules.yml

# Restart Prometheus
docker-compose restart prometheus
```

---

## Security Hardening

### 1. Firewall Configuration

```bash
# Install UFW
sudo apt install ufw -y

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (important!)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow monitoring (restricted to internal network)
sudo ufw allow from 10.0.0.0/8 to any port 9090
sudo ufw allow from 10.0.0.0/8 to any port 3001

# Enable firewall
sudo ufw enable

# Verify
sudo ufw status verbose
```

### 2. Fail2Ban

```bash
# Install Fail2Ban
sudo apt install fail2ban -y

# Configure for nginx
sudo nano /etc/fail2ban/jail.local
```

Add:

```ini
[nginx-http-auth]
enabled = true
port = http,https
logpath = /opt/robustidps/logs/nginx-error.log

[nginx-noscript]
enabled = true
port = http,https
logpath = /opt/robustidps/logs/nginx-access.log
```

```bash
# Restart Fail2Ban
sudo systemctl restart fail2ban
```

### 3. Regular Updates

```bash
# Create update script
nano /opt/robustidps/update.sh
```

```bash
#!/bin/bash
cd /opt/robustidps/robustidps_web_app
git pull
docker-compose pull
docker-compose up -d
docker-compose exec backend alembic upgrade head
```

```bash
chmod +x /opt/robustidps/update.sh

# Schedule with cron
crontab -e
# Add: 0 2 * * 0 /opt/robustidps/update.sh >> /opt/robustidps/logs/update.log 2>&1
```

---

## Scaling

### Horizontal Scaling

**Scale detection workers:**

```bash
# Scale to 8 workers
docker-compose up -d --scale detection-worker=8

# Verify
docker-compose ps
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n robustidps
kubectl get svc -n robustidps
```

### Load Balancing

Use nginx upstream:

```nginx
upstream backend {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

---

## Troubleshooting

### Issue: Services won't start

```bash
# Check logs
docker-compose logs backend
docker-compose logs postgres

# Check disk space
df -h

# Check memory
free -h

# Restart services
docker-compose restart
```

### Issue: Database connection errors

```bash
# Check PostgreSQL
docker-compose exec postgres psql -U robustidps -d robustidps_db

# Reset database
docker-compose down -v
docker-compose up -d postgres
docker-compose exec backend alembic upgrade head
```

### Issue: High memory usage

```bash
# Check resource usage
docker stats

# Reduce workers
# In .env: DETECTION_WORKERS=2

# Restart
docker-compose restart
```

### Issue: Packets not being captured

```bash
# Check interface
ip link show

# Check permissions
docker-compose exec capture-agent tcpdump -D

# Verify network mode
docker inspect robustidps-capture | grep NetworkMode
```

---

## Maintenance

### Backup

```bash
# Database backup
docker-compose exec postgres pg_dump -U robustidps robustidps_db > backup_$(date +%Y%m%d).sql

# Full backup
tar -czf backup_full_$(date +%Y%m%d).tar.gz \
  --exclude='logs' \
  --exclude='data/pcaps' \
  /opt/robustidps/
```

### Restore

```bash
# Restore database
docker-compose exec -T postgres psql -U robustidps robustidps_db < backup_20240101.sql

# Restore full
tar -xzf backup_full_20240101.tar.gz -C /
```

---

## Support

For deployment assistance:
- Documentation: https://docs.robustidps.ai
- Email: support@robustidps.ai
- Issues: https://github.com/rogerpanel/CV/issues

---

**Successfully deployed? ✅**

Your RobustIDPS.ai instance should now be:
- Accessible at https://robustidps.ai
- Capturing network traffic
- Detecting threats in real-time
- Sending alerts
- Blocking malicious IPs

**Monitor system health at:**
- Dashboard: https://robustidps.ai
- Metrics: https://robustidps.ai/grafana
- Health: https://robustidps.ai/health
