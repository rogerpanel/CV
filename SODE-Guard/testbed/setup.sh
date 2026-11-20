#!/usr/bin/env bash
# One-time setup on a dedicated testbed VM (Debian/Ubuntu). Installs the host-side
# tools. The attacker tools and CICFlowMeter live in containers and are built by
# cluster/up.sh and features/Dockerfile.
set -euo pipefail

echo "== apt packages =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  docker.io tcpdump iproute2 iptables suricata zeek python3 python3-pip curl ca-certificates

echo "== k3d + kubectl =="
if ! command -v k3d >/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.6.3 bash
fi
if ! command -v kubectl >/dev/null; then
  curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/v1.30.0/bin/linux/amd64/kubectl"
  sudo install /tmp/kubectl /usr/local/bin/kubectl
fi

echo "== suricata ET Open rules =="
sudo suricata-update || echo "run 'suricata-update' manually if this failed"

echo "== python deps for scoring/analysis (host) =="
pip3 install --user --break-system-packages \
  numpy pandas scikit-learn requests torch --extra-index-url https://download.pytorch.org/whl/cpu

echo "== build the CICFlowMeter image =="
docker build -t idps-testbed/cfm "$(dirname "$0")/features"

cat <<'EOF'

Setup done. Next:
  export IDPS_TESTBED_CONFIRM=yes
  sudo -E testbed/cluster/00-isolation.sh
  testbed/cluster/up.sh
  testbed/run.py collect --repetitions 5 --out testbed/runs/clean
EOF
