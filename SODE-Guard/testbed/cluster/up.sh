#!/usr/bin/env bash
# Bring up the testbed cluster: internal network, victims, benign load, attacker.
# Assumes cluster/00-isolation.sh has already run (it checks).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# refuse to continue if the isolation rule is not present
sub=${IDPS_SUBNET:-10.77.0.0/16}
if ! sudo iptables -C FORWARD -s "$sub" ! -d "$sub" -o "$(ip route show default | awk '/default/{print $5;exit}')" -j DROP 2>/dev/null; then
  echo "isolation rule not found; run: sudo testbed/cluster/00-isolation.sh" >&2
  exit 1
fi

k3d cluster create --config "$HERE/k3d-config.yaml" || k3d cluster start idps-testbed

# build and import the attacker image
docker build -t idps-testbed/attacker:latest "$HERE/../attacker"
k3d image import idps-testbed/attacker:latest -c idps-testbed

kubectl apply -f "$HERE/10-victims.yaml"
kubectl apply -f "$HERE/20-benign-load.yaml"
kubectl apply -f "$HERE/30-attacker.yaml"

echo "waiting for victims..."
kubectl -n victims rollout status deploy/node-red --timeout=180s
kubectl -n victims rollout status deploy/grafana  --timeout=180s
kubectl -n victims rollout status deploy/influxdb  --timeout=180s
kubectl -n victims wait --for=condition=Ready pod/attacker --timeout=180s

echo "cluster up. Victim ClusterIPs:"
kubectl -n victims get svc -o wide
