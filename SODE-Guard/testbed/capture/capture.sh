#!/usr/bin/env bash
# Capture one pcap per run on the cluster bridge. The bridge name is derived from
# the internal docker network so we only ever see testbed traffic.
set -euo pipefail
OUT=${1:?usage: capture.sh RUN_OUT_DIR [DURATION_SEC]}
DUR=${2:-0}       # 0 = until stopped (SIGINT); else fixed duration
NET=${IDPS_NET:-idps-internal}

mkdir -p "$OUT"
# find the host bridge interface for the internal docker network
BRIDGE="br-$(docker network inspect "$NET" -f '{{.Id}}' | cut -c1-12)"
if ! ip link show "$BRIDGE" >/dev/null 2>&1; then
  # k3d uses its own bridge; fall back to the k3d network bridge
  BRIDGE=$(ip -o link | awk -F': ' '/br-/{print $2}' | head -1)
fi
echo "capturing on $BRIDGE -> $OUT/capture.pcap"

args=(-i "$BRIDGE" -w "$OUT/capture.pcap" -s 0 -n)
if [ "$DUR" -gt 0 ]; then
  timeout "$DUR" tcpdump "${args[@]}" || true
else
  echo $$ > "$OUT/capture.pid"
  exec tcpdump "${args[@]}"
fi
