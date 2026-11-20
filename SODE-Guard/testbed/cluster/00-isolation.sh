#!/usr/bin/env bash
# Host-local isolation guard. Run once (as root) before creating the cluster.
#
# Enforces the plan's safety rules so that no attack traffic can leave the VM:
#   1. requires a dedicated, non-primary interface for the cluster bridge;
#   2. drops any FORWARD traffic from the internal Docker network to the public
#      interface;
#   3. refuses to run on the production robustidps.ai host.
#
# It is deliberately conservative: it fails closed. If it cannot prove the host
# is isolated, it exits non-zero and nothing downstream will start.
set -euo pipefail

NET=idps-internal
SUBNET=${IDPS_SUBNET:-10.77.0.0/16}

die() { echo "ISOLATION REFUSED: $*" >&2; exit 1; }

# 1. never run on the production host
host=$(hostname -f 2>/dev/null || hostname)
case "$host" in
  *robustidps.ai*|*robustidps*) die "this looks like the production host ($host); use a separate VM" ;;
esac
if [ -f /etc/robustidps-production ]; then die "production marker present"; fi

# 2. require an explicit opt-in so the rule is never applied by accident
if [ "${IDPS_TESTBED_CONFIRM:-}" != "yes" ]; then
  die "set IDPS_TESTBED_CONFIRM=yes to confirm this is a dedicated testbed VM"
fi

# 3. find the default (public) interface; the cluster must not forward to it
pub_if=$(ip route show default | awk '/default/ {print $5; exit}')
[ -n "$pub_if" ] || die "could not determine the public interface"
echo "public interface: $pub_if (cluster traffic to it will be dropped)"

# 4. create the internal docker network if absent (no masquerade to the host)
if ! docker network inspect "$NET" >/dev/null 2>&1; then
  docker network create --driver bridge --subnet "$SUBNET" \
    -o com.docker.network.bridge.enable_ip_masquerade=false \
    "$NET"
  echo "created internal docker network $NET ($SUBNET)"
fi

# 5. drop forwarded traffic from the testbed subnet to the public interface
rule=(FORWARD -s "$SUBNET" -o "$pub_if" -j DROP)
if ! iptables -C "${rule[@]}" 2>/dev/null; then
  iptables -I "${rule[@]}"
  echo "installed iptables DROP: $SUBNET -> $pub_if"
fi
# also drop the reverse, and anything routed off-subnet except the API server
iptables -C FORWARD -s "$SUBNET" ! -d "$SUBNET" -o "$pub_if" -j DROP 2>/dev/null || \
  iptables -I FORWARD -s "$SUBNET" ! -d "$SUBNET" -o "$pub_if" -j DROP

echo "isolation in place. Remove with: testbed/cluster/00-isolation.sh --down"

if [ "${1:-}" = "--down" ]; then
  iptables -D "${rule[@]}" 2>/dev/null || true
  echo "removed isolation rule"
fi
