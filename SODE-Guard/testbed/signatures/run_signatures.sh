#!/usr/bin/env bash
# Run Suricata (ET Open) and Zeek on a run's pcap, so the signature baselines see
# exactly the same traffic as the ML detectors (plan Q4). Writes alert JSON that
# parse_alerts.py turns into per-flow detections.
set -euo pipefail
RUN=${1:?usage: run_signatures.sh RUN_DIR}
PCAP="$RUN/capture.pcap"
[ -f "$PCAP" ] || { echo "no pcap at $PCAP" >&2; exit 1; }
mkdir -p "$RUN/suricata" "$RUN/zeek"

if command -v suricata >/dev/null; then
  suricata -r "$PCAP" -l "$RUN/suricata" \
    -S "${SURICATA_RULES:-/var/lib/suricata/rules/suricata.rules}" \
    -k none >/dev/null 2>&1 || echo "suricata returned non-zero (continuing)"
  echo "suricata -> $RUN/suricata/eve.json"
else
  echo "suricata not installed; skipping" >&2
fi

if command -v zeek >/dev/null; then
  ( cd "$RUN/zeek" && zeek -r "$PCAP" LogAscii::use_json=T >/dev/null 2>&1 ) || \
    echo "zeek returned non-zero (continuing)"
  echo "zeek -> $RUN/zeek/{conn,notice}.log"
else
  echo "zeek not installed; skipping" >&2
fi
