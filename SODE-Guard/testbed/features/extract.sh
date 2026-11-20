#!/usr/bin/env bash
# Turn a pcap into a flow CSV with CICFlowMeter.
#   docker run --rm -v $PWD/runs/<id>:/data idps-testbed/cfm /data/capture.pcap /data/flows.csv
set -euo pipefail
PCAP=${1:?usage: extract.sh PCAP OUT_CSV}
OUT=${2:?usage: extract.sh PCAP OUT_CSV}
JAR=$(find /opt/cfm/build -name "*.jar" | grep -iE "fat|all|cicflowmeter" | head -1)
[ -n "$JAR" ] || JAR=$(find /opt/cfm/build -name "*.jar" | head -1)
WORK=$(mktemp -d)
java -Djava.library.path=/usr/lib/x86_64-linux-gnu -jar "$JAR" "$PCAP" "$WORK/" || {
  echo "CICFlowMeter failed" >&2; exit 1; }
# CICFlowMeter writes <pcapname>_Flow.csv; normalise the name
csv=$(find "$WORK" -name "*.csv" | head -1)
cp "$csv" "$OUT"
echo "wrote $(wc -l < "$OUT") rows -> $OUT"
