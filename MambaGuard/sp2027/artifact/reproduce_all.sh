#!/usr/bin/env bash
# Regenerate the paper's numbers.
#
#   bash scripts/reproduce_all.sh                               # tests + synthetic checks (CPU)
#   bash scripts/reproduce_all.sh --full --data-root data/raw   # + benchmark runs (GPU)
#
# --full trains/evaluates/certifies on every benchmark clone found under
# --data-root (see reproducibility/PROTOCOL_BENCHMARKS.md) and on the A2A
# sandbox. EMBEDDER=hash runs the pipeline offline for a smoke test only.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
FULL=0
DATA_ROOT="data/raw"
SEEDS="${SEEDS:-0 1 2 3 4}"
EMBEDDER="${EMBEDDER:-minilm}"
EPOCHS="${EPOCHS:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1 ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "== [1/2] unit and end-to-end tests"
"$PYTHON" -m pytest tests -q -p no:cacheprovider

echo "== [2/2] certificates, controller, latency and attack checks (synthetic inputs)"
"$PYTHON" scripts/run_sp2027_experiments.py
echo "   -> reproducibility/experiments_sp2027.json"

if [ "$FULL" -eq 0 ]; then
  echo "Done. Re-run with --full --data-root <dir> for the benchmark-scale results."
  exit 0
fi

# registry name : folder under DATA_ROOT ("builtin" = generated, no download)
BENCHES="mcptox:MCPTox-Benchmark mcp_safetybench:MCPSafety agentdojo:agentdojo injecagent:InjecAgent a2a_sandbox:builtin"
extra=()
[ -n "$EPOCHS" ] && extra+=(--epochs "$EPOCHS")

ran=0
for pair in $BENCHES; do
  name="${pair%%:*}"; folder="${pair#*:}"
  root="$DATA_ROOT/$folder"; [ "$folder" = builtin ] && root=builtin
  if [ "$root" != builtin ] && [ ! -d "$root" ]; then
    echo "-- $name: $root not found, skipped (see reproducibility/PROTOCOL_BENCHMARKS.md)"
    continue
  fi
  if [ "$name" = mcpsecbench ]; then
    # 11 attack episodes: evaluation only, using the model trained on MCPTox.
    continue
  fi
  for seed in $SEEDS; do
    out="outputs/$name/seed$seed"
    echo "-- $name seed $seed"
    "$PYTHON" -m scripts.train --config configs/default.yaml --dataset "$name" --data-root "$root" \
        --seed "$seed" --output-dir "$out" --embedder "$EMBEDDER" "${extra[@]}"
    "$PYTHON" -m scripts.evaluate --checkpoint "$out/checkpoints/best.pt" \
        --dataset "$name" --data-root "$root" --split test --output-dir "$out/eval"
    "$PYTHON" -m scripts.certify --checkpoint "$out/checkpoints/best.pt" --output-dir "$out/certify"
    ran=$((ran + 1))
  done
done

if [ -d "$DATA_ROOT/MCPSecBench" ] && [ -f outputs/mcptox/seed0/checkpoints/best.pt ]; then
  echo "-- mcpsecbench (evaluation with the MCPTox model, seed 0)"
  "$PYTHON" -m scripts.evaluate --checkpoint outputs/mcptox/seed0/checkpoints/best.pt \
      --dataset mcpsecbench --data-root "$DATA_ROOT/MCPSecBench" --split all --output-dir outputs/mcpsecbench/eval
fi
"$PYTHON" -m scripts.benchmark_latency || true
echo "Done: $ran benchmark/seed runs under outputs/."
