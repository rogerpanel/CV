#!/usr/bin/env bash
# Regenerate the paper's numbers.
#
#   bash scripts/reproduce_all.sh                      # tests + synthetic checks (CPU)
#   bash scripts/reproduce_all.sh --full --data-root data/raw   # + benchmark runs (GPU)
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
FULL=0
DATA_ROOT="data/raw"
SEEDS="${SEEDS:-0 1 2 3 4}"

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

echo "== benchmark runs (data root: $DATA_ROOT)"
ran=0
for bench in "$DATA_ROOT"/*/; do
  name="$(basename "$bench")"
  for seed in $SEEDS; do
    out="outputs/$name/seed$seed"
    echo "-- $name seed $seed"
    if ! "$PYTHON" -m scripts.train --config configs/default.yaml \
          --dataset "$name" --data-root "$bench" --seed "$seed" --output-dir "$out"; then
      echo "   no loader for '$name' (see mambaguard/data/datasets.py DATASET_REGISTRY); skipped"
      continue
    fi
    "$PYTHON" -m scripts.evaluate --checkpoint "$out/checkpoints/best.pt" \
        --dataset "$name" --data-root "$bench" --output-dir "$out/eval"
    "$PYTHON" -m scripts.certify --checkpoint "$out/checkpoints/best.pt" \
        --output-dir "$out/certify"
    ran=$((ran + 1))
  done
done
"$PYTHON" -m scripts.benchmark_latency || true
echo "Done: $ran benchmark/seed runs under outputs/."
