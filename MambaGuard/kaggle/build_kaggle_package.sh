#!/usr/bin/env bash
# Build kaggle/dist/mambaguard-kaggle.zip: the code needed on Kaggle, to upload
# as a Kaggle Dataset (Option B in KAGGLE_GUIDE.md). No benchmark data inside.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=kaggle/dist/mambaguard-kaggle.zip
STAGE="$(mktemp -d)/mambaguard-kaggle"
mkdir -p "$STAGE" kaggle/dist

cp -r mambaguard scripts configs tests "$STAGE"/
mkdir -p "$STAGE/kaggle" "$STAGE/reproducibility"
cp kaggle/run_kaggle_study.py kaggle/setup_benchmarks.sh kaggle/requirements-kaggle.txt \
   kaggle/mambaguard_kaggle.ipynb kaggle/KAGGLE_GUIDE.md "$STAGE/kaggle/"
cp reproducibility/PROTOCOL_BENCHMARKS.md reproducibility/benchmark_results.json "$STAGE/reproducibility/"
cp setup.py pyproject.toml requirements.txt LICENSE "$STAGE"/ 2>/dev/null || true
cp kaggle/KAGGLE_GUIDE.md "$STAGE/README.md"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name "*.pyc" -delete

rm -f "$OUT"
(cd "$(dirname "$STAGE")" && zip -qr - "$(basename "$STAGE")") > "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
