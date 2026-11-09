#!/usr/bin/env bash
# Regenerate the paper's results in three stages.
#
#   bash scripts/reproduce_all.sh                       # stage 1: tests + synthetic checks (CPU, ~10 min)
#   bash scripts/reproduce_all.sh --audit  data/raw     # + stage 2: template audit and calibration (CPU, ~30 min)
#   bash scripts/reproduce_all.sh --gpu    data/raw     # + stage 3: five-seed GPU study, independent detectors, analysis (one GPU, ~4.5 h)
#
# data/raw must hold the benchmark clones made by `bash kaggle/setup_benchmarks.sh data/raw`.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
AUDIT=0
GPU=0
DATA_ROOT="data/raw"

while [ $# -gt 0 ]; do
  case "$1" in
    --audit) AUDIT=1; [ $# -gt 1 ] && [ "${2#--}" = "$2" ] && { DATA_ROOT="$2"; shift; } ;;
    --gpu)   GPU=1;   [ $# -gt 1 ] && [ "${2#--}" = "$2" ] && { DATA_ROOT="$2"; shift; } ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs

echo "== [1] unit and end-to-end tests"
"$PYTHON" -m pytest tests -q -p no:cacheprovider

echo "== [1] certificates, controller, Lipschitz probe and attack checks (synthetic inputs)"
MG_OUT_DIR=outputs "$PYTHON" scripts/run_sp2027_experiments.py
echo "   -> outputs/experiments_sp2027.json (compare with reproducibility/experiments_sp2027.json)"

if [ "$AUDIT" -eq 1 ] || [ "$GPU" -eq 1 ]; then
  [ -d "$DATA_ROOT" ] || { echo "data root $DATA_ROOT not found: run  bash kaggle/setup_benchmarks.sh $DATA_ROOT" >&2; exit 1; }
fi

if [ "$AUDIT" -eq 1 ]; then
  echo "== [2] template audit (top features, feature removal, similarity to training attacks)"
  "$PYTHON" scripts/template_audit.py --data-root "$DATA_ROOT" --out outputs/template_audit.json
  echo "   -> outputs/template_audit.json (compare with reproducibility/template_audit.json)"
  echo "== [2] calibration (ECE) of the recorded three-seed predictions"
  "$PYTHON" scripts/calibration_v2.py reproducibility/kaggle_gpu_v2.zip --data-root "$DATA_ROOT" \
      --out outputs/calibration_v2.json
  echo "   -> outputs/calibration_v2.json (compare with reproducibility/calibration_v2.json)"
fi

if [ "$GPU" -eq 1 ]; then
  echo "== [3] five-seed GPU study (main runs, held-out folds, certificates, latency)"
  "$PYTHON" kaggle/run_kaggle_study.py --study all --data-root "$DATA_ROOT" --out outputs/results_v3_all
  echo "== [3] summary tables and pre-registered analysis"
  "$PYTHON" scripts/summarise_kaggle_results.py outputs/results_v3_all.zip | tee outputs/summary_v3.md
  "$PYTHON" scripts/analyse_v3.py outputs/results_v3_all.zip --json outputs/analysis_v3.json | tee outputs/analysis_v3.md
  echo "   -> compare outputs/summary_v3.md and outputs/analysis_v3.md with reproducibility/kaggle_gpu_v3/RESULTS_TRANSCRIBED.md"
  echo "== [3b] independent detectors (Prompt-Guard-2 needs HF_TOKEN; it is skipped otherwise)"
  "$PYTHON" kaggle/run_detector_baselines.py --data-root "$DATA_ROOT" --out outputs/results_baselines
  "$PYTHON" scripts/summarise_detectors.py outputs/results_v3_all.zip outputs/results_baselines.zip | tee outputs/detectors.md
fi
echo "Done."
