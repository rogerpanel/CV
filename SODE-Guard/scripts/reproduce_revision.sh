#!/usr/bin/env bash
# Experiments E1–E4 of the revised paper (needs trained checkpoints and a GPU).
#   CKPT_DIR     directory produced by scripts/reproduce_paper.sh
#   ENS7B_CKPT   state dict of the strongest non-SDE baseline (transfer / E1 rows)
#   SMOOTH_CKPT  state dict of a Gaussian-noise-trained base classifier (E4)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/sode_guard.yaml}"
CKPT_DIR="${CKPT_DIR:-experiments/sode_guard_tnnls_v3}"
OUT="${OUT:-experiments/revision}"
BENCH="${BENCH:-ics3d}"
mkdir -p "$OUT"

EXTRA_E1=()
[[ -n "${ENS7B_CKPT:-}" ]] && EXTRA_E1+=(--baseline "ens7b=$ENS7B_CKPT")

for SEED in 42 137 271 1729 2026; do
    CKPT="$CKPT_DIR/$BENCH/seed-$SEED/best.pt"
    [[ -f "$CKPT" ]] || { echo "missing $CKPT"; continue; }
    python scripts/run_revision_experiments.py e4 --config "$CONFIG" --checkpoint "$CKPT" \
        --benchmark "$BENCH" --seed "$SEED" --out "$OUT" \
        ${SMOOTH_CKPT:+--smoothing-base "ens7b=$SMOOTH_CKPT"}
    python scripts/run_revision_experiments.py e1 --config "$CONFIG" --checkpoint "$CKPT" \
        --benchmark "$BENCH" --seed "$SEED" --out "$OUT" "${EXTRA_E1[@]}"
done

CKPT="$CKPT_DIR/$BENCH/seed-42/best.pt"
python scripts/run_revision_experiments.py e2 --config "$CONFIG" --checkpoint "$CKPT" \
    --benchmark "$BENCH" --out "$OUT" ${ENS7B_CKPT:+--surrogate "ens7b=$ENS7B_CKPT"}
python scripts/run_revision_experiments.py e3 --config "$CONFIG" --benchmark "$BENCH" --out "$OUT"
echo "Results and LaTeX rows written under $OUT"
