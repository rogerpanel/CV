#!/usr/bin/env bash
# Reproduce the full Tables 2–4 and Figure 5 of the manuscript.
#
# Sweeps  benchmarks ∈ {ics3d, iis3d, ids_pqc}
#       × seeds      ∈ {42, 137, 271, 1729, 2026}
#       × ε ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.10}  for PGD-40.
#
# Expects an Ampere-class GPU. Pass GPU=1 BATCH=128 EPOCHS=40 to override.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/sode_guard.yaml}"
RESULTS="${RESULTS:-experiments/sode_guard_tnnls_v3}"
mkdir -p "$RESULTS"

for BENCH in ics3d iis3d ids_pqc; do
    for SEED in 42 137 271 1729 2026; do
        echo ">>> Training SODE-Guard on $BENCH (seed=$SEED)"
        python -m src.training.train --config "$CONFIG" \
            --benchmark "$BENCH" --seed "$SEED"
    done
done

for BENCH in ics3d iis3d ids_pqc; do
    for SEED in 42 137 271 1729 2026; do
        CKPT="$RESULTS/$BENCH/seed-$SEED/best.pt"
        [[ -f "$CKPT" ]] || { echo "Missing $CKPT, skipping"; continue; }
        python -m src.evaluation.run_eval --config "$CONFIG" \
            --checkpoint "$CKPT" --benchmark "$BENCH" \
            --attacks pgd40 fgsm cw deepfool \
            --epsilons 0.005 0.01 0.02 0.03 0.05 0.10 \
            --certify --chaos-degree 4
    done
done

echo "Aggregating tables → $RESULTS/tables/"
python scripts/aggregate_results.py --root "$RESULTS"
