#!/bin/bash
# Build Overleaf-ready zips for both manuscripts. Run from the repository root of mlmh-appraisal.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-dist}
mkdir -p "$OUT"
rm -f "$OUT"/paperA_review_overleaf.zip "$OUT"/paperB_empirical_overleaf.zip
( cd paper/review && zip -qr "../../$OUT/paperA_review_overleaf.zip" manuscript_A.tex prisma_counts.tex prisma_flow_template.tex protocol.md prospero_registration.md search_log.csv P1_extraction_appraisal.xlsx )
( cd paper/empirical && zip -qr "../../$OUT/paperB_empirical_overleaf.zip" manuscript.tex supplement_tripod_checklist.tex tables figures )
ls -la "$OUT"
