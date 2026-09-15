#!/bin/bash
# Build Overleaf-ready zips for both manuscripts. Run from the repository root of mlmh-appraisal.
set -e
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-dist}")
mkdir -p "$OUT"
rm -f "$OUT"/paperA_review_overleaf.zip "$OUT"/paperB_empirical_overleaf.zip "$OUT"/integrated_AB_overleaf.zip
( cd paper/review && zip -qr "$OUT/paperA_review_overleaf.zip" manuscript_A.tex prisma_counts.tex prisma_flow_template.tex protocol.md prospero_registration.md search_log.csv P1_extraction_appraisal.xlsx )
( cd paper/empirical && zip -qr "$OUT/paperB_empirical_overleaf.zip" manuscript.tex supplement_tripod_checklist.tex tables figures )
# Integrated paper: flatten the ../empirical references so the zip is self-contained on Overleaf
TMP=$(mktemp -d); mkdir -p "$TMP/tables" "$TMP/figures"
sed -e 's|\\graphicspath{.*}|\\graphicspath{{figures/}}|' -e 's|\\makeatletter\\def\\input@path{.*}\\makeatother||' paper/integrated/manuscript_AB.tex > "$TMP/manuscript_AB.tex"
for f in paper/integrated/*.tex; do [ "$(basename "$f")" = manuscript_AB.tex ] || cp "$f" "$TMP/"; done
cp paper/integrated/part1_studies.csv paper/integrated/PART1_INSTRUCTIONS.md "$TMP/"
cp paper/empirical/tables/*.tex "$TMP/tables/"; cp paper/empirical/figures/*.pdf paper/integrated/figures/*.pdf "$TMP/figures/"; cp paper/empirical/supplement_*.tex "$TMP/"
( cd "$TMP" && zip -qr "$OUT/integrated_AB_overleaf.zip" . )
rm -rf "$TMP"
ls -la "$OUT"
