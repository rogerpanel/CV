#!/usr/bin/env bash
# Build the anonymised review artifact as a zip.
#
#   bash sp2027/build_artifact.sh            # from the MambaGuard/ directory
#
# Output: sp2027/dist/mambaguard-artifact.zip (+ an unpacked copy beside it).
# Fails if any identifying string or stale claim survives in the copy, or if
# the test suite fails inside the copy.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
DIST="$HERE/dist"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PYTHON="${PYTHON:-python3}"

echo "== anonymise"
bash "$HERE/anonymize.sh" "$SRC" "$WORK" >/dev/null
A="$WORK/anon_build"

echo "== prune"
find "$A" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$A" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
find "$A" -type d \( -name .pytest_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +
rm -rf "$A/docs" "$A/notebooks" "$A/outputs" "$A/data"
rm -f  "$A/scripts/run_v4_experiments.py" "$A/scripts/reproduce_table.py" \
       "$A/reproducibility/experiments.json" "$A/reproducibility/REPRODUCE.md" \
       "$A/reproducibility/ARTIFACTS.md" "$A/reproducibility/HARDWARE.md" \
       "$A/reproducibility/DATASETS_DOWNLOAD.sh" "$A/CITATION.cff"
# Kaggle notebook/guide name the code host; keep only the runner and its helpers.
rm -rf "$A/kaggle/results" "$A/kaggle/dist"
rm -f  "$A/kaggle/mambaguard_kaggle.ipynb" "$A/kaggle/KAGGLE_GUIDE.md" "$A/kaggle/build_kaggle_package.sh"
# A commit hash of the source repository would identify it.
find "$A/reproducibility" -name '*.json' -exec sed -i -E 's/"code_commit": *"[0-9a-f]+"/"code_commit": null/' {} +

echo "== scrub identifying strings"
# Platform-specific taxonomy and dataset names.
grep -rlZ -i 'robustidps' "$A" | xargs -0 -r sed -i \
  -e 's/the 34-class RobustIDPS\.ai attack taxonomy/the 34-class attack taxonomy/g' \
  -e 's/RobustIDPS\.ai 34-class attack taxonomy/34-class attack taxonomy/g' \
  -e 's/RobustIDPS-PQC/PQC-handshake/g' \
  -e 's/load_robustidps_pqc/load_pqc_handshake/g' \
  -e 's/robustidps_pqc/pqc_handshake/g'
# Author-owned dataset/weight DOIs.
grep -rlZ -E 'zenodo|kaggle/dsv' "$A" | xargs -0 -r sed -i \
  -e '/[Zz]enodo *= *"https/d' \
  -e '/"Zenodo": *"https/d' \
  -e 's#Kaggle DOI 10\.34740/kaggle/dsv/[0-9]*#(link withheld for anonymous review)#g' \
  -e 's#10\.5281/zenodo\.[0-9]*#(link withheld for anonymous review)#g'

echo "== add artifact files"
cp "$HERE/artifact/README.md"       "$A/README.md"
cp "$HERE/artifact/DATASETS.md"     "$A/reproducibility/DATASETS.md"
cp "$HERE/artifact/reproduce_all.sh" "$A/scripts/reproduce_all.sh"
chmod +x "$A/scripts/reproduce_all.sh"
"$PYTHON" "$HERE/review/make_review_notebook.py" "$A/kaggle/run_on_kaggle.ipynb" >/dev/null

echo "== verify: no identifying strings"
IDENT='anaedevha|trofimov|borodachev|mephi|robustidps|rogerpanel|torchroger|ar006|zenodo|kaggle/dsv|37\.27\.31|tnnls|lipmamba|moscow|russia|roger nick|campus\.'
if grep -rniE "$IDENT" "$A"; then echo "FAIL: identifying strings remain" >&2; exit 1; fi

echo "== verify: no stale claims or revision wording"
STALE='0\.978|1\.13 ?Mmsg|3\.7 ?ms|radius.{0,20}0\.041|sqrt\(2 ?ln N ?/ ?T\) */ *B|reviewer ?2|previous version|prior draft|IEEE Access 202[56]|s&p 2027 audit'
if grep -rniE "$STALE" "$A" --include=*.py --include=*.md --include=*.yaml --include=*.toml --include=*.sh; then
  echo "FAIL: stale claims remain" >&2; exit 1
fi

echo "== verify: tests pass inside the copy"
( cd "$A" && PYTHONPATH="$A" "$PYTHON" -m pytest tests -q -p no:cacheprovider )
find "$A" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "== pack recorded runs (keeps every folder under GitHub's 100-files-per-upload limit)"
( cd "$A/reproducibility" && for d in kaggle_gpu kaggle_gpu_v2 cpu_checks_v2; do
    [ -d "$d" ] && zip -qr "$d.zip" "$d" && rm -rf "$d"; done )
find "$A" -type d -name .pytest_cache -prune -exec rm -rf {} +
for d in "$A"/*/; do n=$(find "$d" -type f | wc -l); [ "$n" -le 100 ] || echo "note: $d has $n files"; done

echo "== zip"
rm -rf "$DIST" && mkdir -p "$DIST"
mv "$A" "$DIST/mambaguard-artifact"
( cd "$DIST" && zip -qr mambaguard-artifact.zip mambaguard-artifact )
echo "built: $DIST/mambaguard-artifact.zip ($(du -h "$DIST/mambaguard-artifact.zip" | cut -f1))"
