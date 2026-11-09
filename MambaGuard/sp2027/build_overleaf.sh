#!/usr/bin/env bash
# Build sp2027/dist/paper-overleaf.zip (paper source + figure sources + ready-made figure PDFs/PNGs).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/dist/overleaf"
rm -rf "$OUT" && mkdir -p "$OUT/figures/pdf"
# ready-made figures
( cd "$HERE/figures"
  for f in fig1-framework fig2-encoder-ablation fig3-template-audit fig4-radar-generalisation fig5-alarms-calibration fig6-certified-accuracy; do
    w=505.89pt; case $f in fig2*|fig6*) w=243.91pt;; esac
    for i in 1 2; do pdflatex -interaction=nonstopmode -jobname=$f "\def\figname{$f}\def\figwidth{$w}\input{standalone}" >/dev/null; done
    mv $f.pdf pdf/; rm -f $f.aux $f.log
    pdftoppm -r 300 -png -singlefile pdf/$f.pdf pdf/$f
  done )
cp "$HERE/MambaGuard_sp2027.tex" "$OUT/"
cp "$HERE/figures/"*.tex "$OUT/figures/"
cp "$HERE/figures/pdf/"*.pdf "$HERE/figures/pdf/"*.png "$OUT/figures/pdf/"
cp "$HERE/OVERLEAF_README.md" "$OUT/README.md"
# self-containment check: compile the copy
( cd "$OUT" && for i in 1 2; do pdflatex -interaction=nonstopmode MambaGuard_sp2027.tex >/dev/null; done
  grep -q "^!" MambaGuard_sp2027.log && { echo "FAIL: compile errors" >&2; exit 1; }
  echo "compiled: $(grep -o 'Output written.*' MambaGuard_sp2027.log)"
  rm -f MambaGuard_sp2027.aux MambaGuard_sp2027.log MambaGuard_sp2027.out MambaGuard_sp2027.pdf )
( cd "$HERE/dist" && rm -f paper-overleaf.zip && cd overleaf && zip -qr ../paper-overleaf.zip . )
rm -rf "$OUT"
echo "built: $HERE/dist/paper-overleaf.zip ($(du -h "$HERE/dist/paper-overleaf.zip" | cut -f1))"
