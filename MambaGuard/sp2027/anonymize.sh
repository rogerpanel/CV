#!/usr/bin/env bash
# anonymize.sh --- IEEE S&P 2027 Cycle 2 anonymization harness for MambaGuard.
# Usage: ./anonymize.sh <source_dir> [<output_root>]
# Produces <output_root>/anon_build/, a cleaned mirror of <source_dir> ready
# for anonymous.4open.science or GitFront. Idempotent.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <source_dir> [<output_root>]" >&2
    exit 2
fi

SRC="$(cd "$1" && pwd)"
OUT_ROOT="${2:-$(pwd)}"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
DEST="$OUT_ROOT/anon_build"

MARK='<!-- withheld for anonymous review -->'
HASH_MARK='# withheld for anonymous review'

AUTHOR_PATTERN='Anaedevha|Trofimov|Borodachev|MEPhI|robustidps\.ai|rogerpanel'
DOI_1='10.5281/zenodo.19129512'
DOI_2='10.34740/kaggle/dsv/15424420'
ENDPOINT='37.27.31.70'

FILES_MODIFIED=0
FILES_DELETED=0
BYTES_STRIPPED=0

note_mod() {
    FILES_MODIFIED=$((FILES_MODIFIED + 1))
    local before="$1" after="$2"
    local b a
    b=$(wc -c <"$before" | tr -d ' ')
    a=$(wc -c <"$after"  | tr -d ' ')
    if (( b > a )); then
        BYTES_STRIPPED=$((BYTES_STRIPPED + b - a))
    fi
}

echo "[anonymize] source : $SRC"
echo "[anonymize] output : $DEST"

# ---- 1. refresh the mirror (idempotent: wipe + copy) -----------------------
rm -rf "$DEST"
mkdir -p "$DEST"
# Excluded trees:
#   .git            --- carries author history
#   manuscript_v4   --- full-attribution LaTeX source; the anonymised .tex
#                       lives in the submission package, not in this mirror
#   sp2027          --- the submission package itself (contains this harness
#                       and the author's working checklists)
EXCLUDES=(
    '--exclude=.git'
    '--exclude=manuscript_v4'
    '--exclude=sp2027'
    '--exclude=kaggle/results'
    '--exclude=kaggle/dist'
    '--exclude=kaggle/mambaguard_kaggle.ipynb'
    '--exclude=kaggle/KAGGLE_GUIDE.md'
    '--exclude=kaggle/build_kaggle_package.sh'
)
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${EXCLUDES[@]}" "$SRC"/ "$DEST"/
else
    cp -a "$SRC"/. "$DEST"/
    rm -rf "$DEST/.git" "$DEST/manuscript_v4" "$DEST/sp2027" "$DEST/kaggle/results" "$DEST/kaggle/dist"
    rm -f "$DEST/kaggle/mambaguard_kaggle.ipynb" "$DEST/kaggle/KAGGLE_GUIDE.md" "$DEST/kaggle/build_kaggle_package.sh"
fi

# Record .git deletion for the summary if the source had one.
if [[ -d "$SRC/.git" ]]; then
    FILES_DELETED=$((FILES_DELETED + 1))
    echo "[anonymize] stripped: .git/"
fi

# ---- 2. LICENSE ------------------------------------------------------------
if [[ -f "$DEST/LICENSE" ]]; then
    tmp="$DEST/LICENSE.tmp"
    sed -E 's/Copyright \(c\) 2026 Roger Nick Anaedevha/Copyright (c) 2026 Anonymous Authors (for review)/' \
        "$DEST/LICENSE" >"$tmp"
    if ! cmp -s "$DEST/LICENSE" "$tmp"; then
        note_mod "$DEST/LICENSE" "$tmp"
        mv "$tmp" "$DEST/LICENSE"
        echo "[anonymize] rewrote LICENSE copyright line"
    else
        rm -f "$tmp"
    fi
fi

# ---- 3. CITATION.cff ------------------------------------------------------
if [[ -f "$DEST/CITATION.cff" ]]; then
    rm -f "$DEST/CITATION.cff"
    FILES_DELETED=$((FILES_DELETED + 1))
    echo "[anonymize] deleted CITATION.cff"
fi

# ---- 4. setup.py / pyproject.toml -----------------------------------------
strip_author_meta() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    local tmp="$f.tmp"
    # Strip author=, author_email=, maintainer=, maintainer_email= lines in
    # setup.py-style; collapse the multi-line TOML authors = [...] and
    # maintainers = [...] arrays; redact any remaining line that names the
    # authors or the deployment platform (project_urls, Homepage, ...).
    awk -v pat="$AUTHOR_PATTERN" -v ep="$ENDPOINT" '
        BEGIN { in_array = 0 }
        {
            line = $0
            if (in_array) {
                if (line ~ /\]/) { in_array = 0 }
                next
            }
            if (line ~ /^[[:space:]]*(authors|maintainers)[[:space:]]*=[[:space:]]*\[/) {
                if (line ~ /\]/) { next }
                in_array = 1
                next
            }
            if (line ~ /^[[:space:]]*(author|author_email|maintainer|maintainer_email)[[:space:]]*=/) {
                next
            }
            if (line ~ pat || index(line, ep)) {
                next
            }
            print line
        }' "$f" >"$tmp"
    if ! cmp -s "$f" "$tmp"; then
        note_mod "$f" "$tmp"
        mv "$tmp" "$f"
        echo "[anonymize] stripped author metadata from ${f#$DEST/}"
    else
        rm -f "$tmp"
    fi
}
strip_author_meta "$DEST/setup.py"
strip_author_meta "$DEST/pyproject.toml"

# ---- 5. Markdown docs: README.md and docs/*.md ----------------------------
redact_markdown() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    local tmp="$f.tmp"
    # Any line matching author pattern, DOI, or deployment endpoint -> withheld.
    awk -v pat="$AUTHOR_PATTERN" -v d1="$DOI_1" -v d2="$DOI_2" \
        -v ep="$ENDPOINT" -v mark="$MARK" '
        {
            if ($0 ~ pat || index($0, d1) || index($0, d2) || index($0, ep)) {
                print mark
            } else {
                print
            }
        }' "$f" >"$tmp"
    if ! cmp -s "$f" "$tmp"; then
        note_mod "$f" "$tmp"
        mv "$tmp" "$f"
        echo "[anonymize] redacted ${f#$DEST/}"
    else
        rm -f "$tmp"
    fi
}

if [[ -f "$DEST/README.md" ]]; then
    redact_markdown "$DEST/README.md"
fi
if [[ -d "$DEST/docs" ]]; then
    while IFS= read -r -d '' f; do
        redact_markdown "$f"
    done < <(find "$DEST/docs" -type f -name '*.md' -print0)
fi

# ---- 6. Python source: mambaguard/**/*.py, scripts/**/*.py ----------------
redact_python() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    local tmp="$f.tmp"
    # Strip __author__ = ... lines, and any comment or docstring line that
    # names the authors / institution / platform. We do not try to parse the
    # AST --- line-level redaction is adequate for the reviewer artifact.
    awk -v pat="$AUTHOR_PATTERN" -v ep="$ENDPOINT" -v mark="$HASH_MARK" '
        {
            if ($0 ~ /^[[:space:]]*__author__[[:space:]]*=/) {
                print mark
                next
            }
            # comment line containing identifier
            if ($0 ~ /^[[:space:]]*#/ && ($0 ~ pat || index($0, ep))) {
                print mark
                next
            }
            # single-line docstring containing identifier
            if ($0 ~ /^[[:space:]]*("""|'\'''\'''\'')/ && ($0 ~ pat || index($0, ep))) {
                print mark
                next
            }
            print
        }' "$f" >"$tmp"
    if ! cmp -s "$f" "$tmp"; then
        note_mod "$f" "$tmp"
        mv "$tmp" "$f"
        echo "[anonymize] redacted ${f#$DEST/}"
    else
        rm -f "$tmp"
    fi
}

for root in "$DEST/mambaguard" "$DEST/scripts"; do
    [[ -d "$root" ]] || continue
    while IFS= read -r -d '' f; do
        redact_python "$f"
    done < <(find "$root" -type f -name '*.py' -print0)
done

# ---- 7. Notebook metadata --------------------------------------------------
if [[ -d "$DEST/notebooks" ]] && command -v python3 >/dev/null 2>&1; then
    while IFS= read -r -d '' nb; do
        python3 -I - "$nb" <<'PY'
import json, sys, os
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
except Exception:
    sys.exit(0)
meta = doc.get('metadata', {}) or {}
changed = False
ks = meta.get('kernelspec')
if isinstance(ks, dict) and 'display_name' in ks:
    # Scrub any user-identifying display name (keep neutral kernel name).
    ks['display_name'] = 'Python 3'
    changed = True
if 'authors' in meta:
    meta.pop('authors', None)
    changed = True
if 'widgets' in meta:
    meta.pop('widgets', None)
    changed = True
if changed:
    doc['metadata'] = meta
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"[anonymize] scrubbed notebook metadata: {os.path.basename(path)}")
PY
    done < <(find "$DEST/notebooks" -type f -name '*.ipynb' -print0)
fi

# ---- 8. Anonymous mirror README (overwrite, idempotent) -------------------
cat >"$DEST/README.md.anon_header" <<'EOF'
# MambaGuard --- Anonymous Reviewer Mirror

This repository is the double-blind mirror of the MambaGuard codebase prepared
for IEEE S&P 2027 Cycle 2 review. All author-identifying metadata (names,
emails, institution, DOIs that resolve to known authors, deployment endpoints,
git history) has been stripped by `anonymize.sh`. Please do not attempt to
deanonymize.

Contents below are unchanged from the canonical repository except for the
redactions described above. After acceptance the full-attribution version will
be released at the archival URL given in the paper.

---

EOF
if [[ -f "$DEST/README.md" ]]; then
    cat "$DEST/README.md.anon_header" "$DEST/README.md" >"$DEST/README.md.new"
    # Avoid stacking the header on repeated runs.
    if ! head -n 1 "$DEST/README.md" | grep -q 'Anonymous Reviewer Mirror'; then
        mv "$DEST/README.md.new" "$DEST/README.md"
        FILES_MODIFIED=$((FILES_MODIFIED + 1))
        echo "[anonymize] prepended anonymous-mirror header to README.md"
    else
        rm -f "$DEST/README.md.new"
    fi
    rm -f "$DEST/README.md.anon_header"
else
    mv "$DEST/README.md.anon_header" "$DEST/README.md"
    FILES_MODIFIED=$((FILES_MODIFIED + 1))
    echo "[anonymize] created README.md"
fi

# ---- 9. Summary ------------------------------------------------------------
echo
echo "========== anonymize.sh summary =========="
echo "files modified : $FILES_MODIFIED"
echo "files deleted  : $FILES_DELETED"
echo "bytes stripped : $BYTES_STRIPPED"
echo "output         : $DEST"
echo "=========================================="

# ---- 10. Verification ------------------------------------------------------
echo "[verify] scanning anon copy for residual identifiers ..."
HITS_FILE="$(mktemp)"
trap 'rm -f "$HITS_FILE"' EXIT
# Note: we grep text files only; binary matches would be spurious.
if grep -RInE "$AUTHOR_PATTERN|$ENDPOINT" \
        --binary-files=without-match \
        --exclude-dir='.git' "$DEST" >"$HITS_FILE" 2>/dev/null; then
    echo "[verify] FAIL: residual identifiers found:"
    cat "$HITS_FILE"
    exit 1
fi
echo "[verify] OK: no residual identifiers."
exit 0
