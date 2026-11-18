#!/usr/bin/env bash
# Replace the contents of the anonymised mirror repository with the reviewer
# package built by tools/build_supplement.py, then push. anonymous.4open.science
# serves the mirror, so the anonymous link shows the new files after its next
# refresh (or after "Refresh" in its dashboard).
#
# Needs push access to the mirror. In Colab, store a GitHub token as the secret
# GITHUB_TOKEN and run:  !GITHUB_TOKEN=$(python -c "from google.colab import userdata;print(userdata.get('GITHUB_TOKEN'))") bash tools/sync_anonymous_mirror.sh
set -euo pipefail
MIRROR="${MIRROR:-rogerpanel/SODE-ExtractGuard-Models}"
BRANCH_SRC="${BRANCH_SRC:-claude/implement-ode-guard-A39oM}"
WORK="$(mktemp -d)"
AUTH=""
if [ -n "${GITHUB_TOKEN:-}" ]; then AUTH="x-access-token:${GITHUB_TOKEN}@"; fi
git clone -q --depth 1 -b "$BRANCH_SRC" https://github.com/rogerpanel/CV.git "$WORK/cv"
python3 "$WORK/cv/SODE-Guard/tools/build_supplement.py" --dir "$WORK/pkg" --zip "$WORK/supplement.zip"
git clone -q "https://${AUTH}github.com/${MIRROR}.git" "$WORK/mirror"
cd "$WORK/mirror"
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$WORK/pkg/sode_guard_supplement/." .
git add -A
if git diff --cached --quiet; then echo "mirror already up to date"; exit 0; fi
git -c user.name="anonymous" -c user.email="anonymous@example.com" commit -q -m "Code and results of the revision"
git push -q origin HEAD
echo "pushed to $MIRROR:"; git --no-pager log -1 --stat | tail -5
