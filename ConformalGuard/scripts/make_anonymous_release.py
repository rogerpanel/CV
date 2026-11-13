"""Export an anonymised copy of the code for double-blind review.

    python scripts/make_anonymous_release.py --out /path/to/ConformalGuard-anon

Copies the code, drops everything that identifies the authors (the manuscript
and response folders, the platform-integration document), rewrites the
remaining identifying strings, and then scans the result: if any forbidden
term is still present the script lists the hits and exits non-zero.  Publish
the output directory as its own repository and mirror it through an
anonymising service; do not point reviewers at the development repository.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["conformalguard", "configs", "scripts", "tests", "examples", "notebooks",
           "docs", "README.md", "LICENSE", "pyproject.toml", "requirements.txt", ".gitignore"]
DROP = ["docs/ROBUSTIDPS_INTEGRATION.md", "scripts/make_anonymous_release.py"]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".json", ""}

# (pattern, replacement), applied in order, case-sensitive unless flagged.
REPLACEMENTS = [
    (r"Copyright \(c\) 2026 [^\n]*", "Copyright (c) 2026 Anonymous authors"),
    (r"authors = \[.*?\]\n", 'authors = [{ name = "Anonymous" }]\n'),
    (r"[\w.+-]+@(robustidps\.ai|campus\.mephi\.ru|mephi\.ru)", "anonymous@example.org"),
    (r"https?://(www\.)?robustidps\.ai[^\s)`\"']*", "https://example.org"),
    (r"https?://github\.com/rogerpanel/[^\s)`\"']*", "https://example.org/anonymous"),
    (r"github\.com/rogerpanel/[^\s)`\"']*", "example.org/anonymous"),
    (r"https?://doi\.org/10\.5281/zenodo\.19129512", "https://example.org/anonymised-dataset"),
    (r"https?://doi\.org/10\.34740/kaggle/dsv/15424420", "https://example.org/anonymised-dataset"),
    (r"10\.5281/zenodo\.19129512|10\.34740/kaggle/dsv/15424420", "anonymised"),
    (r"Anaedevha (and|&) Trofimov[^\n\"]*", "Anonymous authors"),
    (r"RobustIDPS(\.ai)?([- ]PCAPs)?", "the host platform"),
    (r"robustidps", "hostplatform"),
    (r"integrated_ai_ids", "host_platform"),
]
FORBIDDEN = re.compile(
    r"anaedevha|trofimov|borodachev|rogerpanel|robustidps|mephi|\bICIS\b|"
    r"zenodo\.19129512|15424420", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"{out} exists and is not empty; choose a new directory", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "runs", "*.egg-info")
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=ignore)
        elif src.exists():
            shutil.copy2(src, out / name)
    for rel in DROP:
        (out / rel).unlink(missing_ok=True)

    n_changed = 0
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text
        for pat, repl in REPLACEMENTS:
            new = re.sub(pat, repl, new, flags=re.S if pat.startswith("authors") else 0)
        if new != text:
            path.write_text(new, encoding="utf-8")
            n_changed += 1

    hits = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if FORBIDDEN.search(path.name):
            hits.append(f"{path.relative_to(out)}: file name")
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    hits.append(f"{path.relative_to(out)}:{i}: {line.strip()[:100]}")
        except UnicodeDecodeError:
            hits.append(f"{path.relative_to(out)}: binary file, inspect by hand")

    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"wrote {n_files} files to {out} ({n_changed} rewritten)")
    if hits:
        print(f"\n{len(hits)} identifying string(s) remain -- NOT safe to publish:")
        print("\n".join("  " + h for h in hits))
        return 1
    print("no identifying strings found by the scan. Still read README.md and "
          "LICENSE once by eye, and check the git history of the repository you "
          "publish it from (commit author names are not scrubbed by this script).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
