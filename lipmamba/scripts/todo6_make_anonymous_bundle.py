#!/usr/bin/env python
"""TODO 6 — build the anonymised code bundle for anonymous.4open.science.

ICLR double-blind review requires that the linked repository not identify
the authors.  This script copies the ``lipmamba/`` tree into a zip with:

  * the LICENSE copyright holder replaced by "Anonymous Authors";
  * README / docs / setup.py stripped of author names, e-mails, affiliations,
    the robustidps.ai platform, the CV repository URL and any self-citation;
  * ``docs/ROBUSTIDPS_INTEGRATION.md`` and ``docs/MODEL_CARD.md`` omitted;
  * a final grep for identifying strings that aborts the build if any remain.

Upload the resulting zip to https://anonymous.4open.science/ and paste the
issued URL into ``\\anonrepo`` in the manuscript.

    python scripts/todo6_make_anonymous_bundle.py --out /tmp/LipMamba-ICLR27-anon.zip
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IDENTIFYING = [
    r"Anaedevha", r"Roger\s+Nick", r"rogerpanel", r"robustidps", r"RobustIDPS", r"MEPhI", r"mephi",
    r"Trofimov", r"Borodachev", r"torchroger", r"campus\.mephi", r"github\.com/rogerpanel", r"MambaShield",
    r"anaedevha20\d\d", r"Moscow",
]
OMIT = {"docs/ROBUSTIDPS_INTEGRATION.md", "docs/MODEL_CARD.md", "docs/ICLR2027_AUDIT_RESPONSE.md", "docs/ICLR2027_PAT_RESPONSE.md",
        "scripts/todo6_make_anonymous_bundle.py", "paper", ".git", "runs", "data_cache", "__pycache__",
        ".pytest_cache", "lipmamba.egg-info"}
REPLACEMENTS = [
    (r"`?(docs/)?ROBUSTIDPS_INTEGRATION\.md`?", "the deployment notes (omitted)"),
    (r"`?(docs/)?MODEL_CARD\.md`?", "the model card (omitted)"),
    (r"Copyright \(c\) \d{4} .*", "Copyright (c) 2026 Anonymous Authors"),
    (r"author=\"[^\"]*\"", 'author="Anonymous Authors"'),
    (r"https://github\.com/rogerpanel/CV/tree/[^\s)>\]]*", "https://anonymous.4open.science/r/LipMamba-ICLR27"),
    (r"https://github\.com/rogerpanel/[^\s)>\]]*", "https://anonymous.4open.science/r/LipMamba-ICLR27"),
    (r"> Roger Nick Anaedevha\..*", "> Anonymous Authors. *LipMamba.* Under review, 2026."),
    (r"Author: Roger Nick Anaedevha", "Author: Anonymous"),
    (r"author\s*=\s*\{Anaedevha, Roger Nick\}", "author = {Anonymous}"),
    (r"@article\{anaedevha2026lipmamba", "@article{anonymous2026lipmamba"),
    (r"\(MambaShield\)", ""), (r"MambaShield", "a prior selective-SSM IDS model"),
    (r"robustidps\.ai", "the deployment platform"), (r"RobustIDPS(\.ai)?", "the deployment platform"),
    (r"`robustidps_web_app/`|`integrated_ai_ids/`", "the deployment code"),
]
TEXT_EXT = {".md", ".py", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".tex", ".bib", ""}


def scrub(text: str) -> str:
    for pat, rep in REPLACEMENTS:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="LipMamba-ICLR27-anon.zip")
    ap.add_argument("--src", default=str(ROOT))
    args = ap.parse_args()
    src = Path(args.src)
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "LipMamba-ICLR27"
        shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if n in OMIT or
                                                             str(Path(d).relative_to(src) / n) in OMIT])
        for p in dst.rglob("*"):
            if p.is_file() and p.suffix in TEXT_EXT:
                try:
                    t = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                p.write_text(scrub(t), encoding="utf-8")
        leaks = []
        for p in dst.rglob("*"):
            if p.is_file() and p.suffix in TEXT_EXT:
                try:
                    t = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for pat in IDENTIFYING:
                    for m in re.finditer(pat, t, flags=re.IGNORECASE):
                        leaks.append((str(p.relative_to(dst)), pat, t[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")))
        if leaks:
            print("ABORT — identifying strings remain:", file=sys.stderr)
            for f, pat, ctx in leaks[:50]:
                print(f"  {f}: /{pat}/ …{ctx}…", file=sys.stderr)
            sys.exit(1)
        out = Path(args.out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in dst.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(dst.parent))
        print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) — no identifying strings found.")
        print("Next: upload at https://anonymous.4open.science/ and set \\anonrepo in the .tex to the issued URL.")


if __name__ == "__main__":
    main()
