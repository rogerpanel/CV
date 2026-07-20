#!/usr/bin/env python3
"""Package the generated artifact into a Zenodo-ready archive.

Bundles the artifact directory + configs + code + metadata into a single
versioned .zip and prints the checksum to record in the Zenodo deposition.

Usage:
  python scripts/package_for_zenodo.py --artifact ./artifact --out ./dist
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from uavbench import __version__

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="./artifact")
    ap.add_argument("--out", default="./dist")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    zip_name = f"UAV-EW-Bench-2026_v{__version__}.zip"
    zip_path = os.path.join(args.out, zip_name)

    include_files = [
        "README.md", "requirements.txt", "CITATION.cff", "LICENSE",
        "config/benchmark.yaml", "config/defenses.yaml",
    ]
    include_dirs = ["uavbench", "scripts", "tests", "docs"]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # code + configs
        for rel in include_files:
            p = os.path.join(_ROOT, rel)
            if os.path.exists(p):
                zf.write(p, os.path.join("UAV-EW-Bench-2026", rel))
        for d in include_dirs:
            base = os.path.join(_ROOT, d)
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                if "__pycache__" in root:
                    continue
                for fn in files:
                    if fn.endswith(".pyc"):
                        continue
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, _ROOT)
                    zf.write(p, os.path.join("UAV-EW-Bench-2026", rel))
        # generated artifact (the actual benchmark data)
        for root, _, files in os.walk(args.artifact):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, args.artifact)
                zf.write(p, os.path.join("UAV-EW-Bench-2026", "data", rel))

    digest = _sha256(zip_path)
    with open(zip_path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {zip_name}\n")

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"Zenodo bundle : {zip_path}  ({size_mb:.2f} MB)")
    print(f"SHA-256       : {digest}")
    print("\nUpload this .zip as the Zenodo deposition file, then paste the")
    print("resulting DOI into the dissertation (Chapter 6 + bibliography).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
