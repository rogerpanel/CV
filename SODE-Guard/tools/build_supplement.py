"""Build the anonymised supplementary package for reviewers.

    python tools/build_supplement.py            # -> supplementary/sode_guard_supplement.zip
    python tools/build_supplement.py --dir DIR  # also leave the unpacked tree in DIR (used to sync the mirror)

The package contains the code, tests, configurations, an anonymised Colab
notebook, the documentation of the v2 pipeline and every per-seed result file
behind the paper's tables and figures. Identifying strings are replaced, and the
build fails if any remain.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "sode_guard_supplement"

COPY = ["src", "tests", "configs", "results/v2", "requirements.txt",
        "scripts/v2_run.py", "scripts/v2_driver.py", "scripts/rebuttal_run.py",
        "notebooks/01_quickstart.ipynb", "docs/V2_RERUN.md", "tools/make_v2_colab_nb.py"]
TEXT = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".ipynb", ".cfg", ".sh"}
REPLACE = [
    (r"RobustIDPS v3", "a deployed IDPS"),
    (r"RobustIDPS internal", "internal deployment"),
    (r"robustidps_category", "deployment_category"),
    (r"RobustIDPS(\.ai)?", "the deployment platform"),
    (r"rogernickanaedevha/", "ANONYMOUS/"),
    (r"https://github\.com/rogerpanel/[A-Za-z0-9_.\-/]+", "<anonymous repository>"),
]
FORBIDDEN = re.compile(r"rogerpanel|rogernick|anaedevha|trofimov|borodachev|mephi|torchroger|gmail\.com|"
                       r"robustidps|claude\.ai|Claude-Session|Co-Authored-By", re.I)

LICENSE = """MIT License

Copyright (c) 2026 Anonymous authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

README = r"""# SODE-Guard: supplementary material (anonymised for review)

Code, configurations and every result file behind the tables and figures of
*Stochastic ODE-Guard: Robust and Certified Neural SDE Intrusion Detection*.

## Contents

| Path | What it is |
|---|---|
| `src/models/` | SODE-Guard (`sode_guard.py`), the deployed-model wrapper with post-hoc bias (`pipeline.py`), drift and diffusion networks |
| `src/sde/` | Euler--Maruyama integrator and virtual Brownian tree |
| `src/training/imbalance.py` | Module M1: logit-adjusted, focal and class-balanced losses, post-hoc logit adjustment |
| `src/training/robust.py` | Module M2: EOT-matched adversarial training (N-FGSM, PGD, TRADES); module M3: noise prefix with consistency regularisation |
| `src/certify/smoothing_sde.py` | Proposition 1: smoothing certificate for a random base classifier |
| `src/certify/mean_certificate.py`, `path_flip.py` | Theorem A with Corollary 1, Proposition 2 |
| `src/attacks/eot_autoattack.py` | PGD, EOT-PGD, APGD-CE/DLR with EOT, expected-margin attack, Square |
| `src/attacks/constrained.py` | feature-constrained attacks (future-work extension) |
| `src/evaluation/stats.py`, `operations.py` | CIs, paired bootstrap, Holm; temperature scaling, conformal sets and alarms |
| `scripts/v2_run.py` | all stages: `prepare`, `train`, `attack`, `certify`, `calibrate`, `constrained`, `aggregate` |
| `scripts/v2_driver.py` | resumable, time-budgeted multi-seed driver |
| `notebooks/v2_rerun_colab.ipynb` | Colab notebook that runs everything end to end |
| `results/v2/` | summaries and per-seed result files (see below) |
| `tests/` | unit tests (`pytest tests`) |

## Where each number in the paper comes from

* Tables 2, 4, 5 and Figures 2a and 3: `results/v2/runs/v2/summary.md` and
  `results/v2/runs/v2/seed_*/results.json` (`attack` entries; masking sweeps in seed 0).
* Table 3 and Figure 2b: the same files (`certify` entries).
* Table 6: `results/v2/runs/v2_fullcheck/` (full-strength re-attack, seed 0) and
  `results/v2/runs/v2_classtime/` (per-class chronological split, three seeds).
* Per-flow predictions used by the paired-bootstrap contrasts: `preds_attack_*.npz` and `preds_cert_*.npz`.
* After `prepare`, `python scripts/v2_run.py aggregate --out <run folder>` recomputes every
  summary from the per-seed files; the paired-bootstrap contrasts need the prepared splits.

## Reproducing

On Colab (GPU runtime):
1. Upload this zip through the Files panel.
2. Open `notebooks/v2_rerun_colab.ipynb` and place the dataset CSV files in `DATA_DIR`. Dataset links are withheld for review.
3. Choose *Runtime → Run all*.

The run is resumable and mirrors its results to Google Drive.

Locally:

```bash
pip install -r requirements.txt lightgbm xgboost
python scripts/v2_run.py prepare --csv "DATA/Containers_Dataset.csv" --label-col Label --drop-cols "Src Port" --out runs/v2
python scripts/v2_driver.py --out runs/v2 --seeds 0 1 2 3 4 --deadline $(( $(date +%s) + 86400 )) \
    --extras fullcheck ablations classtime smoothing calibrate constrained
```

Hardware: one GPU per run. A SODE-Guard model trains in about 6–13 minutes on an A100 or T4. The five-seed core with all extras takes about 15–20 GPU hours on a T4.
"""


def sanitise(text: str) -> str:
    for pat, rep in REPLACE:
        text = re.sub(pat, rep, text)
    return text


def build(out_dir: Path) -> Path:
    pkg = out_dir / NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    for rel in COPY:
        src = ROOT / rel
        dst = pkg / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp.*"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    nb = pkg / "notebooks" / "v2_rerun_colab.ipynb"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_v2_colab_nb.py"), str(nb), "--anonymous"],
                   check=True, capture_output=True)
    (pkg / "LICENSE").write_text(LICENSE)
    (pkg / "README.md").write_text(README)
    (pkg / "pyproject.toml").write_text('[project]\nname = "sode-guard"\nversion = "2.1"\n'
                                         'description = "Robust and certified neural SDE intrusion detection"\n'
                                         'authors = [{name = "Anonymous authors"}]\nrequires-python = ">=3.10"\n')
    bad = []
    for f in pkg.rglob("*"):
        if f.is_file() and f.suffix in TEXT:
            t = sanitise(f.read_text(errors="ignore"))
            f.write_text(t)
            for i, line in enumerate(t.splitlines(), 1):
                if FORBIDDEN.search(line):
                    bad.append(f"{f.relative_to(pkg)}:{i}: {line.strip()[:120]}")
    if bad:
        raise SystemExit("identifying strings remain:\n" + "\n".join(bad))
    return pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="keep the unpacked package in this directory")
    ap.add_argument("--zip", default=str(ROOT / "supplementary" / f"{NAME}.zip"))
    a = ap.parse_args()
    work = Path(a.dir) if a.dir else Path(tempfile.mkdtemp())
    pkg = build(work)
    z = Path(a.zip)
    z.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(work))
    n = sum(1 for f in pkg.rglob("*") if f.is_file())
    print(f"{z}: {n} files, {z.stat().st_size / 1e6:.1f} MB; no identifying strings found")


if __name__ == "__main__":
    main()
