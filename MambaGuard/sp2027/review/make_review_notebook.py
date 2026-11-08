"""Write the reviewers' Kaggle notebook (kaggle/run_on_kaggle.ipynb) for the anonymised artifact.

The notebook never names the authors or their code host: it takes the code
either from a Kaggle Dataset the reviewer attaches (the artifact zip) or from
the anonymous mirror's download link.

    python sp2027/review/make_review_notebook.py <artifact_dir>/kaggle/run_on_kaggle.ipynb
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Reproducing the GPU study on Kaggle

One committed run (about 3 h on a T4) reproduces the paper's five-seed study and prints every table of
`reproducibility/kaggle_gpu_v3/RESULTS_TRANSCRIBED.md`.

**Settings** (right-hand panel): *Accelerator* = **GPU T4 x2**, *Internet* = **On**.

**Run cells 1-7 by hand**, then **Save Version -> Save & Run All (Commit)**. Only a committed version keeps its
files: the editor's `/kaggle/working` is erased whenever the editor session restarts. When the version has
finished, open it from the notebook's *Versions* list; its **Output** section has `results_v3_all.zip`.
""")

md("## 1. Check the GPU")
code("""
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- set Accelerator to GPU T4 x2")
""")

md("""
## 2. Get the code

Either attach the artifact as a Kaggle Dataset (**+ Add Input -> Upload -> the artifact zip**), or paste the
anonymous mirror's download link into `ARTIFACT_ZIP_URL`.
""")
code("""
import glob, io, os, shutil, urllib.request, zipfile
ARTIFACT_ZIP_URL = ""   # e.g. the "Download" link of the anonymous mirror
CODE = "/kaggle/working/artifact"
shutil.rmtree(CODE, ignore_errors=True)
hits = glob.glob("/kaggle/input/**/mambaguard/__init__.py", recursive=True)
if hits:
    shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, ignore=shutil.ignore_patterns("__pycache__"))
    print("using the attached dataset")
elif glob.glob("/kaggle/input/**/*.zip", recursive=True):
    z = glob.glob("/kaggle/input/**/*.zip", recursive=True)[0]
    zipfile.ZipFile(z).extractall("/kaggle/working/_unz")
    root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/working/_unz/**/mambaguard/__init__.py", recursive=True)[0]))
    shutil.move(root, CODE); print("using", z)
elif ARTIFACT_ZIP_URL:
    data = urllib.request.urlopen(ARTIFACT_ZIP_URL).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall("/kaggle/working/_unz")
    root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/working/_unz/**/mambaguard/__init__.py", recursive=True)[0]))
    shutil.move(root, CODE); print("downloaded the artifact")
else:
    raise SystemExit("Attach the artifact zip as a dataset or set ARTIFACT_ZIP_URL")
os.chdir(CODE); sys.path.insert(0, CODE)
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
print(sorted(os.listdir(".")))
""")

md("## 3. Two extra packages (torch is already installed)")
code("""
!pip install -q -r kaggle/requirements-kaggle.txt
""")

md("""
## 4. Benchmarks at their pinned commits

Saved outside `/kaggle/working`, so they are not part of the output (two benchmarks have no licence and must not
be redistributed). The A2A sandbox is generated, no download.
""")
code("""
DATA = "/tmp/mambaguard_data"
!bash kaggle/setup_benchmarks.sh {DATA} mcptox mcp_safetybench agentdojo injecagent
""")

md("## 5. Episode counts (compare with `reproducibility/PROTOCOL_BENCHMARKS.md`)")
code("""
from mambaguard.data.protocol_traces import EPISODE_LOADERS
for name, folder in [("mcptox", "MCPTox-Benchmark"), ("mcp_safetybench", "MCPSafety"),
                     ("agentdojo", "agentdojo"), ("injecagent", "InjecAgent"), ("a2a_sandbox", "builtin")]:
    eps = EPISODE_LOADERS[name](folder if folder == "builtin" else f"{DATA}/{folder}", "all")
    att = sum(e.is_attack for e in eps)
    print(f"{name:16s} episodes={len(eps):5d}  attack={att:5d}  benign={len(eps) - att:5d}")
""")

md("## 6. Smoke test (~5 min, tiny data). It must end without `ERROR` lines; its numbers are meaningless.")
code("""
!python kaggle/run_kaggle_study.py --smoke --study all --data-root {DATA} --out /kaggle/working/results_v3_all
""")

md("""
## 7. CPU audit (~20 min): template audit of Section 4.2; needs no GPU

Its lexical AUROCs should match `reproducibility/template_audit.json` exactly.
""")
code("""
!python scripts/template_audit.py --data-root {DATA} --out /kaggle/working/template_audit.json | tail -3
""")

md("## 8. The full study (~3 h). Run it through **Save Version -> Save & Run All (Commit)**.")
code("""
!python kaggle/run_kaggle_study.py --study all --data-root {DATA} --out /kaggle/working/results_v3_all --time-budget-h 11
""")

md("## 9. Tables and pre-registered analysis (printed here; also saved next to the zip)")
code("""
!python scripts/summarise_kaggle_results.py /kaggle/working/results_v3_all.zip | tee /kaggle/working/summary_v3.md
!python scripts/analyse_v3.py /kaggle/working/results_v3_all.zip --json /kaggle/working/analysis_v3.json | tee /kaggle/working/analysis_v3.md
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", sys.argv[1])
