"""Write kaggle/mambaguard_kaggle.ipynb (version 4: five-seed study with saved scores + independent detectors).

    python kaggle/make_notebook.py kaggle/mambaguard_kaggle.ipynb
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
# GPU study on Kaggle (version 4)

One committed run (about 4-4.5 h on a T4) produces **one file to download, `results_v4_bundle.zip`**, containing:

* `results_v4_all/`: the five-seed study (main + held-out folds, lexical and novelty experts, certificates,
  latency) **with every per-episode score file saved**;
* `results_baselines/`: independent detectors on the same splits, the 31 held-out folds and four cross-benchmark
  transfers: ProtectAI DeBERTa, PIGuard/InjecGuard, Prompt-Guard-2 (if a Hugging Face token is set), a Qwen2.5-1.5B
  LLM judge, a DeBERTa-v3-small fine-tuned on our training splits, and the lexical baseline;
* the printed summaries (`summary_v4.md`, `analysis_v4.md`, `baselines_summary.md`).

**Settings:** Accelerator **GPU T4 x2**, Internet **On**.
**Optional (for Prompt-Guard-2):** accept the licence on the model's Hugging Face page, create a read token, and add it
under *Add-ons -> Secrets* with the name `HF_TOKEN`. Without it that one detector is skipped.

**Run cells 1-7 by hand, then Save Version -> Save & Run All (Commit).** Only a committed version keeps its files:
the editor's `/kaggle/working` is erased when the editor session restarts. When the version finishes, open it from
the notebook's version list and download `results_v4_bundle.zip` from its **Output** section.
""")

md("## 1. Check the GPU")
code("""
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- set Accelerator to GPU T4 x2")
""")

md("## 2. Get the code (clones the branch; or attach the code zip as a Dataset)")
code("""
import os, glob, subprocess, shutil
REPO_URL = "https://github.com/rogerpanel/CV.git"
BRANCH   = "claude/implement-mambaguard-detection-zCgu6"
CODE = None
for init in glob.glob("/kaggle/input/**/mambaguard/__init__.py", recursive=True):
    CODE = "/kaggle/working/mambaguard_code"
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(os.path.dirname(os.path.dirname(init)), CODE, ignore=shutil.ignore_patterns("__pycache__"))
    print("using uploaded dataset"); break
if CODE is None:
    shutil.rmtree("/tmp/mg_repo", ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, "--filter=blob:none",
                    "--sparse", REPO_URL, "/tmp/mg_repo"], check=True)
    subprocess.run(["git", "-C", "/tmp/mg_repo", "sparse-checkout", "set", "MambaGuard"], check=True)
    CODE = "/tmp/mg_repo/MambaGuard"
    print("cloned commit", subprocess.check_output(["git", "-C", "/tmp/mg_repo", "rev-parse", "--short", "HEAD"], text=True).strip())
os.chdir(CODE); sys.path.insert(0, CODE)
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN"); print("HF_TOKEN found")
except Exception:
    print("no HF_TOKEN secret: Prompt-Guard-2 will be skipped")
""")

md("## 3. Extra packages (torch and transformers are already installed)")
code("""
!pip install -q -r kaggle/requirements-kaggle.txt
import transformers, sentence_transformers; print("transformers", transformers.__version__)
""")

md("## 4. Benchmarks at pinned commits (kept outside /kaggle/working: two have no licence)")
code("""
DATA = "/tmp/mambaguard_data"
!bash kaggle/setup_benchmarks.sh {DATA} mcptox mcp_safetybench agentdojo injecagent
""")

md("## 5. Episode counts (compare with reproducibility/PROTOCOL_BENCHMARKS.md)")
code("""
from mambaguard.data.protocol_traces import EPISODE_LOADERS
for name, folder in [("mcptox", "MCPTox-Benchmark"), ("mcp_safetybench", "MCPSafety"),
                     ("agentdojo", "agentdojo"), ("injecagent", "InjecAgent"), ("a2a_sandbox", "builtin")]:
    eps = EPISODE_LOADERS[name](folder if folder == "builtin" else f"{DATA}/{folder}", "all")
    att = sum(e.is_attack for e in eps)
    print(f"{name:16s} episodes={len(eps):5d}  attack={att:5d}  benign={len(eps) - att:5d}")
""")

md("## 6. Pre-download the models (MiniLM, the zero-shot detectors, the judge, DeBERTa-v3-small)")
code("""
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from huggingface_hub import snapshot_download
for m in ["protectai/deberta-v3-base-prompt-injection-v2", "leolee99/PIGuard", "leolee99/InjecGuard",
          "meta-llama/Llama-Prompt-Guard-2-86M", "Qwen/Qwen2.5-1.5B-Instruct", "microsoft/deberta-v3-small"]:
    try:
        snapshot_download(m, token=os.environ.get("HF_TOKEN")); print("ok  ", m)
    except Exception as e:
        print("skip", m, "-", type(e).__name__)
""")

md("## 7. Smoke test (~10 min): both studies on tiny data. It must end without `ERROR` lines.")
code("""
!python kaggle/run_kaggle_study.py --smoke --study all --data-root {DATA} --out /kaggle/working/results_v4_all
!python kaggle/run_detector_baselines.py --smoke --data-root {DATA} --out /kaggle/working/results_baselines
""")

md("""
## 8. The real run -> Save Version -> Save & Run All (Commit)

About 2.8 h for the five-seed study and 1.3 h for the independent detectors. If a session times out, attach the
version's output as input and set `RESUME_V4` / `RESUME_BL` to the folders it contains; finished jobs are skipped.
""")
code("""
RESUME_V4 = ""   # e.g. "/kaggle/input/<notebook>/results_v4_all"
RESUME_BL = ""   # e.g. "/kaggle/input/<notebook>/results_baselines"
r4 = f"--resume-from {RESUME_V4}" if RESUME_V4 else ""
rb = f"--resume-from {RESUME_BL}" if RESUME_BL else ""
!python kaggle/run_kaggle_study.py --study all --data-root {DATA} --out /kaggle/working/results_v4_all --time-budget-h 7 {r4}
!python kaggle/run_detector_baselines.py --data-root {DATA} --out /kaggle/working/results_baselines {rb}
""")

md("## 9. Summaries and the single download file")
code("""
import glob, os, subprocess, sys, zipfile
os.chdir("/kaggle/working")
CODE = next(p for p in ["/tmp/mg_repo/MambaGuard", "/kaggle/working/mambaguard_code"] if os.path.isdir(p))
for d in ["results_v4_all", "results_baselines"]:          # zip folders whose run was interrupted
    if os.path.isdir(d) and not os.path.exists(d + ".zip"):
        subprocess.run(f"zip -qr {d}.zip {d} -x '{d}/checkpoints/*'", shell=True, check=True)
def run(cmd, out):
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, cwd=CODE)
    open(out, "w").write(r.stdout + ("\\n" + r.stderr[-3000:] if r.returncode else ""))
    print(r.stdout[:15000])
if os.path.exists("results_v4_all.zip"):
    run(["scripts/summarise_kaggle_results.py", "/kaggle/working/results_v4_all.zip"], "/kaggle/working/summary_v4.md")
    run(["scripts/analyse_v3.py", "/kaggle/working/results_v4_all.zip", "--json", "/kaggle/working/analysis_v4.json"],
        "/kaggle/working/analysis_v4.md")
if os.path.exists("results_baselines/summary.md"):
    open("baselines_summary.md", "w").write(open("results_baselines/summary.md").read())
    print(open("baselines_summary.md").read())
with zipfile.ZipFile("results_v4_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["results_v4_all.zip", "results_baselines.zip", "summary_v4.md", "analysis_v4.md", "analysis_v4.json", "baselines_summary.md"]:
        if os.path.exists(f):
            z.write(f)
print("wrote /kaggle/working/results_v4_bundle.zip", os.path.getsize("results_v4_bundle.zip") // 1024, "KB")
print("Download it from this version's Output section (not from the editor).")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", sys.argv[1])
