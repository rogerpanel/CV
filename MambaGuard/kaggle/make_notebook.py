"""Write kaggle/mambaguard_kaggle.ipynb (version 4.1: runs on a Kaggle T4, a Kaggle TPU VM's CPU host, or Colab).

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
# Study notebook (version 4.3): Google Colab, Kaggle GPU or Kaggle TPU VM

Two parts, run in this order: **detectors** (ProtectAI, PIGuard, Prompt-Guard-2, a Qwen2.5-1.5B judge, a
fine-tuned DeBERTa-v3-small and the lexical baseline on the same splits, 31 held-out folds and 4 cross-benchmark
transfers) and **neural** (the five-seed SSM-GAT study with saved per-episode scores).

### Google Colab (pay as you go)
1. *Runtime -> Change runtime type ->* **T4 GPU** (about 4.5 h for everything) or **L4 GPU** (faster).
2. Optional, for Prompt-Guard-2: key icon on the left -> add secret `HF_TOKEN` (a Hugging Face read token after
   accepting the model's licence) and switch on *Notebook access*.
3. *Runtime -> Run all.* Cell 1 asks you to allow Google Drive: results go to
   **My Drive / agent_protocol_study**, saved job by job.
4. Keep the browser tab open (and the computer awake) until cell 9 prints `wrote .../results_v4_bundle.zip`.
   If the runtime disconnects, reconnect and **Run all** again with `RUN_SMOKE = False`: finished jobs are
   skipped automatically.
5. Download `results_v4_bundle.zip` from Google Drive and send it.

### Kaggle (GPU T4 x2 or TPU v5e-8)
Run cells 0-7 by hand, then **Save Version -> Save & Run All (Commit)**; download `results_v4_bundle.zip` from the
version's **Output**. To continue a stopped version: **+ Add Input -> Notebook Output**, set `RESUME_FROM` to its
path, commit again.
""")

md("## 0. Settings")
code("""
PART        = "both"   # "both" (detectors, then neural), "detectors" or "neural"
RUN_SMOKE   = True     # 10-min check in session 1; False when resuming
DET_H       = 4.5      # time budget of the detector part (hours)
NEURAL_H    = 4.0      # time budget of the neural part (hours)
RESUME_FROM = ""       # Kaggle only: an attached earlier version (Colab resumes from Google Drive by itself)
""")

md("## 1. Machine")
code("""
import os, sys, subprocess, importlib.util
# Colab images also contain a /kaggle folder, so test for Colab first and use Kaggle's own environment variable
ON_COLAB = importlib.util.find_spec("google.colab") is not None
ON_KAGGLE = (not ON_COLAB) and "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if ON_KAGGLE:
    WORK = "/kaggle/working"
elif ON_COLAB:   # write to Google Drive: survives disconnects, and re-running resumes where it stopped
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/agent_protocol_study"
else:
    WORK = os.path.abspath("study_outputs")
os.makedirs(WORK, exist_ok=True)
if importlib.util.find_spec("torch") is None:   # some TPU images ship without PyTorch: CPU build is enough
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
import torch
GPU = torch.cuda.is_available()
print("platform:", "Kaggle" if ON_KAGGLE else "Colab" if ON_COLAB else "other", "| work dir:", WORK)
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if GPU else "none (CPU run)",
      "| CPU cores:", os.cpu_count())
if not GPU:
    torch.set_num_threads(os.cpu_count() or 1)
    flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
    print("CPU run; native bfloat16:", "amx_bf16" in flags or "avx512_bf16" in flags,
          "| the neural study skips latency timing (the paper keeps the T4 latency table)")
""")

md("## 2. Code")
code("""
import glob, shutil
REPO_URL = "https://github.com/rogerpanel/CV.git"
BRANCH   = "claude/implement-mambaguard-detection-zCgu6"
CODE = None
for init in glob.glob("/kaggle/input/**/mambaguard/__init__.py", recursive=True):
    CODE = f"{WORK}/mambaguard_code"
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(os.path.dirname(os.path.dirname(init)), CODE, ignore=shutil.ignore_patterns("__pycache__"))
    print("using uploaded dataset"); break
if CODE is None and os.path.isdir("/tmp/mg_repo/MambaGuard"):   # Colab "Run all" again: refresh the clone
    shutil.rmtree("/tmp/mg_repo", ignore_errors=True)
if CODE is None:
    shutil.rmtree("/tmp/mg_repo", ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, "--filter=blob:none",
                    "--sparse", REPO_URL, "/tmp/mg_repo"], check=True)
    subprocess.run(["git", "-C", "/tmp/mg_repo", "sparse-checkout", "set", "MambaGuard"], check=True)
    CODE = "/tmp/mg_repo/MambaGuard"
    print("cloned commit", subprocess.check_output(["git", "-C", "/tmp/mg_repo", "rev-parse", "--short", "HEAD"], text=True).strip())
os.chdir(CODE); sys.path.insert(0, CODE)
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # plain LFS downloads; Xet transfers can hang
try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    elif ON_COLAB:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN found")
except Exception:
    print("no HF_TOKEN secret: Prompt-Guard-2 will be skipped")
""")

md("## 3. Packages")
code("""
!{sys.executable} -m pip install -q -r kaggle/requirements-kaggle.txt scikit-learn scipy pyyaml networkx
import transformers, sentence_transformers; print("transformers", transformers.__version__)
""")

md("## 4. Benchmarks at pinned commits (kept outside the output folder: two have no licence)")
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

md("## 6. Pre-download models")
code("""
from huggingface_hub import snapshot_download
models = ["sentence-transformers/all-MiniLM-L6-v2"]
if PART in ("detectors", "both") or RUN_SMOKE:
    models += ["protectai/deberta-v3-base-prompt-injection-v2", "leolee99/PIGuard", "leolee99/InjecGuard",
               "meta-llama/Llama-Prompt-Guard-2-86M", "Qwen/Qwen2.5-1.5B-Instruct", "microsoft/deberta-v3-small"]
for m in models:
    try:
        snapshot_download(m, token=os.environ.get("HF_TOKEN")); print("ok  ", m)
    except Exception as e:
        print("skip", m, "-", type(e).__name__)
""")

md("## 7. Smoke test (only if `RUN_SMOKE`). Must end without `ERROR` lines; its numbers are meaningless.")
code("""
NEURAL_SKIP = "--skip enc arch" + ("" if GPU else " latency")
if RUN_SMOKE:
    !python kaggle/run_kaggle_study.py --smoke --study all --data-root {DATA} --out {WORK}/results_v4_all {NEURAL_SKIP}
    !python kaggle/run_detector_baselines.py --smoke --studies main --benchmarks mcptox injecagent --data-root {DATA} --out {WORK}/results_baselines
""")

md("## 8. The run (commit the notebook for this)")
code("""
def resume_arg(name):
    if not RESUME_FROM:
        return ""
    hits = sorted(glob.glob(f"{RESUME_FROM}/**/{name}", recursive=True)) + sorted(glob.glob(f"{RESUME_FROM}/**/{name}.zip", recursive=True))
    hits = [h for h in hits if "smoke" not in h]
    print("resume", name, "from", hits[0] if hits else "nothing found")
    return f"--resume-from {hits[0]}" if hits else ""
RES_D, RES_N = resume_arg("results_baselines"), resume_arg("results_v4_all")
if PART in ("detectors", "both"):
    !python kaggle/run_detector_baselines.py --data-root {DATA} --out {WORK}/results_baselines --time-budget-h {DET_H} {RES_D}
if PART in ("neural", "both"):
    !python kaggle/run_kaggle_study.py --study all --data-root {DATA} --out {WORK}/results_v4_all --time-budget-h {NEURAL_H} {NEURAL_SKIP} {RES_N}
""")

md("## 9. Summaries and the single download file")
code("""
import zipfile
os.chdir(WORK)
for d in ["results_v4_all", "results_baselines"]:          # zip folders whose run was interrupted
    if os.path.isdir(d) and not os.path.exists(d + ".zip"):
        subprocess.run(f"zip -qr {d}.zip {d} -x '{d}/checkpoints/*'", shell=True, check=True)
def run(cmd, out):
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, cwd=CODE)
    open(out, "w").write(r.stdout + ("\\n" + r.stderr[-3000:] if r.returncode else ""))
    print(r.stdout[:15000])
if os.path.exists("results_v4_all.zip"):
    run(["scripts/summarise_kaggle_results.py", f"{WORK}/results_v4_all.zip"], f"{WORK}/summary_v4.md")
    run(["scripts/analyse_v3.py", f"{WORK}/results_v4_all.zip", "--json", f"{WORK}/analysis_v4.json"], f"{WORK}/analysis_v4.md")
if os.path.exists("results_baselines/summary.md"):
    shutil.copy("results_baselines/summary.md", "baselines_summary.md"); print(open("baselines_summary.md").read())
bundle = "results_v4_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["results_v4_all.zip", "results_baselines.zip", "summary_v4.md", "analysis_v4.md", "analysis_v4.json", "baselines_summary.md"]:
        if os.path.exists(f):
            z.write(f)
print("wrote", os.path.join(WORK, bundle), os.path.getsize(bundle) // 1024, "KB")
print("Kaggle: download it from this version's Output section. Colab: Files panel -> right-click -> Download.")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", sys.argv[1])
