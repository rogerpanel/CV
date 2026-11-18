"""Builds notebooks/v2_rerun_ics3d_colab.ipynb.

    python tools/make_v2_colab_nb.py notebooks/v2_rerun_ics3d_colab.ipynb
    python tools/make_v2_colab_nb.py OUT.ipynb --anonymous   # reviewer copy for the supplement
"""
import json, sys
ANON = "--anonymous" in sys.argv
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# SODE-Guard v2 on ICS3D, IIS3D and IDS-PQC (Colab, disconnect-proof)

One notebook for all datasets: choose `DATASET` in step 0. IIS3D is run per source corpus (CSE-CIC-IDS2018, UNSW-NB15, CIC-IoT-2023), because its files use different schemas; the prepare step refuses to mix schemas.

**Experiments**
* **Core** (12 models × 5 seeds): SODE-Guard with the three modules, the original recipe, neural ODE and MLP baselines (standard, adversarially trained, smoothed), and LightGBM, XGBoost and random forest. Each runs training, the lite attack suite and smoothing certificates.
* **Extras** (`EXTRAS`):
  * `calibrate`: temperature scaling, split-conformal prediction sets (marginal and per class) and conformal false-alarm thresholds, all fitted on the held-out calibration half. No retraining; minutes.
  * `constrained`: feature-constrained attacks on the saved models. Only attacker-controllable statistics may change: no `Bwd`/backward-direction features, ports, protocol, flags or window sizes. Features stay in their training domain, and integer counts change by whole units. No retraining; about 1–2 hours.
  * `fullcheck`: seed 0 re-attacked with the full-strength suite.
  * `ablations`: module ablations on 5 seeds.
  * `classtime`: a per-class chronological split.
  * `smoothing`: the consistency-weight study for the smoothed models.

**How it survives disconnects**
* The run is a background process on the runtime.
* Results go to the runtime's local disk and are mirrored to Google Drive after every job and every 5 minutes.
* Each dataset has its own Drive folder.
* After a disconnect or reset, choose **Runtime → Run all**: it restores from Drive and continues. At most the job in progress is lost.

**Time:** rough guesses on an A100:
* `calibrate` and `constrained` on ICS3D: about 2 hours.
* The core on a new dataset: about 6–12 hours, depending on its size (IIS3D has 34 classes, so it is capped at 5,000 flows per class).

Keep the tab open, and click into it now and then.
""")
md("## 0. Settings")
code(r"""
DATASET = 'ics3d'        # 'ics3d' | 'iis3d_cicids2018' | 'iis3d_unsw' | 'iis3d_ciciot2023' | 'idspqc'

DATASETS = {
    # container flows of ICS3D: core, ablations, checks and smoothing study already done -> run the new extras
    'ics3d':  dict(slug='rogernickanaedevha/integrated-cloud-security-3datasets-ics3d',
                   files='Containers_Dataset.csv', label='Label', cap=20000, drive='sode_guard_v2',
                   benign='0',   # Kaggle data card: 0 = benign traffic
                   extras=['calibrate', 'constrained']),
    # IIS3D ships three source corpora with different schemas, in three variants
    # (optimized / research_grade / ultrafast). Use one source and one variant per run:
    'iis3d_cicids2018': dict(slug='rogernickanaedevha/integrated-idps-security-3datasets',
                   files='integrated_dataset_research_grade/integrated_ids_intrusion_dataset.csv',
                   label='label', cap=5000, drive='sode_guard_v2_iis3d_cicids2018',
                   extras=['calibrate', 'constrained', 'fullcheck']),
    'iis3d_unsw': dict(slug='rogernickanaedevha/integrated-idps-security-3datasets',
                   files='integrated_dataset_research_grade/integrated_unsw_nb15_dataset.csv',
                   label='attack_cat', cap=5000, drive='sode_guard_v2_iis3d_unsw',      # benign = 'Normal'
                   extras=['calibrate', 'constrained', 'fullcheck']),
    'iis3d_ciciot2023': dict(slug='rogernickanaedevha/integrated-idps-security-3datasets',
                   files='integrated_dataset_ultrafast/integrated_ciciot2023_dataset.csv',
                   label='label', cap=5000, drive='sode_guard_v2_iis3d_ciciot2023',  # benign = 'BenignTraffic'
                   extras=['calibrate', 'constrained', 'fullcheck']),
    # IDS-PQC: TLS 1.3 and post-quantum handshake flows
    'idspqc': dict(slug='rogernickanaedevha/ids-encrypted-and-post-quantum-cryptography-datasets',
                   files='*.csv', label='', cap=5000, drive='sode_guard_v2_idspqc',
                   extras=['calibrate', 'constrained', 'fullcheck']),
}
CFG = DATASETS[DATASET]
FILES = CFG['files']     # glob inside the download folder; narrow it after step 4b if several file types exist
LABEL = CFG['label']     # '' = auto-detect (Label, label, Attack_type, Category, class, ...)
CAP = CFG['cap']         # flows kept per class
DROP = ['Src Port', 'Source Port', 'src_port', 'sport']   # client-assigned source port (a shortcut feature)
BENIGN = CFG.get('benign', '')   # benign class for alarm thresholds; '' = auto-detect (benign/normal)
EXTRAS = CFG['extras']   # any of 'calibrate', 'constrained', 'fullcheck', 'ablations', 'classtime', 'smoothing'
CORE_MODELS = None if DATASET == 'ics3d' else \
    'sode_base sode ode mlp lgbm xgb rf sode_at ode_at mlp_at sode_rs_c2 mlp_rs_c2 ode_rs_c2'.split()
SEEDS = '0 1 2 3 4'
HOURS = 23               # stop cleanly after this many hours (free Colab: <= 11.5)
RUN_SMOKE = True         # 5-10 minute check before the first long run on a dataset
DRIVE_ROOT = f"/content/drive/MyDrive/{CFG['drive']}"
LOCAL = f'/content/{DATASET}/v2'
DATA_DIR = f'/content/data/{DATASET}'
print(DATASET, '| Drive:', DRIVE_ROOT, '| extras:', EXTRAS)
""")
md("## 1. GPU check")
code(r"""
!nvidia-smi -L || echo "No GPU: Runtime -> Change runtime type -> GPU"
import torch
print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
""")
md("## 2. Code")
code(r"""
import os, subprocess
def driver_running():
    return subprocess.run(['pgrep', '-f', 'scripts/v2_driver.py'], capture_output=True).returncode == 0
if not os.path.isdir('/content/CV/SODE-Guard'):
    !git clone -q --depth 1 -b claude/implement-ode-guard-A39oM https://github.com/rogerpanel/CV.git /content/CV
elif not driver_running():
    !cd /content/CV && git pull -q --ff-only
%cd /content/CV/SODE-Guard
!git log -1 --format='code commit: %h %ci'
!pip -q install -U kaggle kagglehub lightgbm xgboost scikit-learn scipy pandas
""")
md("## 3. Google Drive and restore")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
import glob, shutil, json, time
os.makedirs(DRIVE_ROOT, exist_ok=True)
if DATASET == 'ics3d' and os.path.exists(f'{DRIVE_ROOT}/full.npz') and not os.path.exists(f'{DRIVE_ROOT}/v2/full.npz'):
    print('moving results of the earliest notebook layout into', f'{DRIVE_ROOT}/v2')   # one-off, ICS3D only
    os.makedirs(f'{DRIVE_ROOT}/v2', exist_ok=True)
    for f in os.listdir(DRIVE_ROOT):
        if not f.startswith('v2'):
            shutil.move(f'{DRIVE_ROOT}/{f}', f'{DRIVE_ROOT}/v2/{f}')
os.makedirs(LOCAL, exist_ok=True)
if not driver_running():
    !python scripts/v2_driver.py --out {LOCAL} --sync-to {DRIVE_ROOT} --restore
print('trained models available:', len(glob.glob(f'{LOCAL}/seed_*/*.pt')) + len(glob.glob(f'{LOCAL}/seed_*/*.pkl')))
""")
md(r"""
## 4. Data: Kaggle download (skipped when the prepared data is in Drive)

The token is the Colab secret `KAGGLE_API_TOKEN` (key icon, *Notebook access* on). Never paste it into a cell. Downloading uses no Kaggle GPU quota.
""")
code(r"""
if os.path.exists(f'{LOCAL}/full.npz'):
    print('prepared data present: no download needed')
else:
    from google.colab import userdata
    token = userdata.get('KAGGLE_API_TOKEN')
    assert token and token.startswith('KGAT_'), 'Add the KAGGLE_API_TOKEN secret first (key icon, left bar)'
    os.environ['KAGGLE_API_TOKEN'] = token
    os.makedirs('/root/.kaggle', exist_ok=True)
    open('/root/.kaggle/access_token', 'w').write(token); os.chmod('/root/.kaggle/access_token', 0o600)
    os.makedirs(DATA_DIR, exist_ok=True)
    single = '*' not in FILES
    cmd = ['kaggle', 'datasets', 'download', '-d', CFG['slug'], '-p', DATA_DIR, '--unzip', '-q'] + (['-f', FILES] if single else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    for z in glob.glob(f'{DATA_DIR}/**/*.zip', recursive=True):
        subprocess.run(['unzip', '-o', '-q', z, '-d', os.path.dirname(z)]); os.remove(z)
    if not glob.glob(f'{DATA_DIR}/**/{FILES}', recursive=True):
        print('kaggle CLI did not deliver the files, using kagglehub:', r.stderr[-300:])
        import kagglehub
        path = kagglehub.dataset_download(CFG['slug'])
        for f in glob.glob(f'{path}/**/*', recursive=True):
            if os.path.isfile(f):
                dst = os.path.join(DATA_DIR, os.path.relpath(f, path))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst): os.symlink(f, dst)
    for f in sorted(glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)):
        print(f'{os.path.getsize(f)/1e6:9.1f} MB  {f}')
""")
md(r"""
### 4b. Inspect the CSV files (first run on a dataset)

Lists every CSV with its number of numeric columns and candidate label columns. If the dataset holds files with different schemas, narrow `FILES` in step 0 to the flow files (for example `'*Flow*.csv'`), set `LABEL` if auto-detection picks the wrong column, then re-run steps 0 and 4b.
""")
code(r"""
import pandas as pd
CANDS = ['label','Label','LABEL','attack_cat','Attack_type','attack_type','Category','category','class','Class','Attack','attack','type']
if not os.path.exists(f'{LOCAL}/full.npz'):
    for f in sorted(glob.glob(f'{DATA_DIR}/**/{FILES}', recursive=True)):
        try:
            h = pd.read_csv(f, nrows=300, low_memory=False)
        except Exception as e:
            print('unreadable', f, e); continue
        labs = [c for c in CANDS if c in h.columns]
        ex = {c: h[c].astype(str).unique()[:6].tolist() for c in labs[:1]}
        print(f"{h.select_dtypes('number').shape[1]:3d} numeric | labels {labs} {ex} | {os.path.relpath(f, DATA_DIR)}")
""")
md("## 5. Prepare (cap per class, drop identifiers and the source port, remove duplicates)")
code(r"""
if not os.path.exists(f'{LOCAL}/full.npz'):
    lab = f"--label-col '{LABEL}'" if LABEL else ''
    drop = ' '.join(f'"{c}"' for c in DROP)
    !python scripts/v2_run.py prepare --csv "{DATA_DIR}/**/{FILES}" {lab} --drop-cols {drop} --cap-per-class {CAP} --out {LOCAL} 2>&1 | tail -5
    !python scripts/v2_driver.py --out {LOCAL} --sync-to {DRIVE_ROOT} --sync-only
meta = json.load(open(f'{LOCAL}/meta.json'))
print(json.dumps({k: meta[k] for k in ('label_column', 'classes', 'class_counts_after_dedup', 'dropped_small_classes', 'dedup')}, indent=1))
print(len(meta['raw_features']), 'raw features')
""")
md("## 6. Smoke test (first run on a dataset; its numbers mean nothing)")
code(r"""
if RUN_SMOKE and not glob.glob(f'{LOCAL}/seed_*/results.json') and not driver_running():
    TEST = f'/content/{DATASET}_smoke'
    !rm -rf {TEST} && mkdir -p {TEST} && cp {LOCAL}/full.npz {LOCAL}/meta.json {TEST}/
    core = '--core-models sode mlp lgbm sode_at mlp_at sode_rs_c2'
    !python scripts/v2_driver.py --out {TEST} --seeds 0 --deadline {time.time() + 2400} --margin 60 --heartbeat 120 {core} --extras calibrate constrained --extra-args "--epochs 1 --n-val-robust 200 --n-attack 100 --restarts 1 --long-steps 5 --eot 2 --eot-apgd 2 --apgd-iters 5 --eot-steps 5 --square-queries 20 --em-paths 16 --n-cert 50 --smooth-n 200 --sanity-seeds" | tail -16
else:
    print('skipped')
""")
md("## 7. Start the background run (core, then the extras in the order of `EXTRAS`)")
code(r"""
if driver_running():
    print('background run already active: go to step 8')
elif not torch.cuda.is_available():
    print('No GPU on this runtime: the background run was NOT started. Results on Drive are restored; step 9 still works.')
else:
    deadline = time.time() + HOURS * 3600
    cmd = ['python', 'scripts/v2_driver.py', '--out', LOCAL, '--sync-to', DRIVE_ROOT, '--seeds', *SEEDS.split(),
           '--deadline', str(deadline), '--heartbeat', '300']
    if CORE_MODELS:
        cmd += ['--core-models', *CORE_MODELS]
    if EXTRAS:
        cmd += ['--extras', *EXTRAS]
    if BENIGN:
        cmd += ['--extra-args', f'--benign-class {BENIGN}']
    log = open(f'{LOCAL}/driver.log', 'a')
    log.write(f'\n===== started {time.ctime()} =====\n'); log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd='/content/CV/SODE-Guard')
    print('started background run, pid', p.pid, '| stops by', time.ctime(deadline))
""")
md("## 8. Monitor (safe to stop this cell; the background run continues)")
code(r"""
from IPython.display import clear_output
def show():
    clear_output(wait=True)
    print(time.ctime(), '|', DATASET, '| background run active:', driver_running())
    pj = f'{LOCAL}/progress.json'
    if os.path.exists(pj):
        st = json.load(open(pj))['status']
        counts = {}
        for v in st.values():
            counts[v.split(' (')[0]] = counts.get(v.split(' (')[0], 0) + 1
        print('jobs:', counts)
        for k, v in st.items():
            if v != 'done':
                print(f'  {v[:50]:50s} {k}')
    lines = [l for l in open(f'{LOCAL}/driver.log', errors='ignore').read().splitlines() if l.strip()] if os.path.exists(f'{LOCAL}/driver.log') else []
    print('\n'.join(lines[-25:]))
    !nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
show()
while driver_running():
    time.sleep(300)
    show()
print('\nbackground run finished')
""")
md("## 9. Results: summaries, zip to Drive and download (upload the zip to the chat)")
code(r"""
import zipfile
base = os.path.dirname(LOCAL)
for name in ('v2', 'v2_fullcheck', 'v2_classtime'):
    f = f'{base}/{name}/summary.md'
    if os.path.exists(f):
        print(f'\n######## {DATASET} / {name}\n'); print(open(f).read())
zname = f'v2_results_{DATASET}.zip'
with zipfile.ZipFile(f'/content/{zname}', 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ('v2', 'v2_fullcheck', 'v2_classtime'):
        for f in glob.glob(f'{base}/{name}/**/*', recursive=True):
            if f.endswith(('.md', '.json', '.log', '.npz')) and os.path.basename(f) not in ('full.npz', 'data.npz') and '.tmp' not in f:
                z.write(f, os.path.relpath(f, base))
shutil.copy2(f'/content/{zname}', f'{DRIVE_ROOT}/{zname}')
print(f'\n{zname}:', round(os.path.getsize(f'/content/{zname}') / 1e6, 1), 'MB, also saved to', DRIVE_ROOT)
from google.colab import files
files.download(f'/content/{zname}')
""")
md("## 10. (Only if needed) stop the background run")
code(r"""
STOP = False   # set to True and run this cell to stop; finished work is already on Drive
if STOP:
    !pkill -f scripts/v2_driver.py; pkill -f scripts/v2_run.py
    !python scripts/v2_driver.py --out {LOCAL} --sync-to {DRIVE_ROOT} --sync-only
    print('stopped; running background:', driver_running())
""")

if ANON:
    for c in cells:
        src = c["source"]
        src = src.replace("rogernickanaedevha/", "ANONYMOUS/")
        if "git clone" in src:
            src = (
                "import os, subprocess, glob\n"
                "def driver_running():\n"
                "    return subprocess.run(['pgrep', '-f', 'scripts/v2_driver.py'], capture_output=True).returncode == 0\n"
                "# Reviewers: upload the supplementary zip through the Files panel (left bar) to /content, then run this cell.\n"
                "if not os.path.isdir('/content/CV/SODE-Guard'):\n"
                "    z = sorted(glob.glob('/content/*supplement*.zip'))\n"
                "    assert z, 'Upload the supplementary zip to /content first'\n"
                "    !mkdir -p /content/CV && unzip -q -o \"{z[0]}\" -d /content/CV && mv /content/CV/sode_guard_supplement /content/CV/SODE-Guard\n"
                "%cd /content/CV/SODE-Guard\n"
                "!pip -q install -U kaggle kagglehub lightgbm xgboost scikit-learn scipy pandas")
        if "single = '*' not in FILES" in src:
            src = ("# Dataset links are withheld for review. Place the dataset CSV files under DATA_DIR\n"
                   "# (Files panel, or copy them from your Drive), then run steps 4b and 5.\n"
                   "os.makedirs(DATA_DIR, exist_ok=True)\n"
                   "for f in sorted(glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)):\n"
                   "    print(f'{os.path.getsize(f)/1e6:9.1f} MB  {f}')\n"
                   "if not os.path.exists(f'{LOCAL}/full.npz') and not glob.glob(f'{DATA_DIR}/**/{FILES}', recursive=True):\n"
                   "    print('No CSV files found in', DATA_DIR)")
        if "Kaggle download" in src or "KAGGLE_API_TOKEN" in src:
            src = src.replace("## 4. Data: Kaggle download (skipped when the prepared data is in Drive)",
                              "## 4. Data (dataset links withheld for review)")
            src = src.replace("The token is the Colab secret `KAGGLE_API_TOKEN` (key icon, *Notebook access* on). Never paste it into a cell. Downloading uses no Kaggle GPU quota.",
                              "Place the CSV files of the chosen dataset under `DATA_DIR`; the prepared data is cached in Drive afterwards.")
        c["source"] = src

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print(len(cells), "cells")
