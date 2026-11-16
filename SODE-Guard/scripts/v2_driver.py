"""Time-budgeted, multi-GPU driver for ``v2_run.py`` (built for Kaggle).

Runs the whole v2 plan inside a fixed time budget and stops cleanly before it
runs out, so a Kaggle "Save & Run All" job (12-hour limit) always finishes and
saves its output. Every stage of ``v2_run.py`` skips finished work, so the next
run (with this run's output attached as input) carries on where this one
stopped.

Seeds are spread over the visible GPUs (Kaggle's "GPU T4 x2" gives two), one
process per GPU. Per seed the order is train -> attack -> certify, so finished
seeds accumulate complete results. Then ``aggregate`` runs, followed by the
optional extras (ablations, full-profile attack check on seed 0, per-class
chronological split).

    python scripts/v2_driver.py --out /kaggle/working/v2 --deadline <unix time>
"""
from __future__ import annotations

import argparse
import json
import shlex
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = [sys.executable, str(HERE / "v2_run.py")]
ABLATIONS = ("sode_noposthoc sode_perclass sode_ce sode_focal sode_cb sode_sampler sode_quant "
             "sode_at_pgd sode_at_trades mlp_at_pgd ode_rs sdetgnn").split()
STATUS: dict[str, str] = {}
LOCK = threading.Lock()
SYNC_LOCK = threading.Lock()
SYNC: dict = {"out": None, "dst": None}       # set in main() when --sync-to is given


def sync_tree(src: Path, dst: Path) -> int:
    """Copy new or changed files from src to dst (temporary file, then rename).
    Half-written ``.tmp.`` files are never copied. Returns the number copied."""
    n = 0
    if not src.is_dir():
        return 0
    for f in src.rglob("*"):
        if not f.is_file() or ".tmp." in f.name or "__pycache__" in f.parts:
            continue
        d = dst / f.relative_to(src)
        try:
            st = f.stat()
            if d.exists():
                ds = d.stat()
                if ds.st_size == st.st_size and ds.st_mtime >= st.st_mtime - 1:
                    continue
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(d.name + ".tmp.sync")
            shutil.copy2(f, tmp)
            tmp.replace(d)
            n += 1
        except (FileNotFoundError, OSError) as e:      # file replaced while copying: next sync gets it
            print(f"[{now()}] sync skipped {f.name}: {e}", flush=True)
    return n


def _readable(f: Path) -> bool:
    try:
        if f.suffix == ".pt":
            import torch
            torch.load(f, map_location="cpu")
        elif f.suffix == ".pkl":
            import pickle
            with open(f, "rb") as fh:
                pickle.load(fh)
        elif f.suffix == ".npz":
            import numpy as np
            with np.load(f) as z:
                for k in z.files:
                    z[k]
        elif f.suffix == ".json":
            json.loads(f.read_text())
        return True
    except Exception:
        return False


def check_tree(local: Path, remote: Path | None = None) -> list[str]:
    """Set aside unreadable checkpoints, predictions and results (e.g. from a
    session that died mid-write) and forget everything derived from them, so
    the next run redoes that work. Returns the bad paths."""
    bad = []
    if not local.is_dir():
        return bad
    for f in sorted(local.rglob("*")):
        if not f.is_file() or f.suffix not in (".pt", ".pkl", ".npz", ".json") or ".tmp." in f.name or "logs" in f.parts:
            continue
        if _readable(f):
            continue
        rel = f.relative_to(local)
        bad.append(str(rel))
        for root in (local, remote):
            if root is not None and (root / rel).exists():
                (root / rel).replace(root / rel.with_name(rel.name + ".corrupt"))
        model = f.stem
        if f.name.startswith(("preds_attack_", "preds_cert_")):   # forget the stage that wrote it
            sec, model = ("attack", f.stem[len("preds_attack_"):]) if f.name.startswith("preds_attack_") \
                else ("certify", f.stem[len("preds_cert_"):])
            res = f.parent / "results.json"
            if res.exists() and _readable(res):
                r = json.loads(res.read_text())
                r.get(sec, {}).pop(model, None)
                res.write_text(json.dumps(r, indent=2))
        if f.suffix in (".pt", ".pkl"):          # drop results and predictions of that model in this seed
            res = f.parent / "results.json"
            if res.exists() and _readable(res):
                r = json.loads(res.read_text())
                for sec in ("train", "attack", "certify"):
                    r.get(sec, {}).pop(model, None)
                res.write_text(json.dumps(r, indent=2))
            for pf in f.parent.glob(f"preds_*_{model}.npz"):
                pf.unlink()
                if remote is not None and (remote / pf.relative_to(local)).exists():
                    (remote / pf.relative_to(local)).unlink()
    if remote is not None:                   # make the cleaned results the persistent copy too
        sync_tree(local, remote)
    return bad


def companions(out: Path) -> list[Path]:
    return [out, out.parent / (out.name + "_fullcheck"), out.parent / (out.name + "_classtime")]


def sync_all(reason: str = ""):
    """Mirror the local results folders to the persistent folder (e.g. Google Drive)."""
    if not SYNC["dst"]:
        return
    with SYNC_LOCK:
        n = sum(sync_tree(src, Path(SYNC["dst"]) / src.name) for src in companions(Path(SYNC["out"])))
    if n:
        print(f"[{now()}] synced {n} file(s) to {SYNC['dst']} {reason}", flush=True)


def write_progress(final: bool = False):
    if SYNC["out"]:
        p = Path(SYNC["out"]) / "progress.json"
        tmp = p.with_name("progress.tmp.json")
        tmp.write_text(json.dumps({"updated": time.ctime(), "finished": final, "status": STATUS}, indent=2))
        tmp.replace(p)


def now():
    return time.strftime("%H:%M:%S")


def run(cmd: list[str], gpu: int | None, log: Path, deadline: float, margin: float, tag: str) -> bool:
    left = deadline - time.time() - margin
    if left < 60:
        with LOCK:
            STATUS[tag] = "skipped (no time left)"
        return False
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with LOCK:
        STATUS[tag] = "running"
        print(f"[{now()}] start {tag} (gpu {gpu}, {left / 3600:.1f} h left)", flush=True)
        write_progress()
    with open(log, "a") as fh:
        fh.write(f"\n### {time.ctime()} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE.parent)
        try:
            rc = p.wait(timeout=left)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            with LOCK:
                STATUS[tag] = "stopped at the time budget (resumes next run)"
                print(f"[{now()}] {tag}: stopped at the time budget", flush=True)
                write_progress()
            sync_all("at the time budget")
            return False
    with LOCK:
        STATUS[tag] = "done" if rc == 0 else f"failed (exit {rc}, see {log.name})"
        print(f"[{now()}] {tag}: {STATUS[tag]}", flush=True)
        write_progress()
    sync_all(f"after {tag}")
    return rc == 0


def seed_jobs(a, seed, models=None, certify_models=None, out=None, extra=()):
    out = out or a.out
    m = ["--models", *models] if models else []
    cm = ["--models", *certify_models] if certify_models else m
    prof = ["--profile", a.profile]
    extra = (*extra, *shlex.split(a.extra_args))
    return [
        (f"train seed {seed}", RUN + ["train", "--out", out, "--seeds", str(seed), *m, *extra]),
        (f"attack seed {seed}", RUN + ["attack", "--out", out, "--seeds", str(seed), *prof, *m, *extra]),
        (f"certify seed {seed}", RUN + ["certify", "--out", out, "--seeds", str(seed), *prof, *cm, *extra]),
    ]


def run_groups(groups: dict[int | None, list], a, logs: Path, label: str):
    def worker(gpu, jobs):
        for tag, cmd in jobs:
            ok = run(cmd, gpu, logs / f"{label}_gpu{gpu}.log", a.deadline, a.margin, f"{label}: {tag}")
            if not ok and STATUS[f"{label}: {tag}"].startswith(("stopped", "skipped")):
                return
    threads = [threading.Thread(target=worker, args=(g, j)) for g, j in groups.items() if j]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=a.heartbeat)
        if any(t.is_alive() for t in threads):
            tail(logs)
            sync_all("(periodic)")


def tail(logs: Path, n=3):
    """Print the last informative lines of every log (Kaggle shows stdout in the run log)."""
    for f in sorted(logs.glob("*.log")):
        lines = [ln.rstrip() for ln in f.read_text(errors="ignore").splitlines()
                 if ln.strip() and " epoch " not in ln and not ln.startswith("###")]
        for ln in lines[-n:]:
            print(f"   {f.stem}: {ln[:160]}", flush=True)


def split_seeds(seeds, gpus):
    groups = {g: [] for g in gpus}
    for i, s in enumerate(seeds):
        groups[gpus[i % len(gpus)]].append(s)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=None, help="unix time by which every process must have stopped")
    ap.add_argument("--sync-to", default=None, help="persistent folder (e.g. on Google Drive) mirrored after every job")
    ap.add_argument("--sync-only", action="store_true", help="mirror --out to --sync-to once and exit")
    ap.add_argument("--restore", action="store_true", help="copy --sync-to back into --out (and companions) and exit")
    ap.add_argument("--margin", type=float, default=600, help="seconds kept free before the deadline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--profile", default="lite")
    ap.add_argument("--extras", nargs="*", default=[], choices=["ablations", "fullcheck", "classtime"])
    ap.add_argument("--gpus", type=int, default=None, help="number of GPUs to use (default: all visible)")
    ap.add_argument("--heartbeat", type=float, default=600)
    ap.add_argument("--extra-args", default="", help="extra v2_run.py options for every train/attack/certify call")
    a = ap.parse_args()
    out = Path(a.out)
    SYNC.update(out=str(out), dst=a.sync_to)
    if a.restore or a.sync_only:
        if not a.sync_to:
            raise SystemExit("--restore/--sync-only need --sync-to")
        if a.restore:
            n = sum(sync_tree(Path(a.sync_to) / d.name, d) for d in companions(out))
            print(f"restored {n} file(s) from {a.sync_to}")
            bad = [b for d in companions(out) for b in check_tree(d, Path(a.sync_to) / d.name)]
            print(f"unreadable files set aside (their work will be redone): {bad}" if bad else "all restored files readable")
        else:
            sync_all("(sync-only)")
        return
    if a.deadline is None:
        raise SystemExit("--deadline is required")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    n = min(n, a.gpus) if a.gpus else n
    gpus = list(range(n)) or [None]
    print(f"[{now()}] {len(gpus)} GPU worker(s); budget ends {time.ctime(a.deadline)}", flush=True)

    # 1. core models, seeds spread over GPUs
    by_gpu = split_seeds(a.seeds, gpus)
    run_groups({g: [j for s in ss for j in seed_jobs(a, s)] for g, ss in by_gpu.items()}, a, logs, "core")
    run(RUN + ["aggregate", "--out", a.out], None, logs / "aggregate.log", a.deadline, 60, "core: aggregate")

    # 2. extras, only once the time allows
    if "ablations" in a.extras:
        run_groups({g: [j for s in ss for j in seed_jobs(a, s, ABLATIONS, ["ode_rs"])] for g, ss in by_gpu.items()},
                   a, logs, "ablations")
        run(RUN + ["aggregate", "--out", a.out], None, logs / "aggregate.log", a.deadline, 60, "ablations: aggregate")
    extra_groups = {g: [] for g in gpus}
    if "fullcheck" in a.extras and (out / "seed_0" / "results.json").exists():
        full = out.parent / (out.name + "_fullcheck")
        (full / "seed_0").mkdir(parents=True, exist_ok=True)
        for f in ("full.npz", "meta.json"):
            if not (full / f).exists():
                shutil.copy2(out / f, full / f)
        for f in (out / "seed_0").iterdir():
            if f.suffix in (".pt", ".pkl") or f.name in ("data.npz", "split.json"):
                if not (full / "seed_0" / f.name).exists():
                    shutil.copy2(f, full / "seed_0" / f.name)
        extra_groups[gpus[0]] += [
            ("fullcheck attack seed 0", RUN + ["attack", "--out", str(full), "--seeds", "0", "--profile", "full",
                                               "--models", "sode", "sode_at", "ode_at", "mlp_at", "mlp", "lgbm"]),
            ("fullcheck certify seed 0", RUN + ["certify", "--out", str(full), "--seeds", "0", "--profile", "full",
                                                "--models", "sode_rs", "mlp_rs"]),
            ("fullcheck aggregate", RUN + ["aggregate", "--out", str(full), "--n-boot", "200"])]
    if "classtime" in a.extras:
        ct = out.parent / (out.name + "_classtime")
        ct.mkdir(parents=True, exist_ok=True)
        for f in ("full.npz", "meta.json"):
            if not (ct / f).exists():
                shutil.copy2(out / f, ct / f)
        models = ["sode", "ode", "mlp", "lgbm", "sode_at", "mlp_at"]
        g = gpus[-1]
        for s in (0, 1, 2):
            extra_groups[g] += [(f"classtime {t}", c) for t, c in seed_jobs(a, s, models, out=str(ct),
                                                              extra=("--split", "class_time", "--sanity-seeds"))
                                if not t.startswith("certify")]
        extra_groups[g].append(("classtime aggregate", RUN + ["aggregate", "--out", str(ct), "--n-boot", "500"]))
    if any(extra_groups.values()):
        run_groups(extra_groups, a, logs, "extras")

    write_progress(final=True)
    print("\n=== status ===")
    for k, v in STATUS.items():
        print(f"{v:48s} {k}")
    pending = [k for k, v in STATUS.items() if v != "done"]
    sync_all("(final)")
    print("\nALL DONE" if not pending else f"\n{len(pending)} job(s) not finished: run the notebook again: it resumes from the saved results.")


if __name__ == "__main__":
    main()
