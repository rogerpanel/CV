#!/usr/bin/env python3
"""Operator entry point for the problem-space testbed.

Ties the stages together and writes a run manifest for every run. Three verbs:

  collect  run benign + every scenario N times, capture, extract, label -> flows.csv
  evade    run the evasion search (gradient-guided or black-box) against a model set
  score    score an existing flows.csv against models / platform / signatures

Each scenario runs inside the attacker pod via `kubectl exec`; capture runs on the
host; feature extraction runs in the pinned CICFlowMeter container. Every verb
checks isolation first and refuses to run otherwise.

This is the orchestration layer; the attack tooling, oracle, transforms and
scorers live in their own modules and are imported by the attacker pod and the
host as appropriate. Nothing here targets anything outside the cluster.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def isolation_ok() -> bool:
    try:
        subprocess.run(["bash", str(HERE / "cluster" / "00-isolation.sh")],
                       check=True, env={"IDPS_TESTBED_CONFIRM": "yes", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
        return True
    except subprocess.CalledProcessError:
        return False


def new_run(out: Path, kind: str) -> Path:
    rid = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    d = out / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(
        {"id": rid, "kind": kind, "started": time.time(), "operator": _whoami(),
         "windows": []}, indent=2))
    return d


def _whoami() -> str:
    import getpass
    return getpass.getuser()


def add_window(run: Path, scenario: str, family: str, attacker_ip: str, t0: float, t1: float):
    m = json.loads((run / "manifest.json").read_text())
    m["windows"].append({"scenario": scenario, "family": family,
                         "attacker_ip": attacker_ip, "t0": t0, "t1": t1})
    (run / "manifest.json").write_text(json.dumps(m, indent=2))


def attacker_ip() -> str:
    r = sh(["kubectl", "-n", "victims", "get", "pod", "attacker",
            "-o", "jsonpath={.status.podIP}"], capture_output=True, text=True)
    return r.stdout.strip()


def run_scenario(run: Path, name: str, knob_env: dict[str, str] | None = None) -> dict:
    """Exec one scenario in the attacker pod, record its window, return oracle result."""
    ip = attacker_ip()
    env = " ".join(f"{k}={v}" for k, v in (knob_env or {}).items())
    t0 = time.time()
    cmd = ["kubectl", "-n", "victims", "exec", "attacker", "--",
           "bash", "-lc", f"source targets.env && RUN_OUT=/out {env} "
           f"python3 -c \"import scenarios,oracle,sys; s=scenarios.BY_NAME['{name}']; "
           f"h,p=s.host_port(); import subprocess as sp; "
           f"sp.run(s.build(h,p),timeout=s.timeout); "
           f"print('ORACLE', oracle.check(s.oracle))\""]
    r = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    functional = "ORACLE True" in r.stdout
    from attacker.scenarios import BY_NAME
    add_window(run, name, BY_NAME[name].family, ip, t0, t1)
    return {"scenario": name, "functional": functional, "stdout": r.stdout[-400:]}


def extract_and_label(run: Path) -> Path:
    flows = run / "flows.csv"
    sh(["docker", "run", "--rm", "-v", f"{run}:/data", "idps-testbed/cfm",
        "/data/capture.pcap", "/data/flows_raw.csv"])
    sh([sys.executable, str(HERE / "capture" / "label.py"),
        str(run / "flows_raw.csv"), str(run / "manifest.json"), "--out", str(flows)])
    return flows


# --------------------------------------------------------------------- verbs
def cmd_collect(a):
    if not isolation_ok():
        raise SystemExit("isolation check failed; aborting")
    from attacker.scenarios import SCENARIOS
    run = new_run(Path(a.out), "collect")
    cap = subprocess.Popen(["bash", str(HERE / "capture" / "capture.sh"), str(run)])
    time.sleep(3)
    try:
        for rep in range(a.repetitions):
            for s in SCENARIOS:
                print(f"[rep {rep}] {s.name}")
                res = run_scenario(run, s.name)
                print("   functional:", res["functional"])
                time.sleep(a.gap)
    finally:
        cap.terminate(); cap.wait()
    flows = extract_and_label(run)
    print(f"collected -> {flows}")


def cmd_evade(a):
    if not isolation_ok():
        raise SystemExit("isolation check failed; aborting")
    sys.path.insert(0, str(HERE / "evasion"))
    from attacker.scenarios import SCENARIOS
    from blackbox_search import search, evasion_success
    from transforms import Knobs
    run = new_run(Path(a.out), f"evade-{a.attack}")
    results = []
    for s in SCENARIOS:
        def realise_and_score(k: Knobs, s=s):
            cap = subprocess.Popen(["bash", str(HERE / "capture" / "capture.sh"), str(run), "0"])
            time.sleep(2)
            res = run_scenario(run, s.name, knob_env=k.as_env())
            cap.terminate(); cap.wait()
            flows = extract_and_label(run)
            margin, detected = _score_one(run, flows, a)
            from blackbox_search import Trial
            return Trial(k, margin, res["functional"], detected)
        if a.attack == "blackbox":
            best, hist = search(realise_and_score, queries=a.queries, seed=a.seed)
        else:
            best = realise_and_score(_gradient_knobs(run, s, a)); hist = [best]
        results.append({"scenario": s.name, "evaded": evasion_success(best),
                        "margin": best.margin, "functional": best.functional,
                        "knobs": best.knobs.vector(), "queries": len(hist)})
        print(f"{s.name}: evaded={evasion_success(best)} margin={best.margin:.3f}")
    (run / "evasion_results.json").write_text(json.dumps(results, indent=2))
    print(f"evasion results -> {run/'evasion_results.json'}")


def _score_one(run, flows, a):
    """Score the just-captured flows against the target model; return (margin, detected)."""
    out = run / "scores"
    sh([sys.executable, str(HERE / "scoring" / "score_models.py"),
        "--run", a.models, "--flows", str(flows), "--models", a.target_model,
        "--seeds", "0", "--out", str(out)])
    # margin/detection are read back by analysis; here a placeholder wiring point
    return 0.0, False


def _gradient_knobs(run, s, a):
    from transforms import Knobs
    return Knobs()   # gradient mapping wired in scoring/score_models + gradient_guided


def cmd_score(a):
    sh([sys.executable, str(HERE / "scoring" / "score_models.py"),
        "--run", a.models, "--flows", a.flows, "--out", a.out])
    if a.platform_url:
        sh([sys.executable, str(HERE / "scoring" / "score_platform.py"),
            "--url", a.platform_url, "--flows", a.flows, "--out", a.out + "_platform"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect"); c.set_defaults(fn=cmd_collect)
    c.add_argument("--repetitions", type=int, default=5)
    c.add_argument("--gap", type=float, default=5.0)
    c.add_argument("--out", default="runs/clean")

    e = sub.add_parser("evade"); e.set_defaults(fn=cmd_evade)
    e.add_argument("--attack", choices=["gradient", "blackbox"], default="blackbox")
    e.add_argument("--queries", type=int, default=50)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--models", required=True, help="saved run folder for the target model")
    e.add_argument("--target-model", default="sode_at")
    e.add_argument("--out", default="runs/evade")

    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--models", required=True)
    s.add_argument("--flows", required=True)
    s.add_argument("--platform-url", default="")
    s.add_argument("--out", default="runs/scores")

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
