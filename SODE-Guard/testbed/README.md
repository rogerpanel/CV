# Problem-space testbed

A self-contained, host-local testbed that turns adversarial evasions into **real
network traffic** and measures whether the robustness rankings from feature space
survive. It implements the design in
[`../docs/PROBLEM_SPACE_ATTACK_PLAN.md`](../docs/PROBLEM_SPACE_ATTACK_PLAN.md).

The pipeline is:

```
k3d cluster (internal network only)
  victims + benign load + attacker pod
        │  tcpdump on the cluster bridge
        ▼
   one pcap per run  ──►  CICFlowMeter  ──►  flows.csv (+ labels)
        │                                         │
        │                                         ├─►  saved v2 models      (scoring/score_models.py)
        │                                         ├─►  RobustIDPS.ai API     (scoring/score_platform.py)
        │                                         └─►  Suricata / Zeek       (signatures/)
        ▼
   evasion loop: transform traffic, re-capture, re-score, keep only functional runs
        ▼
   analysis/  ──►  detection, evasion success, cost, paired-bootstrap tables
```

## Safety and compliance (read before running anything)

This testbed exists to **measure a defender**, not to attack anyone. Every rule
in the plan's safety section is enforced here, and the harness refuses to start
otherwise:

* **Isolation.** It runs on a dedicated VM, never on the production
  `robustidps.ai` host. `cluster/k3d-config.yaml` creates an **internal-only**
  Docker network; `cluster/00-isolation.sh` adds an `iptables` rule that drops
  forwarded traffic from the cluster bridge to the public interface and refuses
  to continue if the host has no second, isolated interface.
* **Targets are your own containers only.** Every attacker target is a
  cluster-internal ClusterIP. `attacker/targets.env` holds only RFC1918
  addresses, and `attacker/guard.py` aborts any scenario whose target resolves
  to a public address.
* **No attack traffic leaves the host.** Floods are rate-limited with `tc`
  (`evasion/transforms.py`), and the isolation check runs before each scenario.
* **Vulnerable images** run only on the internal network, with no published
  ports (`cluster/10-victims.yaml`).
* **Logs are kept.** Every run writes a manifest (`runs/<id>/manifest.json`)
  recording the scenario, knobs, target, start/stop times and the operator.

These scripts orchestrate **standard, public** tools (nmap, nuclei, hydra,
slowloris, hping3) against **your own** deliberately vulnerable containers. They
contain no new exploit code and no capability to target external systems.

## Requirements

A single VM, ~8–16 vCPU and 32 GB RAM, no GPU needed for scoring. Install:

```bash
testbed/setup.sh      # k3d, kubectl, docker, tcpdump, suricata, zeek, tc, python deps
```

Training (5 seeds) runs on Colab with the existing `scripts/v2_driver.py`; only
scoring and signature evaluation run here.

## Quick start

```bash
# 0. one-time isolation + cluster
sudo testbed/cluster/00-isolation.sh
testbed/cluster/up.sh                       # k3d cluster + victims + benign load

# 1. collect a clean corpus (Track A): benign + every scenario, 5 repetitions
testbed/run.py collect --repetitions 5 --out runs/clean

# 2. train on it (on Colab): upload runs/clean/flows.csv, run v2_driver, bring the
#    WHOLE run folder back (not just the summaries) -> testbed/scoring/models/<corpus>/
#    It must contain meta.json, full.npz, and seed_*/{data.npz, <model>.pt|.pkl};
#    score_models.py recomputes the training-set scaling from full.npz.

# 3. evasions against the trained models + platform + signatures
testbed/run.py evade --attack gradient --models scoring/models/testbed --out runs/evade_gg
testbed/run.py evade --attack blackbox --queries 50 --out runs/evade_bb

# 4. transfer: score the saved ICS3D models on the testbed traffic (Track B)
testbed/run.py score --models scoring/models/ics3d --flows runs/clean/flows.csv --out runs/transfer

# 5. tables
testbed/analysis/make_tables.py runs/ --out analysis/results
```

## Layout

| Path | What it is |
|---|---|
| `cluster/` | k3d config, isolation guard, victim and benign-load and attacker manifests |
| `attacker/` | scenario definitions, the functionality oracle, the public-target guard |
| `evasion/` | traffic transformations (padding, timing, splitting, chaff, rate), gradient-guided and black-box search |
| `capture/` | pcap capture on the cluster bridge and flow labelling from (attacker IP, run window) |
| `features/` | a pinned CICFlowMeter container and the extraction wrapper (same extractor as the ICS3D container flows) |
| `scoring/` | load the saved v2 models and score; score through the RobustIDPS.ai API; conformal alarm thresholds |
| `signatures/` | run Suricata (ET Open) and Zeek on the same pcaps and parse their alerts |
| `analysis/` | detection, evasion-success and cost metrics, paired bootstrap with Holm |
| `run.py` | the operator entry point that ties the stages together and writes run manifests |

Nothing here writes to the `SODE-Guard` results tree; each run is a fresh folder
under `testbed/runs/` (git-ignored).
