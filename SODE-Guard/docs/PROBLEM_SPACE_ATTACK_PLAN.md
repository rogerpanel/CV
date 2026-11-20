# Problem-space attack experiment: design

Goal: show that the robustness results hold when evasions are carried out as real
network traffic, not only as changes to extracted flow statistics. This answers the
main question a USENIX Security reviewer will ask ("who changes the features?"),
adds a recent, self-collected corpus, and gives a measured deployment cost.

## Questions

| # | Question | Measured by |
|---|---|---|
| Q1 | Do the robustness rankings from feature space (adversarially trained neural detectors robust; trees brittle) persist when the attack is realised as packets? | detection rate and macro-F1 on realised evasive traffic |
| Q2 | Can an attacker evade while the attack still works? | evasion success = attack functional and undetected |
| Q3 | What does deployment cost? | flows/s, per-flow latency and memory on the server CPU; certification time |
| Q4 | Do the signature and built-in detectors of the platform catch the evasive variants? | Suricata/Zeek alerts and the RobustIDPS.ai detectors on the same pcaps |

## Safety and compliance (read first)

* **Isolation.** Use a dedicated Hetzner VM, separate from the production robustidps.ai host; nothing touches the live service.
* **No attack traffic leaves the host.** Everything runs on internal Docker or k3d networks (`--internal`, plus an iptables rule that drops `FORWARD` traffic to the public interface). Floods are rate-limited with `tc`.
* **Targets are your own containers only.** No scanning of any external address. Hetzner's acceptable-use policy forbids attacks on external systems, and unexplained flood traffic can trigger its abuse handling. Keep the experiment host-local and keep logs.
* **Vulnerable images** (for example old Grafana or InfluxDB versions) run only on the internal network, without published ports.
* **Write-up.** Record all of this in the paper's ethics appendix.

## Testbed (one VM, about 8-16 vCPU and 32 GB RAM; scoring needs no GPU)

```
k3d cluster (internal network only)
 ├── victim services: a microservice shop (e.g. Google Online Boutique),
 │   Node-RED, Grafana, InfluxDB (versions with public CVEs, as in ICS3D)
 ├── benign load: Locust user sessions against the shop (realistic browsing and purchases)
 ├── attacker pod: nmap, nuclei, hydra, slowloris / hping3 (rate-limited),
 │   public PoC scripts for the CVEs
 ├── evasion layer on the attacker pod: tc/netem (delays, jitter), an HTTP
 │   padding proxy, MSS clamping / small writes (packet splitting), chaff requests
 └── capture: tcpdump on the cluster bridge, one pcap per run,
     labels from (attacker IP, run window)
features: one CICFlowMeter build for every run (the ICS3D container flows use CICFlowMeter too)
scoring: the v2 pipeline's saved models, and the RobustIDPS.ai backend API
signature baselines: Suricata (ET Open rules) and Zeek on the same pcaps
```

Scenarios, chosen to mirror the ICS3D container classes and the common IIS3D
families:

* reconnaissance (nmap host and port scans);
* vulnerability scan (nuclei);
* SSH and HTTP brute force (hydra);
* slow and volumetric DoS (slowloris, rate-limited floods);
* path traversal (Grafana CVE-2021-43798);
* authentication bypass (InfluxDB CVE-2019-20933);
* Node-RED remote code execution.

Benign traffic runs alongside every scenario.

## Two evaluation tracks

* **Track A (primary, in-domain).** Collect a clean testbed corpus first: benign plus every scenario, with 5 repetitions on different days and load levels. Train every model with the v2 pipeline on it (5 seeds, on Colab). Then attack the trained models with fresh evasive runs. Training and test come from the same environment, so domain shift cannot explain the results.
* **Track B (transfer).** Score the saved ICS3D models on the testbed traffic without retraining. This measures cross-environment transfer and is reported separately.

## The problem-space attacker

Traffic transformations, each with a budget:

| Transformation | Knob | Budget |
|---|---|---|
| payload padding | bytes added per request (junk HTTP header, extra parameters) | at most +50% bytes |
| timing | netem delay and jitter per packet | at most +50 ms per packet, at most 3x run duration |
| packet splitting | MSS clamp / write size | at most 4x packet count |
| chaff | benign-looking requests interleaved with the attack | at most 1 chaff request per attack request |
| rate shaping | attack rate (low-and-slow) | the attack must stay functional |

A **functionality oracle** keeps every evasion honest. An evasive run counts only if
its attack still achieves its goal:

* brute force: a valid credential is still found;
* exploit: the file is still read, or the RCE marker is still written;
* scan: the same open ports are still reported;
* DoS: the benign 95th-percentile latency still rises by at least 2x.

Two attack algorithms:

1. **Gradient-guided.** Compute the constrained feature-space adversarial direction on the target model (white box). Map the sign of each feature change to a knob: for example, a higher mean forward packet length maps to padding, and higher inter-arrival times map to delay. Realise the knob setting, re-extract the features, then score.
2. **Black-box search.** Evolutionary or random search over the knob vector, with about 50 queries per scenario and model. It minimises the model's margin on the realised flows and is constrained by the budget and the oracle. It needs only scores, so the same search also runs against the platform's detectors and the trees.

## Models and baselines

* **Neural detectors:** SODE-AT, MLP-AT, ODE-AT, SODE (M1) and MLP.
* **Trees:** LightGBM, XGBoost and the random forest.
* **Smoothed models:** SODE-RS and MLP-RS, for certified radii on realised flows.
* **Platform baselines:** the RobustIDPS.ai built-in detectors, plus Suricata and Zeek.

## Metrics and statistics

* **Detection:**
  * per flow: the detection rate;
  * per attack instance: an attack run counts as detected if any of its flows raises an alarm;
  * macro-F1 on realised attacks.
* **Evasion success:** the fraction of evasive runs that stay functional and undetected, together with their overhead (bytes, delay, duration).
* **Alarms:** conformal thresholds at 1% and 5% false-alarm rate, calibrated on benign testbed traffic. Report the test false-alarm rate alongside detection.
* **Cost:** per-flow latency, throughput and memory on the server CPU, for every model, Suricata and the platform detectors.
* **Statistics:** 5 repetitions times 5 seeds. Paired bootstrap with Holm correction over contrasts declared in advance:
  * SODE-AT vs MLP-AT under the gradient-guided and black-box attacks;
  * SODE-AT vs LightGBM;
  * the neural detectors vs Suricata.

## Work plan

| Step | Where | Effort |
|---|---|---|
| Testbed manifests, attacker and evasion scripts, capture and labelling, CICFlowMeter container | repository (`testbed/`) | about 2 days of code |
| Collect the clean corpus (Track A) | Hetzner VM | about 1 day of runs |
| Train on the testbed corpus | Colab, with the existing driver | about 3-5 GPU hours |
| Gradient-guided and black-box evasions, scoring, Suricata/Zeek | Hetzner VM | about 1-2 days of runs |
| Analysis and the paper section "Problem-space evaluation", plus a cost table | repository | about 1 day |

## Questions for the platform owner

1. VM size, OS, and whether Docker/k3d may run vulnerable images on an internal-only network.
2. Does the RobustIDPS.ai backend expose a scoring API for an external model, or should SODE-Guard be deployed as an additional detector?
3. Which signature engines (Suricata/Zeek versions and rule sets) and which breach-and-attack-simulation tools are already installed?
4. May the experiment VM be a separate Hetzner instance? Recommended.
