# Field Deployment Kit — Robust UAV Harvest-Readiness Monitoring

**From bench to farm: assembling an industrial test run of the RobustIDPS.ai /
M1+M4+M6+M7 network-layer robustness models on a crop-monitoring UAV.**

- Mission: fly an agricultural field, capture multispectral imagery, decide
  **harvest-due / not-due per zone** — while the research models keep the UAV
  network layer (C2 link, telemetry, GNSS) alive under interference, jamming
  and spoofing.
- Value proposition proven on the bench: the M1+M4+M6+M7 stack sustains ≥ 90 %
  mission completion up to **~25 dB J/S** vs **~7.7 dB** undefended — so in a
  noisy/contested rural RF environment the drone still finishes the survey and
  still delivers the NDVI maps.

> Prices below are **indicative USD ranges** (2026), for planning only. Exact
> parts depend on field size, crop, region, and drone-law regime. Anything
> that transmits interference (jammer/spoofer for the EW test) is **strictly
> regulated** — see §11.

---

## 1. System architecture — three tiers

The dissertation's three-tier resource budget maps directly onto hardware:

```
 TIER 1 — ONBOARD (on the UAV, flight-critical, lowest latency)
   Flight controller (PX4) ── companion edge AI module
     • M4 MambaShield (INT8)  → real-time C2/telemetry intrusion detection
     • fast NDVI/NDRE screen  → quick harvest-readiness flag per frame
     • anti-spoof GNSS + secure C2 radio

 TIER 2 — EDGE GATEWAY (field base: van/tractor/tent, at the field edge)
   Rugged edge server / AGX-class box
     • Full RobustIDPS.ai stack
     • M1 CT/SDE-TGNN (temporal graph over telemetry/fleet)
     • M6 uncertainty calibration (trust score on detections)
     • M7 game-theoretic certification (decision policy + certificate)
     • Full crop-maturity model (multispectral → harvest-due map)
     • Federated-learning client, RTK base / NTRIP caster

 TIER 3 — CLOUD / REGIONAL (office or cloud, offline-tolerant)
   GPU server
     • FedGTD federated aggregation across farms/fleets
     • PQC-signed model-update channel (Kyber-1024 / Dilithium)
     • Long-term storage, agronomy dashboards
```

**Which model runs where:**

| Model | Role | Tier | Why there |
|---|---|---|---|
| **M4 MambaShield (INT8)** | link/telemetry intrusion detection | 1 onboard | lowest latency, flight-critical, cheap in TOPS |
| **M1 CT/SDE-TGNN** | temporal-graph anomaly over telemetry | 2 edge | needs more compute, not per-packet-critical |
| **M6 uncertainty calib.** | trust score on detections | 2 edge | feeds the decision, tolerates ~100 ms |
| **M7 game-theoretic cert.** | certified decision policy | 2 edge / 3 cloud | offline-computable certificate |
| **FedGTD** | federated model aggregation | 3 cloud | cross-farm, periodic |
| **Harvest-readiness NDVI** | quick per-zone maturity flag | 1 onboard | immediate in-flight feedback |
| **Crop-maturity model** | full harvest-due map | 2 edge | fuses calibrated reflectance + agronomy model |

---

## 2. The embedded chip (what you asked about specifically)

### Primary onboard compute — recommended

**NVIDIA Jetson Orin NX 16 GB** (module) on a UAV-grade carrier board.

| Spec | Value |
|---|---|
| AI throughput | ~100 TOPS (INT8) |
| Power | 10–25 W (configurable) |
| Weight | module + carrier ~45–120 g |
| Runs | MambaShield INT8 + NDVI inference **concurrently, real-time** |
| Software | JetPack/L4T, **TensorRT INT8** engines, Docker, MAVLink/ROS 2 |
| Carriers | Seeed reComputer, ConnectTech Boson/Photon, Auvidea |

Rationale: the M4 MambaShield model is quantised to **INT8** in the
dissertation precisely so it fits an onboard budget like this; Orin NX has the
headroom to also run the onboard NDVI screen and the video pipeline, at a power
and weight a multirotor can carry.

### Ultra-low-power alternative (security detector only)

If you want the intrusion detector on a **separate, ~2–3 W** co-processor and
keep a lighter main CPU:

| Option | TOPS | Power | Note |
|---|---|---|---|
| **Hailo-8** M.2 | 26 | ~2.5 W | excellent perf/W for the INT8 detector |
| **Google Coral Edge TPU** | 4 | ~2 W | cheapest, needs INT8 TFLite |
| Jetson **Orin Nano 8 GB** | ~40 | 7–15 W | if you want one small NVIDIA module |

### Edge gateway (Tier 2) compute

| Option | TOPS | Note |
|---|---|---|
| **Jetson AGX Orin 64 GB** | ~275 | single-box, rugged, runs full RobustIDPS.ai + M1/M6/M7 |
| Industrial x86 + **RTX A2000/4000** | — | if you prefer x86 + more RAM/storage at the base |

### Cloud/regional (Tier 3)

Any GPU server (1× RTX 6000 Ada / L40S class) for FedGTD aggregation and PQC
signing; can be a cloud VM.

---

## 3. UAV platform & flight hardware

| Item | Example | Indicative $ |
|---|---|---|
| Airframe (multirotor, ≥ 30 min, payload ≥ 1.5 kg) | DJI M350 RTK, or Freefly Astro, or custom hexacopter | 5,000–14,000 |
| Flight controller (open, PX4/ArduPilot) | Holybro Pixhawk 6X / Cube Orange+ | 250–450 |
| Companion computer | **Jetson Orin NX 16 GB + carrier** (see §2) | 700–1,100 |
| Redundant IMU / airspeed / barometer | (often integrated in FC) | — |
| Gimbal for payload | integrated or Gremsy | 300–2,500 |
| Spare props, arms, landing gear | — | 200–500 |

> Fixed-wing/VTOL (e.g., WingtraOne, Quantum Trinity) is better for very large
> fields (>200 ha) — swap the airframe row, keep everything else.

---

## 4. GNSS anti-spoof / anti-jam (the EW-critical part)

This is where the J/S results become real — pick a receiver with spoofing/jam
detection, ideally the exact models the benchmark simulated:

| Item | Example | Note |
|---|---|---|
| Multi-band RTK receiver | **u-blox ZED-F9P** (= benchmark "u-blox F9P") | cm-level RTK, spoof flags |
| — higher-assurance option | **Septentrio mosaic-X5** (AIM+ anti-jam/spoof) | strong EW resilience |
| — or | **NovAtel OEM7** (= benchmark "NovAtel OEM7") | matches benchmark model |
| Anti-jam antenna (serious EW) | CRPA / controlled-radiation-pattern antenna | nulls jammers |
| Downstream | receiver C/N₀ + spoof flags feed **M4/M6** as features | closes the loop |

---

## 5. Data links — the "network layer" the models monitor

| Link | Example | Purpose |
|---|---|---|
| Primary C2 (encrypted MANET/mesh) | **Silvus StreamCaster**, Doodle Labs Smart Radio, or Microhard pDDL | the link M1/M4 inspect for intrusion |
| BVLOS backhaul | 4G/5G modem (Sierra/Quectel) + NTRIP for RTK corrections | telemetry + updates |
| Telemetry radio (backup) | RFD900x | fail-safe C2 |
| Ground antennas / mast / tracker | directional + omni, 3–6 m mast | range & link margin |

The traffic across these radios is exactly what RobustIDPS.ai classifies —
this is the "UAV networks layer" from the research.

---

## 6. Agricultural sensing payload (harvest-due decision)

| Item | Example | Gives you |
|---|---|---|
| Multispectral camera | **MicaSense Altum-PT** or RedEdge-P, Sentera 6X | NDVI, NDRE, red-edge → maturity |
| Downwelling light sensor + calibrated panel | MicaSense DLS 2 + reflectance panel | radiometric calibration (essential) |
| High-res RGB | Sony a7R / integrated 20–61 MP | scouting, ground-truth |
| Thermal (optional) | FLIR Vue TZ20 | crop water stress |
| LiDAR (optional, biomass/lodging) | Livox / DJI Zenmuse L2 | canopy height, biomass |

**Harvest-due logic:** calibrated reflectance → NDVI/NDRE per zone → crop-specific
maturity threshold / trained classifier → per-zone "due / not-due / N days"
map. Onboard gives an instant flag; the edge gateway produces the full
agronomic map.

---

## 7. Ground segment & networking

| Item | Example | $ |
|---|---|---|
| Ground control station | rugged laptop/tablet + **QGroundControl / Mission Planner** | 1,500–4,000 |
| Edge gateway box (Tier 2) | Jetson AGX Orin dev/rugged, or x86+GPU | 2,000–5,000 |
| RTK base station | Emlid Reach RS3 / RS2+, or NTRIP subscription | 2,000–3,500 |
| Field router + PoE switch + UPS | industrial (Teltonika RUTX) | 400–900 |
| Power in the field | 2 kWh portable power station + solar, or generator | 800–2,500 |

---

## 8. Security & key management (PQC update channel)

| Item | Purpose |
|---|---|
| Secure element on the UAV | Microchip **ATECC608** / TPM 2.0 — stores keys, verifies signed model updates |
| HSM at the gateway | **YubiHSM 2** or equivalent — signs/verifies **Kyber-1024 / Dilithium** (and GOST R 34.10) model & config updates |
| Encrypted storage | LUKS on gateway, signed containers for RobustIDPS.ai |

This implements the dissertation's **PQC-secured update channel** so models
pushed to the drone can't be tampered with in the field.

---

## 9. Tools, materials & test instrumentation

| Category | Items |
|---|---|
| Assembly tools | precision hex/torx drivers, torque driver, soldering station, heat-shrink, Loctite, multimeter, crimpers, zip ties, vibration-damping mounts |
| **RF test gear (validate against the benchmark!)** | handheld **spectrum analyzer** (e.g., Signal Hound BB60 / tinySA Ultra) to measure the real J/S and noise floor during the run; SDR (USRP/HackRF) for RF characterisation |
| Controlled EW emulation | signal generator / GNSS simulator (**LabSat**, Spirent) and jammer emulator — **licensed test range / Faraday enclosure ONLY** (see §11) |
| Field logistics | rugged transit cases (Pelican), landing pad, tie-downs, tent/shade, ND filters, calibration/reflectance panel, GCP targets + survey markers |
| Batteries | 6–10 smart flight batteries + multi-charger + fireproof LiPo bags |
| Safety | fire extinguisher (Li-ion rated), first-aid, hi-vis, comms |

---

## 10. Software stack — flashing the models onto the chip

1. Flash **JetPack/L4T** on the Jetson; enable max-power then tune the power
   profile for flight endurance.
2. Convert each model **PyTorch → ONNX → TensorRT INT8**; build the
   **MambaShield INT8 engine** for the onboard module (calibrate with a field
   traffic sample).
3. Package **RobustIDPS.ai** as signed Docker containers; onboard runs the
   detection container, gateway runs the full stack + M1/M6/M7 + federated
   client.
4. Integrate with the FC over **MAVLink** (MAVSDK/pymavlink); optionally ROS 2
   for the sensor/inference graph.
5. Wire the **GNSS C/N₀ + spoof flags** and **radio-link stats** in as live
   features to M4/M6.
6. Verify the **PQC update path**: sign a test model with Dilithium at the
   HSM, push, confirm the UAV's secure element accepts only valid signatures.

---

## 11. Regulatory, safety & the EW test caveat

- **Drone ops:** register the aircraft, Remote ID, pilot licence, and a
  **BVLOS authorization** if flying beyond line of sight; file NOTAMs; carry
  liability insurance. Requirements differ by country — confirm with your CAA.
- **EW testing is regulated.** Deliberately transmitting GNSS spoofing or
  jamming **on open spectrum is illegal** in most jurisdictions. Do the
  jamming/spoofing test run **only**: (a) in a **licensed EW/RF test range**,
  or (b) inside a **shielded/Faraday enclosure**, or (c) with a **GNSS record
  &amp; replay simulator** (LabSat/Spirent) injected by cable — never radiated
  over the air without authorization.
- For the **non-EW agronomy validation**, no special RF licence is needed —
  fly normally, monitor the link passively, and only *characterise* the ambient
  J/S with the spectrum analyzer.

---

## 12. Indicative bill of materials (starter test run, one UAV)

| # | Subsystem | Representative choice | Qty | Indicative $ |
|---|---|---|---:|---:|
| 1 | Airframe + FC | M350 RTK / Freefly Astro + Pixhawk 6X | 1 | 6,000–14,000 |
| 2 | **Onboard AI chip** | **Jetson Orin NX 16 GB + carrier** | 1 | 700–1,100 |
| 3 | Onboard co-accel (opt.) | Hailo-8 M.2 | 1 | 200–450 |
| 4 | Anti-spoof GNSS + antenna | Septentrio mosaic-X5 / u-blox ZED-F9P + CRPA | 1 | 500–3,500 |
| 5 | C2 MANET radio pair | Silvus / Doodle Labs | 1 set | 3,000–9,000 |
| 6 | 4G/5G + RFD900x backup | modem + telemetry radio | 1 | 400–900 |
| 7 | Multispectral payload | MicaSense Altum-PT + DLS 2 + panel | 1 | 8,000–13,000 |
| 8 | RGB / thermal / LiDAR (opt.) | Zenmuse / FLIR / Livox | — | 1,500–12,000 |
| 9 | Edge gateway (Tier 2) | Jetson AGX Orin 64 GB (rugged) | 1 | 2,000–5,000 |
| 10 | RTK base / NTRIP | Emlid Reach RS3 | 1 | 2,000–3,500 |
| 11 | GCS + networking + power | laptop, router, power station | 1 set | 3,000–7,000 |
| 12 | Security (secure element + HSM) | ATECC608 + YubiHSM 2 | 1 set | 700–1,200 |
| 13 | RF test gear | spectrum analyzer + GNSS sim (rental ok) | 1 | 1,500–20,000 |
| 14 | Batteries, tools, cases, safety | — | lot | 3,000–6,000 |
|   | **Cloud (Tier 3)** | GPU VM for FedGTD + PQC signing | — | ~ /month |
|   | **Indicative total (1-UAV pilot)** | | | **~ $40k–90k** |

Cloud aggregation and the GNSS simulator can be **rented**, and the co-accel,
thermal and LiDAR are optional — a lean first test run lands nearer the low end.

---

## 13. Field test-run procedure (one sortie)

1. **Calibrate** — image the reflectance panel; set RTK base / NTRIP; verify
   secure boot + valid model signatures on the UAV.
2. **Baseline flight** — fly the survey grid; onboard NDVI screen + edge
   maturity model produce the first **harvest-due map**; RobustIDPS.ai logs a
   clean link baseline.
3. **Characterise the RF environment** — spectrum analyzer records ambient
   noise / any interference; log receiver C/N₀ → this is your measured J/S.
4. **(Authorized range only) EW stress** — inject spoofing/jamming via
   simulator or in the shielded range across a J/S sweep; confirm the UAV holds
   mission completion in line with the benchmark curve, and that M4/M6/M7 flag
   and ride through the attack.
5. **Decision + audit** — export the per-zone due/not-due map; export the
   security event log and the completion-vs-J/S record; compare field results
   against `UAV-EW-Bench-2026` to validate the models in the real environment.

---

## 14. What each research contribution buys you in the field

| Research result | Field benefit |
|---|---|
| M4 MambaShield INT8 onboard | detects C2/telemetry intrusion in real time, on a low-power chip |
| M1 CT/SDE-TGNN | catches slow, evolving anomalies across the flight/fleet |
| M6 uncertainty calibration | avoids false "abort" decisions on ambiguous signals |
| M7 game-theoretic certificate | a *provable* safe-operating envelope for the operator/regulator |
| FedGTD federated learning | farms improve the model together without sharing raw data |
| +17 dB J/S mission-completion margin | the survey **finishes** — and the harvest-due map arrives — even under heavy interference |
