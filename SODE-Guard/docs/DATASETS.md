# Dataset Cards

All datasets are loaded through [`src/data/registry.py`](../src/data/registry.py).
Download instructions are in [`scripts/download_data.sh`](../scripts/download_data.sh).
Splits are stratified 70 / 15 / 15, seeded by `experiment.seeds`.

---

### ICS3D — Cloud + Edge-IIoT + Kubernetes flow corpus (primary benchmark)

* Flows: **18 906 414** · Classes: **17** · 83-dim CICFlowMeter-compatible features
* DOI: <https://doi.org/10.34740/kaggle/dsv/12483891>
* Captures: Azure VM east-US, Azure IoT Edge gateway with three industrial
  testbeds (Modbus, OPC-UA, MQTT), and a 12-node K8s cluster running OWASP
  Juice Shop and a synthetic ICS twin. Labels include benign, port-scan,
  command-injection, OPC-UA fuzzing, K8s exec-injection, lateral movement,
  C2 beaconing, exfil-over-DNS, ransomware staging.
* Licence: CC-BY 4.0
* Recommended uses: long-tail attack robustness, encrypted-traffic stress,
  K8s-side-channel detection.

### IIS3D — UNSW-NB15 ∪ CIC-IDS2018 ∪ CIC-IDS2023 unified corpus

* Flows: **13 414 882** · Classes: **23** (union taxonomy)
* DOI: <https://doi.org/10.34740/kaggle/dsv/12479689>
* The three constituent CSVs are aligned to the 83-feature schema and
  re-labelled via the taxonomy in `docs/iis3d_taxonomy.csv` (shipped with the
  Kaggle dataset). Useful for cross-dataset generalisation experiments.
* Licence: CC-BY 4.0

### IDS-PQC — TLS 1.3 + post-quantum handshake corpus

* Flows: **3 102 477** · Classes: **9**
* DOI: <https://doi.org/10.34740/kaggle/dsv/15424420>
* Adds 12 TLS-metadata and 8 handshake fields (`pqc_kx_id`, `pqc_sig_id`,
  `kem_round`, `hybrid_kx`, ...) on top of the 63 base features. Generated
  with `OpenSSL 3.4 + oqs-provider` and Cloudflare's `boringssl-pqc` fork.
* Licence: CC-BY 4.0
* Recommended uses: post-quantum readiness, hybrid KEM anomaly detection,
  ECH detection.

---

### Canonical public corpora (Table 1)

| Key | Flows | Classes | URL |
|---|---|---|---|
| `cicids2017` | 2 830 743 | 15 | <https://www.unb.ca/cic/datasets/ids-2017.html> |
| `cicids2018` | 16 232 943 | 14 | <https://www.unb.ca/cic/datasets/ids-2018.html> |
| `cicids2023` | 13 401 122 | 33 | <https://www.unb.ca/cic/datasets/ids-2023.html> |
| `ciciot2023` | 46 686 579 | 33 | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> |
| `cicddos2019` | 50 063 112 | 13 | <https://www.unb.ca/cic/datasets/ddos-2019.html> |
| `unswnb15`   | 2 540 044 | 10 | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> |
| `nslkdd`     |   148 517 |  5 | <https://www.unb.ca/cic/datasets/nsl.html> |
| `nftoniotv2` | 16 940 496 | 10 | <https://staff.itee.uq.edu.au/marius/NIDS_datasets/> |

Each corpus uses the canonical training/test split shipped by the provider
where one is supplied (NSL-KDD, UNSW-NB15); otherwise the manuscript's 70 /
15 / 15 stratified protocol applies.

### Cross-domain validation datasets

* `lmsys_chat_1m` — used only by the **ODE-ExtractGuard** companion proposal
  for LLM-extraction detection. Not part of SODE-Guard's training pipeline.
* `wildchat` — same as above.
* `tgb2` — Temporal Graph Benchmark 2.0; can be plugged through
  `src/data/registry.py` by adding a `DatasetSpec` and a custom
  `_load_csvs` adapter.
