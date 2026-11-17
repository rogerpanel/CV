# Datasets

CT-Explain uses the six benchmarks catalogued in the manuscript
(84.2M records, 83 unique attack classes). They are distributed as two
Kaggle bundles.

## Bundle 1 — General Network Traffic

Kaggle DOI: `10.34740/kaggle/dsv/12483891`

| Dataset          | Year | Records   | Attack classes |
| ---------------- | ---- | --------- | -------------- |
| CIC-IoT-2023     | 2023 | 46.6M     | 33             |
| CSE-CICIDS2018   | 2018 | 16.2M     | 14             |
| UNSW-NB15        | 2015 | 2.5M      | 9              |

## Bundle 2 — Cloud, Microservices & Edge

Kaggle DOI: `10.34740/KAGGLE/DSV/12479689`

| Dataset        | Year | Records | Attack classes |
| -------------- | ---- | ------- | -------------- |
| MS GUIDE       | 2024 | 13.0M   | 15             |
| Container-NID  | 2023 | 4.8M    | 8              |
| Edge-IIoT      | 2022 | 1.1M    | 5              |

## Preprocessing

* Format: NetFlow-v2 (83 engineered features).
* Identifier columns (`src_ip`, `dst_ip`, ports, `timestamp`, `flow_id`)
  are excluded from the learnable feature vector to prevent label leakage.
* Per-feature z-score standardisation fitted on training only.
* Chronological 70/15/15 split (no random shuffling).
* Cross-dataset transfer: the `transform()` step reuses the fit statistics
  of dataset A when loading dataset B.

## Downloading

```bash
python scripts/download_data.py --bundle all --dest data/
```

The script delegates to the Kaggle CLI when available; otherwise it prints
the DOIs so the bundles can be fetched manually.

## Layout on disk

```
data/
  cic-iot-2023/
    *.csv
  cse-cicids-2018/
    *.csv
  unsw-nb15/
    *.csv
  ms-guide-2024/
    *.csv
  container-nid/
    *.csv
  edge-iiot/
    *.csv
```

## Synthetic dry-run

For pipeline smoke tests (CI, first-time setup on an offline node) each
loader supports a ``synthetic=True`` switch that yields a small compliant
frame. The data loaders in `ct_explain.data.datasets` fall back to
synthetic data when `root` is `None`.

## Related data

The paper also references PQC dataset `10.34740/kaggle/dsv/15424420`
(used by the Multi-Agent PQC-IDS model in the RobustIDPS.ai platform) and
three multi-agent-safety benchmarks consumed by ConformalGuard: AgentDojo
(97 tasks, 629 cases), InjecAgent (1054 cases), R-Judge (569 records).
