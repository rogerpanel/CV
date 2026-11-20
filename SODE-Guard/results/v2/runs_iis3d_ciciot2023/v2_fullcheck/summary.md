# SODE-Guard v2 re-run: 1 seed(s) [0]

Data: 41719 flows after deduplication, 27 classes. Mean ± half-width of the 95% t-interval over seeds.

## Clean and adversarial (l-inf eps = 0.03, z-units)

| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |
|---|---|---|---|---|---|
| lgbm | 0.931 ± 0.000 | 0.958 ± 0.000 | 0.084 ± 0.000 | – | 0.084 ± 0.000 |
| mlp | 0.689 ± 0.000 | 0.739 ± 0.000 | 0.502 ± 0.000 | – | 0.499 ± 0.000 |
| mlp_at | 0.647 ± 0.000 | 0.709 ± 0.000 | 0.593 ± 0.000 | – | 0.592 ± 0.000 |
| ode_at | 0.636 ± 0.000 | 0.709 ± 0.000 | 0.585 ± 0.000 | – | 0.585 ± 0.000 |
| sode | 0.720 ± 0.000 | 0.777 ± 0.000 | 0.466 ± 0.000 | 0.462 ± 0.000 | 0.465 ± 0.000 |
| sode_at | 0.650 ± 0.000 | 0.716 ± 0.000 | 0.611 ± 0.000 | 0.601 ± 0.000 | 0.609 ± 0.000 |

## Per-class clean F1 (mean over seeds)

| model | BenignTraffic | DDoS-ACK_Fragmentation | DDoS-HTTP_Flood | DDoS-ICMP_Flood | DDoS-ICMP_Fragmentation | DDoS-PSHACK_Flood | DDoS-RSTFINFlood | DDoS-SYN_Flood | DDoS-SlowLoris | DDoS-SynonymousIP_Flood | DDoS-TCP_Flood | DDoS-UDP_Flood | DDoS-UDP_Fragmentation | DNS_Spoofing | DictionaryBruteForce | DoS-HTTP_Flood | DoS-SYN_Flood | DoS-TCP_Flood | DoS-UDP_Flood | MITM-ArpSpoofing | Mirai-greeth_flood | Mirai-greip_flood | Mirai-udpplain | Recon-HostDiscovery | Recon-OSScan | Recon-PortScan | VulnerabilityScan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.85 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 | 0.77 | 0.65 | 1.00 | 1.00 | 1.00 | 1.00 | 0.87 | 1.00 | 1.00 | 1.00 | 0.83 | 0.63 | 0.61 | 1.00 |
| mlp | 0.70 | 0.99 | 0.71 | 1.00 | 0.98 | 0.99 | 1.00 | 0.71 | 0.64 | 0.60 | 0.70 | 0.70 | 0.97 | 0.48 | 0.25 | 0.83 | 0.29 | 0.61 | 0.60 | 0.71 | 0.57 | 0.55 | 0.99 | 0.67 | 0.21 | 0.37 | 0.77 |
| mlp_at | 0.63 | 0.99 | 0.68 | 1.00 | 0.98 | 0.99 | 1.00 | 0.70 | 0.58 | 0.61 | 0.18 | 0.72 | 0.97 | 0.49 | 0.25 | 0.82 | 0.12 | 0.65 | 0.44 | 0.68 | 0.56 | 0.51 | 0.99 | 0.63 | 0.19 | 0.31 | 0.79 |
| ode_at | 0.62 | 0.99 | 0.61 | 1.00 | 0.98 | 0.99 | 1.00 | 0.70 | 0.60 | 0.64 | 0.67 | 0.71 | 0.98 | 0.51 | 0.24 | 0.78 | 0.10 | 0.13 | 0.55 | 0.67 | 0.65 | 0.21 | 0.99 | 0.62 | 0.10 | 0.33 | 0.78 |
| sode | 0.71 | 0.99 | 0.70 | 1.00 | 0.98 | 0.99 | 1.00 | 0.71 | 0.69 | 0.71 | 0.40 | 0.73 | 0.98 | 0.55 | 0.23 | 0.86 | 0.41 | 0.64 | 0.56 | 0.71 | 0.90 | 0.90 | 0.99 | 0.70 | 0.23 | 0.35 | 0.79 |
| sode_at | 0.65 | 0.99 | 0.70 | 1.00 | 0.98 | 0.99 | 1.00 | 0.69 | 0.64 | 0.64 | 0.67 | 0.73 | 0.98 | 0.51 | 0.25 | 0.82 | 0.13 | 0.14 | 0.48 | 0.69 | 0.65 | 0.25 | 0.99 | 0.62 | 0.21 | 0.33 | 0.79 |

## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)

| contrast | difference A − B | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| sode vs mlp (clean) | +0.031 | [+0.025, +0.037] | 0.010 | 0.050 |
| sode vs lgbm (clean) | -0.211 | [-0.226, -0.199] | 0.010 | 0.050 |
| module 2: sode_at vs sode (worst case) | +0.143 | [+0.122, +0.164] | 0.010 | 0.050 |
| sode_at vs mlp_at (worst case) | +0.016 | [-0.003, +0.035] | 0.090 | 0.090 |
| sode_at vs ode_at (worst case) | +0.024 | [+0.008, +0.040] | 0.020 | 0.050 |
