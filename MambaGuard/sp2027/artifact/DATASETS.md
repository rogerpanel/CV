# Datasets

Every dataset used in the paper is public. Nothing is redistributed in this
artifact. Place each one under `data/raw/<name>/` before running
`scripts/reproduce_all.sh --full`.

## Protocol-native benchmarks (headline results)

| Name | Reference | What it contains |
|---|---|---|
| MCPSecBench | arXiv:2508.13220 | 17 MCP attack types over 4 surfaces, with servers, clients and attack scripts |
| MCP-SafetyBench | arXiv:2512.15163 (ICLR 2026) | 245 multi-turn cases, 5 domains, 20 attack types, real MCP servers |
| MCPTox | arXiv:2508.14925 (AAAI 2026) | 1,312 tool-poisoning cases over 45 live MCP servers and 353 tools |
| A2A sandbox | built from the 11 vulnerability classes in arXiv:2609.10871 | generated locally, no outbound network access |

Download commands, pinned commits, licenses, the episode construction for each
benchmark and the expected counts are in `PROTOCOL_BENCHMARKS.md`. MCPTox and
MCP-SafetyBench have no license file: use your own clone and do not
redistribute them. The A2A sandbox is generated locally and needs no download.

## Agent-tool corpora

| Name | Source |
|---|---|
| AgentDojo | https://github.com/ethz-spylab/agentdojo |
| InjecAgent | https://github.com/uiuc-kang-lab/InjecAgent |

```bash
git clone --depth 1 https://github.com/ethz-spylab/agentdojo.git data/raw/agentdojo
git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git data/raw/injecagent
```

## Network-IDS transfer (appendix only)

These need registration with the dataset owners.

| Name | Portal |
|---|---|
| CSE-CIC-IDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html |
| CIC-IoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |

## Canonicalisation

Every record is mapped to the six-tuple `(tau, s, d, p, mu, t_m)` by
`mambaguard/data/canonicalize.py`. Splits are strictly temporal (70/15/15),
with duplicate flows sharing the same five-tuple and timestamp collapsed
before splitting (`mambaguard/data/splits.py`).
