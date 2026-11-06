# Protocol-native benchmarks: download and loading

The four headline benchmarks are read directly from the authors' public
repositories. Clone each one under `data/raw/`, then load it with
`mambaguard.data.protocol_traces` (or by name through `DATASET_REGISTRY`).

The commits below are the ones the loaders were written and checked against
(November 2026). Pin them so later changes upstream cannot alter the counts.

## 1. Download

```bash
mkdir -p data/raw && cd data/raw

# MCPSecBench (MIT)
git clone https://github.com/AIS2Lab/MCPSecBench.git
git -C MCPSecBench checkout 7612c5a3e811dcf01f64e4f2bb324591a2feaaf4

# MCP-SafetyBench (no license file: use, do not redistribute)
git clone https://github.com/xjzzzzzzzz/MCPSafety.git
git -C MCPSafety checkout 7872437b6369aac1150e3a19e350a981dc554f81

# MCPTox (no license file: use, do not redistribute)
git clone https://github.com/zhiqiangwang4/MCPTox-Benchmark.git
git -C MCPTox-Benchmark checkout f85189f9ad12504c197c7f920ab818a40657b1fa

# A2ABreak (MIT). Optional: the A2A sandbox is generated, and the clone is
# only used to check the number of findings. 177 MB, most of it unrelated
# RFC benchmarks; a shallow clone is enough.
git clone --depth 1 https://github.com/arlotfi79/A2ABreak.git
cd ../..
```

Only data files are read (`json`, `csv`, `yaml`), plus server source files,
which are parsed with Python's `ast` and never executed. Nothing in these
repositories needs to be installed or run.

## 2. Load and check

```bash
python -m mambaguard.data.protocol_traces mcpsecbench     data/raw/MCPSecBench
python -m mambaguard.data.protocol_traces mcp_safetybench data/raw/MCPSafety
python -m mambaguard.data.protocol_traces mcptox          data/raw/MCPTox-Benchmark
python -m mambaguard.data.protocol_traces a2a_sandbox     data/raw/A2ABreak \
    --out data/processed/a2a_sandbox.jsonl
```

Add `--show 2` to print two episodes message by message, and `--split test` to
restrict to one split. In Python:

```python
from mambaguard.data.protocol_traces import load_mcptox_episodes
test = load_mcptox_episodes("data/raw/MCPTox-Benchmark", split="test")
```

## 3. Expected counts (default options)

| Benchmark | Attack episodes | Benign episodes | Messages | Groups | Split unit |
|---|---|---|---|---|---|
| MCPSecBench | 11 | 7 | 136 | 11 | none (evaluation only) |
| MCP-SafetyBench | 238 | 162 | 2,359 | 238 | case |
| MCPTox | 1,312 | 533 | 5,535 | 45 | server |
| A2A sandbox | 2,200 | 2,200 | 48,947 | 2,200 | attack/benign pair |

Train / val / test sizes (episodes): MCP-SafetyBench 281 / 60 / 59,
MCPTox 1,263 / 283 / 299, A2A sandbox 3,080 / 660 / 660. Splits are
deterministic, grouped (no group appears in two splits) and stratified.

## 4. What each loader builds

**Episode** = one detection window: an ordered list of JSON-RPC messages
around a single user task. `ProtocolMessage.label` is 1 on messages that carry
attacker-controlled content; `Episode.is_attack` is 1 if any message does.
Timestamps are synthetic (none of the sources records time), so use the
loaders' group splits, not `temporal_split`.

**MCPTox.** `response_all.json`: 45 servers, 353 tools, 1,312 poisoned-tool
instances (`wrong_data == 0`; the 36 others are excluded by default). An
attack episode is `tools/list` request, `tools/list` result (clean tools plus
the poisoned tool, at a hash-chosen position, label 1), then the benign user
query. Benign episodes use the same queries with the clean tool list. Labels:
11 risk categories (`attack_type`), paradigm (Template-1/2/3), wrapper style,
and per-LLM outcomes with a success rate. Option `with_model_call="<model>"`
appends that model's recorded tool call.

**MCP-SafetyBench.** 245 attack cases in 5 domains. Clean tool lists come from
the repository's own server sources (`yfinance`, `google-maps`, `github`,
`playwright`, `google-search`, ...). The `filesystem`, `fetch` and
`calculator` servers are external npm/PyPI packages without local
definitions and are omitted from both attack and benign windows
(`extras["servers_without_local_tools"]`). Each attack field is placed on the
message an observer would see (poisoned description, added tool, injected or
forged tool result, replayed or spoofed call, malicious arguments, or the user
question for user-layer attacks). The 7 Rug Pull cases change tool code, not
any message, and are excluded unless `include_unobservable=True`. Benign twins
(162) exist only for description-borne attacks, where they are exact: same
question, unmodified tools. The Server/Host/User layer is inferred from where
the benchmark injects each attack and stored in `extras["layer"]`.

**MCPSecBench.** The repository records no MCP messages: it has 11 attack
prompts, a CSV of 15 runs × 3 platforms × 17 attacks, and the benign and
malicious server sources. Episodes list the tools of every server the
playground configures, then send the prompt. Server names are replaced by
neutral identifiers (the attacker's server is literally named `malicious`).
Benign twins send the same prompt with only the benign servers, so **episode
length alone separates attack from benign**; use MCPSecBench for
message-level evaluation (is the attacker's tool list flagged among the other
lists in the same window?), not episode-level accuracy. Six attacks exist only
as client/transport code and are not represented; their run outcomes are
available from `load_mcpsecbench_outcomes`.

**A2A sandbox.** A2ABreak contains no message traces, exploit code or benign
traffic, so the traces are generated (`a2a_sandbox.py`), one template per
finding, in A2A v1.0 vocabulary (`SendMessage`, `GetTask`, `SubscribeToTask`,
`CancelTask`, `CreateTaskPushNotificationConfig`, `TASK_STATE_*`). Every
attack episode has a benign twin with the same participants and task; every
unlabelled attack message appears verbatim, in order, in the twin (checked for
all 2,200 pairs). Both twins get the same random background exchanges. This
is synthetic data and must be described as such.

## 5. Shortcut check

Balanced test accuracy of a one-threshold classifier on episode length:

| | message count | payload characters |
|---|---|---|
| MCPSecBench | 1.00 | 1.00 |
| MCP-SafetyBench | 0.50 | 0.52 |
| MCPTox | 0.50 | 0.54 |
| A2A sandbox (all) | 0.53 | 0.56 |

Within the A2A sandbox, length is informative for `unbounded_delegation_depth`
(0.80), `indefinite_interrupted_tasks` (0.86), `post_revocation_stream` and
`stale_task_state` (0.63) because the attack itself adds or removes messages.
Report these baselines next to the detector, and add a leave-one-finding-out
evaluation (`split_by="finding"`, `held_out_finding=...`).

## 6. Citations

- Y. Yang et al., MCPSecBench, arXiv:2508.13220.
- T. Zong et al., MCP-SafetyBench, ICLR 2026, arXiv:2512.15163.
- Z. Wang et al., MCPTox, AAAI 2026, arXiv:2508.14925.
- A. Lotfi, M. M. Rahman, I. Karim, E. Bertino, A2ABreak, ACSAC 2026, arXiv:2609.10871.
