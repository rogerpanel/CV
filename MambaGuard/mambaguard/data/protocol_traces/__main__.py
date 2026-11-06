"""Inspect or cache a protocol benchmark.

    python -m mambaguard.data.protocol_traces mcptox data/raw/MCPTox-Benchmark
    python -m mambaguard.data.protocol_traces a2a_sandbox builtin --out data/processed/a2a_sandbox.jsonl
    python -m mambaguard.data.protocol_traces mcp_safetybench data/raw/MCPSafety --split test --show 1
"""
from __future__ import annotations

import argparse
import json

from . import EPISODE_LOADERS, episodes_to_jsonl, summarise


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m mambaguard.data.protocol_traces")
    p.add_argument("benchmark", choices=sorted(EPISODE_LOADERS))
    p.add_argument("root", help="dataset clone (or 'builtin' for a2a_sandbox)")
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--out", help="write episodes to this JSONL file")
    p.add_argument("--show", type=int, default=0, help="print the first N episodes")
    a = p.parse_args()
    eps = EPISODE_LOADERS[a.benchmark](a.root, split=a.split)
    print(json.dumps(summarise(eps), indent=2))
    for e in eps[: a.show]:
        print(f"\n== {e.episode_id}  attack={e.is_attack}  type={e.attack_type}  group={e.group}")
        for m in e.messages:
            print(f"  [{m.label}] {m.tau:6s} {m.src} -> {m.dst}: {m.payload[:160]}")
    if a.out:
        print(f"wrote {episodes_to_jsonl(eps, a.out)} episodes to {a.out}")


if __name__ == "__main__":
    main()
