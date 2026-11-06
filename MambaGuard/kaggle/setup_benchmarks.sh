#!/usr/bin/env bash
# Download the benchmark repositories at the commits the loaders were checked
# against (reproducibility/PROTOCOL_BENCHMARKS.md).
#
#   bash kaggle/setup_benchmarks.sh [DATA_ROOT] [benchmark ...]
#
# DATA_ROOT defaults to /tmp/mambaguard_data (outside /kaggle/working, so the
# clones are not saved as notebook output; MCPTox and MCP-SafetyBench have no
# licence and must not be redistributed). Names: mcptox mcp_safetybench
# agentdojo injecagent mcpsecbench a2abreak. Default: the first four.
# The A2A sandbox is generated and needs no download.
set -euo pipefail

ROOT="${1:-/tmp/mambaguard_data}"
shift || true
WANT="${*:-mcptox mcp_safetybench agentdojo injecagent}"
mkdir -p "$ROOT"

# name  folder  url  commit ("-" = default branch, shallow)
TABLE="
mcptox          MCPTox-Benchmark https://github.com/zhiqiangwang4/MCPTox-Benchmark.git f85189f9ad12504c197c7f920ab818a40657b1fa
mcp_safetybench MCPSafety        https://github.com/xjzzzzzzzz/MCPSafety.git          7872437b6369aac1150e3a19e350a981dc554f81
agentdojo       agentdojo        https://github.com/ethz-spylab/agentdojo.git         089ed468cf3ed0322acc66b0211f26d9d90dbf60
injecagent      InjecAgent       https://github.com/uiuc-kang-lab/InjecAgent.git      f19c9f2c79a41046eb13c03c51a24c567a8ffa07
mcpsecbench     MCPSecBench      https://github.com/AIS2Lab/MCPSecBench.git           7612c5a3e811dcf01f64e4f2bb324591a2feaaf4
a2abreak        A2ABreak         https://github.com/arlotfi79/A2ABreak.git            -
"

fetch() {  # folder url commit
  local dir="$ROOT/$1" url="$2" sha="$3"
  if [ -d "$dir/.git" ] && { [ "$sha" = - ] || [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; }; then
    echo "   already present"; return 0
  fi
  rm -rf "$dir"
  if [ "$sha" = - ]; then
    git clone -q --depth 1 "$url" "$dir"; return 0
  fi
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  if git -C "$dir" fetch -q --depth 1 origin "$sha"; then
    git -C "$dir" checkout -q FETCH_HEAD
  else  # server refused a by-SHA fetch: full clone, then check out
    rm -rf "$dir"; git clone -q "$url" "$dir"; git -C "$dir" checkout -q "$sha"
  fi
}

for name in $WANT; do
  line="$(echo "$TABLE" | awk -v n="$name" '$1 == n')"
  [ -n "$line" ] || { echo "unknown benchmark: $name" >&2; exit 2; }
  read -r _ folder url sha <<< "$line"
  echo "-- $name -> $ROOT/$folder"
  fetch "$folder" "$url" "$sha"
done
du -sh "$ROOT"/* 2>/dev/null || true
