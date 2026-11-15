"""Generate multi-agent sessions with a Claude model playing every agent (Anthropic SDK).

    export ANTHROPIC_API_KEY=...          # or use `ant auth login`
    python scripts/mas_generate_claude.py --n 400 --out runs/mas_claude

Each session is one simulated workspace (conformalguard/mas/world.py) in which a coordinator,
a researcher and an acting agent work on one to three concurrent tasks with mock tools.
Nothing has a real side effect.  Finished sessions are appended to <out>/sessions.jsonl; an
interrupted run resumes, and a session whose calls failed is retried on the next run.
Labels are computed later, by code, from the event log (conformalguard/mas/labels.py).

Notes
- Every agent step is one Messages request with the agent's own history.  The reply is a
  JSON action, requested as structured output (pass --no-schema to parse plain JSON instead).
- Refusals: some workspaces contain an injected instruction, so a request can be declined by
  safety classifiers.  Server-side fallbacks are on by default (`fallbacks="default"`), so a
  declined request is re-run on a fallback model in the same call; the session records which
  models served it.  A step that is still refused ends that agent's part and is counted.
  Haiku models have no server-side fallback, so it is switched off for them.
- Prompt caching: the last turn of each request is a cache breakpoint, so an agent's later
  steps read its earlier history from cache once the history is long enough to be cached.
- Request construction and reply handling are unit-tested with a stand-in client.  This
  script has not been run against the live API in the environment where it was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import generate as G  # noqa: E402

ARG_NAMES = ("doc_id", "name", "to", "subject", "body", "channel", "text", "url", "content",
             "with_user", "user", "resource")
SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "kind": {"type": "string", "enum": ["tool", "message", "finish"]},
        "tool": {"type": "string"},
        "args": {"type": "object", "properties": {n: {"type": "string"} for n in ARG_NAMES},
                 "required": list(ARG_NAMES), "additionalProperties": False},
        "to": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["thought", "kind", "tool", "args", "to", "content"],
    "additionalProperties": False,
}
ARGS_NOTE = ("\nIn \"args\", fill in only the arguments of the tool you call and leave every other "
             "field an empty string.")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_request(system: str, turns: list[dict], model: str, effort: str, fallbacks: bool, schema: bool) -> dict:
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    messages[-1] = {"role": messages[-1]["role"],
                    "content": [{"type": "text", "text": messages[-1]["content"],
                                 "cache_control": {"type": "ephemeral"}}]}
    output_config: dict = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": SCHEMA}
    req = dict(model=model, max_tokens=8000, system=system + (ARGS_NOTE if schema else ""),
               output_config=output_config, messages=messages)
    if fallbacks:
        req.update(betas=[FALLBACK_BETA], fallbacks="default")
    return req


def read_action(resp) -> tuple[dict, bool]:
    """(action, refused) from a Messages response."""
    if resp.stop_reason == "refusal":
        return dict(G.DECLINED), True
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return G.parse_action(text), False


class ClaudePolicy:
    """One agent step = one request.  Counters belong to the session the policy is used for."""

    def __init__(self, client, model: str, effort: str, fallbacks: bool, schema: bool):
        self.client, self.model, self.effort = client, model, effort
        self.fallbacks, self.schema = fallbacks, schema
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
        self.refusals = 0
        self.served_by: set[str] = set()

    def act(self, system: str, turns: list[dict]) -> dict:
        req = build_request(system, turns, self.model, self.effort, self.fallbacks, self.schema)
        try:
            resp = self.client.beta.messages.create(**req) if "betas" in req else self.client.messages.create(**req)
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"request rejected (400): {e.message}") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"rate limited after retries: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"connection error: {e}") from e
        self.usage["calls"] += 1
        usage = getattr(resp, "usage", None)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.usage[k] += int(getattr(usage, k, 0) or 0)
        if getattr(resp, "model", None):
            self.served_by.add(resp.model)
        action, refused = read_action(resp)
        self.refusals += refused
        return action


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="number of sessions")
    ap.add_argument("--seed-start", type=int, default=0, help="first workspace seed")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--no-schema", action="store_true", help="ask for JSON in the prompt only")
    ap.add_argument("--workers", type=int, default=8, help="sessions generated in parallel")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fallbacks = not args.no_fallbacks and "haiku" not in args.model
    if not args.no_fallbacks and not fallbacks:
        print("note: no server-side refusal fallback for this model; a refused step ends that agent's part")
    client = anthropic.Anthropic(max_retries=6)
    path = Path(args.out) / "sessions.jsonl"
    counts = G.run_many(lambda: ClaudePolicy(client, args.model, args.effort, fallbacks, not args.no_schema),
                        range(args.seed_start, args.seed_start + args.n), path, args.workers,
                        {"agent_model": args.model, "provider": "anthropic", "effort": args.effort})
    print(f"done: {counts['written']} sessions written this run, {counts['failed']} failed (rerun to retry); file {path}")
    print("usage:", G.usage_report(path))


if __name__ == "__main__":
    main()
