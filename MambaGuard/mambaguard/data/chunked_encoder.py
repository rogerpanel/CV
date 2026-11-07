"""Chunked payload encoder: keeps sentence-level detail inside long messages.

A single sentence embedding per message averages a poisoned tool description
with every clean description in the same ``tools/list`` result, and an
injected sentence with the rest of a long tool result. This encoder instead

1. splits each JSON payload into chunks: one per tool in a tool list, one per
   short run of sentences in long text fields, plus one structural "header"
   chunk (method, keys, short values);
2. embeds every chunk with a frozen sentence encoder (MiniLM by default);
3. hashes each chunk's word uni/bigrams into ``d_lex`` unsigned buckets
   (log counts, L2-normalised per chunk);
4. returns, per message, ``[mean of chunk embeddings | element-wise max of
   chunk embeddings | element-wise max of chunk lexical vectors]``.

Max pooling lets one unusual chunk (the poisoned tool, the injected sentence)
dominate a coordinate instead of being averaged away. ``which_chunk`` gives the
chunk texts and per-chunk embeddings so a detection can be localised.

Variants: ``mode="chunk"`` drops the lexical block; ``mode="chunk_lex"`` adds
it; ``mode="chunk_mil"`` returns the same pooled vector and also keeps each
message's per-chunk features ``[sentence embedding | lexical vector]`` in
``last_chunks``, so the detector can score every chunk itself and pool the
scores (multiple-instance pooling, ``MambaGuardConfig.chunk_dim``).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Optional

MAX_CHUNKS = 64
SENT_WORDS = 48          # target words per text chunk
SHORT_VALUE = 40         # characters kept per value in the header chunk
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9_]+")


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
def _tool_text(tool: dict[str, Any]) -> str:
    parts = [str(tool.get("name", ""))]
    if tool.get("description"):
        parts.append(str(tool["description"]))
    schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters") or {}
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if isinstance(props, dict):
        for k, v in props.items():
            d = v.get("description", "") if isinstance(v, dict) else ""
            parts.append(f"{k}: {d}".strip())
    return " | ".join(p for p in parts if p)


def _split_text(text: str) -> list[str]:
    """Split long text into runs of whole sentences of about SENT_WORDS words."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > SENT_WORDS:
            out.append(" ".join(cur))
            cur, n = [], 0
        if w > 2 * SENT_WORDS:  # one very long "sentence": cut by words
            words = s.split()
            for i in range(0, len(words), SENT_WORDS):
                out.append(" ".join(words[i:i + SENT_WORDS]))
            continue
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def _header(obj: Any) -> Any:
    """Structure-only view: short values kept, long strings truncated, tool lists named."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "tools" and isinstance(v, list):
                out[k] = [t.get("name", "?") if isinstance(t, dict) else "?" for t in v]
            else:
                out[k] = _header(v)
        return out
    if isinstance(obj, list):
        return [_header(v) for v in obj[:8]]
    if isinstance(obj, str):
        return obj[:SHORT_VALUE]
    return obj


def chunk_payload(payload: str) -> list[str]:
    """Return the chunk texts of one payload (header first)."""
    try:
        obj = json.loads(payload)
    except (TypeError, ValueError):
        obj = None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        text = payload if obj is None else str(obj)
        return (_split_text(text) or [text[:200]])[:MAX_CHUNKS]

    chunks = [json.dumps(_header(obj), ensure_ascii=False, sort_keys=True)]

    def walk(o: Any) -> None:
        if len(chunks) >= MAX_CHUNKS:
            return
        if isinstance(o, dict):
            tools = o.get("tools")
            if isinstance(tools, list) and tools and all(isinstance(t, dict) for t in tools):
                for t in tools:
                    chunks.append(_tool_text(t))
            for k, v in o.items():
                if k != "tools":
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and len(o) > SHORT_VALUE:
            text = o
            try:  # tool results often carry JSON inside a string
                inner = json.loads(o)
                if isinstance(inner, (dict, list)):
                    walk(inner)
                    return
            except ValueError:
                pass
            chunks.extend(_split_text(text))

    walk(obj)
    return [c for c in chunks if c][:MAX_CHUNKS]


# ---------------------------------------------------------------------------
# lexical hashing
# ---------------------------------------------------------------------------
def _lex_vector(text: str, d_lex: int) -> list[tuple[int, float]]:
    toks = _TOKEN.findall(text.lower())
    counts: dict[int, int] = {}
    for f in toks + [a + " " + b for a, b in zip(toks, toks[1:])]:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little") % d_lex
        counts[h] = counts.get(h, 0) + 1
    vals = {k: 1.0 + math.log(c) for k, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vals.values())) or 1.0
    return [(k, v / norm) for k, v in vals.items()]


# ---------------------------------------------------------------------------
class ChunkedEncoder:
    """Drop-in replacement for :class:`MessageCanonicaliser` (``encode_batch``)."""

    def __init__(self, sentence_encoder: Any, mode: str = "chunk_lex", d_lex: int = 512,
                 d_sent: Optional[int] = None) -> None:
        if mode not in ("chunk", "chunk_lex", "chunk_mil"):
            raise ValueError(f"mode must be 'chunk', 'chunk_lex' or 'chunk_mil', got {mode!r}")
        self.sent = sentence_encoder
        self.mode = mode
        self.d_lex = d_lex if mode in ("chunk_lex", "chunk_mil") else 0
        self.d_sent = int(d_sent or getattr(sentence_encoder, "d_p", 384))
        self.d_p = 2 * self.d_sent + self.d_lex
        self.chunk_dim = self.d_sent + self.d_lex if mode == "chunk_mil" else 0
        self.last_chunks: list = []
        self.stats: dict[str, float] = {}

    def encode_batch(self, payloads: list[str]) -> "Any":
        import torch

        if not payloads:
            return torch.zeros((0, self.d_p), dtype=torch.float32)
        per_msg = [chunk_payload(p) for p in payloads]
        uniq: dict[str, int] = {}
        for cs in per_msg:
            for c in cs:
                uniq.setdefault(c, len(uniq))
        texts = list(uniq)
        emb = self.sent.encode_batch(texts)
        emb = emb.detach().float().cpu() if hasattr(emb, "detach") else torch.as_tensor(emb, dtype=torch.float32)
        # per-unique-chunk feature rows: [sentence embedding | lexical vector]
        feats = torch.zeros((len(texts), self.d_sent + self.d_lex), dtype=torch.float32)
        feats[:, : self.d_sent] = emb
        if self.d_lex:
            rows, cols, vals = [], [], []
            for u, c in enumerate(texts):
                for k, v in _lex_vector(c, self.d_lex):
                    rows.append(u)
                    cols.append(k)
                    vals.append(v)
            if rows:
                feats[torch.tensor(rows), self.d_sent + torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
        # pool chunk rows into message vectors, a block of messages at a time
        out = torch.zeros((len(payloads), self.d_p), dtype=torch.float32)
        block = 2048
        for lo in range(0, len(per_msg), block):
            sub = per_msg[lo: lo + block]
            occ = torch.tensor([uniq[c] for cs in sub for c in cs], dtype=torch.long)
            seg = torch.tensor([i for i, cs in enumerate(sub) for _ in cs], dtype=torch.long)
            x = feats[occ]
            n = len(sub)
            cnt = torch.bincount(seg, minlength=n).clamp(min=1).unsqueeze(1).float()
            mean = torch.zeros((n, self.d_sent)).index_add_(0, seg, x[:, : self.d_sent]) / cnt
            mx = torch.full((n, x.shape[1]), -1e9).scatter_reduce(
                0, seg.unsqueeze(1).expand(-1, x.shape[1]), x, reduce="amax", include_self=True)
            out[lo: lo + n, : self.d_sent] = mean
            out[lo: lo + n, self.d_sent: 2 * self.d_sent] = mx[:, : self.d_sent]
            if self.d_lex:
                out[lo: lo + n, 2 * self.d_sent:] = mx[:, self.d_sent:]
        if self.chunk_dim:
            self.last_chunks = [feats[torch.tensor([uniq[c] for c in cs], dtype=torch.long)] for cs in per_msg]
        n = [len(cs) for cs in per_msg]
        self.stats = {"messages": len(payloads), "unique_chunks": len(texts),
                      "chunks_per_message_mean": sum(n) / len(n), "chunks_per_message_max": max(n)}
        return out

    def which_chunk(self, payload: str) -> list[str]:
        """Chunk texts of one payload, for localising a detection."""
        return chunk_payload(payload)
