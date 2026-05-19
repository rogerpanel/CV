"""Llama Guard 3 (8B) adapter for network intrusion classification.

The original Llama Guard is a content-safety policy classifier. We adapt it
by serialising each flow as natural-language ``key: value`` text under a
re-written policy schema (the **IDS policy rewrite** of §4.3.4 in the
manuscript) and parsing the binary safe/unsafe verdict into an attack vs
benign label.

Inference is delegated to ``transformers`` if installed; otherwise the class
raises NotImplementedError so callers can skip gracefully.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

from ..data.feature_engineering import ALL_FEATURES


IDS_POLICY_PROMPT = """You are a network-intrusion classifier. Output exactly
one word: ``benign`` or ``attack``.

The user message describes a single network flow as ``feature=value`` pairs.
A flow is ``attack`` if any of the following hold:
  S1 — duration ≤ 0.01 s with > 100 packets sent (volumetric DoS),
  S2 — high SYN flag count with no completed handshake (SYN scan),
  S3 — encrypted-traffic anomaly (cipher_suite_id unseen, ja3_hash_mod outlier),
  S4 — payload entropy ≥ 7.5 with rare alpn_id (covert-channel),
  S5 — post-quantum key-exchange anomalies (pqc_kx_id mismatch),
otherwise the flow is ``benign``.
"""


def _flow_to_text(row: np.ndarray) -> str:
    return "; ".join(f"{n}={float(v):.4g}" for n, v in zip(ALL_FEATURES, row))


class LlamaGuardIDS:
    def __init__(self, model_id: str = "meta-llama/Llama-Guard-3-8B",
                 device: str = "cuda", load_in_4bit: bool = True):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._tok = self._model = None

    def _lazy_load(self):
        if self._tok is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as exc:                       # pragma: no cover
            raise ImportError("Install transformers + bitsandbytes to use LlamaGuardIDS") from exc
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = {"torch_dtype": "auto", "device_map": self.device}
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def predict(self, X: np.ndarray, batch_size: int = 4,
                max_new_tokens: int = 4) -> np.ndarray:
        self._lazy_load()
        preds = np.zeros(X.shape[0], dtype=np.int64)
        for i in range(0, X.shape[0], batch_size):
            chunk = X[i:i + batch_size]
            prompts = [IDS_POLICY_PROMPT + "\n\nFlow: " + _flow_to_text(r) + "\nVerdict:"
                       for r in chunk]
            enc = self._tok(prompts, return_tensors="pt", padding=True).to(self.device)
            out = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            txt = self._tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for j, t in enumerate(txt):
                preds[i + j] = 1 if "attack" in t.lower() else 0
        return preds
