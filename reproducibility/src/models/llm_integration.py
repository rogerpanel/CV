"""
LLM Integration for Zero-Shot Detection
========================================
Implements Section VII of the paper:
  - Temporal point process prompting            (Section VII-A)
  - Chain-of-thought reasoning scaffolds        (Section VII-A)
  - Zero-shot detection protocol                (Section VII-B)
  - LLM configuration: Meta-Llama-3.1-8B-Instruct

Note: Requires a running LLM inference endpoint (e.g., vLLM, Ollama,
or HuggingFace Transformers). Falls back to template-based heuristics
when no LLM endpoint is available.
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Prompt Templates  (Section VII-A)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a network security analyst specializing in intrusion detection.
You analyze sequences of network events with timestamps and features to identify threats.
For each event sequence, assess the threat level as one of: benign, suspicious, critical.
Provide structured reasoning following the analysis framework below."""

EVENT_NARRATIVE_TEMPLATE = """At time {time:.6f}s ({delta:.6f}s since previous), observed [{event_type}] with features:
  - Source port: {src_port}, Destination port: {dst_port}
  - Protocol: {protocol}
  - Packet size: {pkt_size} bytes
  - Flow duration: {flow_dur:.4f}s
  - Flags: {flags}"""

CHAIN_OF_THOUGHT_SCAFFOLD = """Analyze the event sequence systematically:
1) Identify any reconnaissance activities (port scans, service enumeration)
2) Check for privilege escalation attempts (unusual auth patterns, exploit signatures)
3) Assess lateral movement patterns (internal scanning, credential reuse)
4) Evaluate data exfiltration indicators (large outbound transfers, encoding)
5) Consider benign alternative explanations

Based on temporal patterns, inter-arrival timing, and attack staging:
- Threat assessment: [benign/suspicious/critical]
- Confidence: [0.0-1.0]
- Primary attack category (if applicable): [category]
- Reasoning: [explanation]"""


# ---------------------------------------------------------------------------
# LLM Temporal Reasoner
# ---------------------------------------------------------------------------
class LLMTemporalReasoner:
    """LLM-based zero-shot intrusion detection with temporal reasoning.

    Converts event sequences into natural language narratives and uses
    chain-of-thought prompting for threat assessment.

    Configuration (Section VII-B):
        Model: Meta-Llama-3.1-8B-Instruct
        Context window: 128000 tokens
        Temperature: 0.2
        Top-p: 0.9
        Max output: 256 tokens
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 temperature: float = 0.2, top_p: float = 0.9,
                 max_tokens: int = 256, max_events: int = 64,
                 endpoint_url: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_events = max_events
        self.endpoint_url = endpoint_url

        # Feature name mappings for narrative construction
        self.feature_names = [
            "src_port", "dst_port", "protocol", "pkt_size",
            "flow_dur", "flags", "fwd_pkts", "bwd_pkts",
        ]

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the LLM (HuggingFace Transformers fallback)."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading LLM: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True
            )
            print("LLM loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load LLM ({e}). "
                  f"Using rule-based fallback for zero-shot detection.")
            self._model = "fallback"

    def construct_event_narrative(
            self, timestamps: np.ndarray, features: np.ndarray,
            event_types: Optional[List[str]] = None
    ) -> str:
        """Convert event sequence to natural language narrative.

        Args:
            timestamps: (n_events,) normalised timestamps
            features: (n_events, n_features) feature matrix
            event_types: optional string labels per event
        Returns:
            narrative: formatted event description string
        """
        n = min(len(timestamps), self.max_events)
        lines = []

        for i in range(n):
            delta = timestamps[i] - timestamps[i - 1] if i > 0 else 0.0
            etype = event_types[i] if event_types else "network_event"

            feat = features[i]
            line = EVENT_NARRATIVE_TEMPLATE.format(
                time=float(timestamps[i]),
                delta=float(delta),
                event_type=etype,
                src_port=int(feat[0]) if len(feat) > 0 else 0,
                dst_port=int(feat[1]) if len(feat) > 1 else 0,
                protocol=int(feat[2]) if len(feat) > 2 else 0,
                pkt_size=int(feat[3]) if len(feat) > 3 else 0,
                flow_dur=float(feat[4]) if len(feat) > 4 else 0.0,
                flags=int(feat[5]) if len(feat) > 5 else 0,
            )
            lines.append(line)

        return "\n".join(lines)

    def build_prompt(self, narrative: str) -> str:
        """Construct full prompt with system instructions + CoT scaffold."""
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Event Sequence ---\n{narrative}\n\n"
            f"--- Analysis Instructions ---\n{CHAIN_OF_THOUGHT_SCAFFOLD}"
        )

    def predict(self, timestamps: np.ndarray, features: np.ndarray,
                event_types: Optional[List[str]] = None
                ) -> Dict[str, object]:
        """Zero-shot threat assessment for an event sequence.

        Returns:
            dict with keys: threat_level, confidence, category, reasoning
        """
        narrative = self.construct_event_narrative(
            timestamps, features, event_types
        )
        prompt = self.build_prompt(narrative)

        self._load_model()

        if self._model == "fallback":
            return self._rule_based_predict(timestamps, features)

        return self._llm_predict(prompt)

    def _llm_predict(self, prompt: str) -> Dict[str, object]:
        """Run prediction through the LLM."""
        if self.endpoint_url:
            return self._api_predict(prompt)

        # Local HuggingFace inference
        inputs = self._tokenizer(prompt, return_tensors="pt",
                                  truncation=True, max_length=4096)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=self.max_tokens,
                temperature=self.temperature, top_p=self.top_p,
                do_sample=True,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return self._parse_response(response)

    def _api_predict(self, prompt: str) -> Dict[str, object]:
        """Call external API endpoint (vLLM, Ollama, etc.)."""
        import requests

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(
            f"{self.endpoint_url}/v1/chat/completions",
            json=payload, timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return self._parse_response(text)

    def _parse_response(self, text: str) -> Dict[str, object]:
        """Parse structured LLM response."""
        result = {
            "threat_level": "benign",
            "confidence": 0.5,
            "category": "unknown",
            "reasoning": text,
        }

        text_lower = text.lower()
        if "critical" in text_lower:
            result["threat_level"] = "critical"
        elif "suspicious" in text_lower:
            result["threat_level"] = "suspicious"
        else:
            result["threat_level"] = "benign"

        # Extract confidence if present
        import re
        conf_match = re.search(r"confidence[:\s]+([0-9.]+)", text_lower)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        return result

    def _rule_based_predict(self, timestamps: np.ndarray,
                            features: np.ndarray) -> Dict[str, object]:
        """Heuristic fallback when LLM is unavailable."""
        n = len(timestamps)
        if n < 2:
            return {"threat_level": "benign", "confidence": 0.6,
                    "category": "insufficient_data",
                    "reasoning": "Too few events for assessment."}

        inter_arrival = np.diff(timestamps)
        mean_iat = inter_arrival.mean()
        std_iat = inter_arrival.std()

        # Heuristic scoring
        score = 0.0

        # Rapid bursts suggest scanning/DoS
        if mean_iat < 0.01 and n > 10:
            score += 0.4

        # Regular inter-arrival suggests automated tool
        if std_iat < mean_iat * 0.1 and n > 5:
            score += 0.3

        # Many unique destination ports suggest scanning
        if features.shape[1] > 1:
            unique_dst = len(np.unique(features[:, 1].astype(int)))
            if unique_dst > n * 0.5:
                score += 0.3

        if score >= 0.6:
            level = "critical"
        elif score >= 0.3:
            level = "suspicious"
        else:
            level = "benign"

        return {
            "threat_level": level,
            "confidence": min(0.5 + score, 0.95),
            "category": "heuristic",
            "reasoning": f"Rule-based: IAT={mean_iat:.4f}±{std_iat:.4f}, "
                         f"n_events={n}, score={score:.2f}",
        }

    def batch_predict(self, batch_timestamps: List[np.ndarray],
                      batch_features: List[np.ndarray],
                      batch_event_types: Optional[List[List[str]]] = None
                      ) -> List[Dict[str, object]]:
        """Batch zero-shot prediction."""
        results = []
        for i in range(len(batch_timestamps)):
            etypes = batch_event_types[i] if batch_event_types else None
            result = self.predict(
                batch_timestamps[i], batch_features[i], etypes
            )
            results.append(result)
        return results
