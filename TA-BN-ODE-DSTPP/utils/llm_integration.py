"""
LLM Integration for Zero-Shot Threat Analysis.

Uses Meta-Llama-3.1-8B-Instruct with chain-of-thought prompting
for structured temporal reasoning on novel attack families.

Section 4.6 and Supplementary Section S10:
  - Temperature: 0.2, top-p: 0.9, max 256 output tokens
  - 128K context window
  - Chain-of-thought scaffold with 5-step threat assessment:
    reconnaissance, privilege escalation, lateral movement,
    data exfiltration, benign alternatives
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import warnings

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Event narrative template (Supplementary Section S10)
EVENT_NARRATIVE_TEMPLATE = """You are an expert cybersecurity analyst performing real-time threat assessment.

## Event Summary
- Timestamp: {timestamp}
- Event Type: {event_type}
- Source IP: {src_ip}
- Destination IP: {dst_ip}
- Protocol: {protocol}
- Key Features: {feature_summary}

## Neural ODE Assessment
- Model Confidence: {model_confidence:.2%}
- Predicted Class: {predicted_class}
- Uncertainty (epistemic): {uncertainty:.4f}
- Temporal Intensity: {intensity:.4f}

## Chain-of-Thought Threat Assessment

Analyze this network event through each stage of a potential attack kill chain:

1. **Reconnaissance**: Does this event indicate network scanning, port probing, or information gathering? What specific indicators suggest this?

2. **Privilege Escalation**: Are there signs of credential abuse, exploit attempts, or unauthorized access elevation?

3. **Lateral Movement**: Does the traffic pattern suggest movement between internal systems, pass-the-hash, or remote service exploitation?

4. **Data Exfiltration**: Are there indicators of data staging, unusual outbound transfers, or covert channel communication?

5. **Benign Alternatives**: What legitimate business processes could explain this traffic pattern?

Based on your analysis, provide:
- **Threat Level**: Critical / High / Medium / Low / Benign
- **Confidence**: Your confidence in this assessment (0-100%)
- **Recommended Action**: Block / Alert / Monitor / Allow
- **Reasoning**: Brief justification"""


class LLMThreatAnalyzer:
    """LLM-based zero-shot threat analysis.

    Integrates with the TA-BN-ODE model outputs to provide
    interpretable threat assessments for novel attack families.
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 temperature: float = 0.2, top_p: float = 0.9,
                 max_new_tokens: int = 256, device: str = "cuda"):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._pipe = None

    def _load_model(self):
        """Lazy-load the LLM."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError(
                "transformers library required. Install: pip install transformers accelerate"
            )
        if self._pipe is not None:
            return

        print(f"Loading LLM: {self.model_name}...")
        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("LLM loaded.")

    def analyze_event(self, event_info: Dict,
                      model_output: Dict) -> str:
        """Analyze a single event using the LLM.

        Args:
            event_info: Dictionary with event metadata
                (timestamp, src_ip, dst_ip, protocol, etc.)
            model_output: Dictionary with model predictions
                (logits, confidence, uncertainty, intensity)

        Returns:
            LLM-generated threat analysis string
        """
        self._load_model()

        prompt = EVENT_NARRATIVE_TEMPLATE.format(
            timestamp=event_info.get("timestamp", "N/A"),
            event_type=event_info.get("event_type", "Unknown"),
            src_ip=event_info.get("src_ip", "N/A"),
            dst_ip=event_info.get("dst_ip", "N/A"),
            protocol=event_info.get("protocol", "N/A"),
            feature_summary=event_info.get("feature_summary", "N/A"),
            model_confidence=model_output.get("confidence", 0.0),
            predicted_class=model_output.get("predicted_class", "Unknown"),
            uncertainty=model_output.get("uncertainty", 0.0),
            intensity=model_output.get("intensity", 0.0),
        )

        outputs = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )

        return outputs[0]["generated_text"][len(prompt):]

    def batch_analyze(self, events: List[Dict],
                      model_outputs: List[Dict]) -> List[str]:
        """Analyze a batch of events."""
        return [
            self.analyze_event(e, m)
            for e, m in zip(events, model_outputs)
        ]

    def analyze_without_llm(self, model_output: Dict) -> Dict:
        """Rule-based fallback analysis when LLM is not available.

        Uses model confidence and uncertainty thresholds to provide
        basic threat categorization.
        """
        confidence = model_output.get("confidence", 0.0)
        uncertainty = model_output.get("uncertainty", 1.0)
        predicted_class = model_output.get("predicted_class", "unknown")

        if predicted_class.lower() in ("benign", "normal", "0"):
            if confidence > 0.95 and uncertainty < 0.05:
                return {"threat_level": "Benign", "action": "Allow",
                        "confidence": confidence}
            else:
                return {"threat_level": "Low", "action": "Monitor",
                        "confidence": confidence}

        # Attack predicted
        if confidence > 0.9 and uncertainty < 0.1:
            return {"threat_level": "Critical", "action": "Block",
                    "confidence": confidence}
        elif confidence > 0.7:
            return {"threat_level": "High", "action": "Alert",
                    "confidence": confidence}
        else:
            return {"threat_level": "Medium", "action": "Monitor",
                    "confidence": confidence}
