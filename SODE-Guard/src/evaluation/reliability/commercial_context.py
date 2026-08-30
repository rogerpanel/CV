"""Context object for commercial-baseline comparisons.

Reviewer 1 asked that Snort, Suricata, and Llama Guard be moved to a
separate *reference* category rather than mixed onto the same Pareto
plot as the neural baselines whose threat model matches SODE-Guard's.

This module simply tags each baseline with its threat model so that
downstream plotting code can group them correctly.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CommercialBaselineContext:
    name: str
    category: str            # "neural" | "signature" | "llm_policy"
    threat_model: str        # description
    fair_pgd_grid: bool      # whether the same PGD grid is meaningful

    def as_dict(self) -> dict:
        return dict(name=self.name, category=self.category,
                    threat_model=self.threat_model,
                    fair_pgd_grid=self.fair_pgd_grid)


REGISTRY = {
    "egraphsage":    CommercialBaselineContext("E-GraphSAGE",      "neural",       "ℓ∞ gradient-based",     True),
    "rtids":         CommercialBaselineContext("RTIDS",             "neural",       "ℓ∞ gradient-based",     True),
    "cnn_lstm":      CommercialBaselineContext("CNN-LSTM",          "neural",       "ℓ∞ gradient-based",     True),
    "ids_graphmamba":CommercialBaselineContext("IDS-GraphMamba",    "neural",       "ℓ∞ gradient-based",     True),
    "surrogate_7b":  CommercialBaselineContext("SurrogateIDS-7B",   "neural",       "ℓ∞ gradient-based",     True),
    "sde_tgnn":      CommercialBaselineContext("SDE-TGNN",          "neural",       "ℓ∞ gradient-based",     True),
    "snort3":        CommercialBaselineContext("Snort 3 + SnortML", "signature",    "signature/regex",       False),
    "suricata7":     CommercialBaselineContext("Suricata 7",        "signature",    "signature/regex",       False),
    "llama_guard":   CommercialBaselineContext("Llama Guard 3",     "llm_policy",   "text policy classifier", False),
    "sode_guard":    CommercialBaselineContext("SODE-Guard (ours)", "neural",       "ℓ∞ gradient-based",     True),
}
