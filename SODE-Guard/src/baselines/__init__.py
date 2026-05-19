"""Baseline detectors compared against in the manuscript (Table 2).

Nine baselines in total:

  Academic neural IDPS:
    - E-GraphSAGE                     (Lopez-Martin et al., 2020)
    - RTIDS Transformer               (Wu et al., 2022)
    - CNN-LSTM                        (Vinayakumar et al., 2017)
    - IDS-GraphMamba                  (this work / RobustIDPS internal)

  Deployed detectors (RobustIDPS internal):
    - SurrogateIDS-7B                 (7-branch ensemble)
    - SDE-TGNN                        (strongest baseline)

  Commercial:
    - Snort 3 + SnortML stub          (rules + ML head)
    - Suricata 7 stub                 (signature-based)

  LLM-safety:
    - Llama Guard 3 (8B) adapter      (policy-rewritten for IDS)
"""
from .egraphsage_baseline import EGraphSAGEBaseline
from .rtids import RTIDSTransformer
from .cnn_lstm import CNN_LSTM
from .ids_graphmamba import IDSGraphMamba
from .surrogate_ids import SurrogateIDS7B
from .sde_tgnn import SDE_TGNN
from .snort_stub import SnortML
from .suricata_stub import Suricata7
from .llama_guard_adapter import LlamaGuardIDS

__all__ = [
    "EGraphSAGEBaseline", "RTIDSTransformer", "CNN_LSTM", "IDSGraphMamba",
    "SurrogateIDS7B", "SDE_TGNN", "SnortML", "Suricata7", "LlamaGuardIDS",
]
