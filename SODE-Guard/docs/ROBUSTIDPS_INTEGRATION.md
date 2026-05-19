# RobustIDPS Integration

SODE-Guard is part of the RobustIDPS.ai v3 deployment. This note explains how
the model registers with the platform, how it is served, and what guarantees
remain in deployment.

## 1. Model registration

```yaml
deployment:
  registry_id: sode_guard
  robustidps_category: temporal
  served_via: fastapi
  endpoint: /api/predict/sode_guard
  fast_mode: false        # always run full N_mc=8 for the AC certificate
```

The RobustIDPS registry (in the upstream `robustidps_web_app/backend`) loads
the model via `create-alt-weights` distillation from the SurrogateIDS-7B
teacher, then warm-starts SODE-Guard's encoder with E-GraphSAGE weights.

## 2. Co-deployed detectors

| Registry id | Display name | Category |
|---|---|---|
| `surrogate` | 7-Branch Ensemble | ensemble |
| `neural_ode` | Neural ODE (TA-BN + Point Process) | temporal |
| `optimal_transport` | PPFOT-IDS | federated |
| `fedgtd` | Graph Temporal Dynamics | federated |
| `sde_tgnn` | Stochastic Differential Equation TGNN | temporal |
| `cybersec_llm` | Mamba–CrossAttn–MoE | foundation |
| `clrl_unified` | Continual Learning + RL | clrl |
| `lipmamba` | Certified Poisoning Defense | certified |
| `multi_agent_pqc` | Post-Quantum IDS | pqc |
| `ssl_graph_anomaly` | E-GraphSAGE + Transformer Autoencoder | self_supervised |
| **`sode_guard`** | **Stochastic ODE-Guard** | **temporal** |

## 3. Inference path inside the platform

```
PCAP / CSV upload
   ↓
backend/feature_extractor (CICFlowMeter-equivalent → 83-dim vector)
   ↓
backend/model_router (selects detector(s) from registry)
   ↓
sode_guard.forward_mc(x, n_paths=8)
   ↓
certified_radius via Carbery–Wright invert
   ↓
WebSocket /ws/live → React 18 UI (RobustIDPS Active Defence pane)
```

## 4. Defences exposed via the platform

* **Adversarial sweep**: PGD ε-grid surfaced through `/api/eval/pgd`.
* **MCP Security tests**: tool-poisoning, supply-chain, exfiltration, sandbox
  escape — orthogonal to SODE-Guard's flow classification but documented for
  completeness in `robustidps_documentation_v3.tex`.
* **LLM DefensePipeline v2**: prompt-injection, jailbreak, RAG poisoning —
  also orthogonal; SODE-Guard composes with these by being one of the
  network-side detectors that feeds the SOC Intelligence pages.

## 5. Operational guardrails

* SODE-Guard does **not** enable fast mode (which disables MC dropout). The
  `N_mc = 8` MC paths are required for the anti-concentration certificate to
  remain meaningful.
* Throughput on Hetzner CCX23 (8 vCPU / 32 GB RAM) without GPU is
  ~9 400 flows/s; on a single A100 we measured ~9.4 M flows/s with
  batch size 512.
* When the registry detects PQC features in the live stream, the
  `multi_agent_pqc` detector is added in parallel; SODE-Guard's certificate
  remains valid because it is computed on the same 83-dim feature vector.
