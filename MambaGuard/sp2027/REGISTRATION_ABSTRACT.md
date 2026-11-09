# Abstract registration (IEEE S&P 2027 Cycle 2)

**Title**

What LLM Agent-Protocol Intrusion Detectors Actually Learn

**Abstract** (230 words)

Large-language-model (LLM) agents exchange tool descriptions, tool results and delegated tasks over the Model Context Protocol and the Agent-to-Agent protocol, and learned intrusion detectors are being proposed to flag malicious messages in this traffic. Whether such detectors learn attacks, or only the wording of benchmark attacks, is unknown. We build a measurement framework that renders five benchmarks as protocol episodes, splits them by server, case or template, holds out entire attack wordings and families, and transfers detectors across benchmarks written by different authors. On identical episodes we compare a structured neural detector (selective state-space model, graph attention and chunk-level payload scoring), off-the-shelf prompt-injection detectors, a fine-tuned transformer and an LLM judge with episode-length, lexical (TF-IDF) and label-free novelty controls, over five seeds and under validation-tuned, split-conformal and adaptive-conformal alarm rules. Chunk scoring adds up to 0.38 macro-F1 to the neural detector, yet the lexical classifier matches or beats it on every benchmark; with an attack family held out, AUROC falls to 0.60-0.67 for the neural, lexical and hybrid detectors; a template audit shows that a few dozen n-grams carry most of the lexical signal; and split-conformal thresholds give 17-19% false alarms against a 5% target when test servers are new, which adaptive conformal inference restores to 6-7%. Current agent-security benchmarks reward template recognition, so detectors, and claims about them, must be judged on held-out attack families with drift-aware alarm control.

**Notes**

- Every number above is already measured: the five-seed study and the CPU template audit.
- The off-the-shelf detectors, the fine-tuned transformer, the LLM judge and the cross-benchmark transfer are
  named as methods only. Their results come from the version-4 Kaggle run and go into the paper before the
  submission deadline.
