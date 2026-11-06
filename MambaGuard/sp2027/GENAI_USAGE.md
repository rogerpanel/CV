# Generative AI Usage Statement --- MambaGuard

In line with the IEEE S&P 2027 Call for Papers requirement to disclose
use of generative AI, we declare the following.

## Research use

Two closed, commercial large language models were used as data-generation
oracles for the synthetic agent traces evaluated in Section 4 of the
paper: **Anthropic Claude 3.5 Sonnet** and **OpenAI GPT-4o**. Both were
accessed via their official APIs during August-October 2026. Their role
was limited to generating candidate agent utterances, tool-call plans,
and adversarial paraphrases used as inputs to the MCP/A2A protocol
simulators; they did not author any part of the manuscript's claims or
analysis and did not run the attacks or the defence.

## Reproducibility limitation

Closed models may be updated by their providers and may return different
outputs for identical prompts over time. To let independent researchers
reproduce our benchmark inputs, we release (i) the random seeds used for
each generation, (ii) the full prompt templates in
`mambaguard/data/prompts/`, (iii) the complete API-call logs, and
(iv) the cached responses themselves, so that evaluation can proceed
without re-querying the vendor models.

## Compute footprint

Total queries are approximately 2k per benchmark across the six
benchmarks, for an order-of-magnitude cost of USD 40 and carbon impact
that is negligible compared with the training-time footprint reported
in `COMPUTE_BUDGET.md`.

## Editorial use

The authors used a general-purpose LLM assistant to polish prose in
non-technical passages of the manuscript and this supplementary material.
All claims, proofs, experiments, numbers, and figures are the authors'
own and have been reviewed by the authors.
