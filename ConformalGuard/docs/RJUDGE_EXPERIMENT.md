# R-Judge experiment

A real-data experiment for the conformal guard. Everything here runs on public
data with labels produced by other people, and every number it reports comes
out of `scripts/rjudge_run.py`.

## What it tests

| Question | How it is tested |
|---|---|
| Does the false-block guarantee hold on real agent records? | Over many random splits, the share of *safe* test records with a blocked step must average at most alpha, for every scorer. |
| How many unsafe records does a guard block at that budget? | Block rate on the unsafe test records at alpha = 0.05, 0.10, 0.20. Empirical; depends on the scorer. |
| Does a graph help beyond history? | Same features, same loss, same calibration, close parameter counts: a causal Transformer over the node sequence (`seq`) against relation-aware attention over the typed execution graph (`graph`), with paired per-split differences and a sign test. |
| Do the types matter, or only the topology? | `graph_untyped`: same graph and layers, one node type and one relation. |
| Does history matter at all? | `step`: the current step only. |
| How far do the results depend on templates and on the application domain? | Prompt-grouped splits; leave-one-category-out. |
| How does a full-history LLM judge compare? | The per-record predictions released by the R-Judge authors (Llama-3.1-8B), and optionally a judge you run with `scripts/rjudge_llm_judge.py`. |

## Data

[R-Judge](https://github.com/Lordog/R-Judge) (Yuan et al., Findings of EMNLP
2024): 571 records in the repository at the commit recorded in `results.json`,
301 labelled unsafe and 270 safe by its annotators, in five application
categories and two attack types (injection, unintended). The data are not
copied into this repository.

Each record is converted to an execution graph (`conformalguard/rjudge/data.py`):
agent invocations, tool calls, memory entries (environment returns) and
messages, linked by `invokes`, `writes`, `reads`, `replies` and `next`. A
decision step is one agent message; the proposed action is its tool call or
its reply.

## Protocol

For each split (50% train, 25% calibration, 25% test):

1. each scorer is trained on the training records only;
2. a record's score is its maximum step score;
3. the threshold is the ceil((n+1)(1-alpha))-th smallest score of the n safe
   calibration records;
4. a test record is blocked if its score exceeds the threshold.

Step 3 is the trace-level rule (Corollary 2 of the paper). R-Judge labels
records, not steps, so the step-level statement (Proposition 1) cannot be
tested on it.

## Limits of this experiment

State these whenever the numbers are quoted.

- **Single agent.** R-Judge records have one agent with tools. The graphs are
  small (at most 37 nodes) and contain no delegation between agents, so this
  is not a test of multi-agent composition.
- **In-domain training.** Scorers are trained and tested on R-Judge by
  cross-fitting. This is not transfer from another benchmark.
- **Record-level labels.** Nothing is known about which step is the unsafe
  one, so "blocked at or before the violating action" cannot be measured.
- **A dataset artefact.** Unsafe records usually end on the risky tool call
  and safe records on a final answer. The rule "the record ends on a tool
  call" separates the labels well. A runtime guard cannot use it, because it
  does not know which step is the last, and all four neural scorers are
  causal (unit-tested). The pooled logistic-regression baseline and the rule
  itself do see the end of the record and are reported for that reason.
- **Templates.** Only 151 distinct user prompts occur in 571 records. Random
  splits share prompts between training and testing. The grouped splits do
  not, and are the better guide to behaviour on unseen tasks.
- **Small test parts.** About 68 safe and 75 unsafe test records per split;
  the Web and IoT categories have 35 and 30 records in total.
- **No online adaptation.** The record stream is too short for adaptive
  conformal inference to be informative (the Proposition 2 band at T = 100 is
  20 points wide).

## Running it

On a laptop CPU, no downloads beyond R-Judge (TF-IDF + SVD node features):

```bash
git clone https://github.com/Lordog/R-Judge
pip install numpy torch scikit-learn
python scripts/rjudge_run.py --rjudge-dir R-Judge --features lsa --reps 40 --out runs/rjudge_lsa
```

On a GPU, with a frozen sentence encoder for node features: open
`notebooks/rjudge_colab.ipynb` in Colab, or

```bash
pip install sentence-transformers scikit-learn
python scripts/rjudge_run.py --rjudge-dir R-Judge --features st --reps 100 --loco-reps 10 --out runs/rjudge_st
python scripts/rjudge_llm_judge.py --rjudge-dir R-Judge --model Qwen/Qwen2.5-7B-Instruct --out runs/rjudge_judge
```

`--quick` runs three splits with six epochs to check the installation; its
output is marked as a smoke run and is not a result.

Outputs: `results.json` (every per-split number, aggregates, paired tests,
configuration, code and data commits) and `summary.md` (the tables).

`scripts/rjudge_llm_judge.py` has unit tests for its prompt construction and
verdict parsing; its model-loading path needs a GPU and the Hugging Face hub
and has not been executed in the environment where this code was written.
