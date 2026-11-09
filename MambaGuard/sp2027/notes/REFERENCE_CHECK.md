# Reference and detector-model check

Checked on 2026-11-09 from a cloud sandbox that could reach arXiv, GitHub and
the Hugging Face API, but **not** the Hugging Face file CDN (`us.aws.cdn.hf.co`
was refused by the sandbox proxy). Small repo files (config.json,
tokenizer.json, tokenizer_config.json) downloaded; model weights and
`spm.model` did not. So the real-classifier scores and the LLM-judge scores
in Task 2 could **not** be produced here (details below).

---

## Task 1: references (from export.arxiv.org/api/query)

### a) arXiv 2510.05244

- **Title:** Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks?
- **Authors (8):** Rishika Bhagwatkar, Kevin Kasa, Abhay Puri, Gabriel Huang, Irina Rish, Graham W. Taylor, Krishnamurthy Dj Dvijotham, Alexandre Lacoste → *R. Bhagwatkar et al.*
- **Year:** 2025 (v1 2025-10-06; v2 2026-03-23)
- **Venue:** none given (no comment or journal-ref on arXiv) → cite as arXiv preprint. Any conference version is **UNVERIFIED**.
- IEEE: R. Bhagwatkar *et al.*, "Indirect prompt injections: Are firewalls all you need, or stronger benchmarks?" arXiv:2510.05244, 2025.

**Summary:** The paper shows that a simple, model-agnostic defence placed at the agent–tool interface works well. It uses a Tool-Input Firewall ("Minimizer") and a Tool-Output Firewall ("Sanitizer"), and it reaches perfect security with high utility on all four public benchmarks (AgentDojo, Agent Security Bench, InjecAgent, tau-Bench). The authors argue this mostly shows the benchmarks are weak. They report flawed success metrics, implementation bugs and, above all, weak attacks, and they give fixes for AgentDojo and ASB plus best practices for benchmark design. They also propose a three-stage attack (standard injections, then second-order attacks, then adaptive attacks) and conclude that current agent-security benchmarks are easy to saturate and need strong adaptive attacks and better metrics.

### b) arXiv 2410.22770 (InjecGuard → PIGuard)

- **arXiv title:** InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models
- **arXiv authors:** Hao Li, Xiaogeng Liu (v3, 2025-03-30; v1 2024-10-30)
- **Renamed and published:** yes. The official GitHub README (leolee99/PIGuard) says: "Due to some licensing issues, the model name has been changed from InjecGuard to PIGuard" and "[2025.5.15] Our paper has been accepted to ACL 2025", linking to `aclanthology.org/2025.acl-long.1468.pdf`.
- **Published title (from the README BibTeX):** PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free
- **Published authors (from the README BibTeX):** Hao Li, Xiaogeng Liu, Ning Zhang, Chaowei Xiao
- IEEE: H. Li, X. Liu, N. Zhang, and C. Xiao, "PIGuard: Prompt injection guardrail via mitigating overdefense for free," in *Proc. 63rd Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2025.
- **UNVERIFIED:** The ACL Anthology page itself (aclanthology.org was blocked here), so the exact Anthology title, the page numbers and the "long paper / 2025.acl-long.1468" id are taken only from the authors' README. The arXiv abstract still uses the InjecGuard title and lists only two authors.

### c) PromptShield (Jacob et al.)

- **arXiv id:** 2501.15145 (correct; `ti:PromptShield` search returns it plus an unrelated 2026 "PromptShield Home" paper, arXiv 2608.05495, UbiComp 2026 poster)
- **Title:** PromptShield: Deployable Detection for Prompt Injection Attacks
- **Authors (5):** Dennis Jacob, Hend Alzahrani, Zhanhao Hu, Basel Alomair, David Wagner → *D. Jacob et al.*
- **Year:** 2025 (v1 2025-01-25, v2 2025-04-12)
- **Venue:** arXiv comment: "ACM CODASPY 2025; extended technical report".
- IEEE: D. Jacob *et al.*, "PromptShield: Deployable detection for prompt injection attacks," in *Proc. 15th ACM Conf. Data Appl. Secur. Privacy (CODASPY)*, 2025. (The arXiv version is the extended tech report. The edition number "15th" is from memory → **UNVERIFIED**; pages are **UNVERIFIED**.)

### d) arXiv 2405.11449 NetMamba

- **Title:** NetMamba: Efficient Network Traffic Classification via Pre-training Unidirectional Mamba
- **Authors (6):** Tongze Wang, Xiaohui Xie, Wenduo Wang, Chuyi Wang, Youjian Zhao, Yong Cui → *T. Wang et al.*
- **Venue:** **ICNP 2024**, not INFOCOM. arXiv has no journal-ref, but the official repo (wangtz19/NetMamba) README says "ICNP 2024" and gives BibTeX with `booktitle={2024 IEEE 32nd International Conference on Network Protocols (ICNP)}, pages={1--11}, year={2024}`.
- IEEE: T. Wang *et al.*, "NetMamba: Efficient network traffic classification via pre-training unidirectional Mamba," in *Proc. IEEE 32nd Int. Conf. Netw. Protocols (ICNP)*, 2024, pp. 1–11.

---

## Task 2: detector models

### Availability without a token (Hugging Face API, `token=False`)

| model | gated? | config.json without token | id2label |
|---|---|---|---|
| protectai/deberta-v3-base-prompt-injection-v2 | no | yes | `{0: SAFE, 1: INJECTION}` |
| leolee99/PIGuard | no | yes | `{0: benign, 1: injection}` |
| leolee99/InjecGuard | no | yes | `{0: benign, 1: injection}` (same files as PIGuard) |
| meta-llama/Llama-Prompt-Guard-2-86M | **gated (manual)** → 401 without token | no | not readable here (model card, from memory: `LABEL_0` = benign, `LABEL_1` = malicious → **UNVERIFIED**) |
| meta-llama/Prompt-Guard-86M (fallback) | **gated (manual)** → 401 | no | not readable (from memory: `BENIGN, INJECTION, JAILBREAK` → **UNVERIFIED**) |
| Qwen/Qwen2.5-1.5B-Instruct | no | yes | n/a (causal LM) |
| microsoft/deberta-v3-small | no | yes | n/a (base model) |

**Important, PIGuard/InjecGuard need `trust_remote_code=True`.** Their config has
`model_type: "piguard"`, `architectures: ["PIGuard"]` and an `auto_map` pointing to
`modeling_piguard.PIGuard`. `load_classifier()` in `kaggle/run_detector_baselines.py`
does not pass `trust_remote_code`. With transformers 5.x this triggers an interactive
"Do you wish to run the custom code? [y/N]" prompt. In a notebook with no stdin it will
fail or hang, and without remote code transformers warns "model of type `piguard` to
instantiate a model of type ``". The fix (left to the owning session, since I must not
edit files) is `trust_remote_code=True` for the leolee99 repos. The remote code is
small and readable: `PIGuard(DebertaV2ForSequenceClassification)` with a fresh
`Linear(hidden, num_labels)` on the [CLS] token, returning `SequenceClassifierOutput(logits=...)`.
Its `forward(input_ids, attention_mask, **kwargs)` absorbs `token_type_ids`, so the
script's `self.model(**enc).logits` call is compatible.

### Smoke run

Benchmarks fetched with `bash kaggle/setup_benchmarks.sh /tmp/mgdata mcptox injecagent` (OK:
`/tmp/mgdata/MCPTox-Benchmark` 24 MB, `/tmp/mgdata/InjecAgent` 8.3 MB). Needed
`pip install einops scikit-learn` as well (einops is imported via run_kaggle_study.py).
Note the script writes to `<--out>_smoke`, i.e. `/tmp/bl_smoke`, not `/tmp/bl`.

```
python kaggle/run_detector_baselines.py --smoke --studies main --benchmarks mcptox injecagent \
    --detectors protectai piguard promptguard2 lexical --data-root /tmp/mgdata --out /tmp/bl
```

Printed SKIP lines (no HF_TOKEN):

```
SKIP protectai: protectai/deberta-v3-base-prompt-injection-v2: OSError: Can't load tokenizer for 'protectai/deberta-v3-base-prompt-injection-v2'. ...
SKIP piguard: leolee99/PIGuard: OSError: Can't load tokenizer for 'leolee99/PIGuard'. ...; leolee99/InjecGuard: OSError: Can't load tokenizer for 'leolee99/InjecGuard'. ...
SKIP promptguard2: meta-llama/Llama-Prompt-Guard-2-86M: OSError: You are trying to access a gated repo. ... 401 ...; meta-llama/Prompt-Guard-86M: OSError: You are trying to access a gated repo. ... 401 ...
main__mcptox__lexical__all__seed0: AUROC 1.000 [1.000, 1.000]  macro-F1 1.000  (0.8s)
main__injecagent__lexical__all__seed0: AUROC 1.000 [1.000, 1.000]  macro-F1 1.000  (0.0s)
done in 10.2 min
```

Summary table:

```
| case | lexical |
|---|---|
| injecagent | 1.000 (1.000) |
| mcptox | 1.000 (1.000) |
```

(smoke test sizes: injecagent n_test=15 with 3 benign; mcptox n_test=24 with 12 benign.)

**Why protectai and piguard were skipped: this sandbox, not the script.** The tokenizer
error comes from `spm.model` failing to download: the file is stored in LFS and served from
`us.aws.cdn.hf.co`, which this sandbox's proxy refuses (403). With Xet enabled (the
default) the download first hangs inside `huggingface_hub.xet_get`, which is most of
the 10.2 min runtime. `config.json` / `tokenizer.json` (non-LFS) download fine. On
Kaggle with internet on, these models should load. If downloads hang there too, set
`HF_HUB_DISABLE_XET=1`. promptguard2 is a genuine gating skip and needs `HF_TOKEN` plus an
accepted Llama licence.

**Are the scores sensible?** Only lexical ran. AUROC 1.000 on both benchmarks is
"sensible" in direction (attack episodes score higher), but on 15–24 test episodes it
says nothing beyond perfect separability by surface tokens. That is consistent with the
paper's shortcut-learning point, not evidence of a strong detector. I could not check
the transformer detectors' score direction here. Do this on Kaggle: mean prob of
label=1 episodes should exceed that of label=0.

### Benign-label detection (`ZeroShot.__init__`, `benign = [...]`)

```python
benign = [i for i, l in id2label.items() if any(w in l for w in ("safe", "benign", "legit", "label_0", "negative"))]
self.benign = benign[0] if benign else 0
```

| model | id2label (lower-cased) | picked index | correct? |
|---|---|---|---|
| protectai v2 | {0: safe, 1: injection} | 0 ("safe") | **yes** |
| PIGuard / InjecGuard | {0: benign, 1: injection} | 0 ("benign") | **yes** (once it loads with `trust_remote_code=True`) |
| Llama-Prompt-Guard-2-86M | {0: label_0, 1: label_1} (UNVERIFIED, gated) | 0 ("label_0") | **yes**, if the card's LABEL_0 = benign holds |
| Prompt-Guard-86M (v1 fallback) | {0: benign, 1: injection, 2: jailbreak} (UNVERIFIED) | 0 | **yes**. Score = P(injection)+P(jailbreak), which is fine for this use |

So it picks the right index for every model in the list. Two latent fragilities, neither triggered by these models:
(1) substring `"safe"` also matches `"unsafe"`, so a model with `{0: UNSAFE, 1: SAFE}` would pick 0 (wrong). An exact-match or `startswith` test would be safer.
(2) the fallback `0` is silent; an assert or log line printing `self.labels` and `self.benign` would make a wrong pick visible in the run log.

### LLM judge (Qwen/Qwen2.5-1.5B-Instruct)

**Skipped: not runnable here.** The weights (`model.safetensors`, LFS) cannot be downloaded
from this sandbox, for the same CDN block as above. Not a CPU-speed issue. What I could check
with the downloadable tokenizer:

- Yes/No are single tokens: `'Yes'`→[9454], `' Yes'`→[7414], `'No'`→[2753], `' No'`→[2308]. So
  `enc(w)[-1]` takes the whole word and the `{Yes, ' Yes'}` vs `{No, ' No'}` logsumexp is well formed.
- `pad_token` is `<|endoftext|>` (already set). The script sets `padding_side="left"`, which is right
  for reading `logits[:, -1, :]`.
- **Possible bug:** `tok.truncation_side` is `right` (default). In `Judge.chunk_scores` the
  prompt is tokenized with `truncation=True, max_length=judge_max_len` (768). If a chunk is
  long in tokens (up to 2000 chars; dense JSON, base64 or CJK text can exceed ~700 tokens),
  right truncation cuts off the question and the `<|im_start|>assistant` generation
  prompt. The last-token logits would then not be an answer to the yes/no question.
  Switching to `truncation_side = "left"` would not help, because it would cut the
  instruction header instead. Better to truncate the chunk text by tokens before formatting the
  prompt, e.g. `c = tok.decode(tok.encode(c, add_special_tokens=False)[:judge_max_len-200])`,
  and call the tokenizer without truncation.

The 20 benign/injected judge scores requested are therefore **not available** from this
session. They need to be rerun where `huggingface.co` LFS downloads work (Kaggle).
