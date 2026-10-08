# FedLoRAGuard — video script (≈7 minutes)

**Format.** An in-person Findings paper only needs the poster PDF; the video is optional.
- If you record one, keep it **under 8 minutes**, the limit for Findings videos.
- Each module below is timed. Read it while pointing at the poster panel shown in brackets.
- At a calm pace (about 130 words per minute) the whole script runs about 7 minutes.

**Recording checklist (from the Underline instructions).**
- Light from the front, slightly to the side.
- Plain wall or bookcase behind you; no striped or checked clothing.
- Use a headset or USB microphone, not the laptop microphone.
- Turn phone and computer notifications off.
- Frame yourself from the shoulders to the top of your head.
- Record with PowerPoint or ScreenPal. ScreenPal doesn't run on Linux, so use OBS there.
- Use the laser pointer to highlight what you're talking about.
- Export as MP4 and name the file **EMNLP 2026_Find-5651**.

---

## Module 1: Opening (0:00–0:30) · [Header]

Hello, I'm Roger Nick Anaedevha from the National Research Nuclear University MEPhI in Moscow. This is joint work with Alexander Trofimov and Yuri Borodachev.

Our paper is **FedLoRAGuard**. It verifies whether LoRA adapters shared on model marketplaces are backdoored, without any marketplace having to share its adapter weights. It also comes with a formal guarantee.

## Module 2: The problem (0:30–1:30) · [Panel 1]

LoRA is now the standard way to fine-tune large language models. Public hubs host tens of thousands of community adapters, and people download and merge them freely.

That makes them a supply-chain attack surface:
- A backdoored adapter can be merged into a normal task adapter and keep its hidden behaviour while still looking useful.
- A small, cheap LoRA fine-tune can remove most of a chat model's safety training.

Existing detectors such as PEFTGuard work well, but they are centralised: every marketplace must hand its raw adapter weights to one detector. Commercial platforms are unwilling to do that, and it creates a single point of trust.

A detector that looks at one adapter at a time also misses a strong signal: where the adapter came from, and what it was derived from.

## Module 3: Our idea and the graph (1:30–2:30) · [Panel 2, Panel 3 bottom line]

Our idea has three parts:
- Model the whole adapter ecosystem as a graph that changes over time.
- Learn a detector on that graph collaboratively across marketplaces, under differential privacy.
- Prove how many dishonest marketplaces the detector can withstand.

The graph has four node types: adapters, base models, contributors and downstream applications. Six kinds of time-stamped edges connect them, such as *derives from*, *fine-tunes*, *deploys* and *cites*.

The lineage captured by these edges lets a suspicious adapter raise suspicion about its relatives, even across marketplaces.

## Module 4: How it works (2:30–3:45) · [Panel 3 diagram, Panel 4]

Inside each marketplace, a multimodal encoder reads three things about every adapter:
- the singular-value spectrum of its weight update,
- its model card and contributor text,
- its usage statistics.

A dynamic graph network then aggregates information from recent neighbours over time. It combines a DyGFormer-style temporal encoder with relation-aware attention from HGT.

Training is federated. In each round:
1. About a fifth of the marketplaces are sampled.
2. Each one clips its entire model update and adds Gaussian noise. This is client-level differential privacy.
3. The updates are combined with secure aggregation, so the server never sees any single update.
4. FLTrust down-weights updates that point away from a small vetted reference set.

A privacy accountant tracks the cumulative budget. Measured on our runtime, the whole privacy stack adds only about four percent to the time per round.

## Module 5: The guarantee (3:45–4:45) · [Panel 5]

We prove two results.

The first bounds how much one client's update can change once it is clipped. That is what lets us add a calibrated amount of noise.

The second is a certified bound on colluding marketplaces. Using the margin between the top two class probabilities and the privacy budget of a single smoothing step, it gives an integer k-star. No coalition of up to k-star attackers can flip the verdict on a given adapter, as long as each attacker's update stays within a bounded distance of an honest one.

With fifty marketplaces, our models give **k-star equals twenty-five**. This certified result is separate from our empirical stress test, where label-flipping attackers controlled up to a third of the clients without hurting the certificate.

## Module 6: Results (4:45–6:00) · [KPI tiles, Panel 7, Panel 8, Panel 6]

Our benchmark is **LoRAchain-2026**:
- 13,500 benign and backdoored adapters,
- four base-model families,
- ten attack types.

FedLoRAGuard reaches **96.4% macro-F1** and an AUROC of 0.984. That is within **1.7 points** of centralised PEFTGuard, without anyone sharing weights, and it beats every federated baseline.

The ablation shows two things:
- The temporal graph matters most. Replacing it with a static graph costs over four points.
- Removing differential privacy gains only half a point of F1, but the certificate drops to zero. The privacy cost is what buys the guarantee.

For the mechanism: backdoored adapters have a top singular value about **2.4 times larger** than benign ones. A spectral-only detector catches simple backdoors, but drops to about 71% on composite and trigger-free attacks. That is where the graph helps.

## Module 7: Robustness and limits (6:00–6:40) · [Panel 9, Panel 11]

Results hold under stricter tests:
- **Lineage-disjoint and attack-family-held-out splits:** above 92%.
- **Half the lineage edges removed:** about 91%.
- **Stronger privacy budget:** about 94%.

Scanning one adapter takes 0.7 seconds on an A100.

The limits are worth stating plainly:
- The certificate covers bounded-update attackers, not arbitrary ones.
- Part of the lineage graph is synthesised.
- Deployment should use a low false-positive threshold with human review.

## Module 8: Close (6:40–7:00) · [Panel 10, QR code]

To summarise: adapter integrity can be verified across marketplaces without sharing weights, lineage over time is the key signal, and differential privacy becomes a certified guarantee.

The code and benchmark are linked from the QR code on the poster.

Thank you, and I look forward to your questions at the poster session.

---

## Quick facts for live Q&A at the poster

| Question | Answer |
|---|---|
| Privacy level? | Client-level DP: noise σ=1.1, sampling q=0.2, T=100 rounds → ε_T ≈ 13.6 at δ=10⁻⁵ (Opacus PRV). Stronger privacy ε_T=1.0 still gives 94.1% macro-F1. |
| Why is k* bigger than ε_T would suggest? | The certificate uses the per-query smoothing budget ε_r=0.5, not the cumulative training budget. |
| Why not just use the SVD spectrum? | It reaches 90.2% on simple rank-1 backdoors, but 71.4% on composite, weight-poisoning and trigger-free attacks, and has no certificate. |
| Doesn't FLTrust's vetted root set defeat the purpose? | No. The 200 root adapters are vetted once at setup; the detector then screens an unbounded upload stream. |
| Latency? | 0.7 s per adapter on A100, 1.6 s on A10G, 2.9 s on T4. |
| Benchmark realism? | Ten publicly documented attacks; lineage partly synthesised from HuggingGraph and PADBench; the 50/50 split is a design choice, not a base rate. |
