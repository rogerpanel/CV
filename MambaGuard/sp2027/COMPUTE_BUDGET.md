# Compute Budget --- MambaGuard

This one-pager documents the hardware, wall-clock, energy, and carbon
footprint of all experiments that back the paper, consistent with the
IEEE S&P 2027 reporting guidance.

## Hardware

- Training and adaptive attacks: 1 x NVIDIA A100 80GB SXM (TDP 400 W),
  on a single node with an AMD EPYC host CPU.
- Inference latency tests: Intel Xeon host CPU (single socket, 16
  physical cores), no GPU.
- No multi-node training; no distributed data-parallel runs.

## Training (per dataset)

| Quantity                        | Value                                 |
|---------------------------------|---------------------------------------|
| Seeds per run                   | 5                                     |
| Epochs per seed                 | 30                                    |
| Wall-clock per epoch            | ~27 min                               |
| Wall-clock per seed             | 5 x 27 min x 30 ~= 67.5 GPU-h / seed? |
| Correct per-dataset total       | 5 seeds x 30 epochs x 27 min ~= 67 h  |

Reading: for one dataset we burn five seeds x thirty epochs at roughly
twenty-seven minutes per epoch, which is approximately 67 GPU-hours of
A100 time. The same recipe runs over three datasets, so training
contributes approximately 3 x 67 ~= 200 GPU-hours.

## Adaptive attacks

The adaptive-attack sweeps (PGD on the detector, GCG on the projector,
LLM-driven rewriting, graph evasion) share the same A100 and total
approximately 50 GPU-hours across all attacks and seeds reported in the
paper.

## Inference latency

CPU-only microbenchmarks (batch size 1, batch size 32, varying sequence
length) complete in well under one hour on the Xeon host; they are not
included in the GPU total.

## Total

| Bucket                      | GPU-hours (A100) |
|-----------------------------|------------------|
| Training (3 datasets)       | ~200             |
| Adaptive attacks            | ~50              |
| Ablations and sweeps        | ~50              |
| **Total**                   | **~300**         |

## Energy and carbon estimate

Using the ML CO2 Impact calculator with a representative US-average grid
mix and A100 draw at ~400 W:

- Energy: 300 h x 0.4 kW ~= 120 kWh.
- Carbon: 120 kWh x ~0.4 kg CO2e / kWh ~= **~50 kg CO2e**.

For calibration, a single large-scale open LLM pre-training run (think
Llama-class, O(10^5) GPU-h) emits several orders of magnitude more; our
footprint is well below one round-trip transatlantic flight for a single
passenger, and well below any model-training-scale baseline in the
related literature.

## Caveats

- The per-epoch wall-clock number is averaged across datasets; the
  largest dataset is ~15% slower per epoch, the smallest ~20% faster.
- The adaptive-attack figure excludes failed attack-config sweeps that
  were pruned early; those add at most ~10 GPU-h and are included in
  the "ablations and sweeps" bucket.
- Generative-AI data synthesis is reported separately in
  `GENAI_USAGE.md` and is not double-counted here.
