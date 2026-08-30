# Packet-Semantic Constraints for Problem-Space Attacks

Reviewer 2 asked whether adversarial perturbations that live in the
83-dim feature space can actually be realised by network traffic. The
answer is: not all of them. Certain features are *constrained* by the
packet grammar; others are *derived* from raw counts and must remain
consistent with them.

This document lists the constraint families implemented by
`src/attacks/semantic/feasibility.py`. All constraints are applied at
every step of the PGD / C&W loops so the reported robust accuracy uses
only *feasible* perturbations.

## Constraint families

1. **Box constraints.** Every feature has a natural non-negative
   lower bound (byte counts, packet counts, durations, TLS extension
   counts) or a flag bit domain `{0, 1}`.
2. **Integer constraints.** Packet counts, byte totals, TLS enum
   indices (`cipher_suite_id`, `pqc_kx_id`, `alpn_id`, ...) round
   to the nearest non-negative integer.
3. **Flag constraints.** All `flag_*` features (`flag_psh`, `flag_syn`,
   `flag_ack`, `flag_fin`, ...) are Bernoulli — clipped and rounded.
4. **Derived-ratio recomputation.** Whenever both numerator and
   denominator of a derived ratio (`fwd_bwd_byte_ratio`,
   `fwd_bwd_pkt_ratio`, `down_up_ratio`) are present, the ratio is
   recomputed from the projected raw counts so the adversary cannot
   desynchronise it.
5. **Protocol structural constraints.** `min_seg_size_fwd ≤
   avg_fwd_seg_size ≤ fwd_pkt_len_max`; `total_pkts = fwd_packets +
   bwd_packets`; `handshake_rtt ≥ 0`; TLS/PQC identifiers ≥ 0.

## What the constraints do NOT enforce

The projector is *conservative* — it is permissive of features that
cannot be tied to a single raw value by the CICFlowMeter grammar, such
as `entropy_payload` or the JA4 hash bucket. These are allowed to move
freely inside their box because a real attacker can, in principle,
manipulate them via payload crafting. Table 9 of the revised
manuscript reports both the unconstrained ("feature-space") and the
constrained ("problem-space") attack results so the reader can see the
delta.

## Empirical impact

Under `ε = 0.03` PGD-40 on the ICS3D test partition:

| Threat model  | SDE-TGNN F1 | SODE-Guard F1 | Gap  |
|---            |---          |---            |---   |
| Feature-space | 0.858       | 0.922         | +6.4 |
| Problem-space | 0.842       | 0.914         | +7.2 |

SODE-Guard's advantage *widens* under the tighter constraints, which
is consistent with the anti-concentration argument: the effective
chaos degree of the smoothed margin is unchanged by projecting to a
smaller feasibility set, so the certificate's ε^{1/d*} tail decay
still applies.
