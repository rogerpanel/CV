"""Demo: adaptive conformal inference on a stream with an abrupt shift.

Shows the Proposition 2 band and that the static threshold drifts while the
adaptive one stays inside the band at every horizon.  Synthetic scores only.
"""

from __future__ import annotations

import numpy as np

from conformalguard.adaptive.gibbs_candes import aci_bound, run_stream


def main():
    rng = np.random.default_rng(0)
    cal = rng.normal(0.0, 1.0, 2000)
    stream = np.concatenate([rng.normal(0.0, 1.0, 10_000),
                             rng.normal(0.5, 1.2, 10_000)])
    aci = run_stream(cal, stream, alpha=0.05, gamma=0.05)
    static = run_stream(cal, stream, alpha=0.05, gamma=0.05, adaptive=False)
    for T in (5_000, 10_000, 11_000, 20_000):
        print(f"T={T:>6}  ACI {aci['cum_err'][T-1]:.4f} "
              f"(|.-0.05| <= {aci_bound(T, 0.05, 0.05):.4f})   "
              f"static {static['cum_err'][T-1]:.4f}")


if __name__ == "__main__":
    main()
