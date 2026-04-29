"""Reproduce the manuscript's Kambélé children-cadmium HQ run."""
from __future__ import annotations

from cmhr.health_risk import MonteCarloRiskAssessment


def main():
    mc = MonteCarloRiskAssessment(receptor="children", n_iterations=10_000, seed=20260429)
    cd = mc.run_ingestion(concentration=0.011, element="Cd", matrix="water")
    pb = mc.run_ingestion(concentration=0.038, element="Pb", matrix="water")
    print("Children Cd HQ (water ingestion):", cd.percentiles())
    print("Children Pb HQ (water ingestion):", pb.percentiles())

    as_cr = mc.run_ingestion(concentration=0.058, element="As", matrix="water",
                             carcinogenic=True)
    print("Children As ILCR (water ingestion):", as_cr.percentiles())


if __name__ == "__main__":
    main()
