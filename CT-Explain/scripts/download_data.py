"""CLI: stub for downloading the two Kaggle bundles.

Kaggle requires an authenticated API key (~/.kaggle/kaggle.json). This
script assumes the `kaggle` CLI is installed and delegates to it. If it
isn't present, the script just prints the DOIs so the user can download
manually. The destination is the same layout the dataset loaders expect:

    data/
      cic-iot-2023/
      cse-cicids-2018/
      unsw-nb15/
      ms-guide-2024/
      container-nid/
      edge-iiot/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BUNDLES = {
    "general": {
        "doi": "10.34740/kaggle/dsv/12483891",
        "kaggle_id": "kaggle-dataset-id-placeholder/general-network-traffic",
        "datasets": ["cic-iot-2023", "cse-cicids-2018", "unsw-nb15"],
    },
    "cloud-edge": {
        "doi": "10.34740/KAGGLE/DSV/12479689",
        "kaggle_id": "kaggle-dataset-id-placeholder/cloud-microservices-edge",
        "datasets": ["ms-guide-2024", "container-nid", "edge-iiot"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=list(BUNDLES) + ["all"], default="all")
    ap.add_argument("--dest", default="data")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    targets = list(BUNDLES) if args.bundle == "all" else [args.bundle]
    kaggle = shutil.which("kaggle")

    for key in targets:
        info = BUNDLES[key]
        print(f"\n=== Bundle '{key}' — DOI {info['doi']} ===")
        if kaggle is None:
            print("(kaggle CLI not found — resolve the DOI manually and unpack to data/)")
            print("Expected subdirectories under data/:",
                  ", ".join(info["datasets"]))
            continue
        cmd = [
            kaggle, "datasets", "download", "-d", info["kaggle_id"],
            "--unzip", "-p", str(dest),
        ]
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print("Kaggle download failed:", exc, file=sys.stderr)
            print(f"Download manually from https://doi.org/{info['doi']}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
