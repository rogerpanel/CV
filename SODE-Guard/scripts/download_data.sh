#!/usr/bin/env bash
# Download SODE-Guard benchmark datasets.
#
# Requires:
#   - kaggle CLI configured with ~/.kaggle/kaggle.json for the three Kaggle DSVs,
#   - curl/wget for the canonical CIC and UNSW corpora.
#
# Usage:
#   bash scripts/download_data.sh --datasets ics3d iis3d ids_pqc
#   bash scripts/download_data.sh --all
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${SODE_GUARD_DATA_DIR:-$ROOT/data/raw}"
mkdir -p "$DATA_DIR"

declare -A KAGGLE_DSV=(
  [ics3d]="rogerpanel/ics3d-cloud-edge-iiot-k8s/12483891"
  [iis3d]="rogerpanel/iis3d-unsw-cicids2018-2023/12479689"
  [ids_pqc]="rogerpanel/ids-pqc-tls13-post-quantum/15424420"
)

declare -A URL_TARGETS=(
  [cicids2017]="https://www.unb.ca/cic/datasets/ids-2017.html"
  [cicids2018]="https://www.unb.ca/cic/datasets/ids-2018.html"
  [cicids2023]="https://www.unb.ca/cic/datasets/ids-2023.html"
  [ciciot2023]="https://www.unb.ca/cic/datasets/iotdataset-2023.html"
  [cicddos2019]="https://www.unb.ca/cic/datasets/ddos-2019.html"
  [unswnb15]="https://research.unsw.edu.au/projects/unsw-nb15-dataset"
  [nslkdd]="https://www.unb.ca/cic/datasets/nsl.html"
  [nftoniotv2]="https://staff.itee.uq.edu.au/marius/NIDS_datasets/"
)

ALL_KEYS=("ics3d" "iis3d" "ids_pqc" "cicids2017" "cicids2018" "cicids2023" \
          "ciciot2023" "cicddos2019" "unswnb15" "nslkdd" "nftoniotv2")

REQUESTED=()
if [[ "${1:-}" == "--all" ]]; then
    REQUESTED=("${ALL_KEYS[@]}")
elif [[ "${1:-}" == "--datasets" ]]; then
    shift; REQUESTED=("$@")
else
    echo "Usage: $0 --all | --datasets <key> [<key> ...]"
    echo "Available keys: ${ALL_KEYS[*]}"
    exit 1
fi

fetch_kaggle() {
    local key="$1"; local slug="${KAGGLE_DSV[$key]}"
    local out="$DATA_DIR/$key"
    mkdir -p "$out"
    echo ">> kaggle datasets download -d $slug → $out"
    if ! command -v kaggle >/dev/null; then
        echo "kaggle CLI not found. pip install kaggle and place credentials in ~/.kaggle/."
        exit 1
    fi
    kaggle datasets download -d "${slug%/*}" -p "$out" --unzip
}

fetch_url_landing() {
    local key="$1"; local url="${URL_TARGETS[$key]}"
    echo ">> $key requires manual download from $url"
    echo "   After download, place CSV/parquet files under $DATA_DIR/$key/ to match"
    echo "   the glob declared in src/data/registry.py."
}

for key in "${REQUESTED[@]}"; do
    if [[ -n "${KAGGLE_DSV[$key]:-}" ]]; then
        fetch_kaggle "$key"
    elif [[ -n "${URL_TARGETS[$key]:-}" ]]; then
        fetch_url_landing "$key"
    else
        echo "Unknown dataset key: $key"; exit 1
    fi
done
echo "Datasets staged under $DATA_DIR"
