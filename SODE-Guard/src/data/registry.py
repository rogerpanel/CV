"""Dataset registry.

The manuscript trains on three Kaggle benchmark suites (ICS3D, IIS3D, IDS-PQC)
totalling 1.84×10⁷ flows, plus the canonical public IDS corpora referenced in
the literature review (Section 2.3 / Table 1). Each entry returns a torch
``DataLoader`` over the 83-dim feature vector and the integer label.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .feature_engineering import FlowFeatureExtractor, standardize
from .splits import stratified_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_glob: str        # relative to data_root
    num_classes: int
    doi: str             # Kaggle DOI or URL
    label_column: str = "label"


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "ics3d":      DatasetSpec("ICS3D",       "ics3d/*.csv",       17, "10.34740/kaggle/dsv/12483891"),
    "iis3d":      DatasetSpec("IIS3D",       "iis3d/*.csv",       23, "10.34740/kaggle/dsv/12479689"),
    "ids_pqc":    DatasetSpec("IDS-PQC",     "ids_pqc/*.csv",      9, "10.34740/kaggle/dsv/15424420"),
    "cicids2017": DatasetSpec("CICIDS2017",  "cicids2017/*.csv",  15, "https://www.unb.ca/cic/datasets/ids-2017.html"),
    "cicids2018": DatasetSpec("CIC-IDS2018", "cicids2018/*.csv",  14, "https://www.unb.ca/cic/datasets/ids-2018.html"),
    "cicids2023": DatasetSpec("CIC-IDS2023", "cicids2023/*.csv",  33, "https://www.unb.ca/cic/datasets/ids-2023.html"),
    "ciciot2023": DatasetSpec("CIC-IoT-2023","ciciot2023/*.csv",  33, "https://www.unb.ca/cic/datasets/iotdataset-2023.html"),
    "cicddos2019":DatasetSpec("CIC-DDoS-2019","cicddos2019/*.csv",13, "https://www.unb.ca/cic/datasets/ddos-2019.html"),
    "unswnb15":   DatasetSpec("UNSW-NB15",   "unswnb15/*.csv",    10, "https://research.unsw.edu.au/projects/unsw-nb15-dataset"),
    "nslkdd":     DatasetSpec("NSL-KDD",     "nslkdd/*.csv",       5, "https://www.unb.ca/cic/datasets/nsl.html"),
    "nftoniotv2": DatasetSpec("NF-ToN-IoT-V2","nftoniotv2/*.csv", 10, "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"),
}


def list_datasets() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def _load_csvs(spec: DatasetSpec, data_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    files = sorted(Path(data_root).glob(spec.csv_glob))
    if not files:
        raise FileNotFoundError(
            f"No CSVs matched {spec.csv_glob} under {data_root}. "
            f"Run scripts/download_data.sh --datasets {spec.name.lower()}"
        )
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if spec.label_column not in df.columns:
        raise ValueError(f"Dataset {spec.name} missing column '{spec.label_column}'")
    y = df[spec.label_column]
    X = df.drop(columns=[spec.label_column])
    return X, y


def get_loader(name: str, *,
               data_root: str | os.PathLike = "data/raw",
               split: str = "train",
               batch_size: int = 128,
               num_workers: int = 4,
               seed: int = 42,
               return_stats: bool = False
               ) -> DataLoader:
    """Return a stratified, standardised DataLoader for one of the registered
    benchmarks. The dataset is materialised in RAM as ``TensorDataset`` —
    acceptable up to ICS3D (≈18 M rows × 83 floats ≈ 6 GB)."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list_datasets()}")
    spec = DATASET_REGISTRY[name]
    X_df, y_ser = _load_csvs(spec, Path(data_root))

    # Label encoding
    labels, _ = pd.factorize(y_ser, sort=True)
    extractor = FlowFeatureExtractor()
    X = extractor(X_df)
    X, mean, std = standardize(X)

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratified_split(X, labels, seed=seed)
    parts = {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)}
    Xp, yp = parts[split]

    ds = TensorDataset(torch.from_numpy(Xp).float(),
                       torch.from_numpy(np.asarray(yp)).long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=num_workers, pin_memory=True, drop_last=(split == "train"))
    if return_stats:
        return loader, {"mean": mean, "std": std, "num_classes": spec.num_classes,
                        "doi": spec.doi, "name": spec.name}
    return loader
