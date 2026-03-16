"""
Dataset loading and sequential task splitting for the 6 benchmark datasets.

Datasets (84.2M total records, 83 unique attack classes):
  General Network Traffic (DOI: 10.34740/kaggle/dsv/12483891):
    - CIC-IoT-2023:      1,621,834 flows, 46 features, 33 attack classes
    - CSE-CIC-IDS2018:  16,232,943 flows, 79 features, 14 attack classes
    - UNSW-NB15:         2,540,044 flows, 49 features,  9 attack classes

  Cloud, Microservices & Edge (DOI: 10.34740/KAGGLE/DSV/12479689):
    - CloudSec-2024:    38,412,201 flows, 62 features, 11 attack classes
    - ContainerIDS:     18,764,592 flows, 54 features,  8 attack classes
    - EdgeIIoT-2024:     6,628,386 flows, 61 features,  8 attack classes
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ── Dataset Metadata ─────────────────────────────────────────────────
DATASET_REGISTRY = {
    "cic_iot_2023": {
        "name": "CIC-IoT-2023",
        "group": "general_network",
        "num_flows": 1_621_834,
        "num_features": 46,
        "num_attack_classes": 33,
        "benign_pct": 12.4,
        "label_column": "label",
        "kaggle_doi": "10.34740/kaggle/dsv/12483891",
    },
    "cse_cic_ids2018": {
        "name": "CSE-CIC-IDS2018",
        "group": "general_network",
        "num_flows": 16_232_943,
        "num_features": 79,
        "num_attack_classes": 14,
        "benign_pct": 83.1,
        "label_column": "Label",
        "kaggle_doi": "10.34740/kaggle/dsv/12483891",
    },
    "unsw_nb15": {
        "name": "UNSW-NB15",
        "group": "general_network",
        "num_flows": 2_540_044,
        "num_features": 49,
        "num_attack_classes": 9,
        "benign_pct": 55.0,
        "label_column": "attack_cat",
        "kaggle_doi": "10.34740/kaggle/dsv/12483891",
    },
    "cloudsec_2024": {
        "name": "CloudSec-2024",
        "group": "cloud_edge",
        "num_flows": 38_412_201,
        "num_features": 62,
        "num_attack_classes": 11,
        "benign_pct": 71.3,
        "label_column": "label",
        "kaggle_doi": "10.34740/KAGGLE/DSV/12479689",
    },
    "container_ids": {
        "name": "ContainerIDS",
        "group": "cloud_edge",
        "num_flows": 18_764_592,
        "num_features": 54,
        "num_attack_classes": 8,
        "benign_pct": 68.7,
        "label_column": "label",
        "kaggle_doi": "10.34740/KAGGLE/DSV/12479689",
    },
    "edge_iiot_2024": {
        "name": "EdgeIIoT-2024",
        "group": "cloud_edge",
        "num_flows": 6_628_386,
        "num_features": 61,
        "num_attack_classes": 8,
        "benign_pct": 42.1,
        "label_column": "Attack_type",
        "kaggle_doi": "10.34740/KAGGLE/DSV/12479689",
    },
}

# Cross-dataset sequential order (Section IV-D of the paper)
CROSS_DATASET_ORDER = [
    "cic_iot_2023",       # 33 classes
    "unsw_nb15",          # 9 classes, 7 novel
    "cse_cic_ids2018",    # 14 classes, 5 novel
    "cloudsec_2024",      # 11 classes, 4 novel
    "container_ids",      # 8 classes, 2 novel
    "edge_iiot_2024",     # 8 classes, 2 novel
]


class NIDSDataset(Dataset):
    """PyTorch dataset for network intrusion detection flows."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class DatasetLoader:
    """
    Load and preprocess the 6 benchmark datasets.

    Handles CSV/Parquet loading, feature cleaning (inf/NaN removal),
    label encoding, and standardisation.
    """

    def __init__(self, data_dir: str, max_features: int = 79):
        self.data_dir = Path(data_dir)
        self.max_features = max_features
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def load_dataset(
        self,
        dataset_key: str,
        subsample: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """
        Load a single dataset by key.

        Args:
            dataset_key: Key from DATASET_REGISTRY.
            subsample: Fraction to subsample (for development/testing).

        Returns:
            features (N, D), labels (N,), label_encoder
        """
        meta = DATASET_REGISTRY[dataset_key]
        dataset_dir = self.data_dir / dataset_key
        label_col = meta["label_column"]

        logger.info(f"Loading {meta['name']} from {dataset_dir}")

        # Try multiple file formats
        df = self._load_files(dataset_dir)

        if subsample is not None and subsample < 1.0:
            df = df.sample(frac=subsample, random_state=42)
            logger.info(f"Subsampled to {len(df)} records")

        # Extract labels
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available: {list(df.columns[:10])}..."
            )
        labels_raw = df[label_col].astype(str).values
        df = df.drop(columns=[label_col], errors="ignore")

        # Clean features: drop non-numeric, handle inf/NaN
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Pad or truncate to max_features
        features = df.values.astype(np.float32)
        if features.shape[1] < self.max_features:
            pad = np.zeros(
                (features.shape[0], self.max_features - features.shape[1]),
                dtype=np.float32,
            )
            features = np.hstack([features, pad])
        elif features.shape[1] > self.max_features:
            features = features[:, : self.max_features]

        # Encode labels
        le = LabelEncoder()
        labels = le.fit_transform(labels_raw)
        self.label_encoders[dataset_key] = le

        # Standardise features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        self.scalers[dataset_key] = scaler

        logger.info(
            f"Loaded {meta['name']}: {features.shape[0]} samples, "
            f"{features.shape[1]} features, {len(le.classes_)} classes"
        )
        return features, labels, le

    def _load_files(self, dataset_dir: Path) -> pd.DataFrame:
        """Load all CSV/Parquet files in a directory."""
        dfs = []
        for ext in ["*.csv", "*.parquet", "*.csv.gz"]:
            for f in sorted(dataset_dir.glob(ext)):
                logger.info(f"  Reading {f.name}")
                if f.suffix == ".parquet":
                    dfs.append(pd.read_parquet(f))
                else:
                    dfs.append(
                        pd.read_csv(f, low_memory=False, on_bad_lines="skip")
                    )
        if not dfs:
            raise FileNotFoundError(
                f"No CSV/Parquet files found in {dataset_dir}. "
                f"Download datasets from Kaggle using the DOIs in the config."
            )
        return pd.concat(dfs, ignore_index=True)

    def load_all_datasets(
        self, subsample: Optional[float] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, LabelEncoder]]:
        """Load all 6 datasets."""
        results = {}
        for key in DATASET_REGISTRY:
            try:
                results[key] = self.load_dataset(key, subsample=subsample)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {key}: {e}")
        return results


class SequentialTaskSplitter:
    """
    Create sequential temporal splits for continual learning evaluation.

    From Section IV-D: Each dataset is partitioned into 5 temporal splits.
    Flows are sorted by timestamp (or index), divided into 5 equal partitions,
    and each partition gets an 80/20 stratified train/test split.
    """

    def __init__(
        self,
        num_splits: int = 5,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.num_splits = num_splits
        self.train_ratio = train_ratio
        self.seed = seed

    def split_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Split a dataset into sequential tasks.

        Returns:
            List of dicts, each with keys:
                'train_X', 'train_y', 'test_X', 'test_y'
        """
        n = len(labels)
        split_size = n // self.num_splits
        tasks = []

        for i in range(self.num_splits):
            start = i * split_size
            end = start + split_size if i < self.num_splits - 1 else n

            X_split = features[start:end]
            y_split = labels[start:end]

            # Stratified 80/20 split
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - self.train_ratio,
                random_state=self.seed + i,
            )
            train_idx, test_idx = next(sss.split(X_split, y_split))

            tasks.append({
                "train_X": X_split[train_idx],
                "train_y": y_split[train_idx],
                "test_X": X_split[test_idx],
                "test_y": y_split[test_idx],
                "split_id": i + 1,
            })
            logger.info(
                f"  Split {i+1}: {len(train_idx)} train, "
                f"{len(test_idx)} test, "
                f"{len(np.unique(y_split[train_idx]))} classes"
            )

        return tasks

    def create_cross_dataset_tasks(
        self,
        all_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, LabelEncoder]],
    ) -> List[Dict]:
        """
        Create cross-dataset sequential tasks following the paper's order.

        The 6 datasets are presented sequentially:
        CIC-IoT-2023 -> UNSW-NB15 -> CSE-CIC-IDS2018 ->
        CloudSec-2024 -> ContainerIDS -> EdgeIIoT-2024
        """
        tasks = []
        seen_classes = set()

        for dataset_key in CROSS_DATASET_ORDER:
            if dataset_key not in all_datasets:
                logger.warning(f"Skipping {dataset_key}: not loaded")
                continue

            features, labels, le = all_datasets[dataset_key]
            class_names = set(le.classes_)
            novel = class_names - seen_classes
            seen_classes |= class_names

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - self.train_ratio,
                random_state=self.seed,
            )
            train_idx, test_idx = next(sss.split(features, labels))

            tasks.append({
                "dataset": dataset_key,
                "train_X": features[train_idx],
                "train_y": labels[train_idx],
                "test_X": features[test_idx],
                "test_y": labels[test_idx],
                "num_classes": len(class_names),
                "novel_classes": len(novel),
                "label_encoder": le,
            })
            logger.info(
                f"Cross-dataset task: {dataset_key} - "
                f"{len(class_names)} classes ({len(novel)} novel)"
            )

        return tasks


def create_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    dataset = NIDSDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
