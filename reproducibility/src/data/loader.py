"""
Dataset Loading
===============
ICS3D (DOI: 10.34740/kaggle/dsv/12483891):
  - Container Security: 697,289 flows, Kubernetes, 12 attack scenarios
  - Edge-IIoTset:       4,000,000 records, 7-layer IoT testbed
  - GUIDE (SOC):        1,000,000 incidents, 6100 orgs, MITRE ATT&CK

Standard Benchmarks (DOI: 10.34740/KAGGLE/DSV/12479689):
  - CIC-IDS2018, UNSW-NB15, CIC-IoT-2023
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict


class SecurityDataset(Dataset):
    """PyTorch Dataset for security event data."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 timestamps: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.timestamps = (torch.FloatTensor(timestamps)
                           if timestamps is not None else None)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"features": self.X[idx], "label": self.y[idx]}
        if self.timestamps is not None:
            item["timestamp"] = self.timestamps[idx]
        return item


class ICS3DDataLoader:
    """Loader for the Integrated Cloud Security 3Datasets (ICS3D).

    Downloads via kagglehub if not present locally.
    """

    # Kaggle dataset identifiers
    ICS3D_DOI = "10.34740/kaggle/dsv/12483891"
    BENCHMARK_DOI = "10.34740/KAGGLE/DSV/12479689"

    def __init__(self, data_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/ics3d")

    def _download_ics3d(self) -> str:
        """Download ICS3D via kagglehub."""
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                "rogernickanaedevha/integrated-cloud-security-3-datasets-ics3d"
            )
            print(f"ICS3D downloaded to: {path}")
            return path
        except ImportError:
            print("kagglehub not installed. Install with: pip install kagglehub")
            print(f"Or download manually from DOI: {self.ICS3D_DOI}")
            raise
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Download manually from: https://doi.org/{self.ICS3D_DOI}")
            raise

    def _find_data_path(self) -> str:
        """Locate dataset directory."""
        if self.data_dir and os.path.exists(self.data_dir):
            return self.data_dir
        return self._download_ics3d()

    def load_containers(self, mode: str = "DNN") -> Tuple[np.ndarray, np.ndarray]:
        """Load Container Security dataset (697,289 flows).

        Args:
            mode: 'DNN' for numeric features, 'raw' for all columns
        Returns:
            X: features, y: labels
        """
        path = self._find_data_path()

        # Try common file patterns
        for fname in ["container_dataset.csv", "containers.csv",
                      "Container_Security.csv"]:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                break
        else:
            # Search recursively
            import glob
            csvs = glob.glob(os.path.join(path, "**/*ontainer*.csv"),
                             recursive=True)
            if not csvs:
                raise FileNotFoundError(
                    f"Container dataset not found in {path}. "
                    f"Available files: {os.listdir(path)}"
                )
            df = pd.read_csv(csvs[0])
            print(f"Loaded: {csvs[0]} ({len(df)} rows)")

        return self._prepare_df(df, mode)

    def load_edge_iiot(self, mode: str = "DNN") -> Tuple[np.ndarray, np.ndarray]:
        """Load Edge-IIoTset dataset (4M records, 7-layer IoT testbed)."""
        path = self._find_data_path()

        for fname in ["Edge-IIoTset.csv", "edge_iiot.csv",
                      "DNN-EdgeIIoT-dataset.csv",
                      "Selected_dataset_DNN.csv"]:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, low_memory=False)
                break
        else:
            import glob
            csvs = glob.glob(os.path.join(path, "**/*IIoT*.csv"),
                             recursive=True)
            csvs += glob.glob(os.path.join(path, "**/*DNN*.csv"),
                              recursive=True)
            if not csvs:
                raise FileNotFoundError(
                    f"Edge-IIoT dataset not found in {path}"
                )
            df = pd.read_csv(csvs[0], low_memory=False)
            print(f"Loaded: {csvs[0]} ({len(df)} rows)")

        return self._prepare_df(df, mode)

    def load_guide(self, mode: str = "DNN") -> Tuple[np.ndarray, np.ndarray]:
        """Load GUIDE SOC dataset (1M incidents, MITRE ATT&CK)."""
        path = self._find_data_path()

        for fname in ["GUIDE_dataset.csv", "guide.csv",
                      "GUIDE_Train.csv", "guide_incidents.csv"]:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, low_memory=False)
                break
        else:
            import glob
            csvs = glob.glob(os.path.join(path, "**/*GUIDE*.csv"),
                             recursive=True)
            csvs += glob.glob(os.path.join(path, "**/*guide*.csv"),
                              recursive=True)
            if not csvs:
                raise FileNotFoundError(
                    f"GUIDE dataset not found in {path}"
                )
            df = pd.read_csv(csvs[0], low_memory=False)
            print(f"Loaded: {csvs[0]} ({len(df)} rows)")

        return self._prepare_df(df, mode)

    def _prepare_df(self, df: pd.DataFrame,
                    mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from DataFrame."""
        # Identify label column
        label_col = None
        for col in ["Attack_type", "attack_type", "label", "Label",
                     "Attack", "attack_label", "class", "IncidentGrade",
                     "Triage"]:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            # Use last column as label
            label_col = df.columns[-1]
            print(f"Using '{label_col}' as label column")

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].astype(str).fillna("Unknown"))

        # Select numeric features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in feature_cols:
            feature_cols.remove(label_col)

        X = df[feature_cols].fillna(0).values.astype(np.float32)

        # Handle infinities
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        print(f"Features: {X.shape}, Classes: {len(le.classes_)} "
              f"({', '.join(le.classes_[:5])}{'...' if len(le.classes_) > 5 else ''})")

        return X, y

    def load_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load all three ICS3D domains."""
        datasets = {}
        for name, loader in [("containers", self.load_containers),
                              ("edge_iiot", self.load_edge_iiot),
                              ("guide", self.load_guide)]:
            try:
                datasets[name] = loader()
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        return datasets


class BenchmarkDataLoader:
    """Loader for standard benchmark datasets.

    CIC-IDS2018, UNSW-NB15, CIC-IoT-2023
    DOI: 10.34740/KAGGLE/DSV/12479689
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir

    def _download_benchmarks(self) -> str:
        try:
            import kagglehub
            path = kagglehub.dataset_download(
                "rogernickanaedevha/integrated-benchmark-datasets-for-ids"
            )
            print(f"Benchmarks downloaded to: {path}")
            return path
        except Exception as e:
            print(f"Download failed: {e}")
            raise

    def _find_data_path(self) -> str:
        if self.data_dir and os.path.exists(self.data_dir):
            return self.data_dir
        return self._download_benchmarks()

    def load_cicids2018(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CIC-IDS2018 (16.2M records)."""
        path = self._find_data_path()
        return self._load_benchmark(path, "CIC", "CICIDS")

    def load_unsw_nb15(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load UNSW-NB15 (257,673 records)."""
        path = self._find_data_path()
        return self._load_benchmark(path, "UNSW", "NB15")

    def load_ciciot2023(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CIC-IoT-2023."""
        path = self._find_data_path()
        return self._load_benchmark(path, "IoT", "CICIoT")

    def _load_benchmark(self, path: str,
                        *patterns: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generic benchmark loader by filename pattern."""
        import glob
        for pat in patterns:
            csvs = glob.glob(os.path.join(path, f"**/*{pat}*.csv"),
                             recursive=True)
            if csvs:
                df = pd.read_csv(csvs[0], low_memory=False)
                print(f"Loaded: {csvs[0]} ({len(df)} rows)")

                loader = ICS3DDataLoader()
                return loader._prepare_df(df, "DNN")

        raise FileNotFoundError(
            f"Benchmark not found for patterns {patterns} in {path}"
        )


def create_dataloaders(
        X: np.ndarray, y: np.ndarray,
        train_ratio: float = 0.7, val_ratio: float = 0.15,
        batch_size: int = 256, num_workers: int = 4,
        max_samples: Optional[int] = None,
        timestamps: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with temporal ordering.

    Uses temporal split (not random) to prevent data leakage,
    as specified in Section VIII-C.
    """
    n = len(X)
    if max_samples and n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]
        if timestamps is not None:
            timestamps = timestamps[idx]
        n = max_samples

    # Temporal split (70/15/15)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    ts_train = timestamps[:n_train] if timestamps is not None else None
    ts_val = (timestamps[n_train:n_train + n_val]
              if timestamps is not None else None)
    ts_test = (timestamps[n_train + n_val:]
               if timestamps is not None else None)

    train_ds = SecurityDataset(X_train, y_train, ts_train)
    val_ds = SecurityDataset(X_val, y_val, ts_val)
    test_ds = SecurityDataset(X_test, y_test, ts_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 4, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
