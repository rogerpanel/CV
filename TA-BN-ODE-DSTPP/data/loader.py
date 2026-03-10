"""
Dataset loaders for ICS3D and standard benchmark datasets.

ICS3D (DOI: 10.34740/kaggle/dsv/12483891):
  - Container Security: 697,289 Kubernetes flows, 12 attack scenarios
  - Edge-IIoTset: 4M records, 7-layer IoT testbed
  - GUIDE: 1M SOC incidents from 6,100 organizations, 441 MITRE ATT&CK techniques

Standard benchmarks (DOI: 10.34740/KAGGLE/DSV/12479689):
  - CIC-IDS2018: 16.2M records
  - UNSW-NB15: 175,341 train / 82,332 test
  - CIC-IoT-2023: 33 attack types
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False
    warnings.warn("kagglehub not installed. Install with: pip install kagglehub")


def _download_dataset(slug: str) -> str:
    """Download dataset from Kaggle via kagglehub."""
    if not HAS_KAGGLEHUB:
        raise RuntimeError("kagglehub required. Install: pip install kagglehub")
    path = kagglehub.dataset_download(slug)
    print(f"Dataset downloaded to: {path}")
    return path


def _find_csv_files(root: str) -> Dict[str, str]:
    """Recursively find CSV files in downloaded dataset directory."""
    csv_files = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".csv"):
                csv_files[f] = os.path.join(dirpath, f)
    return csv_files


class ICS3DLoader:
    """Loader for the Integrated Cloud Security 3Datasets (ICS3D).

    DOI: 10.34740/kaggle/dsv/12483891
    """

    def __init__(self, kaggle_slug: str = "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d",
                 local_path: Optional[str] = None):
        self.slug = kaggle_slug
        self.local_path = local_path
        self._root = None

    def _ensure_downloaded(self) -> str:
        if self._root is not None:
            return self._root
        if self.local_path and os.path.exists(self.local_path):
            self._root = self.local_path
        else:
            self._root = _download_dataset(self.slug)
        return self._root

    def load_container_security(self) -> Tuple[pd.DataFrame, str]:
        """Load Container Security domain (697,289 Kubernetes flows)."""
        root = self._ensure_downloaded()
        csv_files = _find_csv_files(root)

        # Search for container/kubernetes related files
        candidates = [f for f in csv_files if any(
            kw in f.lower() for kw in ["container", "kubernetes", "k8s"]
        )]
        if not candidates:
            # Fall back to first available file matching common patterns
            candidates = sorted(csv_files.keys())

        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {root}")

        target = candidates[0]
        print(f"Loading Container Security: {target}")
        df = pd.read_csv(csv_files[target], low_memory=False)
        return df, "container"

    def load_edge_iiot(self) -> Tuple[pd.DataFrame, str]:
        """Load Edge-IIoTset domain (4M records, 7-layer IoT testbed)."""
        root = self._ensure_downloaded()
        csv_files = _find_csv_files(root)

        candidates = [f for f in csv_files if any(
            kw in f.lower() for kw in ["edge", "iiot", "iot"]
        )]
        if not candidates:
            candidates = sorted(csv_files.keys())

        target = candidates[0] if candidates else sorted(csv_files.keys())[0]
        print(f"Loading Edge-IIoTset: {target}")
        df = pd.read_csv(csv_files[target], low_memory=False)
        return df, "edge_iiot"

    def load_guide_soc(self) -> Tuple[pd.DataFrame, str]:
        """Load GUIDE SOC domain (1M incidents, 441 MITRE ATT&CK techniques)."""
        root = self._ensure_downloaded()
        csv_files = _find_csv_files(root)

        candidates = [f for f in csv_files if any(
            kw in f.lower() for kw in ["guide", "soc", "mitre"]
        )]
        if not candidates:
            candidates = sorted(csv_files.keys())

        target = candidates[-1] if len(candidates) > 2 else candidates[0]
        print(f"Loading GUIDE SOC: {target}")
        df = pd.read_csv(csv_files[target], low_memory=False)
        return df, "guide_soc"

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all three ICS3D domains."""
        datasets = {}
        for loader_fn, name in [
            (self.load_container_security, "container"),
            (self.load_edge_iiot, "edge_iiot"),
            (self.load_guide_soc, "guide_soc"),
        ]:
            try:
                df, _ = loader_fn()
                datasets[name] = df
                print(f"  {name}: {len(df):,} records, {df.shape[1]} features")
            except Exception as e:
                print(f"  Warning: Could not load {name}: {e}")
        return datasets


class BenchmarkLoader:
    """Loader for standard benchmark datasets.

    DOI: 10.34740/KAGGLE/DSV/12479689
    """

    def __init__(self, kaggle_slug: str = "rogernickanaedevha/integrated-cloud-security-3datasets-benchmarks",
                 local_path: Optional[str] = None):
        self.slug = kaggle_slug
        self.local_path = local_path
        self._root = None

    def _ensure_downloaded(self) -> str:
        if self._root is not None:
            return self._root
        if self.local_path and os.path.exists(self.local_path):
            self._root = self.local_path
        else:
            self._root = _download_dataset(self.slug)
        return self._root

    def load_cic_ids2018(self) -> Tuple[pd.DataFrame, str]:
        """Load CIC-IDS2018 (16.2M records)."""
        root = self._ensure_downloaded()
        csv_files = _find_csv_files(root)

        candidates = [f for f in csv_files if any(
            kw in f.lower() for kw in ["cic-ids2018", "cicids2018", "ids2018", "cic_ids"]
        )]
        if not candidates:
            candidates = sorted(csv_files.keys())

        target = candidates[0]
        print(f"Loading CIC-IDS2018: {target}")
        df = pd.read_csv(csv_files[target], low_memory=False)
        return df, "cic_ids2018"

    def load_unsw_nb15(self) -> Tuple[pd.DataFrame, str]:
        """Load UNSW-NB15 (175,341 train / 82,332 test)."""
        root = self._ensure_downloaded()
        csv_files = _find_csv_files(root)

        candidates = [f for f in csv_files if any(
            kw in f.lower() for kw in ["unsw", "nb15"]
        )]
        if not candidates:
            candidates = sorted(csv_files.keys())

        target = candidates[0]
        print(f"Loading UNSW-NB15: {target}")
        df = pd.read_csv(csv_files[target], low_memory=False)
        return df, "unsw_nb15"

    def load_cic_iot2023(self) -> Tuple[pd.DataFrame, str]:
        """Load CIC-IoT-2023 (33 attack types)."""
        root = self._ensure_downloaded()
        csv_files = _find_csv_files(root)

        candidates = [f for f in csv_files if any(
            kw in f.lower() for kw in ["cic-iot", "ciciot", "iot2023", "iot-2023"]
        )]
        if not candidates:
            candidates = sorted(csv_files.keys())

        target = candidates[0]
        print(f"Loading CIC-IoT-2023: {target}")
        df = pd.read_csv(csv_files[target], low_memory=False)
        return df, "cic_iot2023"

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all benchmark datasets."""
        datasets = {}
        for loader_fn, name in [
            (self.load_cic_ids2018, "cic_ids2018"),
            (self.load_unsw_nb15, "unsw_nb15"),
            (self.load_cic_iot2023, "cic_iot2023"),
        ]:
            try:
                df, _ = loader_fn()
                datasets[name] = df
                print(f"  {name}: {len(df):,} records, {df.shape[1]} features")
            except Exception as e:
                print(f"  Warning: Could not load {name}: {e}")
        return datasets


def get_all_datasets(ics3d_slug: str = "rogernickanaedevha/integrated-cloud-security-3datasets-ics3d",
                     benchmarks_slug: str = "rogernickanaedevha/integrated-cloud-security-3datasets-benchmarks",
                     ics3d_local: Optional[str] = None,
                     benchmarks_local: Optional[str] = None
                     ) -> Dict[str, pd.DataFrame]:
    """Load all six datasets used in the paper."""
    print("=" * 60)
    print("Loading ICS3D datasets (DOI: 10.34740/kaggle/dsv/12483891)")
    print("=" * 60)
    ics3d = ICS3DLoader(ics3d_slug, ics3d_local)
    datasets = ics3d.load_all()

    print("\n" + "=" * 60)
    print("Loading benchmark datasets (DOI: 10.34740/KAGGLE/DSV/12479689)")
    print("=" * 60)
    benchmarks = BenchmarkLoader(benchmarks_slug, benchmarks_local)
    datasets.update(benchmarks.load_all())

    print(f"\nTotal: {len(datasets)} datasets loaded")
    return datasets
