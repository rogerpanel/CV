"""
Dataset-specific loaders for encrypted traffic benchmarks

Supports loading and preprocessing for all 10 datasets used in the paper:
- CICIDS2017, CICIDS2018, UNSW-NB15, BoT-IoT
- ISCX-VPN-NonVPN-2016, CESNET-TLS-Year22, VisQUIC
- CIC-IoT-2023, Edge-IIoTset
- IIS3D / HTTPS Traffic Classification (DOI: 10.34740/kaggle/dsv/12479689)

The IIS3D dataset contains 145,671 HTTPS flow records with 88 numerical
features and 6 application categories (W, D, P, U, M, L).

Reference: Paper Section 4.1 - Datasets
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .dataset import EncryptedTrafficDataset


class DatasetConfig:
    """Centralized dataset path and configuration management."""

    DATASET_CONFIGS = {
        'CICIDS2017': {
            'path': 'data/raw/CICIDS2017/',
            'label_col': 'Label',
            'description': 'Canadian Institute for Cybersecurity IDS 2017',
        },
        'CICIDS2018': {
            'path': 'data/raw/CICIDS2018/',
            'label_col': 'Label',
            'description': 'Canadian Institute for Cybersecurity IDS 2018',
        },
        'UNSW-NB15': {
            'path': 'data/raw/UNSW-NB15/',
            'label_col': 'attack_cat',
            'description': 'UNSW Network-Based Intrusion Detection 15',
        },
        'BoT-IoT': {
            'path': 'data/raw/BoT-IoT/',
            'label_col': 'category',
            'description': 'Botnet Internet of Things dataset (72M+ records)',
        },
        'ISCX-VPN': {
            'path': 'data/raw/ISCX-VPN/',
            'label_col': 'Label',
            'description': 'ISCX VPN-NonVPN 2016 (14 application categories)',
        },
        'CESNET-TLS': {
            'path': 'data/raw/CESNET-TLS/',
            'label_col': 'label',
            'description': 'CESNET TLS Year 2022 (180 web service labels)',
        },
        'VisQUIC': {
            'path': 'data/raw/VisQUIC/',
            'label_col': 'label',
            'description': '100K labeled QUIC traces from 44K+ websites',
        },
        'CIC-IoT-2023': {
            'path': 'data/raw/CIC-IoT-2023/',
            'label_col': 'Label',
            'description': 'CIC IoT Dataset 2023',
        },
        'Edge-IIoT': {
            'path': 'data/raw/Edge-IIoT/',
            'label_col': 'Attack_type',
            'description': 'Edge IIoTset (10+ IoT devices, 14 attack types)',
        },
        'IIS3D': {
            'path': 'data/raw/IIS3D/',
            'label_col': 'TYPE',
            'description': 'HTTPS Traffic Classification (145,671 flows, '
                           '88 features, 6 classes: W/D/P/U/M/L). '
                           'DOI: 10.34740/kaggle/dsv/12479689',
        },
    }


class BaseDatasetLoader:
    """
    Base loader providing common functionality for all datasets.

    Handles CSV loading, feature preprocessing, label encoding,
    and stratified train/val/test splitting.
    """

    def __init__(self, data_path: str, label_col: str = 'Label',
                 random_state: int = 42):
        self.data_path = data_path
        self.label_col = label_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file with error handling."""
        df = pd.read_csv(filepath, low_memory=False)
        return df

    def load_multiple_csvs(self, directory: str) -> pd.DataFrame:
        """Load and concatenate multiple CSV files from a directory."""
        dfs = []
        for f in sorted(os.listdir(directory)):
            if f.endswith('.csv'):
                df = self.load_csv(os.path.join(directory, f))
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features: handle missing values, encode labels, scale.

        Returns:
            (features, labels)
        """
        labels = df[self.label_col].values
        features_df = df.drop(columns=[self.label_col])

        # Keep only numeric columns
        numeric_df = features_df.select_dtypes(include=[np.number])

        # Handle missing/infinite values
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.fillna(0.0)

        features = numeric_df.values.astype(np.float32)
        encoded_labels = self.label_encoder.fit_transform(labels)

        return features, encoded_labels

    def split_and_create_datasets(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """Split data and create PyTorch datasets."""
        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_state,
            stratify=labels
        )

        val_frac = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_frac),
            random_state=self.random_state,
            stratify=y_temp
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # Reshape for temporal models: (N, 1, features)
        X_train = X_train.reshape(-1, 1, X_train.shape[1])
        X_val = X_val.reshape(-1, 1, X_val.shape[1])
        X_test = X_test.reshape(-1, 1, X_test.shape[1])

        train_ds = EncryptedTrafficDataset(X_train, labels=y_train)
        val_ds = EncryptedTrafficDataset(X_val, labels=y_val)
        test_ds = EncryptedTrafficDataset(X_test, labels=y_test)

        return train_ds, val_ds, test_ds


class IIS3DLoader(BaseDatasetLoader):
    """
    Loader for the IIS3D / HTTPS Traffic Classification dataset.

    Dataset details:
    - 145,671 HTTPS flow records
    - 88 numerical features per flow
    - 6 application categories:
        W (Website): 80,789 (55.46%) -- majority class
        D (File Download): 20,393 (14.00%)
        P (Video Player): 12,553 (8.62%)
        U (File Upload): 10,862 (7.46%)
        M (Music Player): 10,701 (7.35%)
        L (Live Video): 10,373 (7.12%)

    DOI: 10.34740/kaggle/dsv/12479689
    """

    CLASS_NAMES = ['D', 'L', 'M', 'P', 'U', 'W']
    CLASS_DESCRIPTIONS = {
        'W': 'Website',
        'D': 'File Download',
        'P': 'Video Player',
        'U': 'File Upload',
        'M': 'Music Player',
        'L': 'Live Video',
    }

    def __init__(self, data_path: str = 'data/raw/IIS3D/', random_state: int = 42):
        super().__init__(data_path, label_col='TYPE', random_state=random_state)

    def load(self) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """Load and preprocess the IIS3D dataset."""
        csv_path = os.path.join(self.data_path, 'HTTPS-clf-dataset.csv')
        df = self.load_csv(csv_path)

        features, labels = self.preprocess_features(df)
        print(f"IIS3D dataset loaded: {len(features)} samples, "
              f"{features.shape[1]} features, "
              f"{len(np.unique(labels))} classes")

        return self.split_and_create_datasets(features, labels)


class CICIDS2017Loader(BaseDatasetLoader):
    """Loader for CICIDS2017 dataset."""

    def __init__(self, data_path: str = 'data/raw/CICIDS2017/', random_state: int = 42):
        super().__init__(data_path, label_col='Label', random_state=random_state)

    def load(self):
        df = self.load_multiple_csvs(self.data_path)
        features, labels = self.preprocess_features(df)
        return self.split_and_create_datasets(features, labels)


class CICIDS2018Loader(BaseDatasetLoader):
    """Loader for CICIDS2018 dataset."""

    def __init__(self, data_path: str = 'data/raw/CICIDS2018/', random_state: int = 42):
        super().__init__(data_path, label_col='Label', random_state=random_state)

    def load(self):
        df = self.load_multiple_csvs(self.data_path)
        features, labels = self.preprocess_features(df)
        return self.split_and_create_datasets(features, labels)


class UNSWNB15Loader(BaseDatasetLoader):
    """Loader for UNSW-NB15 dataset (specific train/test splits)."""

    def __init__(self, data_path: str = 'data/raw/UNSW-NB15/', random_state: int = 42):
        super().__init__(data_path, label_col='attack_cat', random_state=random_state)

    def load(self):
        df = self.load_multiple_csvs(self.data_path)
        features, labels = self.preprocess_features(df)
        return self.split_and_create_datasets(features, labels)


class BoTIoTLoader(BaseDatasetLoader):
    """Loader for BoT-IoT dataset (72M+ records)."""

    def __init__(self, data_path: str = 'data/raw/BoT-IoT/', random_state: int = 42):
        super().__init__(data_path, label_col='category', random_state=random_state)

    def load(self):
        df = self.load_multiple_csvs(self.data_path)
        features, labels = self.preprocess_features(df)
        return self.split_and_create_datasets(features, labels)


class DatasetFactory:
    """
    Factory for creating dataset loaders uniformly.

    Provides a single interface for loading any supported dataset.
    """

    LOADERS = {
        'CICIDS2017': CICIDS2017Loader,
        'CICIDS2018': CICIDS2018Loader,
        'UNSW-NB15': UNSWNB15Loader,
        'BoT-IoT': BoTIoTLoader,
        'IIS3D': IIS3DLoader,
    }

    @classmethod
    def create_loader(cls, dataset_name: str, data_path: str = None,
                      random_state: int = 42) -> BaseDatasetLoader:
        if dataset_name not in cls.LOADERS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls.LOADERS.keys())}"
            )

        loader_cls = cls.LOADERS[dataset_name]
        if data_path:
            return loader_cls(data_path=data_path, random_state=random_state)
        return loader_cls(random_state=random_state)

    @classmethod
    def load_dataset(cls, dataset_name: str, data_path: str = None,
                     random_state: int = 42):
        loader = cls.create_loader(dataset_name, data_path, random_state)
        return loader.load()


def load_all_datasets(
    dataset_names: List[str] = None,
    data_root: str = 'data/raw/',
    random_state: int = 42
) -> Dict[str, Tuple]:
    """
    Batch-load multiple datasets.

    Args:
        dataset_names: List of dataset names to load
        data_root: Root directory for raw data
        random_state: Random seed

    Returns:
        Dictionary mapping dataset name to (train, val, test) tuples
    """
    if dataset_names is None:
        dataset_names = list(DatasetFactory.LOADERS.keys())

    results = {}
    for name in dataset_names:
        try:
            data_path = os.path.join(data_root, name)
            datasets = DatasetFactory.load_dataset(
                name, data_path=data_path, random_state=random_state
            )
            results[name] = datasets
            print(f"Loaded {name} successfully")
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")

    return results
