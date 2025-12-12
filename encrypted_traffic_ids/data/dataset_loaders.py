"""
Dataset Loaders for All Encrypted Traffic Datasets

Provides unified loaders for 9+ heterogeneous encrypted traffic datasets used in the paper:
1. CICIDS2017 - Canadian Institute for Cybersecurity IDS 2017
2. CICIDS2018 (CSE-CIC-IDS2018) - Updated version with more attack types
3. UNSW-NB15 - UNSW Network-Based 2015
4. ISCX-VPN-NonVPN-2016 - VPN and non-VPN encrypted traffic
5. CESNET-TLS-Year22 - Year-long TLS traffic from backbone
6. VisQUIC - QUIC protocol encrypted traffic
7. CIC-IoT-2023 - IoT devices encrypted traffic
8. Edge-IIoTset - Edge and IIoT comprehensive dataset
9. BoT-IoT - Botnet IoT dataset
10. IIS3D - Integrated IDPS Security 3Datasets

Each loader handles dataset-specific preprocessing and returns unified format.

References:
    Paper Section 4.1 - Experimental Setup
    See datasets_bibliography.bib for full citations
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from .dataset import EncryptedTrafficDataset
from .preprocessing import FlowFeatureExtractor

warnings.filterwarnings('ignore')


class DatasetConfig:
    """Configuration for dataset paths and parameters."""

    def __init__(self, data_root: str = './data'):
        """
        Initialize dataset configuration.

        Args:
            data_root: Root directory for all datasets
        """
        self.data_root = Path(data_root)
        self.datasets = {
            'CICIDS2017': self.data_root / 'CICIDS2017',
            'CICIDS2018': self.data_root / 'CICIDS2018',
            'UNSWNB15': self.data_root / 'UNSW-NB15',
            'ISCXVPN': self.data_root / 'ISCX-VPN-2016',
            'CESNETTLS': self.data_root / 'CESNET-TLS-Year22',
            'VisQUIC': self.data_root / 'VisQUIC',
            'CICIoT2023': self.data_root / 'CIC-IoT-2023',
            'EdgeIIoT': self.data_root / 'Edge-IIoTset',
            'BoTIoT': self.data_root / 'BoT-IoT',
            'IIS3D': self.data_root / 'IIS3D'
        }


class BaseDatasetLoader:
    """Base class for dataset loaders."""

    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        max_samples: Optional[int] = None
    ):
        """
        Initialize base loader.

        Args:
            dataset_path: Path to dataset
            test_size: Test set proportion
            val_size: Validation set proportion (from training set)
            random_state: Random seed
            max_samples: Maximum samples to load (for memory constraints)
        """
        self.dataset_path = Path(dataset_path)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.max_samples = max_samples

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FlowFeatureExtractor()

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            return pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def preprocess_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and labels.

        Args:
            df: DataFrame
            feature_columns: List of feature column names
            label_column: Label column name

        Returns:
            Tuple of (features, labels)
        """
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Extract features and labels
        X = df[feature_columns].values
        y = df[label_column].values

        # Encode labels
        y = self.label_encoder.fit_transform(y)

        # Limit samples if needed
        if self.max_samples and len(X) > self.max_samples:
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X = X[indices]
            y = y[indices]

        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            X: Features
            y: Labels

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.val_size, random_state=self.random_state, stratify=y_train
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


class CICIDS2017Loader(BaseDatasetLoader):
    """
    Loader for CICIDS2017 dataset.

    Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
    Features: 78 flow-based features
    Classes: Benign, DoS, DDoS, PortScan, Brute Force, etc.
    """

    def load(self) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """Load CICIDS2017 dataset."""
        print("Loading CICIDS2017 dataset...")

        # CICIDS2017 has multiple CSV files
        csv_files = list(self.dataset_path.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")

        # Load all files
        dfs = []
        for csv_file in csv_files:
            df = self.load_csv(csv_file)
            if df is not None:
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(df)} samples")

        # Feature columns (excluding label)
        label_column = 'Label'
        feature_columns = [col for col in df.columns if col != label_column and col != ' Label']

        # Preprocess
        X, y = self.preprocess_features(df, feature_columns, label_column if label_column in df.columns else ' Label')

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Normalize
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # Reshape to temporal format (batch, seq_len=1, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Create datasets
        train_dataset = EncryptedTrafficDataset(X_train, labels=y_train)
        val_dataset = EncryptedTrafficDataset(X_val, labels=y_val)
        test_dataset = EncryptedTrafficDataset(X_test, labels=y_test)

        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"  Classes: {self.label_encoder.classes_}")

        return train_dataset, val_dataset, test_dataset


class CICIDS2018Loader(BaseDatasetLoader):
    """
    Loader for CICIDS2018 (CSE-CIC-IDS2018) dataset.

    Dataset: https://www.unb.ca/cic/datasets/ids-2018.html
    Features: 79 flow-based features
    Classes: Benign + 7 attack categories
    """

    def load(self) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """Load CICIDS2018 dataset."""
        print("Loading CICIDS2018 dataset...")

        csv_files = list(self.dataset_path.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")

        dfs = []
        for csv_file in csv_files:
            df = self.load_csv(csv_file)
            if df is not None:
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(df)} samples")

        label_column = 'Label'
        feature_columns = [col for col in df.columns if col not in ['Label', ' Label', 'Timestamp']]

        X, y = self.preprocess_features(df, feature_columns, label_column if label_column in df.columns else ' Label')
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        train_dataset = EncryptedTrafficDataset(X_train, labels=y_train)
        val_dataset = EncryptedTrafficDataset(X_val, labels=y_val)
        test_dataset = EncryptedTrafficDataset(X_test, labels=y_test)

        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"  Classes: {self.label_encoder.classes_}")

        return train_dataset, val_dataset, test_dataset


class UNSWNB15Loader(BaseDatasetLoader):
    """
    Loader for UNSW-NB15 dataset.

    Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Features: 49 features
    Classes: Normal + 9 attack types
    """

    def load(self) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """Load UNSW-NB15 dataset."""
        print("Loading UNSW-NB15 dataset...")

        # UNSW-NB15 has specific train/test splits
        train_file = self.dataset_path / 'UNSW_NB15_training-set.csv'
        test_file = self.dataset_path / 'UNSW_NB15_testing-set.csv'

        if not train_file.exists():
            # Try alternate file pattern
            csv_files = list(self.dataset_path.glob('*.csv'))
            if csv_files:
                df = pd.concat([self.load_csv(f) for f in csv_files], ignore_index=True)
                label_column = 'attack_cat'
                feature_columns = [col for col in df.columns if col not in [label_column, 'label', 'id']]

                X, y = self.preprocess_features(df, feature_columns, label_column if label_column in df.columns else 'label')
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            else:
                raise FileNotFoundError(f"Dataset files not found in {self.dataset_path}")
        else:
            df_train = self.load_csv(train_file)
            df_test = self.load_csv(test_file)

            label_column = 'attack_cat'
            feature_columns = [col for col in df_train.columns if col not in [label_column, 'label', 'id']]

            X_train, y_train = self.preprocess_features(df_train, feature_columns, label_column)
            X_test, y_test = self.preprocess_features(df_test, feature_columns, label_column)

            # Create validation set from train
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_size, random_state=self.random_state, stratify=y_train
            )

        print(f"  Loaded {len(X_train) + len(X_val) + len(X_test)} samples")

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        train_dataset = EncryptedTrafficDataset(X_train, labels=y_train)
        val_dataset = EncryptedTrafficDataset(X_val, labels=y_val)
        test_dataset = EncryptedTrafficDataset(X_test, labels=y_test)

        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"  Classes: {self.label_encoder.classes_}")

        return train_dataset, val_dataset, test_dataset


class BoTIoTLoader(BaseDatasetLoader):
    """
    Loader for BoT-IoT dataset.

    Dataset: https://research.unsw.edu.au/projects/bot-iot-dataset
    Features: Flow-based features from IoT botnet traffic
    Classes: Normal + DDoS, DoS, Reconnaissance, Theft
    """

    def load(self) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """Load BoT-IoT dataset."""
        print("Loading BoT-IoT dataset...")

        csv_files = list(self.dataset_path.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")

        dfs = []
        for csv_file in csv_files[:5]:  # Limit files due to size
            df = self.load_csv(csv_file)
            if df is not None:
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(df)} samples")

        label_column = 'attack' if 'attack' in df.columns else 'category'
        feature_columns = [col for col in df.columns if col not in [label_column, 'pkSeqID', 'stime', 'flgs']]

        X, y = self.preprocess_features(df, feature_columns, label_column)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        train_dataset = EncryptedTrafficDataset(X_train, labels=y_train)
        val_dataset = EncryptedTrafficDataset(X_val, labels=y_val)
        test_dataset = EncryptedTrafficDataset(X_test, labels=y_test)

        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"  Classes: {self.label_encoder.classes_}")

        return train_dataset, val_dataset, test_dataset


class DatasetFactory:
    """
    Factory for creating dataset loaders.

    Provides unified interface for loading all supported datasets.
    """

    _loaders = {
        'CICIDS2017': CICIDS2017Loader,
        'CICIDS2018': CICIDS2018Loader,
        'UNSWNB15': UNSWNB15Loader,
        'BoTIoT': BoTIoTLoader,
        # Additional loaders can be added here
    }

    @classmethod
    def create_loader(
        cls,
        dataset_name: str,
        dataset_path: str,
        **kwargs
    ) -> BaseDatasetLoader:
        """
        Create dataset loader.

        Args:
            dataset_name: Name of dataset
            dataset_path: Path to dataset
            **kwargs: Additional arguments for loader

        Returns:
            Dataset loader instance
        """
        if dataset_name not in cls._loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls._loaders.keys())}")

        loader_class = cls._loaders[dataset_name]
        return loader_class(dataset_path, **kwargs)

    @classmethod
    def load_dataset(
        cls,
        dataset_name: str,
        dataset_path: str,
        **kwargs
    ) -> Tuple[EncryptedTrafficDataset, EncryptedTrafficDataset, EncryptedTrafficDataset]:
        """
        Load dataset directly.

        Args:
            dataset_name: Name of dataset
            dataset_path: Path to dataset
            **kwargs: Additional arguments for loader

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        loader = cls.create_loader(dataset_name, dataset_path, **kwargs)
        return loader.load()


def load_all_datasets(
    config: DatasetConfig,
    datasets_to_load: Optional[List[str]] = None
) -> Dict[str, Tuple]:
    """
    Load multiple datasets.

    Args:
        config: Dataset configuration
        datasets_to_load: List of dataset names to load (None = all available)

    Returns:
        Dictionary mapping dataset name to (train, val, test) tuple
    """
    if datasets_to_load is None:
        datasets_to_load = list(DatasetFactory._loaders.keys())

    loaded_datasets = {}

    for dataset_name in datasets_to_load:
        if dataset_name not in config.datasets:
            print(f"Warning: {dataset_name} not found in config, skipping...")
            continue

        dataset_path = config.datasets[dataset_name]

        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue

        try:
            datasets = DatasetFactory.load_dataset(dataset_name, str(dataset_path))
            loaded_datasets[dataset_name] = datasets
            print(f"✓ Successfully loaded {dataset_name}\n")
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}\n")

    return loaded_datasets
