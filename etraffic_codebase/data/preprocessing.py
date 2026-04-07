"""
Feature extraction and preprocessing for encrypted traffic flows

Extracts statistical, temporal, and packet-level features from
encrypted network traffic without accessing payload contents.

Features follow the 88-feature schema used in the HTTPS Traffic
Classification dataset (DOI: 10.34740/kaggle/dsv/12479689):
- Per-flow statistics (packet counts, byte counts, ratios)
- Timing information (inter-arrival times, flow duration)
- Directional features (forward/backward packet stats)

Reference: Paper Section 3.1 - Feature Engineering
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class FlowFeatureExtractor:
    """
    Extract features from encrypted traffic flows.

    Computes 88 statistical features per flow without accessing
    encrypted payload contents, matching the IIS3D / HTTPS Traffic
    Classification dataset schema.
    """

    FEATURE_GROUPS = {
        'packet': [
            'total_fwd_packets', 'total_bwd_packets',
            'total_length_fwd_packets', 'total_length_bwd_packets',
            'fwd_packet_length_max', 'fwd_packet_length_min',
            'fwd_packet_length_mean', 'fwd_packet_length_std',
            'bwd_packet_length_max', 'bwd_packet_length_min',
            'bwd_packet_length_mean', 'bwd_packet_length_std',
        ],
        'timing': [
            'flow_duration', 'flow_iat_mean', 'flow_iat_std',
            'flow_iat_max', 'flow_iat_min',
            'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
            'fwd_iat_max', 'fwd_iat_min',
            'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std',
            'bwd_iat_max', 'bwd_iat_min',
        ],
        'statistical': [
            'flow_bytes_per_s', 'flow_packets_per_s',
            'down_up_ratio', 'average_packet_size',
            'avg_fwd_segment_size', 'avg_bwd_segment_size',
            'fwd_header_length', 'bwd_header_length',
        ],
    }

    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def extract_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features from raw flow data.

        Args:
            raw_data: DataFrame with flow-level features

        Returns:
            Feature array (N, num_features)
        """
        # Select numerical columns
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        features = raw_data[numeric_cols].values

        # Handle missing values and infinities
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        return features

    def fit_transform(self, features: np.ndarray,
                      labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler and encoder, then transform features and labels.

        Args:
            features: Raw feature array (N, num_features)
            labels: Raw label array (N,)

        Returns:
            (scaled_features, encoded_labels)
        """
        scaled = self.scaler.fit_transform(features)
        encoded = self.label_encoder.fit_transform(labels)
        self.is_fitted = True
        return scaled, encoded

    def transform(self, features: np.ndarray,
                  labels: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform features (and optionally labels) using fitted scaler/encoder.

        Args:
            features: Feature array
            labels: Optional label array

        Returns:
            (scaled_features, encoded_labels)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform first")

        scaled = self.scaler.transform(features)
        encoded = self.label_encoder.transform(labels) if labels is not None else None
        return scaled, encoded


def preprocess_dataset(
    data: pd.DataFrame,
    label_column: str = 'TYPE',
    class_mapping: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess an encrypted traffic dataset.

    Handles:
    1. Automatic label column detection
    2. Class normalization to fixed schema (D, L, M, P, U, W for HTTPS dataset)
    3. Missing value imputation
    4. Infinite value replacement
    5. Feature scaling via StandardScaler

    Args:
        data: Raw DataFrame
        label_column: Name of the label column
        class_mapping: Optional mapping from raw labels to normalized labels

    Returns:
        (features, labels, class_names)
    """
    # Detect label column
    if label_column not in data.columns:
        # Try common alternatives
        for col in ['Label', 'label', 'class', 'Class', 'Attack', 'category']:
            if col in data.columns:
                label_column = col
                break
        else:
            raise ValueError(f"Label column '{label_column}' not found in data")

    labels = data[label_column].values
    features_df = data.drop(columns=[label_column])

    # Select only numeric columns
    numeric_df = features_df.select_dtypes(include=[np.number])

    # Handle missing/infinite values
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(0.0)

    features = numeric_df.values.astype(np.float32)

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    class_names = list(le.classes_)

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, encoded_labels, class_names


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple:
    """
    Split dataset into train/val/test with stratification.

    Maintains class distribution across all splits, which is critical
    for imbalanced encrypted traffic datasets.

    Args:
        features: Feature array (N, D)
        labels: Label array (N,)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed for reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )

    # Second split: val vs test
    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction),
        random_state=random_state,
        stratify=y_temp
    )

    print(f"Dataset split:")
    print(f"  Train: {len(X_train):,} samples ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(X_val):,} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(X_test):,} samples ({test_ratio*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    strategy: str = 'focal_loss',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance in encrypted traffic datasets.

    Strategies:
    - 'smote': Synthetic Minority Over-sampling Technique
    - 'undersample': Random undersampling of majority class
    - 'focal_loss': Return as-is (handled by focal loss during training)
    - 'weighted': Return as-is (handled by weighted sampling)

    Args:
        features: Feature array
        labels: Label array
        strategy: Balancing strategy
        random_state: Random seed

    Returns:
        (balanced_features, balanced_labels)
    """
    if strategy in ('focal_loss', 'weighted'):
        return features, labels

    if strategy == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            return smote.fit_resample(features, labels)
        except ImportError:
            print("Warning: imbalanced-learn not installed, skipping SMOTE")
            return features, labels

    if strategy == 'undersample':
        from collections import Counter
        counts = Counter(labels)
        min_count = min(counts.values())

        balanced_features = []
        balanced_labels = []
        for cls in counts:
            mask = labels == cls
            cls_features = features[mask]
            indices = np.random.RandomState(random_state).choice(
                len(cls_features), min_count, replace=False
            )
            balanced_features.append(cls_features[indices])
            balanced_labels.extend([cls] * min_count)

        return np.vstack(balanced_features), np.array(balanced_labels)

    return features, labels
