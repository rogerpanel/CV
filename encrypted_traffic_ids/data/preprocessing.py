"""
Feature extraction and preprocessing for encrypted traffic

This module implements comprehensive feature extraction from encrypted network traffic
without accessing payload contents, as described in the paper:

Features extracted:
- Packet-level: Size, inter-arrival time, direction
- Flow-level: Statistical aggregations (mean, std, min, max)
- Temporal: Packet sequences and timing patterns
- Protocol-specific: TLS handshake metadata (when available)

All feature extraction maintains privacy by operating only on metadata.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class FlowFeatureExtractor:
    """
    Extract features from encrypted traffic flows.

    This class implements the feature extraction pipeline described in the paper,
    processing packet-level metadata into fixed-dimensional feature vectors suitable
    for deep learning models.
    """

    def __init__(self, max_packets: int = 100, feature_dim: int = 64):
        """
        Initialize feature extractor.

        Args:
            max_packets: Maximum number of packets per flow to consider
            feature_dim: Dimensionality of extracted feature vectors
        """
        self.max_packets = max_packets
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.fitted = False

    def extract_packet_features(self, packets: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Extract features from packet sequence.

        Args:
            packets: DataFrame or array with packet metadata
                    Expected columns: ['size', 'timestamp', 'direction']

        Returns:
            Feature array of shape (max_packets, num_packet_features)
        """
        if isinstance(packets, pd.DataFrame):
            packets = packets.values

        num_packets = min(len(packets), self.max_packets)
        features = np.zeros((self.max_packets, 3))  # size, inter_arrival_time, direction

        if num_packets > 0:
            for i in range(num_packets):
                # Packet size
                features[i, 0] = packets[i, 0] if len(packets[i]) > 0 else 0

                # Inter-arrival time (time since previous packet)
                if i > 0 and len(packets[i]) > 1:
                    features[i, 1] = packets[i, 1] - packets[i-1, 1]
                else:
                    features[i, 1] = 0

                # Direction (0: upstream, 1: downstream)
                features[i, 2] = packets[i, 2] if len(packets[i]) > 2 else 0

        return features

    def extract_flow_statistics(self, packets: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Extract statistical features from flow.

        Computes aggregated statistics as described in the paper:
        - Mean, median, std, min, max for packet sizes
        - Mean, median, std, min, max for inter-arrival times
        - Flow duration
        - Total bytes
        - Packet count
        - Upstream/downstream packet ratio

        Args:
            packets: Packet metadata

        Returns:
            Statistical feature vector
        """
        if isinstance(packets, pd.DataFrame):
            sizes = packets['size'].values if 'size' in packets.columns else packets.iloc[:, 0].values
            timestamps = packets['timestamp'].values if 'timestamp' in packets.columns else packets.iloc[:, 1].values
            directions = packets['direction'].values if 'direction' in packets.columns else packets.iloc[:, 2].values
        else:
            sizes = packets[:, 0]
            timestamps = packets[:, 1]
            directions = packets[:, 2]

        features = []

        # Packet size statistics
        if len(sizes) > 0:
            features.extend([
                np.mean(sizes),
                np.median(sizes),
                np.std(sizes),
                np.min(sizes),
                np.max(sizes),
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Inter-arrival time statistics
        if len(timestamps) > 1:
            inter_arrival_times = np.diff(timestamps)
            features.extend([
                np.mean(inter_arrival_times),
                np.median(inter_arrival_times),
                np.std(inter_arrival_times),
                np.min(inter_arrival_times),
                np.max(inter_arrival_times),
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Flow-level features
        features.extend([
            timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,  # Flow duration
            np.sum(sizes),  # Total bytes
            len(sizes),  # Packet count
            np.sum(directions) / len(directions) if len(directions) > 0 else 0.5,  # Downstream ratio
        ])

        return np.array(features, dtype=np.float32)

    def extract_temporal_patterns(self, packets: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Extract temporal patterns from packet sequence.

        Creates a temporal representation suitable for CNN and RNN processing,
        preserving sequential information critical for encrypted traffic analysis.

        Args:
            packets: Packet metadata

        Returns:
            Temporal feature matrix of shape (max_packets, num_temporal_features)
        """
        packet_features = self.extract_packet_features(packets)

        # Compute cumulative statistics over time
        temporal_features = np.zeros((self.max_packets, 5))

        for i in range(min(len(packets), self.max_packets)):
            if i > 0:
                # Cumulative bytes up to packet i
                temporal_features[i, 0] = np.sum(packet_features[:i+1, 0])

                # Cumulative time
                temporal_features[i, 1] = np.sum(packet_features[:i+1, 1])

                # Bytes per second up to this point
                if temporal_features[i, 1] > 0:
                    temporal_features[i, 2] = temporal_features[i, 0] / temporal_features[i, 1]

                # Packet rate
                temporal_features[i, 3] = (i + 1) / (temporal_features[i, 1] + 1e-6)

                # Direction changes
                temporal_features[i, 4] = np.sum(
                    np.abs(np.diff(packet_features[:i+1, 2]))
                )

        # Combine packet features and temporal patterns
        combined = np.concatenate([packet_features, temporal_features], axis=1)

        return combined

    def fit(self, flows: List[Union[pd.DataFrame, np.ndarray]]) -> 'FlowFeatureExtractor':
        """
        Fit feature scaler on training data.

        Args:
            flows: List of packet sequences (one per flow)

        Returns:
            self
        """
        # Extract features from all flows
        all_features = []
        for flow in flows:
            stat_features = self.extract_flow_statistics(flow)
            all_features.append(stat_features)

        all_features = np.array(all_features)

        # Fit scaler
        self.scaler.fit(all_features)
        self.fitted = True

        return self

    def transform(self, flows: List[Union[pd.DataFrame, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform flows to feature representations.

        Args:
            flows: List of packet sequences

        Returns:
            Tuple of (temporal_features, statistical_features)
            - temporal_features: (num_flows, max_packets, num_temporal_features)
            - statistical_features: (num_flows, num_statistical_features)
        """
        if not self.fitted:
            warnings.warn("Scaler not fitted. Call fit() first or use fit_transform().")

        temporal_features = []
        statistical_features = []

        for flow in flows:
            # Extract temporal patterns for RNN/CNN
            temporal = self.extract_temporal_patterns(flow)
            temporal_features.append(temporal)

            # Extract statistical features
            stats = self.extract_flow_statistics(flow)
            statistical_features.append(stats)

        temporal_features = np.array(temporal_features, dtype=np.float32)
        statistical_features = np.array(statistical_features, dtype=np.float32)

        # Normalize statistical features
        if self.fitted:
            statistical_features = self.scaler.transform(statistical_features)

        return temporal_features, statistical_features

    def fit_transform(self, flows: List[Union[pd.DataFrame, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler and transform flows in one step.

        Args:
            flows: List of packet sequences

        Returns:
            Tuple of (temporal_features, statistical_features)
        """
        self.fit(flows)
        return self.transform(flows)


def preprocess_dataset(
    data: pd.DataFrame,
    label_column: str = 'label',
    flow_column: str = 'flow_id',
    normalize: bool = True,
    handle_missing: str = 'drop'
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Preprocess encrypted traffic dataset.

    Args:
        data: Raw dataset
        label_column: Name of label column
        flow_column: Name of flow identifier column
        normalize: Whether to normalize features
        handle_missing: How to handle missing values ('drop', 'fill')

    Returns:
        Tuple of (preprocessed_data, labels, label_encoder)
    """
    # Handle missing values
    if handle_missing == 'drop':
        data = data.dropna()
    elif handle_missing == 'fill':
        data = data.fillna(data.median())

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data[label_column])

    # Remove non-feature columns
    feature_columns = [col for col in data.columns
                      if col not in [label_column, flow_column]]
    features = data[feature_columns]

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        features_normalized = pd.DataFrame(
            scaler.fit_transform(features),
            columns=feature_columns,
            index=features.index
        )
        return features_normalized, labels, label_encoder

    return features, labels, label_encoder


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train/validation/test sets with stratification.

    Args:
        X: Feature array
        y: Label array
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified sampling (maintains class distribution)

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        ...     X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        ... )
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # First split: separate test set
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp if stratify else None
    )

    print(f"Dataset split:")
    print(f"  Training: {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Validation: {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  Testing: {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")

    if stratify:
        print(f"\nClass distribution (training):")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = 'smote',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance imbalanced dataset using various strategies.

    Args:
        X: Feature array
        y: Label array
        strategy: Balancing strategy ('smote', 'random_oversample', 'random_undersample')
        random_state: Random seed

    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if strategy == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif strategy == 'random_oversample':
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == 'random_undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    X_balanced, y_balanced = sampler.fit_resample(X, y)

    print(f"Dataset balanced using {strategy}:")
    print(f"  Original size: {len(y)}")
    print(f"  Balanced size: {len(y_balanced)}")

    return X_balanced, y_balanced
