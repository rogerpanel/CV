"""
Data preprocessing pipeline.

Implements the preprocessing described in Section 5.1:
  - z-score normalization
  - Learned embeddings for categorical features
  - Forward-fill imputation
  - Temporal 70/15/15 split (preserving time ordering)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Optional, List


def _identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], Optional[str]]:
    """Identify numeric, categorical, and label columns."""
    # Common label column names in IDS datasets
    label_candidates = [
        "label", "Label", "attack_cat", "Attack", "class",
        "attack_type", "classification", "target", "Category",
    ]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        # Use last column as label
        label_col = df.columns[-1]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in categorical_cols:
        categorical_cols.remove(label_col)

    return numeric_cols, categorical_cols, label_col


def preprocess_dataset(df: pd.DataFrame,
                       max_samples: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    """Preprocess a raw dataset DataFrame.

    Steps:
      1. Identify feature and label columns
      2. Forward-fill missing values
      3. Encode categorical features via LabelEncoder (embeddings learned in model)
      4. z-score normalize numeric features
      5. Encode labels

    Args:
        df: Raw DataFrame
        max_samples: Subsample if set (for faster experimentation)

    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Label array [n_samples]
        label_encoder: Fitted LabelEncoder for labels
        scaler: Fitted StandardScaler for features
    """
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    numeric_cols, categorical_cols, label_col = _identify_columns(df)

    # Forward-fill then back-fill missing values
    df = df.ffill().bfill()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))

    # Process numeric features: replace inf, fill NaN, z-score
    X_num = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    # Encode categoricals as integers (model will use nn.Embedding)
    X_cat_list = []
    for col in categorical_cols:
        cat_le = LabelEncoder()
        X_cat_list.append(cat_le.fit_transform(df[col].astype(str)).reshape(-1, 1).astype(np.float32))

    if X_cat_list:
        X_cat = np.hstack(X_cat_list)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    print(f"  Preprocessed: {X.shape[0]:,} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes")

    return X, y, le, scaler


def temporal_split(X: np.ndarray, y: np.ndarray,
                   ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
                   ) -> Tuple:
    """Temporal split preserving time ordering (no shuffling).

    Args:
        X, y: Full dataset
        ratios: (train, val, test) ratios summing to 1.0

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))

    return (
        X[:n_train], y[:n_train],
        X[n_train:n_val], y[n_train:n_val],
        X[n_val:], y[n_val:],
    )


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for network traffic time series.

    Supports both single-event and sequence-based access.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int = 1,
                 timestamps: Optional[np.ndarray] = None):
        """
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            seq_len: Sequence length (1 for single-event mode)
            timestamps: Optional real timestamps [n_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = seq_len

        if timestamps is not None:
            self.timestamps = torch.FloatTensor(timestamps)
        else:
            # Synthetic uniform timestamps
            self.timestamps = torch.arange(len(X), dtype=torch.float32)

        self.n_features = X.shape[1]

    def __len__(self):
        if self.seq_len == 1:
            return len(self.X)
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        if self.seq_len == 1:
            return {
                "x": self.X[idx],
                "y": self.y[idx],
                "t": self.timestamps[idx],
            }
        # Sequence mode
        end = idx + self.seq_len
        return {
            "x": self.X[idx:end],
            "y": self.y[end - 1],  # Label of last event
            "t": self.timestamps[idx:end],
            "event_types": self.y[idx:end],  # For point process
        }
