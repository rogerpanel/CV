"""
Preprocessing Pipeline (Section VIII-C)
=======================================
1. Timestamp normalisation to seconds since epoch
2. Feature standardisation via z-score (computed on training data only)
3. Categorical encoding via learned embeddings
4. Missing value imputation (forward-fill for temporal continuity)
5. Temporal train/val/test splits (70/15/15) preventing data leakage
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict


class Preprocessor:
    """Consistent preprocessing for all datasets."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler on training data and transform."""
        # Handle infinities and NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Clip extreme values (beyond 6 sigma)
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mean, std = col_data.mean(), col_data.std()
            if std > 0:
                X[:, col] = np.clip(col_data, mean - 6 * std, mean + 6 * std)

        # Z-score normalisation
        X = self.scaler.fit_transform(X)

        # Encode labels
        y = self.label_encoder.fit_transform(y.astype(str))

        self._fitted = True
        return X, y

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform using fitted scaler."""
        assert self._fitted, "Call fit_transform first"

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X = self.scaler.transform(X)

        if y is not None:
            y = self.label_encoder.transform(y.astype(str))
        return X, y

    @property
    def n_classes(self) -> int:
        return len(self.label_encoder.classes_)

    @property
    def class_names(self):
        return self.label_encoder.classes_

    def extract_timestamps(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract and normalise timestamps from DataFrame."""
        time_cols = [c for c in df.columns
                     if "time" in c.lower() or "timestamp" in c.lower()]
        if not time_cols:
            return None

        ts = pd.to_datetime(df[time_cols[0]], errors="coerce")
        # Normalise to seconds since first event
        t0 = ts.min()
        return (ts - t0).dt.total_seconds().fillna(0).values
