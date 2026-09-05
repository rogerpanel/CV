"""A small 1D-CNN on the raw 1440-minute series, wrapped as an sklearn estimator.

Requires the optional ``torch`` dependency (``pip install -e .[cnn]``).  The
architecture is deliberately modest: three conv blocks with pooling, global
average pooling and a linear head.  Training is fixed-epoch (no early stopping
on the test fold), and the random seed is set for torch and numpy.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Conv1DClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, seed: int = 0, n_classes: int = 2, epochs: int = 30, lr: float = 1e-3, batch_size: int = 32, hidden: int = 32):
        self.seed = seed
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden = hidden

    # ------------------------------------------------------------------ torch
    def _build(self, n_in: int):
        import torch
        import torch.nn as nn

        h = self.hidden
        return nn.Sequential(
            nn.Conv1d(1, h, kernel_size=15, padding=7), nn.BatchNorm1d(h), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(h, h * 2, kernel_size=9, padding=4), nn.BatchNorm1d(h * 2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(h * 2, h * 2, kernel_size=5, padding=2), nn.BatchNorm1d(h * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(h * 2, self.n_classes if self.n_classes > 2 else 1),
        )

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.net_ = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        Xt = torch.from_numpy(X).unsqueeze(1)
        if self.n_classes > 2:
            yt = torch.from_numpy(y.astype(np.int64))
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
            pos = float(y.mean())
            pw = torch.tensor([(1 - pos) / max(pos, 1e-6)])
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        n = len(Xt)
        g = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                out = self.net_(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch

        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.from_numpy(X).unsqueeze(1))
            if self.n_classes > 2:
                p = torch.softmax(out, dim=1).numpy()
            else:
                p1 = torch.sigmoid(out).numpy().ravel()
                p = np.column_stack([1 - p1, p1])
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
