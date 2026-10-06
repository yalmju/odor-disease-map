"""Nonnegative distance fusion. All scaling is fitted on training data only."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .features import CHANNELS, FEATURE_VERSION


def check_channels(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(CHANNELS) or not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"Expected finite nonnegative n x {len(CHANNELS)} channels")
    return x


class DistanceFusion:
    def __init__(self, ridge=0.05):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.ridge = float(ridge)

    def fit(self, x, y):
        x = check_channels(x); y = np.asarray(y, dtype=float)
        if y.shape != (len(x),) or len(x) < 2 or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Need at least two finite nonnegative target distances aligned to rows")
        if np.any((x == 0).all(axis=1) & (y != 0)):
            raise ValueError("Identical representations have nonzero targets; review repeats/representation before fitting")
        scale = np.quantile(x, 0.9, axis=0)
        scale[scale < 1e-8] = 1
        a = (x / scale) ** 2
        fit = lsq_linear(np.vstack([a / np.sqrt(len(x)), np.sqrt(self.ridge) * np.eye(len(CHANNELS))]),
                         np.r_[y ** 2 / np.sqrt(len(x)), np.zeros(len(CHANNELS))], bounds=(0, np.inf), tol=1e-10, max_iter=1000)
        if not fit.success:
            raise RuntimeError(f"Distance fusion did not converge: {fit.message}")
        self.scale_, self.weights_ = scale, fit.x
        self.train_n_ = len(x)
        return self

    def predict(self, x):
        if not hasattr(self, "weights_"):
            raise ValueError("Model is not fitted")
        x = check_channels(x)
        return np.sqrt(((x / self.scale_) ** 2) @ self.weights_)

    def save(self, path):
        if not hasattr(self, "weights_"):
            raise ValueError("Model is not fitted")
        payload = {"schema_version": 1, "feature_version": FEATURE_VERSION, "channels": list(CHANNELS), "ridge": self.ridge,
                   "train_n": self.train_n_, "scale": self.scale_.tolist(), "weights": self.weights_.tolist()}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("schema_version") != 1 or p.get("feature_version") != FEATURE_VERSION or p.get("channels") != list(CHANNELS):
            raise ValueError("Incompatible feature or model schema")
        model = cls(p["ridge"])
        model.scale_, model.weights_ = np.asarray(p["scale"], float), np.asarray(p["weights"], float)
        if any(v.shape != (len(CHANNELS),) or not np.isfinite(v).all() for v in [model.scale_, model.weights_]) or (model.scale_ <= 0).any() or (model.weights_ < 0).any():
            raise ValueError("Invalid serialized model coefficients")
        model.train_n_ = int(p["train_n"])
        return model
