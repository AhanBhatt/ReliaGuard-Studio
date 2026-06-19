from __future__ import annotations

import numpy as np


def inject_missingness(X: np.ndarray, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < rate
    X_missing = X.copy()
    column_means = np.nanmean(X_missing, axis=0)
    for col_idx in range(X.shape[1]):
        X_missing[mask[:, col_idx], col_idx] = column_means[col_idx]
    return X_missing


def inject_noise(X: np.ndarray, noise_scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_scale, size=X.shape)
    return np.clip(X + noise, 0.0, 1.0)
