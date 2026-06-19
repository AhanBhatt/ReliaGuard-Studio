from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray | list[float], statistic: Callable[[np.ndarray], float] | None = None, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    stat_fn = statistic or np.mean
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=data.size, replace=True)
        estimates.append(stat_fn(sample))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def cluster_bootstrap_ci(frame: pd.DataFrame, value_col: str, cluster_col: str, statistic: Callable[[pd.DataFrame], float] | None = None, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    stat_fn = statistic or (lambda df: float(df[value_col].mean()))
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    grouped = {cluster: subframe for cluster, subframe in frame.groupby(cluster_col)}
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = pd.concat([grouped[cluster] for cluster in sampled_clusters], ignore_index=True)
        estimates.append(stat_fn(sampled))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def summarize_rate(frame: pd.DataFrame, group_cols: list[str], value_col: str, cluster_col: str | None = None, seed: int = 42) -> pd.DataFrame:
    rows = []
    for group_values, subframe in frame.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values, strict=True))
        row["n"] = int(len(subframe))
        row["rate"] = float(subframe[value_col].mean())
        if cluster_col and cluster_col in subframe.columns:
            low, high = cluster_bootstrap_ci(subframe, value_col, cluster_col, n_boot=400, seed=seed)
        else:
            low, high = bootstrap_ci(subframe[value_col].to_numpy(dtype=float), n_boot=400, seed=seed)
        row["ci_low"] = low
        row["ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(first: np.ndarray | list[float], second: np.ndarray | list[float]) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)
